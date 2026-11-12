import argparse
import sys
import tempfile
import uuid
from pathlib import Path

from app.config import ConfigError, load_config
from app.db import (
    DEFAULT_DATABASE_PATH,
    DatabaseError,
    get_clips,
    get_transcript_words,
)
from app.pipeline.ingest import IngestError, resolve_video_source
from app.pipeline.render import RenderError, render_clip
from app.pipeline.segment import Clip
from app.pipeline.thumbnail import ThumbnailError, create_thumbnail
from app.providers.base import ProviderError, TranscriptWord
from app.providers.gemini_llm import create_analyzer


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Corta clips guardados en SQLite y quema subtítulos ASS."
    )
    parser.add_argument(
        "source",
        help="La misma ruta o enlace usado al crear la lista de clips.",
    )
    parser.add_argument(
        "--database",
        help="Ruta a la base SQLite (por defecto, data/clipper.sqlite3).",
    )
    parser.add_argument(
        "--clip-id",
        type=int,
        action="append",
        help="ID de un clip que quieres renderizar. Se puede repetir.",
    )
    parser.add_argument(
        "--style",
        choices=("classic", "impact", "both"),
        default="both",
        help="Estilo de subtítulo que se renderiza (por defecto: ambos).",
    )
    parser.add_argument(
        "--reframe",
        choices=("center", "face", "blur"),
        default="center",
        help=(
            "Encuadre vertical: centrado, seguimiento de cara con fallback "
            "desenfocado o fondo desenfocado (por defecto: center)."
        ),
    )
    args = parser.parse_args()

    try:
        config = load_config()
        database = Path(args.database) if args.database else DEFAULT_DATABASE_PATH
        rows = get_clips(args.source, args.clip_id, database)
        if not rows:
            raise DatabaseError(
                "No se encontraron clips para ese origen. Comprueba el origen y los IDs."
            )
        words = get_transcript_words(args.source, database)
        if not words:
            raise DatabaseError(
                "No se encontró la transcripción con tiempos por palabra. "
                "Vuelve a ejecutar la Fase 1 para este origen."
            )

        styles = ("classic", "impact") if args.style == "both" else (args.style,)
        output_dir = config.output_folder / "clips"
        output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f"render-{uuid.uuid4().hex[:8]}-",
            dir=config.output_folder,
        ) as temporary:
            work_dir = Path(temporary)
            video_path = resolve_video_source(args.source, work_dir / "source")
            analyzer = create_analyzer(config)
            thumbnail_dir = config.output_folder / "thumbnails"
            thumbnail_dir.mkdir(parents=True, exist_ok=True)
            rendered: list[Path] = []
            thumbnails: list[Path] = []
            try:
                for row in rows:
                    clip = Clip(
                        start=row["start"],
                        end=row["end"],
                        title=row["title"],
                        reason=row["reason"],
                        score=row["score"],
                    )
                    thumbnail_path = thumbnail_dir / f"clip_{row['id']:05d}.png"
                    thumbnails.append(
                        create_thumbnail(video_path, clip, words, analyzer, thumbnail_path)
                    )
                    for style in styles:
                        output_path = output_dir / (
                            f"clip_{row['id']:05d}_{args.reframe}_{style}.mp4"
                        )
                        rendered.append(
                            render_clip(
                                video_path,
                                clip,
                                words,
                                output_path,
                                work_dir,
                                style,
                                config,
                                reframe_mode=args.reframe,
                            )
                        )
            finally:
                analyzer.close()
    except (
        ConfigError,
        DatabaseError,
        IngestError,
        RenderError,
        ThumbnailError,
        ProviderError,
    ) as exc:
        print(f"ERROR: {exc}")
        return 1
    except OSError as exc:
        if getattr(exc, "errno", None) == 28:
            print("ERROR: El disco está lleno. Libera espacio y vuelve a intentarlo.")
        else:
            print("ERROR: No se pudo leer o guardar uno de los archivos.")
        return 1

    for path in rendered:
        print(f"Clip creado: {path}")
    for path in thumbnails:
        print(f"Miniatura creada: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
