import argparse
import json
import sqlite3
import sys
import tempfile
import uuid
from pathlib import Path

from app.config import ConfigError, load_config
from app.db import DEFAULT_DATABASE_PATH, DatabaseError, save_results
from app.pipeline.analyze import AnalysisError, analyze_transcript
from app.pipeline.audio import AudioProcessingError, create_audio_chunks
from app.pipeline.ingest import IngestError, resolve_video_source
from app.pipeline.segment import snap_clips_to_sentences
from app.pipeline.transcribe import transcribe_chunks
from app.providers.base import ProviderError
from app.providers.gemini_llm import create_analyzer
from app.providers.groq_stt import create_transcriber


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Transcribe un vídeo y guarda en SQLite los momentos destacados."
    )
    parser.add_argument("source", help="Ruta a un vídeo local o enlace de YouTube.")
    parser.add_argument(
        "--database",
        help="Ruta opcional para la base de datos SQLite.",
    )
    parser.add_argument(
        "--language",
        help="Código de idioma de la pista de audio (por ejemplo, es o en).",
    )
    args = parser.parse_args()

    try:
        config = load_config()
        if config.transcription_fallback or config.analysis_fallback:
            raise ProviderError(
                "Hay un proveedor de fallback configurado, pero no está habilitado."
            )
        config.output_folder.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f"job-{uuid.uuid4().hex[:8]}-",
            dir=config.output_folder,
        ) as temporary:
            work_dir = Path(temporary)
            video_path = resolve_video_source(args.source, work_dir / "source")
            chunks = create_audio_chunks(video_path, work_dir / "audio", config)
            words = transcribe_chunks(
                chunks,
                create_transcriber(config),
                language=args.language,
            )
            if not words:
                raise ProviderError(
                    "Groq no reconoció palabras en el audio. Comprueba el idioma y el sonido."
                )
            analyzer = create_analyzer(config)
            try:
                candidates = analyze_transcript(words, analyzer)
            finally:
                analyzer.close()
            clips, warnings = snap_clips_to_sentences(candidates, words)
            database = args.database or DEFAULT_DATABASE_PATH
            saved = save_results(args.source, clips, words, Path(database))
    except (
        ConfigError,
        IngestError,
        AudioProcessingError,
        ProviderError,
        AnalysisError,
        DatabaseError,
    ) as exc:
        print(f"ERROR: {exc}")
        return 1
    except sqlite3.Error:
        print("ERROR: No se pudieron guardar los clips en SQLite.")
        return 1
    except OSError as exc:
        if getattr(exc, "errno", None) == 28:
            print("ERROR: El disco está lleno. Libera espacio y vuelve a intentarlo.")
        else:
            print("ERROR: No se pudo leer o guardar uno de los archivos.")
        return 1

    for warning in warnings:
        print(f"AVISO: {warning}")
    print(f"Se guardaron {saved} clips en {database}.")
    print(json.dumps(
        [
            {
                "start": clip.start,
                "end": clip.end,
                "title": clip.title,
                "reason": clip.reason,
                "score": clip.score,
            }
            for clip in clips
        ],
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
