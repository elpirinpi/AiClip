import sqlite3
import uuid
from pathlib import Path
from typing import Any

from app.config import APP_DATA_DIR
from app.pipeline.segment import Clip
from app.providers.base import TranscriptWord


DEFAULT_DATABASE_PATH = APP_DATA_DIR / "data" / "clipper.sqlite3"


class DatabaseError(RuntimeError):
    """Error legible al guardar resultados en SQLite."""


def connect_database(path: Path | None = None) -> sqlite3.Connection:
    database_path = path or DEFAULT_DATABASE_PATH
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS clips (
            id INTEGER PRIMARY KEY,
            source TEXT NOT NULL,
            start REAL NOT NULL,
            end REAL NOT NULL,
            title TEXT NOT NULL,
            reason TEXT NOT NULL,
            score INTEGER NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            CHECK (start >= 0),
            CHECK (end > start),
            CHECK (score BETWEEN 1 AND 10)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            input_path TEXT NOT NULL,
            status TEXT NOT NULL,
            progress INTEGER NOT NULL DEFAULT 0,
            error TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    columns = {
        row["name"] for row in connection.execute("PRAGMA table_info(clips)")
    }
    if "selected" not in columns:
        connection.execute(
            "ALTER TABLE clips ADD COLUMN selected INTEGER NOT NULL DEFAULT 1"
        )
    if "updated_at" not in columns:
        connection.execute(
            "ALTER TABLE clips ADD COLUMN updated_at TEXT"
        )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS transcript_words (
            source TEXT NOT NULL,
            position INTEGER NOT NULL,
            start REAL NOT NULL,
            end REAL NOT NULL,
            text TEXT NOT NULL,
            PRIMARY KEY (source, position),
            CHECK (start >= 0),
            CHECK (end >= start)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS clip_exports (
            id INTEGER PRIMARY KEY,
            job_id TEXT NOT NULL,
            clip_id INTEGER NOT NULL,
            video_path TEXT NOT NULL,
            thumbnail_path TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_clip_exports_latest
        ON clip_exports (job_id, clip_id, id DESC)
        """
    )


def save_clips(
    source: str, clips: list[Clip], path: Path | None = None
) -> int:
    connection = connect_database(path)
    try:
        initialize_database(connection)
        with connection:
            connection.executemany(
                """
                INSERT INTO clips (source, start, end, title, reason, score)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        source,
                        clip.start,
                        clip.end,
                        clip.title,
                        clip.reason,
                        clip.score,
                    )
                    for clip in clips
                ],
            )
        return len(clips)
    except sqlite3.OperationalError as exc:
        if "full" in str(exc).casefold():
            raise DatabaseError(
                "El disco está lleno. Libera espacio y vuelve a intentarlo."
            ) from exc
        raise DatabaseError("No se pudieron guardar los clips en SQLite.") from exc
    finally:
        connection.close()


def save_results(
    source: str,
    clips: list[Clip],
    words: list[TranscriptWord],
    path: Path | None = None,
) -> int:
    connection = connect_database(path)
    try:
        initialize_database(connection)
        with connection:
            connection.execute("DELETE FROM clips WHERE source = ?", (source,))
            connection.execute(
                "DELETE FROM transcript_words WHERE source = ?", (source,)
            )
            connection.executemany(
                """
                INSERT INTO transcript_words (source, position, start, end, text)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (source, index, word.start, word.end, word.text)
                    for index, word in enumerate(words)
                ],
            )
            connection.executemany(
                """
                INSERT INTO clips (source, start, end, title, reason, score)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        source,
                        clip.start,
                        clip.end,
                        clip.title,
                        clip.reason,
                        clip.score,
                    )
                    for clip in clips
                ],
            )
        return len(clips)
    except sqlite3.OperationalError as exc:
        if "full" in str(exc).casefold():
            raise DatabaseError(
                "El disco está lleno. Libera espacio y vuelve a intentarlo."
            ) from exc
        raise DatabaseError("No se pudieron guardar los resultados en SQLite.") from exc
    finally:
        connection.close()


def get_clips(source: str, clip_ids: list[int] | None = None, path: Path | None = None):
    connection = connect_database(path)
    try:
        initialize_database(connection)
        query = (
            "SELECT id, source, start, end, title, reason, score "
            "FROM clips WHERE source = ?"
        )
        parameters: list[object] = [source]
        if clip_ids:
            placeholders = ",".join("?" for _ in clip_ids)
            query += f" AND id IN ({placeholders})"
            parameters.extend(clip_ids)
        query += " ORDER BY score DESC, id"
        rows = connection.execute(query, parameters).fetchall()
        return rows
    except sqlite3.Error as exc:
        raise DatabaseError("No se pudieron leer los clips de SQLite.") from exc
    finally:
        connection.close()


def get_transcript_words(
    source: str, path: Path | None = None
) -> list[TranscriptWord]:
    connection = connect_database(path)
    try:
        initialize_database(connection)
        rows = connection.execute(
            """
            SELECT start, end, text
            FROM transcript_words
            WHERE source = ?
            ORDER BY position
            """,
            (source,),
        ).fetchall()
        return [
            TranscriptWord(start=row["start"], end=row["end"], text=row["text"])
            for row in rows
        ]
    except sqlite3.Error as exc:
        raise DatabaseError("No se pudo leer la transcripción de SQLite.") from exc
    finally:
        connection.close()


def create_job(job_id: str, source: str, input_path: str, path: Path | None = None) -> None:
    connection = connect_database(path)
    try:
        initialize_database(connection)
        with connection:
            connection.execute(
                """
                INSERT INTO jobs (id, source, input_path, status, progress)
                VALUES (?, ?, ?, 'queued', 0)
                """,
                (job_id, source, input_path),
            )
    except sqlite3.Error as exc:
        raise DatabaseError("No se pudo añadir el trabajo a la cola.") from exc
    finally:
        connection.close()


def update_job(
    job_id: str,
    *,
    status: str | None = None,
    progress: int | None = None,
    error: str | None = None,
    input_path: str | None = None,
    path: Path | None = None,
) -> None:
    fields: list[str] = []
    values: list[Any] = []
    if status is not None:
        fields.append("status = ?")
        values.append(status)
    if progress is not None:
        fields.append("progress = ?")
        values.append(progress)
    if input_path is not None:
        fields.append("input_path = ?")
        values.append(input_path)
    fields.append("error = ?")
    values.append(error)
    fields.append("updated_at = CURRENT_TIMESTAMP")
    if not fields:
        return
    connection = connect_database(path)
    try:
        initialize_database(connection)
        with connection:
            cursor = connection.execute(
                f"UPDATE jobs SET {', '.join(fields)} WHERE id = ?",
                [*values, job_id],
            )
            if cursor.rowcount == 0:
                raise DatabaseError("No se encontró el trabajo.")
    except sqlite3.Error as exc:
        raise DatabaseError("No se pudo actualizar el estado del trabajo.") from exc
    finally:
        connection.close()


def get_job(job_id: str, path: Path | None = None) -> sqlite3.Row | None:
    connection = connect_database(path)
    try:
        initialize_database(connection)
        return connection.execute(
            "SELECT * FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
    except sqlite3.Error as exc:
        raise DatabaseError("No se pudo leer el trabajo.") from exc
    finally:
        connection.close()


def list_jobs(path: Path | None = None) -> list[sqlite3.Row]:
    connection = connect_database(path)
    try:
        initialize_database(connection)
        return connection.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC"
        ).fetchall()
    except sqlite3.Error as exc:
        raise DatabaseError("No se pudo leer la cola de trabajos.") from exc
    finally:
        connection.close()


def list_queued_jobs(path: Path | None = None) -> list[sqlite3.Row]:
    connection = connect_database(path)
    try:
        initialize_database(connection)
        return connection.execute(
            "SELECT id, status FROM jobs WHERE status IN ('queued', 'render_queued') "
            "ORDER BY created_at, id"
        ).fetchall()
    except sqlite3.Error as exc:
        raise DatabaseError("No se pudo recuperar la cola de trabajos.") from exc
    finally:
        connection.close()


def retry_failed_job(job_id: str, path: Path | None = None) -> bool:
    connection = connect_database(path)
    try:
        initialize_database(connection)
        with connection:
            cursor = connection.execute(
                """
                UPDATE jobs
                SET status = 'queued', progress = 0, error = NULL,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ? AND status = 'failed'
                """,
                (job_id,),
            )
            return cursor.rowcount == 1
    except sqlite3.Error as exc:
        raise DatabaseError("No se pudo volver a poner el trabajo en la cola.") from exc
    finally:
        connection.close()


def delete_job_history(path: Path | None = None) -> int:
    connection = connect_database(path)
    try:
        initialize_database(connection)
        with connection:
            job_count = connection.execute(
                "SELECT COUNT(*) FROM jobs"
            ).fetchone()[0]
            connection.execute("DELETE FROM clip_exports")
            connection.execute(
                "DELETE FROM clips WHERE source IN (SELECT id FROM jobs)"
            )
            connection.execute(
                "DELETE FROM transcript_words WHERE source IN (SELECT id FROM jobs)"
            )
            connection.execute("DELETE FROM jobs")
        return int(job_count)
    except sqlite3.Error as exc:
        raise DatabaseError("No se pudo limpiar el historial de trabajos.") from exc
    finally:
        connection.close()


def get_clip(source: str, clip_id: int, path: Path | None = None) -> sqlite3.Row | None:
    connection = connect_database(path)
    try:
        initialize_database(connection)
        return connection.execute(
            """
            SELECT id, source, start, end, title, reason, score, selected
            FROM clips WHERE source = ? AND id = ?
            """,
            (source, clip_id),
        ).fetchone()
    except sqlite3.Error as exc:
        raise DatabaseError("No se pudo leer el clip.") from exc
    finally:
        connection.close()


def get_transcript_duration(source: str, path: Path | None = None) -> float:
    connection = connect_database(path)
    try:
        initialize_database(connection)
        row = connection.execute(
            "SELECT MAX(end) AS duration FROM transcript_words WHERE source = ?",
            (source,),
        ).fetchone()
        return float(row["duration"] or 0)
    except sqlite3.Error as exc:
        raise DatabaseError("No se pudo leer la duración de la transcripción.") from exc
    finally:
        connection.close()


def update_clip(
    source: str,
    clip_id: int,
    *,
    start: float,
    end: float,
    title: str,
    selected: bool,
    path: Path | None = None,
) -> None:
    connection = connect_database(path)
    try:
        initialize_database(connection)
        with connection:
            cursor = connection.execute(
                """
                UPDATE clips
                SET start = ?, end = ?, title = ?, selected = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ? AND source = ?
                """,
                (start, end, title, int(selected), clip_id, source),
            )
            if cursor.rowcount == 0:
                raise DatabaseError("No se encontró el clip.")
    except sqlite3.Error as exc:
        raise DatabaseError("No se pudo actualizar el clip.") from exc
    finally:
        connection.close()


def set_clip_selection(
    source: str, selected: bool, path: Path | None = None
) -> None:
    connection = connect_database(path)
    try:
        initialize_database(connection)
        with connection:
            connection.execute(
                "UPDATE clips SET selected = ? WHERE source = ?",
                (int(selected), source),
            )
    except sqlite3.Error as exc:
        raise DatabaseError("No se pudo actualizar la selección de clips.") from exc
    finally:
        connection.close()


def add_manual_clip(
    source: str,
    start: float,
    end: float,
    title: str,
    path: Path | None = None,
) -> int:
    connection = connect_database(path)
    try:
        initialize_database(connection)
        with connection:
            cursor = connection.execute(
                """
                INSERT INTO clips (source, start, end, title, reason, score, selected)
                VALUES (?, ?, ?, ?, 'Clip añadido manualmente.', 10, 1)
                """,
                (source, start, end, title),
            )
            return int(cursor.lastrowid)
    except sqlite3.Error as exc:
        raise DatabaseError("No se pudo crear el clip manual.") from exc
    finally:
        connection.close()


def save_clip_export(
    job_id: str,
    clip_id: int,
    video_path: Path,
    thumbnail_path: Path,
    path: Path | None = None,
) -> None:
    connection = connect_database(path)
    try:
        initialize_database(connection)
        with connection:
            connection.execute(
                """
                INSERT INTO clip_exports (
                    job_id, clip_id, video_path, thumbnail_path
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    job_id,
                    clip_id,
                    str(video_path),
                    str(thumbnail_path),
                ),
            )
    except sqlite3.Error as exc:
        raise DatabaseError("No se pudo guardar la ruta de exportación del clip.") from exc
    finally:
        connection.close()


def get_clip_export(
    job_id: str,
    clip_id: int,
    path: Path | None = None,
) -> sqlite3.Row | None:
    connection = connect_database(path)
    try:
        initialize_database(connection)
        return connection.execute(
            """
            SELECT video_path, thumbnail_path
            FROM clip_exports
            WHERE job_id = ? AND clip_id = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (job_id, clip_id),
        ).fetchone()
    except sqlite3.Error as exc:
        raise DatabaseError("No se pudo leer la exportación del clip.") from exc
    finally:
        connection.close()


def get_job_clips(job_id: str, path: Path | None = None) -> list[sqlite3.Row]:
    connection = connect_database(path)
    try:
        initialize_database(connection)
        return connection.execute(
            """
            SELECT id, source, start, end, title, reason, score, selected
            FROM clips WHERE source = ? ORDER BY score DESC, id
            """,
            (job_id,),
        ).fetchall()
    except sqlite3.Error as exc:
        raise DatabaseError("No se pudieron leer los clips del trabajo.") from exc
    finally:
        connection.close()
