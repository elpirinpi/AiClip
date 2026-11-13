import shutil
import sys
from pathlib import Path


def find_ffmpeg(value: str) -> str | None:
    executable = shutil.which(value)
    if executable:
        return str(Path(executable).resolve())
    configured_path = Path(value)
    if configured_path.is_file():
        return str(configured_path.resolve())
    bundle_dir = getattr(sys, "_MEIPASS", None)
    if bundle_dir:
        bundled_executable = Path(bundle_dir) / "ffmpeg.exe"
        if bundled_executable.is_file():
            return str(bundled_executable.resolve())
    return None
