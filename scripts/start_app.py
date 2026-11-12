import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

import uvicorn

from app.config import (
    APP_DATA_DIR,
    DEFAULT_CONFIG_PATH,
    DEFAULT_OUTPUT_FOLDER,
    PROJECT_ROOT,
    ConfigError,
    load_config,
)
from app.main import app


HOST = "127.0.0.1"
PORT = 8765
APP_URL = f"http://{HOST}:{PORT}/"


def _initialize_config() -> None:
    if DEFAULT_CONFIG_PATH.is_file():
        return
    bundle_root = Path(getattr(sys, "_MEIPASS", PROJECT_ROOT))
    template_path = bundle_root / "config.example.json"
    try:
        settings = json.loads(template_path.read_text(encoding="utf-8"))
        settings["output_folder"] = str(DEFAULT_OUTPUT_FOLDER)
        APP_DATA_DIR.mkdir(parents=True, exist_ok=True)
        DEFAULT_CONFIG_PATH.write_text(
            json.dumps(settings, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        raise ConfigError("No se pudo crear la configuración inicial de AiClip.") from exc


def _open_browser_when_ready() -> None:
    request = urllib.request.Request(
        f"http://{HOST}:{PORT}/api/health",
        headers={"Accept": "application/json"},
    )
    for _ in range(60):
        try:
            with urllib.request.urlopen(request, timeout=1) as response:
                if response.status == 200:
                    if not webbrowser.open(APP_URL):
                        print(f"Abre esta dirección en tu navegador: {APP_URL}")
                    return
        except (OSError, urllib.error.URLError):
            time.sleep(0.5)
    print(f"No se pudo abrir el navegador automáticamente. Abre: {APP_URL}")


def main() -> int:
    try:
        _initialize_config()
        config = load_config()
        config.output_folder.mkdir(parents=True, exist_ok=True)
    except ConfigError as exc:
        print(f"ERROR: {exc}")
        return 1
    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root:
        os.environ["PATH"] = str(bundle_root) + os.pathsep + os.environ.get("PATH", "")
    print("AiClip se ejecutará solo en este PC.")
    print(f"Abre {APP_URL} en tu navegador para empezar.")
    threading.Thread(target=_open_browser_when_ready, daemon=True).start()
    try:
        uvicorn.run(
            app,
            host=HOST,
            port=PORT,
            log_level="warning",
            access_log=False,
        )
    except OSError as exc:
        print(
            "ERROR: No se pudo iniciar el servidor local. "
            "Comprueba si el puerto 8765 está ocupado."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
