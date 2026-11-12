import json
import re
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from app.version import __version__


LATEST_RELEASE_API = "https://api.github.com/repos/elpirinpi/AiClip/releases/latest"
VERSION_PATTERN = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)$")


class UpdateCheckError(RuntimeError):
    """Error legible al comprobar si existe una actualización."""


def _version_parts(value: str) -> tuple[int, int, int]:
    match = VERSION_PATTERN.fullmatch(value)
    if match is None:
        raise UpdateCheckError(
            "GitHub publicó una versión con un formato que AiClip no reconoce."
        )
    return tuple(int(part) for part in match.groups())


def _github_release_url(value: object) -> str:
    if not isinstance(value, str):
        raise UpdateCheckError("GitHub devolvió un enlace de actualización inválido.")
    parsed = urlparse(value)
    if (
        parsed.scheme != "https"
        or parsed.netloc != "github.com"
        or not parsed.path.startswith("/elpirinpi/AiClip/releases/")
    ):
        raise UpdateCheckError("GitHub devolvió un enlace de actualización inválido.")
    return value


def check_for_updates() -> dict[str, object]:
    request = Request(
        LATEST_RELEASE_API,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": f"AiClip/{__version__}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urlopen(request, timeout=5) as response:
            release = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        if exc.code == 403:
            raise UpdateCheckError(
                "GitHub limitó temporalmente las consultas. Inténtalo más tarde."
            ) from exc
        raise UpdateCheckError(
            "GitHub no pudo proporcionar la información de versiones."
        ) from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise UpdateCheckError(
            "No se pudo conectar con GitHub. Comprueba Internet e inténtalo de nuevo."
        ) from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise UpdateCheckError(
            "GitHub devolvió una respuesta de actualización inválida."
        ) from exc

    if not isinstance(release, dict):
        raise UpdateCheckError("GitHub devolvió una respuesta de actualización inválida.")
    tag = release.get("tag_name")
    if not isinstance(tag, str):
        raise UpdateCheckError("GitHub no devolvió el número de la última versión.")
    current_version = _version_parts(__version__)
    latest_version = _version_parts(tag)
    release_url = _github_release_url(release.get("html_url"))
    assets = release.get("assets", [])
    if not isinstance(assets, list):
        raise UpdateCheckError("GitHub devolvió una lista de descargas inválida.")
    installer_url = release_url
    installer_available = False
    for asset in assets:
        if isinstance(asset, dict) and asset.get("name") == "AiClip-Setup.msi":
            installer_url = _github_release_url(asset.get("browser_download_url"))
            installer_available = True
            break

    return {
        "current_version": __version__,
        "latest_version": tag.removeprefix("v"),
        "update_available": latest_version > current_version,
        "release_url": release_url,
        "download_url": installer_url,
        "installer_available": installer_available,
    }
