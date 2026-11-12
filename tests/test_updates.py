import io
import json
from urllib.error import URLError

import pytest

import app.updates as updates
from app.updates import UpdateCheckError, check_for_updates


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def _release(tag: str, *, installer: bool = False) -> dict[str, object]:
    release: dict[str, object] = {
        "tag_name": tag,
        "html_url": "https://github.com/elpirinpi/AiClip/releases/tag/v0.1.0",
        "assets": [],
    }
    if installer:
        release["assets"] = [
            {
                "name": "AiClip-Setup.msi",
                "browser_download_url": (
                    "https://github.com/elpirinpi/AiClip/releases/download/"
                    "v0.2.0/AiClip-Setup.msi"
                ),
            }
        ]
    return release


def _mock_release(monkeypatch, release: dict[str, object]) -> None:
    payload = json.dumps(release).encode("utf-8")

    def fake_urlopen(request, timeout):
        assert request.full_url == updates.LATEST_RELEASE_API
        assert request.get_header("User-agent") == f"AiClip/{updates.__version__}"
        assert timeout == 5
        return FakeResponse(payload)

    monkeypatch.setattr(updates, "urlopen", fake_urlopen)


def test_update_check_reports_current_version_without_download(
    monkeypatch,
) -> None:
    _mock_release(monkeypatch, _release(f"v{updates.__version__}"))

    result = check_for_updates()

    assert result["current_version"] == updates.__version__
    assert result["latest_version"] == updates.__version__
    assert result["update_available"] is False
    assert result["installer_available"] is False
    assert result["download_url"] == result["release_url"]


def test_update_check_returns_github_msi_link_for_newer_release(
    monkeypatch,
) -> None:
    _mock_release(monkeypatch, _release("v0.2.0", installer=True))

    result = check_for_updates()

    assert result["latest_version"] == "0.2.0"
    assert result["update_available"] is True
    assert result["installer_available"] is True
    assert result["download_url"].endswith("/v0.2.0/AiClip-Setup.msi")


@pytest.mark.parametrize(
    "release",
    [
        _release("latest"),
        {
            "tag_name": "v0.2.0",
            "html_url": "https://example.com/fake",
            "assets": [],
        },
        _release("v0.2.0", installer=True)
        | {
            "assets": [
                {
                    "name": "AiClip-Setup.msi",
                    "browser_download_url": "https://example.com/fake.msi",
                }
            ]
        },
    ],
)
def test_update_check_rejects_malformed_version_or_external_links(
    monkeypatch, release: dict[str, object]
) -> None:
    _mock_release(monkeypatch, release)

    with pytest.raises(UpdateCheckError):
        check_for_updates()


def test_update_check_reports_network_failures_in_spanish(monkeypatch) -> None:
    def fail_urlopen(_request, timeout):
        raise URLError("offline")

    monkeypatch.setattr(updates, "urlopen", fail_urlopen)

    with pytest.raises(UpdateCheckError, match="No se pudo conectar con GitHub"):
        check_for_updates()
