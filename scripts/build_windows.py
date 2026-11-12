import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree


ROOT = Path(__file__).resolve().parents[1]
DIST_DIR = ROOT / "dist"
APP_NAME = "AiClip"
VERSION = "0.1.0"
UPGRADE_CODE = "85DF2B20-1095-4A82-AB2E-66BC0C159F3E"
WIX_NAMESPACE = "http://wixtoolset.org/schemas/v4/wxs"
ElementTree.register_namespace("", WIX_NAMESPACE)


def _find_ffmpeg_files() -> tuple[Path, Path, Path]:
    ffmpeg_value = shutil.which("ffmpeg")
    ffprobe_value = shutil.which("ffprobe")
    if not ffmpeg_value or not ffprobe_value:
        raise RuntimeError(
            "Se necesitan ffmpeg.exe y ffprobe.exe en PATH para compilar."
        )
    ffmpeg = Path(ffmpeg_value).resolve()
    ffprobe = Path(ffprobe_value).resolve()
    if ffmpeg.parent != ffprobe.parent:
        raise RuntimeError("FFmpeg y FFprobe deben estar en la misma carpeta.")
    license_path = ffmpeg.parent.parent / "LICENSE.txt"
    if not license_path.is_file():
        raise RuntimeError(
            "No se encontró LICENSE.txt junto al paquete estático de FFmpeg."
        )
    return ffmpeg, ffprobe, license_path


def _build_executable(
    ffmpeg: Path, ffprobe: Path, ffmpeg_license: Path
) -> Path:
    try:
        import PyInstaller.__main__
    except ImportError as exc:
        raise RuntimeError(
            "Falta PyInstaller. Instálalo con: python -m pip install -r requirements.txt"
        ) from exc

    separator = ";"
    build_args = [
        "--noconfirm",
        "--clean",
        "--onefile",
        "--console",
        "--name",
        APP_NAME,
        "--distpath",
        str(DIST_DIR),
        "--workpath",
        str(ROOT / "build" / "pyinstaller"),
        "--add-data",
        f"{ROOT / 'app' / 'web'}{separator}app/web",
        "--add-data",
        f"{ROOT / 'prompts'}{separator}prompts",
        "--add-data",
        f"{ROOT / 'config.example.json'}{separator}.",
        "--add-data",
        f"{ffmpeg_license}{separator}.",
        "--add-binary",
        f"{ffmpeg}{separator}.",
        "--add-binary",
        f"{ffprobe}{separator}.",
        "--collect-all",
        "mediapipe",
        "--collect-all",
        "cv2",
        str(ROOT / "scripts" / "start_app.py"),
    ]
    PyInstaller.__main__.run(build_args)
    executable = DIST_DIR / f"{APP_NAME}.exe"
    if not executable.is_file():
        raise RuntimeError("PyInstaller terminó sin generar AiClip.exe.")
    return executable


def _write_wix_source(
    executable: Path, ffmpeg_license: Path, path: Path
) -> None:
    wix = f"{{{WIX_NAMESPACE}}}"
    root = ElementTree.Element(f"{wix}Wix")
    package = ElementTree.SubElement(
        root,
        f"{wix}Package",
        {
            "Name": APP_NAME,
            "Manufacturer": APP_NAME,
            "Version": VERSION,
            "UpgradeCode": UPGRADE_CODE,
            "Scope": "perUser",
        },
    )
    ElementTree.SubElement(
        package,
        f"{wix}MajorUpgrade",
        {"DowngradeErrorMessage": "Ya hay instalada una versión más reciente de AiClip."},
    )
    ElementTree.SubElement(package, f"{wix}MediaTemplate", {"EmbedCab": "yes"})
    feature = ElementTree.SubElement(package, f"{wix}Feature", {"Id": "Main"})
    ElementTree.SubElement(feature, f"{wix}ComponentGroupRef", {"Id": "AppFiles"})

    local_app_data = ElementTree.SubElement(
        package, f"{wix}StandardDirectory", {"Id": "LocalAppDataFolder"}
    )
    programs = ElementTree.SubElement(
        local_app_data, f"{wix}Directory", {"Id": "ProgramsFolder", "Name": "Programs"}
    )
    ElementTree.SubElement(
        programs, f"{wix}Directory", {"Id": "INSTALLFOLDER", "Name": APP_NAME}
    )
    menu = ElementTree.SubElement(
        package, f"{wix}StandardDirectory", {"Id": "ProgramMenuFolder"}
    )
    shortcut_dir = ElementTree.SubElement(
        menu,
        f"{wix}Directory",
        {"Id": "ShortcutFolder", "Name": APP_NAME},
    )

    components = ElementTree.SubElement(
        package, f"{wix}ComponentGroup", {"Id": "AppFiles", "Directory": "INSTALLFOLDER"}
    )
    exe_component = ElementTree.SubElement(
        components, f"{wix}Component", {"Id": "ApplicationExecutable", "Guid": "*"}
    )
    ElementTree.SubElement(
        exe_component,
        f"{wix}File",
        {
            "Id": "AiClipExecutable",
            "Source": str(executable),
            "KeyPath": "yes",
        },
    )
    license_component = ElementTree.SubElement(
        components, f"{wix}Component", {"Id": "FFmpegLicenseComponent", "Guid": "*"}
    )
    ElementTree.SubElement(
        license_component,
        f"{wix}File",
        {
            "Id": "FFmpegLicense",
            "Source": str(ffmpeg_license),
            "KeyPath": "yes",
        },
    )

    shortcut_component = ElementTree.SubElement(
        package,
        f"{wix}Component",
        {
            "Id": "ApplicationShortcut",
            "Guid": "*",
            "Directory": "ShortcutFolder",
        },
    )
    ElementTree.SubElement(
        shortcut_component,
        f"{wix}Shortcut",
        {
            "Id": "StartMenuShortcut",
            "Name": APP_NAME,
            "Target": "[INSTALLFOLDER]AiClip.exe",
            "WorkingDirectory": "INSTALLFOLDER",
        },
    )
    ElementTree.SubElement(
        shortcut_component,
        f"{wix}RegistryValue",
        {
            "Root": "HKCU",
            "Key": "Software\\AiClip",
            "Name": "Installed",
            "Type": "integer",
            "Value": "1",
            "KeyPath": "yes",
        },
    )
    ElementTree.SubElement(
        shortcut_component,
        f"{wix}RemoveFolder",
        {"Id": "RemoveShortcutFolder", "On": "uninstall"},
    )
    ElementTree.SubElement(feature, f"{wix}ComponentRef", {"Id": "ApplicationShortcut"})

    ElementTree.indent(root, space="  ")
    ElementTree.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def _build_installer(executable: Path, ffmpeg_license: Path) -> Path:
    wix = shutil.which("wix")
    if not wix:
        raise RuntimeError(
            "No se encontró WiX Toolset v4. Instálalo con "
            "'dotnet tool install --global wix --version 4.0.5'."
        )
    build_dir = ROOT / "build" / "wix"
    build_dir.mkdir(parents=True, exist_ok=True)
    source = build_dir / "Product.wxs"
    _write_wix_source(executable, ffmpeg_license, source)
    installer = DIST_DIR / f"{APP_NAME}-Setup.msi"
    subprocess.run(
        [wix, "build", str(source), "-arch", "x64", "-o", str(installer)],
        check=True,
        cwd=ROOT,
    )
    if not installer.is_file():
        raise RuntimeError("WiX terminó sin generar el instalador MSI.")
    return installer


def main() -> int:
    parser = argparse.ArgumentParser(description="Compila AiClip para Windows.")
    parser.add_argument(
        "--skip-msi",
        action="store_true",
        help="Genera solo el ejecutable cuando WiX no está instalado.",
    )
    args = parser.parse_args()
    try:
        ffmpeg, ffprobe, ffmpeg_license = _find_ffmpeg_files()
        DIST_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ffmpeg_license, DIST_DIR / "FFmpeg-LICENSE.txt")
        executable = _build_executable(ffmpeg, ffprobe, ffmpeg_license)
        print(f"Ejecutable creado: {executable}")
        if not args.skip_msi:
            installer = _build_installer(executable, DIST_DIR / "FFmpeg-LICENSE.txt")
            print(f"Instalador creado: {installer}")
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
