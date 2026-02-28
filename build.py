#!/usr/bin/env python3
"""
Optimized build script for FJ Chat to Speech
Supports: Windows (.exe), Linux, and macOS
"""

import os
import sys
import shutil
import subprocess
import platform
from pathlib import Path
import stat
import tarfile

from app.constants import APP_VERSION, APP_NAME

MAIN_SCRIPT = "main.py"
PLATFORM = platform.system()

if PLATFORM == "Windows":
    FILE_NAME = f"fj_chat_to_speech_{APP_VERSION}_windows"
elif PLATFORM == "Darwin":
    FILE_NAME = f"fj_chat_to_speech_{APP_VERSION}_macos"
else:
    FILE_NAME = f"fj_chat_to_speech_{APP_VERSION}_linux"

ICON_PATH = "img/icon.ico" if PLATFORM == "Windows" else "img/icon.png"

HIDDEN_IMPORTS = [
    "collections",
    "asyncio",
    "datetime",
    "functools",
    "gc",
    "json",
    "html",
    "re",
    "sys",
    "threading",
    "inspect",
    "multiprocessing",
    "time",
    "typing",
    "hashlib",
    "num2words",
    "torch",
    "sounddevice",
    "PyQt6.QtWidgets",
    "PyQt6.QtCore",
    "PyQt6.QtGui",
    "scipy",
    "numpy",
    "googletrans",
    "PIL",
    "PIL.Image",
    "omegaconf",
    "silero",
    "silero.utils",
    "detoxify",
]

EXCLUDES = []


def create_virtual_env():
    """Create virtual environment for building"""
    venv_path = ".venv"
    venv_path = os.path.abspath(venv_path)

    if not os.path.exists(venv_path):
        print(f"[PKG] Creating virtual environment at {venv_path}...")
        try:
            subprocess.run([sys.executable, "-m", "venv", venv_path], check=True)
            print("[OK] Virtual environment created")
        except subprocess.CalledProcessError as e:
            print(f"[ERR] Failed to create virtual environment: {e}")
            return None
    else:
        print(f"[OK] Using existing virtual environment at {venv_path}")

    if PLATFORM == "Windows":
        pip_path = os.path.join(venv_path, "Scripts", "pip.exe")
        python_path = os.path.join(venv_path, "Scripts", "python.exe")
    else:
        pip_path = os.path.join(venv_path, "bin", "pip")
        python_path = os.path.join(venv_path, "bin", "python")

    return venv_path, pip_path, python_path


def clean_build_dirs():
    """Clean build directories"""
    dirs_to_clean = ["build", "dist", "__pycache__"]
    for dir_name in dirs_to_clean:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)

    spec_files = Path(".").glob("*.spec")
    for spec_file in spec_files:
        os.remove(spec_file)

    print("[OK] Build directories cleaned")


def create_spec_file():
    """Create single-file PyInstaller spec for all platforms"""
    data_files = []

    if os.path.exists(ICON_PATH):
        data_files.append((ICON_PATH, "img"))

    stop_words_files = [
        ("spam_filter/banned.txt", "spam_filter"),
        ("spam_filter/ru.txt", "spam_filter"),
        ("spam_filter/en.txt", "spam_filter"),
    ]

    for src, dst in stop_words_files:
        if os.path.exists(src):
            data_files.append((src, dst))
        else:
            print(f"[WARN] Data file not found and will be skipped: {src}")

    excludes_str = str(EXCLUDES).replace("'", '"')
    hidden_imports_str = str(HIDDEN_IMPORTS).replace("'", '"')

    spec_content = f"""# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['{MAIN_SCRIPT}'],
    pathex=[],
    binaries=[],
    datas={data_files},
    hiddenimports={hidden_imports_str},
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes={excludes_str},
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='{FILE_NAME}',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon="{ICON_PATH}",
)
"""

    spec_filename = f"{FILE_NAME}.spec"

    with open(spec_filename, "w", encoding="utf-8") as f:
        f.write(spec_content)

    return spec_filename


def install_dependencies():
    """Install required dependencies including Silero"""
    print("[PKG] Installing dependencies...")

    venv_result = create_virtual_env()
    if not venv_result:
        return None, None, None

    venv_path, pip_path, python_path = venv_result

    if PLATFORM == "Darwin":
        subprocess.run(
            [pip_path, "install", "torch==2.8.0", "torchaudio==2.8.0"], check=True
        )
    else:
        subprocess.run(
            [pip_path, "install", "-r", "torch.requirements.txt"], check=True
        )
    subprocess.run([pip_path, "install", "-r", "requirements.txt"], check=True)

    print("[PKG] Installing PyInstaller...")
    subprocess.run([pip_path, "install", "pyinstaller"], check=True)

    return venv_path, pip_path, python_path


def build():
    """Build with UPX compression"""
    print("\n" + "=" * 50)
    print(f"Building for {PLATFORM}")
    print("=" * 50)

    venv_path, _, _ = install_dependencies()
    if not venv_path:
        return False

    if PLATFORM == "Windows":
        pyinstaller_path = os.path.join(venv_path, "Scripts", "pyinstaller.exe")
    else:
        pyinstaller_path = os.path.join(venv_path, "bin", "pyinstaller")

    if os.path.exists("dist"):
        shutil.rmtree("dist")

    spec_file = create_spec_file()

    cmd = [pyinstaller_path, "--clean", "--noconfirm", spec_file]

    try:
        subprocess.run(cmd, check=True, timeout=1800)  # 30 minutes timeout

        print(f"\n[SUCCESS] Build completed: dist/")

        if PLATFORM == "Darwin":
            app_path = f"dist/{FILE_NAME}"
            if os.path.exists(app_path):
                os.chmod(exe_path, os.stat(exe_path).st_mode | stat.S_IEXEC)
                print(f"[OK] Created macOS bundle: {app_path}")
        elif PLATFORM == "Windows":
            exe_path = f"dist/{FILE_NAME}.exe"
        elif PLATFORM == "Linux":
            exe_path = f"dist/{FILE_NAME}"

            if not os.path.exists(exe_path):
                print("[ERR] Binary not found")
                return False

            os.chmod(exe_path, os.stat(exe_path).st_mode | stat.S_IEXEC)
            create_launcher_script()

            launcher_path = "dist/run_chat_voice.sh"
            archive_path = f"dist/{FILE_NAME}.tar.gz"

            with tarfile.open(archive_path, "w:gz") as tar:
                tar.add(exe_path, arcname=os.path.basename(exe_path))
                tar.add(launcher_path, arcname=os.path.basename(launcher_path))

            print(f"[OK] Created tarball: {archive_path}")

            os.remove(exe_path)
            os.remove(launcher_path)

            print("[OK] Removed original binary and launcher")

        if PLATFORM == "Darwin":
            app_path = f"dist/{FILE_NAME}"
            if os.path.exists(app_path):
                size_mb = os.path.getsize(app_path) / (1024 * 1024)
                print(f"   Bundle size: {size_mb:.2f} MB")
        elif PLATFORM == "Windows":
            exe_path = f"dist/{FILE_NAME}.exe"
            if os.path.exists(exe_path):
                size_mb = os.path.getsize(exe_path) / (1024 * 1024)
                print(f"   Size: {size_mb:.2f} MB")
        elif PLATFORM == "Linux":
            exe_path = f"dist/{FILE_NAME}"
            archive_path = f"dist/{FILE_NAME}.tar.gz"
            if os.path.exists(exe_path):
                size_mb = os.path.getsize(exe_path) / (1024 * 1024)
                print(f"   Binary size: {size_mb:.2f} MB")
            if os.path.exists(archive_path):
                size_mb = os.path.getsize(archive_path) / (1024 * 1024)
                print(f"   Archive size: {size_mb:.2f} MB")

        return True

    except subprocess.TimeoutExpired:
        print("\n[ERR] Build timed out")
        return False
    except Exception as e:
        print(f"\n[ERR] Build failed: {e}")
        return False


def create_launcher_script():
    """Create launcher script for Linux"""
    launcher = f"""#!/bin/bash
# FJ Chat to Speech Launcher
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Run application
"./{FILE_NAME}"
"""
    launcher_path = f"dist/run_chat_voice.sh"
    with open(launcher_path, "w", encoding="utf-8") as f:
        f.write(launcher)

    print("[OK] Launcher script created: dist/run_chat_voice.sh")


def main():
    """Main function"""
    print("=" * 60)
    print(f"Optimized build of {APP_NAME} v{APP_VERSION} for {PLATFORM}")
    print("=" * 60)

    clean_build_dirs()

    success = build()

    print("\n" + "=" * 60)
    print("Build Results")
    print("=" * 60)

    if success:
        print("\n[SUCCESS] Build completed successfully!")
        print(f"   Files are located in the 'dist/' directory")

        print("\n[DIR] Output directory: dist/")
        print("   Files:")

        if os.path.exists("dist"):
            for file in sorted(Path("dist").iterdir()):
                if file.is_file():
                    size = file.stat().st_size / (1024 * 1024)
                    print(f"   - {file.name} ({size:.2f} MB)")
                elif file.is_dir() and PLATFORM == "Darwin":
                    try:
                        size = sum(
                            f.stat().st_size for f in file.rglob("*") if f.is_file()
                        ) / (1024 * 1024)
                        print(f"   - {file.name}/ ({size:.2f} MB)")
                    except:
                        print(f"   - {file.name}/")
    else:
        print("\n[ERR] Build failed")


if __name__ == "__main__":
    main()
