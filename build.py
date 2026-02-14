#!/usr/bin/env python3
"""
Optimized build script for FJ Chat Voice
Supports: Windows (.exe) and Linux
"""

import os
import sys
import shutil
import subprocess
import platform
from pathlib import Path
import stat
import locale

APP_NAME = "FJ_Chat_Voice"
MAIN_SCRIPT = "main.py"
VERSION = "1.0.3"
FILE_NAME = f"{APP_NAME}_{VERSION}"

TORCH_CACHE_DIR = Path.home() / ".cache" / "torch" / "hub"

HIDDEN_IMPORTS = [
    "sys",
    "time",
    "re",
    "json",
    "threading",
    "datetime",
    "os",
    "urllib",
    "pathlib",
    "queue",
    "gc",
    "hashlib",
    "warnings",
    "googleapiclient",
    "googleapiclient.discovery",
    "torch",
    "sounddevice",
    "numpy",
    "customtkinter",
    "tkinter",
    "collections",
    "num2words",
    "protobuf",
    "requests",
    "httplib2",
    "filelock",
    "silero",
    "silero.utils",
]

EXCLUDES = [
    "torch.onnx",
    "torch._inductor",
    "torch._dynamo",
    "torch.contrib",
    # "tensorboard",
    "torchvision",
    "torchaudio.prototype",
    "matplotlib",
    # "scipy.signal.windows",
    "notebook",
    "jupyter",
]


def create_virtual_env():
    """Create virtual environment for building"""
    venv_path = "build_venv"

    # Use absolute path
    venv_path = os.path.abspath(venv_path)

    if not os.path.exists(venv_path):
        print(f"📦 Creating virtual environment at {venv_path}...")
        try:
            if platform.system() == "Windows":
                subprocess.run([sys.executable, "-m", "venv", venv_path], check=True)
            else:
                import tkinter

                subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", venv_path], check=True)
            print("✓ Virtual environment created")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create virtual environment: {e}")
            return None
    else:
        print(f"✓ Using existing virtual environment at {venv_path}")

    # Determine paths
    if platform.system() == "Windows":
        pip_path = os.path.join(venv_path, "Scripts", "pip.exe")
        python_path = os.path.join(venv_path, "Scripts", "python.exe")
    else:
        pip_path = os.path.join(venv_path, "bin", "pip")
        python_path = os.path.join(venv_path, "bin", "python")

    return venv_path, pip_path, python_path


def clean_build_dirs():
    """Clean build directories"""
    dirs_to_clean = ["build", "dist", "__pycache__", "*.spec"]
    for dir_name in dirs_to_clean:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)

    spec_files = Path(".").glob("*.spec")
    for spec_file in spec_files:
        os.remove(spec_file)

    print("✓ Build directories cleaned")


def create_spec_file():
    """Create optimized spec file"""
    excludes_str = str(EXCLUDES).replace("'", '"')
    hidden_imports_str = str(HIDDEN_IMPORTS).replace("'", '"')

    # Check if model cache exists and add to datas
    datas = [("settings.json", ".")]
    datas_str = str(datas).replace("'", '"')

    spec_content = f"""# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['{MAIN_SCRIPT}'],
    pathex=[],
    binaries=[],
    datas={datas_str},
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
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icon.ico' if os.path.exists('icon.ico') else None
)
"""
    with open(f"{FILE_NAME}.spec", "w", encoding="utf-8") as f:
        f.write(spec_content)
    return f"{FILE_NAME}.spec"


def install_dependencies():
    """Install required dependencies including Silero"""
    print("📦 Installing dependencies...")

    venv_result = create_virtual_env()
    if not venv_result:
        return
    venv_path, pip_path, python_path = venv_result

    # Upgrade pip, setuptools, and wheel first
    # subprocess.run([pip_path, "install", "--upgrade", "pip"], check=True)
    subprocess.run([pip_path, "install", "--upgrade", "setuptools", "wheel"], check=True)
    subprocess.run([pip_path, "install", "-r", "requirements.txt"], check=True)
    subprocess.run([pip_path, "install", "-r", "torch.requirements.txt"], check=True)

    return venv_path, pip_path, python_path


def build_with_upx():
    """Build with UPX compression"""
    print("\n" + "=" * 50)
    print(f"Building for {platform.system()}")
    print("=" * 50)

    # Check UPX availability
    upx_available = shutil.which("upx") is not None
    if not upx_available:
        print("⚠️ UPX not found. It is recommended to install UPX for compression:")
        if platform.system() == "Windows":
            print("   Download: https://github.com/upx/upx/releases")
        else:
            print("   sudo apt install upx || sudo dnf install upx")

    # First install dependencies
    venv_path, pip_path, python_path = install_dependencies()

    # Install PyInstaller in venv
    print("📦 Installing PyInstaller...")
    subprocess.run([pip_path, "install", "pyinstaller"], check=True)

    # Create spec file
    spec_file = create_spec_file()

    # Get PyInstaller path
    if platform.system() == "Windows":
        pyinstaller_path = os.path.join(venv_path, "Scripts", "pyinstaller")
    else:

        pyinstaller_path = os.path.join(venv_path, "bin", "pyinstaller")

    # Clean previous dist
    if os.path.exists("dist"):
        shutil.rmtree("dist")

    # Run PyInstaller - IMPORTANT: Use --onefile for Linux
    if platform.system() == "Windows":
        cmd = [pyinstaller_path, "--clean", "--noconfirm", spec_file]
    else:
        # For Linux, use one-file mode
        cmd = [pyinstaller_path, "--clean", "--noconfirm", "--onefile", MAIN_SCRIPT, "--name", FILE_NAME]

        # Add hidden imports
        for imp in HIDDEN_IMPORTS:
            cmd.extend(["--hidden-import", imp])

        # Add excludes
        for exc in EXCLUDES:
            cmd.extend(["--exclude-module", exc])

        # Add datas
        cmd.extend(["--add-data", "settings.json:."])

    try:
        # Add more time for large builds
        subprocess.run(cmd, check=True, timeout=1200)

        # Additional UPX compression
        if upx_available and platform.system() == "Windows":
            exe_path = f"dist/{FILE_NAME}.exe"
            if os.path.exists(exe_path):
                print("🔧 Applying UPX compression...")
                subprocess.run(["upx", "--best", "--lzma", exe_path], check=True)

        print(f"\n✅ Executable built: dist/")

        # Show size
        if platform.system() == "Windows":
            exe_path = f"dist/{FILE_NAME}.exe"
        else:
            exe_path = f"dist/{FILE_NAME}"
            create_launcher_script()

            # Make executable executable
            os.chmod(exe_path, os.stat(exe_path).st_mode | stat.S_IEXEC)

        if os.path.exists(exe_path):
            size_mb = os.path.getsize(exe_path) / (1024 * 1024)
            print(f"   Size: {size_mb:.2f} MB")

        return True
    except subprocess.TimeoutExpired:
        print("\n❌ Build timed out")
        return False
    except Exception as e:
        print(f"\n❌ Build failed: {e}")
        return False


def create_launcher_script():
    """Create launcher script for Linux"""
    launcher = f"""#!/bin/bash
# FJ Chat Voice Launcher
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Run application
"./{FILE_NAME}"
"""
    launcher_path = f"dist/run_chat_voice.sh"
    with open(launcher_path, "w", encoding="utf-8") as f:
        f.write(launcher)

    # Make executable
    st = os.stat(launcher_path)
    os.chmod(launcher_path, st.st_mode | stat.S_IEXEC)
    print("✓ Launcher script created: dist/run_chat_voice.sh")


def main():
    """Main function"""
    print("=" * 60)
    print(f"Optimized build of {APP_NAME} v{VERSION}")
    print("=" * 60)

    # Check if running in correct directory
    if not os.path.exists(MAIN_SCRIPT):
        print(f"❌ {MAIN_SCRIPT} not found in current directory")
        return

    # Clean directories
    clean_build_dirs()

    # Build
    success = build_with_upx()

    print("\n" + "=" * 60)
    print("Build Results")
    print("=" * 60)

    if success:
        print("\n✅ Build completed successfully!")
        print(f"   Files are located in the 'dist/' directory")

        # Show final size
        if platform.system() == "Windows":
            exe_path = f"dist/{FILE_NAME}.exe"
        else:
            exe_path = f"dist/{FILE_NAME}"

        if os.path.exists(exe_path):
            size_mb = os.path.getsize(exe_path) / (1024 * 1024)
            if size_mb > 1024:
                size_gb = size_mb / 1024
                print(f"   Size: {size_gb:.2f} GB")
            else:
                print(f"   Size: {size_mb:.2f} MB")

        print("\n📁 Output directory: dist/")
        print("   Files:")
        for file in sorted(Path("dist").iterdir()):
            if file.is_file():
                size = file.stat().st_size / (1024 * 1024)
                print(f"   - {file.name} ({size:.2f} MB)")

    else:
        print("\n❌ Build failed")


if __name__ == "__main__":
    main()
