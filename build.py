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

APP_NAME = "FJ Chat Voice"
MAIN_SCRIPT = "main.py"
FILE_NAME = "fj_chat_voice"

ICON_PATH = "img/icon.png"
ICON_PATH_WINDOWS = "img/icon.ico"
ICON_PATH_MAC = "img/icon.icns"


def ensure_windows_icon():
    """Convert PNG to ICO if needed"""
    if platform.system() == "Windows" and not os.path.exists(ICON_PATH_WINDOWS):
        if os.path.exists(ICON_PATH):
            try:
                from PIL import Image

                img = Image.open(ICON_PATH)
                img.save(ICON_PATH_WINDOWS, sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
                print(f"✓ Created Windows icon: {ICON_PATH_WINDOWS}")
                return ICON_PATH_WINDOWS
            except ImportError:
                print("⚠️ PIL not installed, skipping icon conversion")
                return ICON_PATH
            except Exception as e:
                print(f"⚠️ Failed to convert icon: {e}")
                return ICON_PATH
    return ICON_PATH_WINDOWS if os.path.exists(ICON_PATH_WINDOWS) else ICON_PATH


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
    "googleapiclient",
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
    "silero",
    "silero.utils",
]

EXCLUDES = []


def create_virtual_env():
    """Create virtual environment for building"""
    venv_path = ".venv"
    venv_path = os.path.abspath(venv_path)

    if not os.path.exists(venv_path):
        print(f"📦 Creating virtual environment at {venv_path}...")
        try:
            subprocess.run([sys.executable, "-m", "venv", venv_path], check=True)
            print("✓ Virtual environment created")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create virtual environment: {e}")
            return None
    else:
        print(f"✓ Using existing virtual environment at {venv_path}")

    if platform.system() == "Windows":
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

    print("✓ Build directories cleaned")


def create_spec_file():
    """Create PyInstaller spec file with proper icon handling"""
    icon_data = []
    icon_path = ICON_PATH

    if os.path.exists(icon_path):
        icon_data = [(icon_path, "img")]

        if platform.system() == "Windows" and os.path.exists(ICON_PATH_WINDOWS):
            icon_data.append((ICON_PATH_WINDOWS, "img"))
            icon_path = ICON_PATH_WINDOWS

    excludes_str = str(EXCLUDES).replace("'", '"')
    hidden_imports_str = str(HIDDEN_IMPORTS).replace("'", '"')

    spec_content = f"""# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['{MAIN_SCRIPT}'],
    pathex=[],
    binaries=[],
    datas={icon_data},
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
    console=platform.system() == "Windows",
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon="{icon_path}",
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

    subprocess.run([pip_path, "install", "pillow"], check=True)
    subprocess.run([pip_path, "install", "-r", "torch.requirements.txt"], check=True)
    subprocess.run([pip_path, "install", "-r", "requirements.txt"], check=True)

    print("📦 Installing PyInstaller...")
    subprocess.run([pip_path, "install", "pyinstaller"], check=True)

    return venv_path, pip_path, python_path


def build():
    """Build with UPX compression"""
    print("\n" + "=" * 50)
    print(f"Building for {platform.system()}")
    print("=" * 50)

    venv_path, pip_path, python_path = install_dependencies()

    if platform.system() == "Windows":
        ensure_windows_icon()

    if platform.system() == "Windows":
        pyinstaller_path = os.path.join(venv_path, "Scripts", "pyinstaller.exe")
    else:
        pyinstaller_path = os.path.join(venv_path, "bin", "pyinstaller")

    if os.path.exists("dist"):
        shutil.rmtree("dist")

    spec_file = create_spec_file()

    # if platform.system() == "Windows":
    #     create_windows_manifest()

    cmd = [pyinstaller_path, "--clean", "--noconfirm", spec_file]

    try:
        subprocess.run(cmd, check=True, timeout=1200)

        print(f"\n✅ Executable built: dist/")

        if platform.system() == "Windows":
            exe_path = f"dist/{FILE_NAME}.exe"
        else:
            exe_path = f"dist/{FILE_NAME}"
            create_launcher_script()
            os.chmod(exe_path, os.stat(exe_path).st_mode | stat.S_IEXEC)

        if os.path.exists(exe_path):
            if os.path.exists(ICON_PATH):
                shutil.copytree("img", "dist/img")

            size_mb = os.path.getsize(exe_path) / (1024 * 1024)
            print(f"   Size: {size_mb:.2f} MB")

        return True
    except subprocess.TimeoutExpired:
        print("\n❌ Build timed out")
        return False
    except Exception as e:
        print(f"\n❌ Build failed: {e}")
        return False


# def create_windows_manifest():
#     """Create Windows manifest file for better icon integration"""
#     manifest_content = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
# <assembly xmlns="urn:schemas-microsoft-com:asm.v1" manifestVersion="1.0">
#     <assemblyIdentity
#         version="1.0.5"
#         processorArchitecture="*"
#         name="FJ Chat Voice"
#         type="win32"
#     />
#     <description>FJ Chat Voice Application</description>
#     <trustInfo xmlns="urn:schemas-microsoft-com:asm.v2">
#         <security>
#             <requestedPrivileges>
#                 <requestedExecutionLevel level="asInvoker" uiAccess="false"/>
#             </requestedPrivileges>
#         </security>
#     </trustInfo>
# </assembly>"""

#     manifest_path = "app.manifest"
#     with open(manifest_path, "w", encoding="utf-8") as f:
#         f.write(manifest_content)
#     print("✓ Windows manifest created")


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


def verify_icon():
    """Verify icon exists and is valid"""
    if not os.path.exists(ICON_PATH):
        print(f"⚠️ Icon not found at {ICON_PATH}")
        return False

    # Check if icon is a valid image file
    try:
        from PIL import Image

        img = Image.open(ICON_PATH)
        print(f"✓ Icon found: {ICON_PATH} ({img.size[0]}x{img.size[1]} pixels)")
        return True
    except Exception as e:
        print(f"⚠️ Icon file may be corrupted: {e}")
        return False


def main():
    """Main function"""
    print("=" * 60)
    print(f"Optimized build of {APP_NAME}")
    print("=" * 60)

    # Check if running in correct directory
    if not os.path.exists(MAIN_SCRIPT):
        print(f"❌ {MAIN_SCRIPT} not found in current directory")
        return

    # Verify icon exists
    verify_icon()

    # Clean directories
    clean_build_dirs()

    # Build
    success = build()

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
