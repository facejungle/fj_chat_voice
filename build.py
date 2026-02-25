#!/usr/bin/env python3
"""
Optimized build script for FJ Chat to Speech
Supports: Windows (.exe), Linux, and macOS (.app)
"""

import os
import sys
import shutil
import subprocess
import platform
from pathlib import Path
import stat

from app.constants import APP_VERSION, APP_NAME

MAIN_SCRIPT = "main.py"
PLATFORM = platform.system()

# Platform-specific filename
if PLATFORM == "Windows":
    FILE_NAME = f"fj_chat_to_speech_{APP_VERSION}_windows"
elif PLATFORM == "Darwin":  # macOS
    FILE_NAME = f"fj_chat_to_speech_{APP_VERSION}_macos"
else:  # Linux
    FILE_NAME = f"fj_chat_to_speech_{APP_VERSION}_linux"

ICON_PATH = "img/icon.png"
ICON_PATH_WINDOWS = "img/icon.ico"
ICON_PATH_MAC = "img/icon.icns"

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
]

EXCLUDES = []


def ensure_icons():
    """Convert PNG to platform-specific formats if needed"""
    if PLATFORM == "Windows" and not os.path.exists(ICON_PATH_WINDOWS):
        if os.path.exists(ICON_PATH):
            try:
                from PIL import Image
                img = Image.open(ICON_PATH)
                img.save(
                    ICON_PATH_WINDOWS,
                    sizes=[
                        (16, 16), (32, 32), (48, 48), (64, 64),
                        (128, 128), (256, 256)
                    ],
                )
                print(f"[OK] Created Windows icon: {ICON_PATH_WINDOWS}")
                return ICON_PATH_WINDOWS
            except Exception as e:
                print(f"[WARN] Failed to convert Windows icon: {e}")
                
    elif PLATFORM == "Darwin" and not os.path.exists(ICON_PATH_MAC):
        if os.path.exists(ICON_PATH):
            try:
                # For macOS, we need to create an iconset
                iconset_dir = "img/icon.iconset"
                os.makedirs(iconset_dir, exist_ok=True)
                
                from PIL import Image
                img = Image.open(ICON_PATH)
                
                # Generate different sizes for macOS
                sizes = [16, 32, 64, 128, 256, 512, 1024]
                for size in sizes:
                    # Regular size
                    resized = img.resize((size, size), Image.Resampling.LANCZOS)
                    resized.save(f"{iconset_dir}/icon_{size}x{size}.png")
                    
                    # Retina size (2x)
                    if size * 2 <= 1024:
                        resized_2x = img.resize((size*2, size*2), Image.Resampling.LANCZOS)
                        resized_2x.save(f"{iconset_dir}/icon_{size}x{size}@2x.png")
                
                # Convert iconset to icns
                subprocess.run([
                    "iconutil", "-c", "icns", iconset_dir,
                    "-o", ICON_PATH_MAC
                ], check=True)
                
                # Clean up
                shutil.rmtree(iconset_dir)
                print(f"[OK] Created macOS icon: {ICON_PATH_MAC}")
                return ICON_PATH_MAC
                
            except Exception as e:
                print(f"[WARN] Failed to convert macOS icon: {e}")
    
    # Return appropriate icon path
    if PLATFORM == "Windows" and os.path.exists(ICON_PATH_WINDOWS):
        return ICON_PATH_WINDOWS
    elif PLATFORM == "Darwin" and os.path.exists(ICON_PATH_MAC):
        return ICON_PATH_MAC
    return ICON_PATH

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
    """Create PyInstaller spec file with proper icon handling"""
    icon_path = ensure_icons()
    
    data_files = []
    
    # Add icon to data files
    if os.path.exists(icon_path):
        data_files.append((icon_path, "img"))
    
    # Add stop words files
    stop_words_files = [
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

    # Base spec content
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
"""

    # Platform-specific EXE/Bundle creation
    if PLATFORM == "Darwin":
        # macOS creates an .app bundle
        spec_content += f"""
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
    icon="{icon_path}",
)

app = BUNDLE(
    exe,
    name='{APP_NAME}.app',
    icon="{icon_path}",
    bundle_identifier=None,
    info_plist={{
        'NSPrincipalClass': 'NSApplication',
        'NSHighResolutionCapable': True,
        'CFBundleShortVersionString': '{APP_VERSION}',
        'CFBundleVersion': '{APP_VERSION}',
        'CFBundleName': '{APP_NAME}',
    }},
)
"""
    else:
        # Windows and Linux create executables
        spec_content += f"""
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
    icon="{icon_path}",
)
"""

    spec_filename = f"{FILE_NAME}.spec" if PLATFORM != "Darwin" else f"{APP_NAME}.spec"
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

    # Install dependencies
    subprocess.run([pip_path, "install", "pillow"], check=True)
    if PLATFORM == "Darwin":
        # For macOS, install PyTorch without CUDA
        subprocess.run([
            pip_path, "install",
            "torch==2.8.0",
            "torchaudio==2.8.0"
        ], check=True)
    else:
        subprocess.run([pip_path, "install", "-r", "torch.requirements.txt"], check=True)
    subprocess.run([pip_path, "install", "-r", "requirements.txt"], check=True)
    
    print("[PKG] Installing PyInstaller...")
    subprocess.run([pip_path, "install", "pyinstaller"], check=True)

    return venv_path, pip_path, python_path

def build():
    """Build with UPX compression"""
    print("\n" + "=" * 50)
    print(f"Building for {PLATFORM}")
    print("=" * 50)

    venv_path, pip_path, python_path = install_dependencies()
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

        # Post-build handling
        if PLATFORM == "Darwin":
            app_path = f"dist/{APP_NAME}.app"
            if os.path.exists(app_path):
                # Create a zip of the .app bundle
                shutil.make_archive(
                    f"dist/{FILE_NAME}",
                    'zip',
                    "dist",
                    f"{APP_NAME}.app"
                )
                print(f"[OK] Created macOS bundle: {app_path}")
        elif PLATFORM == "Windows":
            exe_path = f"dist/{FILE_NAME}.exe"
        else:  # Linux
            exe_path = f"dist/{FILE_NAME}"
            create_launcher_script()
            if os.path.exists(exe_path):
                os.chmod(exe_path, os.stat(exe_path).st_mode | stat.S_IEXEC)

        # Show size information
        if PLATFORM == "Darwin":
            zip_path = f"dist/{FILE_NAME}.zip"
            if os.path.exists(zip_path):
                size_mb = os.path.getsize(zip_path) / (1024 * 1024)
                print(f"   Bundle size: {size_mb:.2f} MB")
        else:
            exe_path = f"dist/{FILE_NAME}.exe" if PLATFORM == "Windows" else f"dist/{FILE_NAME}"
            if os.path.exists(exe_path):
                size_mb = os.path.getsize(exe_path) / (1024 * 1024)
                print(f"   Size: {size_mb:.2f} MB")

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

    # Make executable
    if os.path.exists(launcher_path):
        st = os.stat(launcher_path)
        os.chmod(launcher_path, st.st_mode | stat.S_IEXEC)
    print("[OK] Launcher script created: dist/run_chat_voice.sh")

def verify_assets():
    """Verify required assets exist"""
    assets_ok = True
    
    if not os.path.exists(MAIN_SCRIPT):
        print(f"[ERR] {MAIN_SCRIPT} not found")
        assets_ok = False
    
    if not os.path.exists(ICON_PATH):
        print(f"[WARN] Icon not found at {ICON_PATH}")
    
    return assets_ok

def main():
    """Main function"""
    print("=" * 60)
    print(f"Optimized build of {APP_NAME} v{APP_VERSION} for {PLATFORM}")
    print("=" * 60)

    if not verify_assets():
        return

    clean_build_dirs()
    
    success = build()

    print("\n" + "=" * 60)
    print("Build Results")
    print("=" * 60)

    if success:
        print("\n[SUCCESS] Build completed successfully!")
        print(f"   Files are located in the 'dist/' directory")

        # List all files in dist
        print("\n[DIR] Output directory: dist/")
        print("   Files:")
        
        if os.path.exists("dist"):
            for file in sorted(Path("dist").iterdir()):
                if file.is_file():
                    size = file.stat().st_size / (1024 * 1024)
                    print(f"   - {file.name} ({size:.2f} MB)")
                elif file.is_dir() and PLATFORM == "Darwin":
                    # For macOS .app bundle
                    try:
                        size = sum(f.stat().st_size for f in file.rglob('*') if f.is_file()) / (1024 * 1024)
                        print(f"   - {file.name}/ ({size:.2f} MB)")
                    except:
                        print(f"   - {file.name}/")
    else:
        print("\n[ERR] Build failed")

if __name__ == "__main__":
    main()