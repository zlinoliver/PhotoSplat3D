"""Build script for creating macOS .app bundle using PyInstaller.

Usage:
    python build_app.py
"""

import PyInstaller.__main__
import sys
from pathlib import Path

# Get project root
project_root = Path(__file__).parent

# PyInstaller options
options = [
    # Entry point
    'src/sharp/cli/gui.py',

    # App name
    '--name=PhotoSplat3D',

    # Create .app bundle (macOS)
    '--windowed',

    # One directory mode (easier to debug)
    '--onedir',

    # Clean build
    '--clean',

    # Don't confirm overwrites
    '--noconfirm',

    # Add data files (if needed)
    # '--add-data=data:data',

    # Hidden imports (for modules without data files)
    '--hidden-import=sharp.gui',
    '--hidden-import=sharp.models',
    '--hidden-import=sharp.utils',
    '--hidden-import=sharp.cli',
    '--hidden-import=PIL._tkinter_finder',

    # Collect all from these packages (for complex packages with plugins/data)
    '--collect-all=torch',
    '--collect-all=torchvision',
    '--collect-all=timm',
    '--collect-all=gsplat',
    '--collect-all=imageio',
    '--collect-all=pillow_heif',
    '--collect-all=plyfile',
    '--collect-all=scipy',

    # Output directory
    '--distpath=dist',
    '--workpath=build',
    '--specpath=build',
]

if __name__ == '__main__':
    PyInstaller.__main__.run(options)

    print("\n" + "="*60)
    print("✅ Build complete!")
    print("="*60)
    print(f"\nApp location: {project_root}/dist/PhotoSplat3D.app")
    print("\nTo run:")
    print("  open dist/PhotoSplat3D.app")
    print("\nTo distribute:")
    print("  1. Compress: Right-click PhotoSplat3D.app → Compress")
    print("  2. Share the .zip file")
    print("  3. Recipients: Double-click to extract, then run PhotoSplat3D.app")
    print("="*60)
