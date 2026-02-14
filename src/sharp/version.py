"""Version information for SHARP.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

import sys
from pathlib import Path

__version__ = "1.0.0"


def get_version() -> str:
    """Get the current version of PhotoSplat3D."""
    # Try to read from VERSION file
    try:
        # When running from source
        version_file = Path(__file__).parent.parent.parent / "VERSION"
        if version_file.exists():
            return version_file.read_text().strip()

        # When running from PyInstaller bundle
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
            bundle_dir = Path(sys._MEIPASS)
            version_file = bundle_dir / "VERSION"
            if version_file.exists():
                return version_file.read_text().strip()
    except Exception:
        pass

    # Fallback to hardcoded version
    return __version__


def get_version_info() -> dict:
    """Get detailed version information."""
    return {
        'version': get_version(),
        'name': 'PhotoSplat3D',
        'full_name': 'PhotoSplat3D Gaussian Splatting Generator',
        'description': 'Photorealistic view synthesis from a single image',
    }
