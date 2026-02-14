"""GUI configuration constants.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from pathlib import Path
import platform
import os

# Window configuration
APP_TITLE = "PhotoSplat3D Gaussian Splatting Generator"
WINDOW_SIZE = "1200x700"

# Thumbnails
THUMBNAIL_SIZE = 384  # enlarged preview size

# Colors for status indicators
STATUS_COLORS = {
    'pending': '#808080',      # Gray
    'processing': '#FFA500',   # Orange
    'completed': '#28A745',    # Green
    'failed': '#DC3545',       # Red
    'skipped': '#6C757D'       # Dark gray
}

# Output configuration
def get_output_dir() -> Path:
    """Get the configured output directory (user-configurable via GUI)."""
    from .config_manager import get_config_manager
    return get_config_manager().get_output_directory()

HISTORY_FILE = Path.home() / ".sharp_history.json"


def _app_support_base() -> Path:
    system = platform.system()
    home = Path.home()

    if system == "Darwin":
        return home / "Library" / "Application Support"
    if system == "Windows":
        return Path(os.environ.get("APPDATA", home / "AppData" / "Roaming"))
    return home / ".local" / "share"


APP_SUPPORT_DIR = _app_support_base() / "PhotoSplat3D"
CHECKPOINT_CACHE_DIR = APP_SUPPORT_DIR / "checkpoints"
INSTALL_INFO_FILE = APP_SUPPORT_DIR / "install_path.json"
MODEL_INFO_FILE = APP_SUPPORT_DIR / "model.json"
