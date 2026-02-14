"""Configuration manager for user settings.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

from .i18n import DEFAULT_LANGUAGE

logger = logging.getLogger(__name__)

CONFIG_FILE = Path.home() / ".sharp_config.json"
DEFAULT_OUTPUT_DIR = Path.home() / "Desktop" / "SHARP_Output"
DEFAULT_MODAL_ENDPOINT = os.getenv(
    "SHARP_MODAL_ENDPOINT",
    "",
)


class ConfigManager:
    """Manages application configuration and user settings."""

    def __init__(self):
        """Initialize config manager and load existing config."""
        self.config = self._load_config()
        self._ensure_defaults()

    def _load_config(self) -> dict:
        """Load configuration from file or create default."""
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, 'r') as f:
                    config = json.load(f)
                    logger.info(f"Loaded config from {CONFIG_FILE}")
                    return config
            except Exception as e:
                logger.warning(f"Failed to load config: {e}, using defaults")

        # Return default config
        return {
            "output_directory": str(DEFAULT_OUTPUT_DIR),
            "language": DEFAULT_LANGUAGE,
            "modal_endpoint": DEFAULT_MODAL_ENDPOINT,
            "panorama_face_size": "auto",
        }

    def _ensure_defaults(self):
        """Ensure essential keys exist when older configs are loaded."""
        updated = False
        if "output_directory" not in self.config:
            self.config["output_directory"] = str(DEFAULT_OUTPUT_DIR)
            updated = True
        if "language" not in self.config:
            self.config["language"] = DEFAULT_LANGUAGE
            updated = True
        if not self.config.get("modal_endpoint"):
            self.config["modal_endpoint"] = DEFAULT_MODAL_ENDPOINT
            updated = True
        if "panorama_face_size" not in self.config:
            self.config["panorama_face_size"] = "auto"
            updated = True
        if updated:
            self._save_config()

    def _save_config(self):
        """Save configuration to file."""
        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(self.config, f, indent=2)
                logger.info(f"Saved config to {CONFIG_FILE}")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")

    def get_output_directory(self) -> Path:
        """Get the configured output directory, creating it if needed."""
        output_dir = Path(self.config.get("output_directory", str(DEFAULT_OUTPUT_DIR)))

        # Ensure directory exists
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create output directory {output_dir}: {e}")
            # Fallback to default
            output_dir = DEFAULT_OUTPUT_DIR
            output_dir.mkdir(parents=True, exist_ok=True)

        return output_dir

    def set_output_directory(self, directory: Path):
        """Set the output directory and save config."""
        self.config["output_directory"] = str(directory)
        self._save_config()
        logger.info(f"Output directory set to: {directory}")

    def get_language(self) -> str:
        """Return the UI language."""
        return self.config.get("language", DEFAULT_LANGUAGE)

    def set_language(self, language: str):
        """Persist UI language selection."""
        self.config["language"] = language
        self._save_config()

    def get_modal_endpoint(self) -> Optional[str]:
        """Return configured Modal endpoint, falling back to default/env."""
        endpoint = self.config.get("modal_endpoint") or DEFAULT_MODAL_ENDPOINT
        return endpoint.strip() or None

    def set_modal_endpoint(self, endpoint: str):
        """Persist Modal endpoint URL."""
        self.config["modal_endpoint"] = endpoint
        self._save_config()

    def get_panorama_face_size(self) -> Optional[int]:
        """Return configured panorama face size or None for auto."""
        value = self.config.get("panorama_face_size", "auto")
        if value in (None, "", "auto"):
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None

    def set_panorama_face_size(self, value: Optional[int]):
        """Persist panorama face size or auto."""
        if value is None:
            self.config["panorama_face_size"] = "auto"
        else:
            self.config["panorama_face_size"] = int(value)
        self._save_config()


# Global instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Get or create the global config manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager
