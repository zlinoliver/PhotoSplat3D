"""Manages conversion history to avoid duplicate processing.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

import json
from pathlib import Path
from typing import Dict
from datetime import datetime
import hashlib


class HistoryManager:
    """Tracks successfully converted images."""

    def __init__(self, history_file: Path):
        self.history_file = history_file
        self.history: Dict[str, dict] = self._load_history()

    def _load_history(self) -> Dict[str, dict]:
        """Load history from JSON file."""
        if not self.history_file.exists():
            return {}

        try:
            with open(self.history_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}

    def _save_history(self):
        """Persist history to JSON file."""
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2, default=str)

    def _compute_hash(self, image_path: Path) -> str:
        """Compute file hash for reliable tracking."""
        hasher = hashlib.md5()
        with open(image_path, 'rb') as f:
            # Read in chunks for large files
            for chunk in iter(lambda: f.read(8192), b''):
                hasher.update(chunk)
        return hasher.hexdigest()

    def is_converted(self, image_path: Path) -> bool:
        """Check if image has been successfully converted."""
        key = str(image_path.absolute())
        if key not in self.history:
            return False

        # Verify file still exists and hash matches
        record = self.history[key]
        if not Path(record['output_path']).exists():
            return False

        # Optional: verify hash to detect file changes
        try:
            current_hash = self._compute_hash(image_path)
            return current_hash == record.get('hash', '')
        except (IOError, OSError):
            return False

    def add_conversion(self, image_path: Path, output_path: Path):
        """Record successful conversion."""
        key = str(image_path.absolute())
        self.history[key] = {
            'image_path': str(image_path),
            'output_path': str(output_path),
            'hash': self._compute_hash(image_path),
            'converted_at': datetime.now().isoformat(),
            'file_size': image_path.stat().st_size
        }
        self._save_history()

    def remove_conversion(self, image_path: Path):
        """Remove conversion record (for re-conversion)."""
        key = str(image_path.absolute())
        if key in self.history:
            del self.history[key]
            self._save_history()

    def get_converted_count(self) -> int:
        """Return total number of converted images."""
        return len(self.history)
