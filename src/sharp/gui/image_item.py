"""Data model for tracking individual image conversion state.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from enum import Enum
from datetime import datetime

from .constants import STATUS_COLORS


class ImageStatus(Enum):
    """Possible states for an image in the conversion pipeline."""
    PENDING = "pending"           # Queued, waiting to process
    PROCESSING = "processing"     # Currently being converted
    COMPLETED = "completed"       # Successfully converted
    FAILED = "failed"            # Conversion failed
    SKIPPED = "skipped"          # Skipped (already converted)


@dataclass
class ImageItem:
    """Represents a single image in the conversion queue."""
    path: Path
    status: ImageStatus = ImageStatus.PENDING
    error_message: Optional[str] = None
    output_path: Optional[Path] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    @property
    def name(self) -> str:
        """Return filename."""
        return self.path.name

    @property
    def status_color(self) -> str:
        """Return color for status indicator."""
        return STATUS_COLORS[self.status.value]
