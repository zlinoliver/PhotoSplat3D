"""Left panel displaying image thumbnails and status.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from typing import List
from pathlib import Path

import customtkinter as ctk
from PIL import Image

from .image_item import ImageItem, ImageStatus
from .constants import THUMBNAIL_SIZE
from .i18n import Translator


class ImageListPanel(ctk.CTkFrame):
    """Left panel showing image queue with thumbnails."""

    def __init__(self, parent, translator: Translator):
        super().__init__(parent, fg_color="transparent")

        self.translator = translator
        self.items: List[ImageItem] = []
        self.thumbnail_cache = {}  # Path -> CTkImage
        self.grid_columns = 2

        self.translator.register(self._update_texts)
        self._create_widgets()

    def _create_widgets(self):
        """Create scrollable list of images."""
        self.title_label = ctk.CTkLabel(
            self,
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.title_label.pack(fill="x", pady=(0, 8))

        self.scrollable_frame = ctk.CTkScrollableFrame(
            self,
            fg_color=("gray98", "gray16"),
            corner_radius=12
        )
        self.scrollable_frame.pack(fill="both", expand=True)

        for col in range(self.grid_columns):
            self.scrollable_frame.grid_columnconfigure(col, weight=1)

        self._update_texts()

    def add_images(self, image_paths: List[Path]):
        """Add new images to the queue."""
        for path in image_paths:
            if any(item.path == path for item in self.items):
                continue

            item = ImageItem(path=path)
            self.items.append(item)
            self._create_item_widget(item)

    def _create_item_widget(self, item: ImageItem):
        """Create widget for a single image item laid out in a grid."""
        frame = ctk.CTkFrame(
            self.scrollable_frame,
            corner_radius=14,
            fg_color=("white", "#1f1f1f"),
            border_width=1,
            border_color=("gray85", "#2c2c2c"),
        )

        header = ctk.CTkFrame(frame, fg_color="transparent")
        header.pack(fill="x")

        status_indicator = ctk.CTkFrame(
            header,
            width=22,
            height=22,
            corner_radius=11,
            fg_color=item.status_color
        )
        status_indicator.pack(side="right", pady=4, padx=4)
        status_indicator.pack_propagate(False)

        thumbnail = self._load_thumbnail(item.path)
        thumb_label = ctk.CTkLabel(frame, image=thumbnail, text="")
        thumb_label.image = thumbnail
        thumb_label.pack(pady=(6, 10))

        info_frame = ctk.CTkFrame(frame, fg_color="transparent")
        info_frame.pack(fill="x", padx=6, pady=(0, 6))

        name_label = ctk.CTkLabel(info_frame, text=item.name, font=ctk.CTkFont(size=12, weight="bold"), anchor="w")
        name_label.pack(fill="x")

        size_text = self._format_file_size(item.path.stat().st_size)
        size_label = ctk.CTkLabel(info_frame, text=size_text, font=ctk.CTkFont(size=11), anchor="w")
        size_label.pack(fill="x")

        item.widget_frame = frame
        item.status_indicator = status_indicator
        self._layout_items()

    def _load_thumbnail(self, image_path: Path) -> ctk.CTkImage:
        """Load and cache thumbnail."""
        cached = self.thumbnail_cache.get(image_path)
        if cached:
            return cached

        try:
            img = Image.open(image_path)
            exif = img.getexif()
            orientation = exif.get(274) if exif else None
            if orientation == 3:
                img = img.transpose(Image.ROTATE_180)
            elif orientation == 6:
                img = img.transpose(Image.ROTATE_270)
            elif orientation == 8:
                img = img.transpose(Image.ROTATE_90)

            img.thumbnail((THUMBNAIL_SIZE, THUMBNAIL_SIZE), Image.Resampling.LANCZOS)
        except Exception:
            img = Image.new('RGB', (THUMBNAIL_SIZE, THUMBNAIL_SIZE), color='gray')

        thumbnail = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
        self.thumbnail_cache[image_path] = thumbnail
        return thumbnail

    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size for display."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"

    def update_item_status(self, item: ImageItem):
        """Update visual status for an item."""
        indicator = getattr(item, 'status_indicator', None)
        if indicator:
            indicator.configure(fg_color=item.status_color)

    def clear_completed(self):
        """Remove all items from list and reset grid."""
        for item in self.items:
            frame = getattr(item, "widget_frame", None)
            if frame:
                frame.destroy()
        removed_paths = {item.path for item in self.items}
        self.items.clear()

        for path in removed_paths:
            self.thumbnail_cache.pop(path, None)

        self._layout_items()

    def get_pending_items(self) -> List[ImageItem]:
        """Get all items ready for processing."""
        return [item for item in self.items if item.status == ImageStatus.PENDING]

    def _layout_items(self):
        """Arrange item widgets in a grid layout."""
        for idx, item in enumerate(self.items):
            frame = getattr(item, 'widget_frame', None)
            if not frame:
                continue
            row = idx // self.grid_columns
            col = idx % self.grid_columns
            frame.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")

        for col in range(self.grid_columns):
            self.scrollable_frame.grid_columnconfigure(col, weight=1, uniform="image_col")

    def _update_texts(self):
        """Update static labels based on language."""
        self.title_label.configure(text=self.translator.translate("panel.image_queue"))

    def refresh_language(self):
        """External hook for language changes."""
        self._update_texts()
