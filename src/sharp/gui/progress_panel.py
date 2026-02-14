"""Right panel showing progress and logs with CustomTkinter.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from datetime import datetime
from typing import Dict

import customtkinter as ctk

from .image_item import ImageStatus
from .i18n import Translator


class ProgressPanel(ctk.CTkFrame):
    """Right panel displaying progress and log messages."""

    def __init__(self, parent, translator: Translator):
        super().__init__(parent, fg_color="transparent")

        self.translator = translator
        self.total_count = 0
        self.completed_count = 0
        self.failed_count = 0
        self.current_filename = "None"

        self._create_widgets()
        self.translator.register(self._update_static_texts)
        self._update_static_texts()

    def _create_widgets(self):
        """Create progress indicators and log area."""
        header_font = ctk.CTkFont(size=14, weight="bold")

        self.title_label = ctk.CTkLabel(self, font=header_font, anchor="w")
        self.title_label.pack(fill="x", pady=(0, 10))

        self.stats_frame = ctk.CTkFrame(
            self,
            corner_radius=12,
            fg_color=("white", "#1f1f1f"),
            border_width=1,
            border_color=("gray85", "#2c2c2c"),
        )
        self.stats_frame.pack(fill="x", pady=(0, 12))

        self.stats_label = ctk.CTkLabel(self.stats_frame, font=ctk.CTkFont(size=13, weight="bold"), anchor="w")
        self.stats_label.pack(fill="x", pady=(8, 6), padx=12)

        self.total_label = ctk.CTkLabel(self.stats_frame, anchor="w")
        self.total_label.pack(fill="x", padx=12)

        self.completed_label = ctk.CTkLabel(self.stats_frame, anchor="w")
        self.completed_label.pack(fill="x", padx=12)

        self.failed_label = ctk.CTkLabel(self.stats_frame, anchor="w")
        self.failed_label.pack(fill="x", padx=12)

        self.progress_label = ctk.CTkLabel(self.stats_frame, anchor="w")
        self.progress_label.pack(fill="x", padx=12, pady=(8, 0))

        self.progress_bar = ctk.CTkProgressBar(self.stats_frame, height=12)
        self.progress_bar.pack(fill="x", padx=12, pady=(4, 4))
        self.progress_bar.set(0)

        self.current_label = ctk.CTkLabel(self.stats_frame, anchor="w", wraplength=320)
        self.current_label.pack(fill="x", padx=12, pady=(6, 10))

        self.log_label = ctk.CTkLabel(self, font=ctk.CTkFont(size=13, weight="bold"), anchor="w")
        self.log_label.pack(fill="x")

        self.log_frame = ctk.CTkFrame(
            self,
            corner_radius=12,
            fg_color=("white", "#1f1f1f"),
            border_width=1,
            border_color=("gray85", "#2c2c2c"),
        )
        self.log_frame.pack(fill="both", expand=True, pady=(6, 0))

        self.log_text = ctk.CTkTextbox(
            self.log_frame,
            wrap="word",
            activate_scrollbars=True,
            state="disabled",
        )
        self.log_text.pack(fill="both", expand=True, padx=8, pady=8)
        self._text_widget = getattr(self.log_text, "_textbox", self.log_text)

        self._configure_tags()

    def _configure_tags(self):
        """Configure tag colors on the underlying Tk Text widget."""
        if hasattr(self._text_widget, "tag_config"):
            configure = self._text_widget.tag_config
        elif hasattr(self._text_widget, "tag_configure"):
            configure = self._text_widget.tag_configure
        else:
            return

        mode = ctk.get_appearance_mode()
        if mode == "Dark":
            info = "#f5f5f5"
            success = "#7cd67c"
            warning = "#ffdb7a"
            error = "#ff9ba1"
        else:
            info = "#1f1f1f"
            success = "#1a7f37"
            warning = "#cc8400"
            error = "#c82333"

        configure("INFO", foreground=info)
        configure("WARNING", foreground=warning)
        configure("ERROR", foreground=error)
        configure("SUCCESS", foreground=success)

    def _update_static_texts(self):
        """Refresh static labels when language changes."""
        self.title_label.configure(text=self.translator.translate("progress.title"))
        self.stats_label.configure(text=self.translator.translate("progress.stats"))
        self.log_label.configure(text=self.translator.translate("progress.logs"))
        self._refresh_stats_labels()
        if self.total_count > 0:
            progress = (self.completed_count + self.failed_count) / self.total_count * 100
            self.progress_label.configure(
                text=self.translator.translate("progress.percent", percent=progress)
            )
        else:
            self.progress_label.configure(text=self.translator.translate("progress.percent_zero"))
        self.set_current_file(self.current_filename)

    def _refresh_stats_labels(self):
        """Update statistics labels with current counts."""
        self.total_label.configure(text=self.translator.translate("progress.total", count=self.total_count))
        self.completed_label.configure(
            text=self.translator.translate("progress.completed", count=self.completed_count)
        )
        self.failed_label.configure(text=self.translator.translate("progress.failed", count=self.failed_count))

    def update_stats(self, status_counts: Dict[ImageStatus, int]):
        """Update statistics display."""
        self.total_count = sum(status_counts.values())
        self.completed_count = status_counts.get(ImageStatus.COMPLETED, 0)
        self.failed_count = status_counts.get(ImageStatus.FAILED, 0)

        self._refresh_stats_labels()

        if self.total_count > 0:
            progress = (self.completed_count + self.failed_count) / self.total_count * 100
            self.progress_bar.set(progress / 100)
            self.progress_label.configure(
                text=self.translator.translate("progress.percent", percent=progress)
            )
        else:
            self.progress_bar.set(0)
            self.progress_label.configure(text=self.translator.translate("progress.percent_zero"))

    def set_current_file(self, filename: str):
        """Update currently processing file."""
        self.current_filename = filename
        if filename == "None":
            text = self.translator.translate("progress.current_none")
        else:
            text = self.translator.translate("progress.current", filename=filename)
        self.current_label.configure(text=text)

    def add_log(self, message: str, level: str = "INFO"):
        """Add log message."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted = f"[{timestamp}] {message}\n"

        self.log_text.configure(state="normal")
        try:
            self._text_widget.insert("end", formatted, level)
        except Exception:
            self._text_widget.insert("end", formatted)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def clear_logs(self):
        """Clear all log messages."""
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def refresh_language(self):
        """External hook to force update."""
        self._update_static_texts()
