"""Top control panel with action buttons.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from tkinter import filedialog
from pathlib import Path
from typing import Callable, Dict, List

import customtkinter as ctk

from .constants import get_output_dir
from .i18n import Translator


class ControlPanel(ctk.CTkFrame):
    """Top toolbar with control buttons built with CustomTkinter."""

    def __init__(
        self,
        parent,
        translator: Translator,
        current_language: str,
        on_import: Callable[[List[Path]], None],
        on_start: Callable[[], None],
        on_stop: Callable[[], None],
        on_open_output: Callable[[], None],
        on_clear_completed: Callable[[], None],
        on_select_output: Callable[[Path], None],
        on_language_change: Callable[[str], None],
        on_online_convert: Callable[[], None],
        on_open_panorama_settings: Callable[[], None],
    ):
        super().__init__(parent, fg_color="transparent")

        self.translator = translator
        self.current_language = current_language
        self.on_import = on_import
        self.on_start = on_start
        self.on_stop = on_stop
        self.on_open_output = on_open_output
        self.on_clear_completed = on_clear_completed
        self.on_select_output = on_select_output
        self.on_language_change = on_language_change
        self.on_online_convert = on_online_convert
        self.on_open_panorama_settings = on_open_panorama_settings

        self.language_var = ctk.StringVar(value=self.translator.language_name(current_language))
        self.language_options: Dict[str, str] = {}
        self.button_font = ctk.CTkFont(size=12)

        self.translator.register(self._update_texts)
        self._create_widgets()

    def _create_widgets(self):
        """Create toolbar buttons."""
        padding_opts = {"padx": 3, "pady": 6}
        primary_width = 90
        start_stop_width = 80

        self.import_btn = ctk.CTkButton(
            self,
            command=self._on_import_clicked,
            height=38,
            font=self.button_font,
            width=start_stop_width,
        )
        self.import_btn.pack(side="left", **padding_opts)

        self._add_separator()

        self.start_btn = ctk.CTkButton(
            self,
            command=self.on_start,
            state="disabled",
            height=38,
            font=self.button_font,
            width=start_stop_width,
        )
        self.start_btn.pack(side="left", **padding_opts)

        self.online_btn = ctk.CTkButton(
            self,
            command=self.on_online_convert,
            state="disabled",
            height=38,
            font=self.button_font,
            width=primary_width,
        )
        # Intentionally hidden from the toolbar until ready to expose.

        self.stop_btn = ctk.CTkButton(
            self,
            command=self.on_stop,
            state="disabled",
            fg_color="#d9534f",
            hover_color="#c9302c",
            height=38,
            font=self.button_font,
            width=primary_width,
        )
        self.stop_btn.pack(side="left", **padding_opts)

        self._add_separator()

        self.clear_btn = ctk.CTkButton(
            self,
            command=self.on_clear_completed,
            height=38,
            font=self.button_font,
            width=primary_width,
        )
        self.clear_btn.pack(side="left", **padding_opts)

        # Language selector container
        self.language_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.language_frame.pack(side="right", padx=6, pady=6)

        self.panorama_settings_btn = ctk.CTkButton(
            self.language_frame,
            command=self.on_open_panorama_settings,
            height=32,
            width=120,
            font=self.button_font,
        )
        self.panorama_settings_btn.pack(side="right", padx=(6, 0))

        self.language_label = ctk.CTkLabel(self.language_frame)
        self.language_label.pack(side="left", padx=(0, 6))

        self.language_menu = ctk.CTkOptionMenu(
            self.language_frame,
            variable=self.language_var,
            values=[],
            command=self._on_language_selected,
            width=90,
        )
        self.language_menu.pack(side="left")

        # Output folder controls
        self.output_btn = ctk.CTkButton(
            self,
            command=self.on_open_output,
            height=38,
            font=self.button_font,
            width=0,
        )
        self.output_btn.pack(side="left", **padding_opts)

        self.select_output_btn = ctk.CTkButton(
            self,
            command=self._on_select_output_clicked,
            height=38,
            font=self.button_font,
            width=0,
        )
        self.select_output_btn.pack(side="left", **padding_opts)

        self._update_language_options()
        self._update_texts()

    def _add_separator(self):
        """Add a subtle separator block."""
        separator = ctk.CTkFrame(self, width=2, fg_color=("gray80", "gray30"), height=32)
        separator.pack(side="left", fill="y", padx=4, pady=6)
        separator.pack_propagate(False)

    def _on_import_clicked(self):
        """Handle import button click."""
        file_paths = filedialog.askopenfilenames(
            title=self.translator.translate("dialog.select_images"),
            filetypes=[
                (self.translator.translate("dialog.image_files"), "*.jpg *.jpeg *.png *.heic *.JPG *.JPEG *.PNG *.HEIC"),
                (self.translator.translate("dialog.all_files"), "*.*")
            ]
        )

        if file_paths:
            paths = [Path(p) for p in file_paths]
            self.on_import(paths)

    def _on_select_output_clicked(self):
        """Handle select output folder button click."""
        current_dir = get_output_dir()

        selected_dir = filedialog.askdirectory(
            title=self.translator.translate("dialog.select_output_folder"),
            initialdir=str(current_dir)
        )

        if selected_dir:
            self.on_select_output(Path(selected_dir))

    def set_state(self, has_images: bool, is_running: bool):
        """Update button states based on application state."""
        if has_images and not is_running:
            self.start_btn.configure(state="normal")
            self.online_btn.configure(state="normal")
        else:
            self.start_btn.configure(state="disabled")
            self.online_btn.configure(state="disabled")

        if is_running:
            self.stop_btn.configure(state="normal")
        else:
            self.stop_btn.configure(state="disabled")

    def _update_texts(self):
        """Refresh button texts based on current language."""
        self.current_language = self.translator.language
        self.import_btn.configure(text=self.translator.translate("control.import"))
        self.start_btn.configure(text=self.translator.translate("control.start"))
        self.stop_btn.configure(text=self.translator.translate("control.stop"))
        self.clear_btn.configure(text=self.translator.translate("control.clear"))
        self.output_btn.configure(text=self.translator.translate("control.open_output"))
        self.select_output_btn.configure(text=self.translator.translate("control.change_output"))
        self.online_btn.configure(text=self.translator.translate("control.online"))
        self.panorama_settings_btn.configure(text=self.translator.translate("control.panorama_settings"))
        self.language_label.configure(text=self.translator.translate("control.language"))
        self._update_language_options()

    def _update_language_options(self):
        """Update option menu entries to reflect translations."""
        names = []
        mapping: Dict[str, str] = {}
        for code in self.translator.available_languages():
            name = self.translator.language_name(code)
            names.append(name)
            mapping[name] = code
        self.language_options = mapping
        self.language_menu.configure(values=names)
        self.language_menu.set(self.translator.language_name(self.current_language))

    def _on_language_selected(self, selected_name: str):
        """Handle user selecting a new language."""
        code = self.language_options.get(selected_name)
        if not code or code == self.current_language:
            return

        self.current_language = code
        self.on_language_change(code)

    def refresh_language(self):
        """External hook to refresh labels."""
        self._update_texts()
