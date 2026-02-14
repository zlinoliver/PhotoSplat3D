"""Main application window coordinating all components.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from tkinter import messagebox
from pathlib import Path
from queue import Queue
from typing import List, Optional
import platform
import subprocess
from collections import Counter
import logging
import sys
import shutil
import json
import threading
from datetime import datetime

import customtkinter as ctk

from sharp.utils import io

from .control_panel import ControlPanel
from .image_list_panel import ImageListPanel
from .progress_panel import ProgressPanel
from .worker import WorkerThread
from .history_manager import HistoryManager
from .image_item import ImageItem, ImageStatus
from .constants import (
    WINDOW_SIZE,
    HISTORY_FILE,
    get_output_dir,
    APP_SUPPORT_DIR,
    CHECKPOINT_CACHE_DIR,
    INSTALL_INFO_FILE,
    MODEL_INFO_FILE,
)
from .config_manager import get_config_manager
from .i18n import get_translator
from .modal_client import convert_image_via_modal, ModalConversionError

logger = logging.getLogger(__name__)


ctk.set_appearance_mode("system")
ctk.set_default_color_theme("blue")


class SharpGUI(ctk.CTk):
    """Main GUI application window."""

    def __init__(self):
        super().__init__()

        self.translator = get_translator()
        self.config_manager = get_config_manager()
        initial_language = self.config_manager.get_language()
        self.translator.set_language(initial_language, notify=False)
        self.translator.register(self._on_language_changed)

        self.title(self._t("app.title"))
        self.geometry(WINDOW_SIZE)

        # State
        self.work_queue = Queue()
        self.worker: Optional[WorkerThread] = None
        self.online_thread: Optional[threading.Thread] = None
        self.is_running = False
        self.is_stopping = False
        self.is_online_mode = False
        self.history = HistoryManager(HISTORY_FILE)
        self.online_cancel_event = threading.Event()
        self._panorama_settings_window = None

        # Supported extensions
        self.supported_extensions = set(io.get_supported_image_extensions())
        self._initialize_storage_paths()
        self.checkpoint_path = self._ensure_cached_checkpoint()

        self._create_ui()
        self._setup_logging()

        # Welcome message
        self.progress_panel.add_log(
            self._t("log.welcome", name=self._t("app.title")),
            "INFO"
        )
        self.progress_panel.add_log(
            self._t("log.output_folder", path=get_output_dir()),
            "INFO"
        )

    def _create_ui(self):
        """Create main UI layout."""

        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        self.control_panel = ControlPanel(
            self,
            translator=self.translator,
            current_language=self.translator.language,
            on_import=self.import_images,
            on_start=self.start_conversion,
            on_stop=self.stop_conversion,
            on_open_output=self.open_output_folder,
            on_clear_completed=self.clear_completed,
            on_select_output=self.select_output_folder,
            on_language_change=self.change_language,
            on_online_convert=self.start_online_conversion,
            on_open_panorama_settings=self.show_panorama_settings_dialog,
        )
        self.control_panel.grid(row=0, column=0, sticky="ew", padx=16, pady=(16, 8))

        content = ctk.CTkFrame(self, fg_color="transparent")
        content.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 12))
        content.grid_columnconfigure(0, weight=3, uniform="main_cols")
        content.grid_columnconfigure(1, weight=2, uniform="main_cols")
        content.grid_rowconfigure(0, weight=1)

        self.image_list = ImageListPanel(content, translator=self.translator)
        self.image_list.grid(row=0, column=0, sticky="nsew", padx=(0, 12))

        self.progress_panel = ProgressPanel(content, translator=self.translator)
        self.progress_panel.grid(row=0, column=1, sticky="nsew")

        self.status_bar = ctk.CTkLabel(
            self,
            text=self._t("status.ready"),
            anchor="w",
            fg_color=("gray94", "#151515"),
            corner_radius=8,
            padx=14,
            height=34,
        )
        self.status_bar.grid(row=2, column=0, sticky="ew", padx=16, pady=(0, 16))

        self.notice_label = ctk.CTkLabel(
            self,
            text=self._t("app.notice"),
            anchor="center",
            fg_color="transparent",
            font=ctk.CTkFont(size=12),
            wraplength=760,
            text_color=("gray40", "#d0d0d0"),
        )
        self.notice_label.grid(row=3, column=0, sticky="ew", padx=16, pady=(0, 12))

    def show_panorama_settings_dialog(self):
        """Display configuration dialog for panorama face size."""
        if self._panorama_settings_window is not None and self._panorama_settings_window.winfo_exists():
            self._panorama_settings_window.focus_set()
            self._panorama_settings_window.lift()
            return

        dialog = ctk.CTkToplevel(self)
        dialog.title(self._t("dialog.panorama_title"))
        dialog.geometry("360x220")
        dialog.resizable(False, False)
        dialog.transient(self)
        dialog.grab_set()
        self._panorama_settings_window = dialog

        def _close():
            if self._panorama_settings_window is not None:
                try:
                    self._panorama_settings_window.destroy()
                except Exception:
                    pass
            self._panorama_settings_window = None

        dialog.protocol("WM_DELETE_WINDOW", _close)

        container = ctk.CTkFrame(dialog, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=20, pady=20)

        label = ctk.CTkLabel(
            container,
            text=self._t("dialog.panorama_face_size"),
            anchor="w",
        )
        label.pack(fill="x")

        auto_label = self._t("dialog.panorama_auto")
        options = [auto_label, "1024", "1536", "2048"]
        current_value = self.config_manager.get_panorama_face_size()
        current_label = auto_label if current_value is None else str(current_value)

        selection = ctk.StringVar(value=current_label)
        menu = ctk.CTkOptionMenu(
            container,
            values=options,
            variable=selection,
            width=180,
        )
        menu.pack(fill="x", pady=(6, 12))

        button_frame = ctk.CTkFrame(container, fg_color="transparent")
        button_frame.pack(fill="x", pady=(10, 0))

        def _save():
            value = selection.get()
            face_size = None if value == auto_label else int(value)
            self.config_manager.set_panorama_face_size(face_size)
            messagebox.showinfo(
                self._t("dialog.panorama_title"),
                self._t("dialog.panorama_saved"),
            )
            _close()

        cancel_btn = ctk.CTkButton(
            button_frame,
            text=self._t("dialog.cancel"),
            command=_close,
            width=100,
        )
        cancel_btn.pack(side="right", padx=(8, 0))

        save_btn = ctk.CTkButton(
            button_frame,
            text=self._t("dialog.panorama_save"),
            command=_save,
            width=120,
        )
        save_btn.pack(side="right")

    def _initialize_storage_paths(self):
        """Ensure shared directories exist and record installation metadata."""
        try:
            APP_SUPPORT_DIR.mkdir(parents=True, exist_ok=True)
            CHECKPOINT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logger.warning("Unable to prepare app support directories: %s", exc)
        self.install_info_file = INSTALL_INFO_FILE
        self.model_info_file = MODEL_INFO_FILE
        self._record_install_path()

    def _record_install_path(self):
        bundle_path = self._detect_install_path()
        if not bundle_path:
            return
        data = {
            "path": str(bundle_path),
            "updated_at": datetime.now().isoformat(),
        }
        try:
            self.install_info_file.parent.mkdir(parents=True, exist_ok=True)
            self.install_info_file.write_text(json.dumps(data, indent=2))
        except Exception as exc:
            logger.warning("Could not record install path: %s", exc)

    def _detect_install_path(self) -> Optional[Path]:
        executable = Path(sys.executable).resolve()
        for parent in executable.parents:
            if parent.suffix == ".app":
                return parent
        # Development fallback
        return Path(__file__).resolve().parents[3]

    def _ensure_cached_checkpoint(self) -> Optional[Path]:
        cached = self._discover_checkpoint(CHECKPOINT_CACHE_DIR)
        if cached:
            logger.info("Using cached checkpoint at %s", cached)
            self._write_model_metadata(cached)
            return cached

        bundled = self._find_bundled_checkpoint()
        if bundled:
            target = CHECKPOINT_CACHE_DIR / bundled.name
            try:
                shutil.copy2(bundled, target)
                logger.info("Copied bundled checkpoint to cache: %s", target)
                self._write_model_metadata(target)
                return target
            except Exception as exc:
                logger.error("Failed to cache bundled checkpoint: %s", exc)
                return bundled

        default_cache = Path.home() / ".cache/torch/hub/checkpoints"
        fallback = self._discover_checkpoint(default_cache)
        if fallback:
            try:
                target = CHECKPOINT_CACHE_DIR / fallback.name
                shutil.copy2(fallback, target)
                logger.info("Imported checkpoint from default cache: %s", target)
                self._write_model_metadata(target)
                return target
            except Exception as exc:
                logger.warning("Failed to import fallback checkpoint: %s", exc)
                return fallback

        logger.warning("No local checkpoint found; will download at runtime.")
        return None

    def _discover_checkpoint(self, directory: Optional[Path]) -> Optional[Path]:
        if not directory or not directory.exists():
            return None
        for pattern in ("*.pt", "*.pth", "*.ckpt", "*.safetensors"):
            matches = sorted(directory.glob(pattern))
            if matches:
                return matches[0]
        return None

    def _find_bundled_checkpoint(self) -> Optional[Path]:
        candidates = []
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            candidates.append(Path(meipass) / "checkpoints")

        module_path = Path(__file__).resolve()
        for parent in list(module_path.parents)[:5]:
            candidates.append(parent / "checkpoints")

        seen = set()
        for directory in candidates:
            if directory in seen:
                continue
            seen.add(directory)
            checkpoint = self._discover_checkpoint(directory)
            if checkpoint:
                logger.info("Found bundled checkpoint at %s", checkpoint)
                return checkpoint
        return None

    def _write_model_metadata(self, path: Path):
        try:
            data = {
                "path": str(path),
                "filename": path.name,
                "updated_at": datetime.now().isoformat(),
                "size_bytes": path.stat().st_size,
            }
            self.model_info_file.parent.mkdir(parents=True, exist_ok=True)
            self.model_info_file.write_text(json.dumps(data, indent=2))
        except Exception as exc:
            logger.warning("Unable to store model metadata: %s", exc)

    def _setup_logging(self):
        """Setup logging to GUI."""

        class GUIHandler(logging.Handler):
            def __init__(self, panel):
                super().__init__()
                self.panel = panel

            def emit(self, record):
                msg = self.format(record)
                level = record.levelname
                self.panel.add_log(msg, level)

        handler = GUIHandler(self.progress_panel)
        handler.setFormatter(logging.Formatter('%(message)s'))
        logging.getLogger('sharp').addHandler(handler)
        logging.getLogger('sharp').setLevel(logging.INFO)

    def import_images(self, paths: List[Path]):
        """Import images into queue."""
        valid_paths = []
        unsupported = 0

        for path in paths:
            # Check extension
            if path.suffix not in self.supported_extensions:
                unsupported += 1
                continue

            valid_paths.append(path)

        if valid_paths:
            self.image_list.add_images(valid_paths)
            self.progress_panel.add_log(
                self._t("log.imported", count=len(valid_paths)),
                "INFO"
            )
            self.update_ui_state()
            self.update_statistics()

        if unsupported > 0:
            messagebox.showwarning(
                self._t("dialog.unsupported_title"),
                self._t("dialog.unsupported_message", count=unsupported)
            )

    def start_conversion(self):
        """Start the conversion process."""
        pending = self.image_list.get_pending_items()

        if not pending:
            messagebox.showwarning(
                self._t("dialog.no_images_title"),
                self._t("dialog.no_images_message")
            )
            return

        if self.is_stopping:
            messagebox.showinfo(
                self._t("dialog.stopping_title"),
                self._t("dialog.stopping_message")
            )
            return

        # Add items to work queue
        for item in pending:
            self.work_queue.put(item)

        # Start worker thread
        self.worker = WorkerThread(
            work_queue=self.work_queue,
            status_callback=self.on_status_update,
            event_callback=self.on_worker_event,
            checkpoint_path=self.checkpoint_path,
            panorama_face_size=self.config_manager.get_panorama_face_size(),
        )
        self.worker.start()

        self.is_running = True
        self.update_ui_state()

        self.progress_panel.add_log(
            self._t("log.start", count=len(pending)),
            "INFO"
        )

    def start_online_conversion(self):
        """Start Modal-based conversion for all pending images."""
        if self.is_running or self.is_stopping:
            messagebox.showinfo(
                self._t("dialog.stopping_title"),
                self._t("dialog.stopping_message")
            )
            return

        pending = self.image_list.get_pending_items()
        if not pending:
            messagebox.showwarning(
                self._t("dialog.no_images_title"),
                self._t("dialog.no_images_message")
            )
            return

        endpoint = self.config_manager.get_modal_endpoint()
        if not endpoint:
            messagebox.showerror(
                self._t("dialog.error_title"),
                self._t("dialog.modal_endpoint_missing")
            )
            return

        self.is_running = True
        self.is_online_mode = True
        self.online_cancel_event.clear()
        self.progress_panel.add_log(
            self._t("log.start", count=len(pending)),
            "INFO"
        )
        self.update_ui_state()

        thread = threading.Thread(
            target=self._run_online_conversion,
            args=(list(pending), endpoint),
            daemon=True,
        )
        self.online_thread = thread
        thread.start()

    def stop_conversion(self):
        """Stop the conversion process."""
        worker_active = self.worker and self.worker.is_alive()
        online_active = self.online_thread and self.online_thread.is_alive()

        if not worker_active and not online_active:
            return

        if messagebox.askyesno(
            self._t("dialog.stop_confirm_title"),
            self._t("dialog.stop_confirm_message")
        ):
            self.is_stopping = True
            self.progress_panel.add_log(self._t("log.stopping"), "WARNING")

            if worker_active:
                self.worker.stop()

                # Clear queue
                while not self.work_queue.empty():
                    try:
                        self.work_queue.get_nowait()
                    except Exception:
                        break

                self._wait_for_worker_stop()
            elif online_active:
                self.online_cancel_event.set()

    def _wait_for_worker_stop(self):
        """Poll worker thread until it fully stops before resetting state."""
        if self.worker and self.worker.is_alive():
            self.after(200, self._wait_for_worker_stop)
            return

        self.worker = None
        self.is_running = False
        self.is_stopping = False
        self.progress_panel.add_log(self._t("log.stopped"), "WARNING")
        self.update_ui_state()

    def _run_online_conversion(self, items: List[ImageItem], endpoint: str):
        """Background worker invoking Modal endpoint for each image."""
        output_dir = get_output_dir()
        cancelled = False

        for item in items:
            if self.online_cancel_event.is_set():
                cancelled = True
                break

            try:
                item.status = ImageStatus.PROCESSING
                item.started_at = datetime.now()
                self.on_status_update(item)

                self._log_async(self._t("log.online_uploading", filename=item.name))
                result = convert_image_via_modal(endpoint, item.path)

                output_path = self._save_modal_output(output_dir, item, result.ply_bytes)

                item.status = ImageStatus.COMPLETED
                item.output_path = output_path
                item.completed_at = datetime.now()
                self.history.add_conversion(item.path, output_path)

                self._log_async(self._t("log.online_received", filename=item.name, path=str(output_path)))

                for log_entry in result.logs:
                    self._log_async(str(log_entry))

            except ModalConversionError as exc:
                item.status = ImageStatus.FAILED
                item.error_message = str(exc)
                self._log_async(
                    self._t("log.online_failed", filename=item.name, error=item.error_message),
                    "ERROR",
                )
            except Exception as exc:
                item.status = ImageStatus.FAILED
                item.error_message = str(exc)
                self._log_async(
                    self._t("log.online_failed", filename=item.name, error=item.error_message),
                    "ERROR",
                )
            finally:
                self.on_status_update(item)

        if self.online_cancel_event.is_set():
            cancelled = True

        self.after(0, self._finish_online_conversion, cancelled)

    def _save_modal_output(self, output_dir: Path, item: ImageItem, ply_bytes: bytes) -> Path:
        """Persist Modal PLY output to disk with unique filenames."""
        suffix = "_gaussian"
        candidate = output_dir / f"{item.path.stem}{suffix}.ply"
        counter = 1

        while candidate.exists():
            candidate = output_dir / f"{item.path.stem}{suffix}_{counter}.ply"
            counter += 1

        with open(candidate, "wb") as handle:
            handle.write(ply_bytes)

        return candidate

    def _finish_online_conversion(self, cancelled: bool):
        """Reset UI state after online thread completes."""
        self.online_thread = None
        self.is_online_mode = False
        self.online_cancel_event.clear()
        self.worker = None
        self.is_running = False
        self.is_stopping = False

        if cancelled:
            self.progress_panel.add_log(self._t("log.stopped"), "WARNING")
        else:
            self.progress_panel.add_log(self._t("log.all_done"), "SUCCESS")

        self.progress_panel.set_current_file("None")
        self.update_ui_state()

    def _log_async(self, message: str, level: str = "INFO"):
        """Thread-safe logging helper."""
        self.after(0, self.progress_panel.add_log, message, level)

    def clear_completed(self):
        """Clear completed items from list."""
        self.image_list.clear_completed()
        self.update_statistics()

    def open_output_folder(self):
        """Open output directory in file explorer."""
        output_path = get_output_dir()

        system = platform.system()
        try:
            if system == "Darwin":  # macOS
                subprocess.run(["open", str(output_path)])
            elif system == "Windows":
                subprocess.run(["explorer", str(output_path)])
            else:  # Linux
                subprocess.run(["xdg-open", str(output_path)])
        except Exception as e:
            messagebox.showerror(
                self._t("dialog.error_title"),
                self._t("dialog.open_output_error", error=e)
            )

    def select_output_folder(self, folder: Path):
        """Handle user selecting a new output folder."""
        try:
            # Save the new output directory
            self.config_manager.set_output_directory(folder)

            # Show confirmation
            messagebox.showinfo(
                self._t("dialog.output_changed_title"),
                self._t("dialog.output_changed_message", folder=folder)
            )

            logger.info(f"User changed output folder to: {folder}")

        except Exception as e:
            logger.error(f"Failed to change output folder: {e}")
            messagebox.showerror(
                self._t("dialog.error_title"),
                self._t("dialog.change_output_error", error=e)
            )

    def on_status_update(self, item: ImageItem):
        """Callback when item status changes."""
        # Update UI (must be in main thread)
        self.after(0, self._update_item_ui, item)

    def on_worker_event(self, event_key: str, payload: dict):
        """Receive structured events from worker thread."""
        self.after(0, self._handle_worker_event, event_key, payload)

    def _handle_worker_event(self, event_key: str, payload: dict):
        """Render worker events as localized log messages."""
        key_map = {
            "model_loading": "log.model_loading",
            "model_device": "log.model_device",
            "model_download": "log.model_download",
            "model_checkpoint": "log.model_checkpoint",
            "model_loaded": "log.model_loaded",
            "inference_start": "log.inference_start",
            "inference_end": "log.inference_end",
            "saving_output": "log.saving_output",
            "panorama_detected": "log.panorama_detected",
            "panorama_config": "log.panorama_config",
        }

        translation_key = key_map.get(event_key)
        if not translation_key:
            return

        message = self._t(translation_key, **payload)
        self.progress_panel.add_log(message, "INFO")

        if event_key in {"inference_start", "inference_end"} and "filename" in payload:
            self.progress_panel.set_current_file(payload["filename"])

    def _update_item_ui(self, item: ImageItem):
        """Update UI for status change (runs in main thread)."""
        self.image_list.update_item_status(item)

        if item.status == ImageStatus.PROCESSING:
            self.progress_panel.set_current_file(item.name)
            self.progress_panel.add_log(self._t("log.processing", filename=item.name), "INFO")

        elif item.status == ImageStatus.COMPLETED:
            self.progress_panel.add_log(self._t("log.completed", filename=item.name), "SUCCESS")
            self.history.add_conversion(item.path, item.output_path)

        elif item.status == ImageStatus.FAILED:
            self.progress_panel.add_log(
                self._t("log.failed", filename=item.name, error=item.error_message),
                "ERROR"
            )

        self.update_statistics()

        # Check if all done (no pending or processing items)
        if self.is_running and not self._has_active_items():
            if not self.is_online_mode:
                self.after(300, self._check_completion)

    def _check_completion(self):
        """Check if conversion is complete."""
        if not self.is_running:
            return
        if self._has_active_items():
            return
        self.is_running = False
        self.worker = None
        self.progress_panel.add_log(self._t("log.all_done"), "SUCCESS")
        self.progress_panel.set_current_file("None")
        self.update_ui_state()

    def update_statistics(self):
        """Update progress statistics."""
        status_counts = Counter(item.status for item in self.image_list.items)
        self.progress_panel.update_stats(status_counts)

    def update_ui_state(self):
        """Update button states based on current state."""
        if self.is_stopping:
            self.control_panel.set_state(has_images=False, is_running=True)
            self.status_bar.configure(text=self._t("status.stopping"))
            return

        has_pending = bool(self.image_list.get_pending_items())

        self.control_panel.set_state(
            has_images=has_pending,
            is_running=self.is_running
        )

        # Update status bar
        if self.is_running:
            self.status_bar.configure(text=self._t("status.running"))
        else:
            self.status_bar.configure(text=self._t("status.ready"))

    def change_language(self, language: str):
        """Handle user-requested language change."""
        self.config_manager.set_language(language)
        self.translator.set_language(language)

    def _on_language_changed(self):
        """Refresh top-level UI strings when language changes."""
        self.title(self._t("app.title"))
        self.control_panel.refresh_language()
        self.image_list.refresh_language()
        self.progress_panel.refresh_language()
        self.update_ui_state()
        if hasattr(self, "notice_label"):
            self.notice_label.configure(text=self._t("app.notice"))

    def _t(self, key: str, **kwargs) -> str:
        """Translate helper."""
        return self.translator.translate(key, **kwargs)

    def _has_active_items(self) -> bool:
        """Return True if there are pending or processing items."""
        return any(
            item.status in {ImageStatus.PENDING, ImageStatus.PROCESSING}
            for item in self.image_list.items
        )
