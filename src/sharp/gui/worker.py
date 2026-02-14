"""Background worker for processing images without blocking GUI.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

import logging
import os
import threading
from queue import Queue, Empty
from pathlib import Path
from typing import Callable, Optional
from datetime import datetime
import torch

from sharp.models import PredictorParams, create_predictor
from sharp.utils import io
from sharp.utils.gaussians import save_ply
from sharp.utils import panorama
from sharp.cli.predict import predict_image
from .image_item import ImageItem, ImageStatus
from .constants import get_output_dir

logger = logging.getLogger(__name__)


class WorkerThread(threading.Thread):
    """Background thread for processing image conversions."""

    def __init__(
        self,
        work_queue: Queue[ImageItem],
        status_callback: Callable[[ImageItem], None],
        event_callback: Optional[Callable[[str, dict], None]] = None,
        checkpoint_path: Optional[Path] = None,
        device: str = "default",
        panorama_face_size: Optional[int] = None,
    ):
        super().__init__(daemon=True)
        self.work_queue = work_queue
        self.status_callback = status_callback
        self.event_callback = event_callback
        self.checkpoint_path = checkpoint_path
        self.device_name = device
        self.panorama_face_size = panorama_face_size

        self._stop_event = threading.Event()
        self._stop_sentinel = object()

        self.model = None
        self.device = None

    def stop(self):
        """Signal thread to stop."""
        self._stop_event.set()
        try:
            self.work_queue.put_nowait(self._stop_sentinel)
        except Exception:
            # If queue is full or closed, ignore; loop checks stop event.
            pass

    def _emit_event(self, key: str, **payload):
        if self.event_callback:
            self.event_callback(key, payload)

    def _load_model(self):
        """Load model once at startup."""
        self._emit_event("model_loading")
        logger.info("Loading SHARP model...")

        # Determine device
        if self.device_name == "default":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(self.device_name)

        logger.info(f"Using device: {self.device}")
        self._emit_event("model_device", device=str(self.device))

        # Load checkpoint
        if self.checkpoint_path is None:
            DEFAULT_MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"
            logger.info("Loading model from cache (or downloading if needed)")
            self._emit_event("model_download")
            state_dict = torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL, progress=True)
        else:
            logger.info(f"Loading checkpoint from {self.checkpoint_path}")
            self._emit_event("model_checkpoint", path=str(self.checkpoint_path))
            state_dict = torch.load(self.checkpoint_path, weights_only=True)

        # Create and initialize model
        self.model = create_predictor(PredictorParams())
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model.to(self.device)

        logger.info("Model loaded successfully")
        self._emit_event("model_loaded")

    def _process_image(self, item: ImageItem, output_dir: Path):
        """Process a single image."""
        try:
            # Update status to processing
            item.status = ImageStatus.PROCESSING
            item.started_at = datetime.now()
            self.status_callback(item)

            # Load image
            logger.info(f"Loading image: {item.path}")
            image, icc_profile, f_px = io.load_rgb(item.path)
            height, width = image.shape[:2]

            # Run prediction (panorama or standard)
            logger.info(f"Running inference on: {item.path.name}")
            panorama_mode = panorama.detect_panorama_mode(image)
            if panorama_mode:
                strategy = panorama.get_default_strategy(panorama_mode)
                face_size = self.panorama_face_size or panorama.default_face_size(image)
                flip_poles = os.getenv("SHARP_PANORAMA_FLIP_POLES", "0").lower() in {"1", "true", "yes"}
                flip_y = os.getenv("SHARP_PANORAMA_FLIP_Y", "1").lower() not in {"0", "false", "no"}
                global_flip_y = os.getenv("SHARP_PANORAMA_GLOBAL_FLIP_Y", "1").lower() not in {"0", "false", "no"}
                faces = panorama.generate_faces(
                    image,
                    panorama_mode,
                    strategy=strategy,
                    face_size=face_size,
                    overlap_deg=0.0,
                    flip_poles=flip_poles,
                )
                self._emit_event(
                    "panorama_detected",
                    filename=item.name,
                    mode=panorama_mode,
                    faces=len(faces),
                )
                self._emit_event(
                    "panorama_config",
                    filename=item.name,
                    strategy=strategy,
                    face_size=face_size,
                    flip_poles=flip_poles,
                    flip_y=flip_y,
                    global_flip_y=global_flip_y,
                )
                gaussians_list = []
                for face in faces:
                    self._emit_event(
                        "inference_start",
                        filename=f"{item.name}:{face.name}",
                    )
                    gaussians_face = predict_image(self.model, face.image, face.f_px, self.device)
                    self._emit_event(
                        "inference_end",
                        filename=f"{item.name}:{face.name}",
                    )
                    gaussians_face = panorama.filter_gaussians_by_mask(
                        gaussians_face,
                        face.mask,
                        face.f_px,
                    )
                    gaussians_list.append(
                        panorama.transform_gaussians_to_world(
                            gaussians_face,
                            face.camera_to_world,
                            flip_y=flip_y,
                        )
                    )
                gaussians = panorama.merge_gaussians(gaussians_list)
                if global_flip_y:
                    gaussians = panorama.flip_gaussians_y(gaussians)
                f_px = faces[0].f_px
                height = faces[0].face_size
                width = faces[0].face_size
            else:
                self._emit_event("inference_start", filename=item.name)
                gaussians = predict_image(self.model, image, f_px, self.device)
                self._emit_event("inference_end", filename=item.name)

            # Save output
            output_path = output_dir / f"{item.path.stem}.ply"
            logger.info(f"Saving to: {output_path}")
            self._emit_event("saving_output", path=str(output_path))
            save_ply(gaussians, f_px, (height, width), output_path)

            # Update status
            item.status = ImageStatus.COMPLETED
            item.output_path = output_path
            item.completed_at = datetime.now()

        except Exception as e:
            logger.error(f"Failed to process {item.path}: {e}", exc_info=True)
            item.status = ImageStatus.FAILED
            item.error_message = str(e)

        finally:
            self.status_callback(item)

    def run(self):
        """Main worker loop."""
        try:
            # Load model once
            self._load_model()

            output_dir = get_output_dir()

            while not self._stop_event.is_set():
                try:
                    # Get next item (timeout to check stop event periodically)
                    item = self.work_queue.get(timeout=0.5)

                    if item is self._stop_sentinel:
                        self.work_queue.task_done()
                        break

                    # Process image
                    self._process_image(item, output_dir)

                    # Mark task as done
                    self.work_queue.task_done()

                except Empty:
                    # No items in queue, continue loop
                    continue

        except Exception as e:
            logger.error(f"Worker thread crashed: {e}", exc_info=True)
