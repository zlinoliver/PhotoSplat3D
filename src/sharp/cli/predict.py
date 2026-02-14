"""Contains `sharp predict` CLI implementation.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data

from sharp.models import (
    PredictorParams,
    RGBGaussianPredictor,
    create_predictor,
)
from sharp.utils import io
from sharp.utils import logging as logging_utils
from sharp.utils.gaussians import (
    Gaussians3D,
    SceneMetaData,
    save_ply,
    unproject_gaussians,
)
from sharp.utils import panorama

from .render import render_gaussians

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"


@click.command()
@click.option(
    "-i",
    "--input-path",
    type=click.Path(path_type=Path, exists=True),
    help="Path to an image or containing a list of images.",
    required=True,
)
@click.option(
    "-o",
    "--output-path",
    type=click.Path(path_type=Path, file_okay=False),
    help="Path to save the predicted Gaussians and renderings.",
    required=True,
)
@click.option(
    "-c",
    "--checkpoint-path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Path to the .pt checkpoint. If not provided, downloads the default model automatically.",
    required=False,
)
@click.option(
    "--render/--no-render",
    "with_rendering",
    is_flag=True,
    default=False,
    help="Whether to render trajectory for checkpoint.",
)
@click.option(
    "--device",
    type=str,
    default="default",
    help="Device to run on. ['cpu', 'mps', 'cuda']",
)
@click.option(
    "--panorama",
    "panorama_mode",
    type=click.Choice(["auto", "180", "360", "none"], case_sensitive=False),
    default="auto",
    help="Enable panorama slicing for equirectangular inputs.",
)
@click.option(
    "--panorama-strategy",
    type=str,
    default=None,
    help="Panorama slice strategy (e.g. cube6, ring8, ring12, front4).",
)
@click.option(
    "--panorama-face-size",
    type=int,
    default=None,
    help="Override panorama face size in pixels.",
)
@click.option(
    "--panorama-overlap",
    type=float,
    default=0.0,
    help="Extra overlap degrees between panorama slices.",
)
@click.option(
    "--panorama-flip-poles",
    is_flag=True,
    default=False,
    help="Flip panorama pole orientation (up/down faces).",
)
@click.option(
    "--panorama-flip-y/--no-panorama-flip-y",
    default=True,
    help="Flip Y axis when merging panorama faces.",
)
@click.option(
    "--panorama-global-flip-y/--no-panorama-global-flip-y",
    default=True,
    help="Flip final panorama output along Y axis.",
)
@click.option("-v", "--verbose", is_flag=True, help="Activate debug logs.")
def predict_cli(
    input_path: Path,
    output_path: Path,
    checkpoint_path: Path,
    with_rendering: bool,
    device: str,
    panorama_mode: str,
    panorama_strategy: str | None,
    panorama_face_size: int | None,
    panorama_overlap: float,
    panorama_flip_poles: bool,
    panorama_flip_y: bool,
    panorama_global_flip_y: bool,
    verbose: bool,
):
    """Predict Gaussians from input images."""
    logging_utils.configure(logging.DEBUG if verbose else logging.INFO)

    extensions = io.get_supported_image_extensions()

    image_paths = []
    if input_path.is_file():
        if input_path.suffix in extensions:
            image_paths = [input_path]
    else:
        for ext in extensions:
            image_paths.extend(list(input_path.glob(f"**/*{ext}")))

    if len(image_paths) == 0:
        LOGGER.info("No valid images found. Input was %s.", input_path)
        return

    LOGGER.info("Processing %d valid image files.", len(image_paths))

    if device == "default":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    LOGGER.info("Using device %s", device)

    if with_rendering and device != "cuda":
        LOGGER.warning("Can only run rendering with gsplat on CUDA. Rendering is disabled.")
        with_rendering = False

    # Load or download checkpoint
    if checkpoint_path is None:
        LOGGER.info("No checkpoint provided. Downloading default model from %s", DEFAULT_MODEL_URL)
        state_dict = torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL, progress=True)
    else:
        LOGGER.info("Loading checkpoint from %s", checkpoint_path)
        state_dict = torch.load(checkpoint_path, weights_only=True)

    gaussian_predictor = create_predictor(PredictorParams())
    gaussian_predictor.load_state_dict(state_dict)
    gaussian_predictor.eval()
    gaussian_predictor.to(device)

    output_path.mkdir(exist_ok=True, parents=True)

    for image_path in image_paths:
        LOGGER.info("Processing %s", image_path)
        image, _, f_px = io.load_rgb(image_path)
        height, width = image.shape[:2]
        intrinsics = torch.tensor(
            [
                [f_px, 0, (width - 1) / 2.0, 0],
                [0, f_px, (height - 1) / 2.0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            device=device,
            dtype=torch.float32,
        )
        mode = None
        if panorama_mode.lower() != "none":
            if panorama_mode.lower() == "auto":
                mode = panorama.detect_panorama_mode(image)
            else:
                mode = panorama_mode.lower()
        if mode in {"180", "360"}:
            strategy = panorama_strategy or panorama.get_default_strategy(mode)
            face_size = panorama_face_size or panorama.default_face_size(image)
            faces = panorama.generate_faces(
                image,
                mode,
                strategy=strategy,
                face_size=face_size,
                overlap_deg=panorama_overlap,
                flip_poles=panorama_flip_poles,
            )
            LOGGER.info(
                "Panorama detected (%s) with %d faces. strategy=%s face_size=%s flip_poles=%s flip_y=%s global_flip_y=%s",
                mode,
                len(faces),
                strategy,
                face_size,
                panorama_flip_poles,
                panorama_flip_y,
                panorama_global_flip_y,
            )
            gaussians_list = []
            for face in faces:
                gaussians_face = predict_image(gaussian_predictor, face.image, face.f_px, torch.device(device))
                gaussians_face = panorama.filter_gaussians_by_mask(
                    gaussians_face,
                    face.mask,
                    face.f_px,
                )
                gaussians_list.append(
                    panorama.transform_gaussians_to_world(
                        gaussians_face,
                        face.camera_to_world,
                        flip_y=panorama_flip_y,
                    )
                )
            gaussians = panorama.merge_gaussians(gaussians_list)
            if panorama_global_flip_y:
                gaussians = panorama.flip_gaussians_y(gaussians)
            f_px = faces[0].f_px
            height = faces[0].face_size
            width = faces[0].face_size
        else:
            gaussians = predict_image(gaussian_predictor, image, f_px, torch.device(device))

        LOGGER.info("Saving 3DGS to %s", output_path)
        save_ply(gaussians, f_px, (height, width), output_path / f"{image_path.stem}.ply")

        if with_rendering:
            output_video_path = (output_path / image_path.stem).with_suffix(".mp4")
            LOGGER.info("Rendering trajectory to %s", output_video_path)

            metadata = SceneMetaData(intrinsics[0, 0].item(), (width, height), "linearRGB")
            render_gaussians(gaussians, metadata, output_video_path)


@torch.no_grad()
def predict_image(
    predictor: RGBGaussianPredictor,
    image: np.ndarray,
    f_px: float,
    device: torch.device,
) -> Gaussians3D:
    """Predict Gaussians from an image."""
    internal_shape = (1536, 1536)

    LOGGER.info("Running preprocessing.")
    image_pt = torch.from_numpy(image.copy()).float().to(device).permute(2, 0, 1) / 255.0
    _, height, width = image_pt.shape
    disparity_factor = torch.tensor([f_px / width]).float().to(device)

    image_resized_pt = F.interpolate(
        image_pt[None],
        size=(internal_shape[1], internal_shape[0]),
        mode="bilinear",
        align_corners=True,
    )

    # Predict Gaussians in the NDC space.
    LOGGER.info("Running inference.")
    gaussians_ndc = predictor(image_resized_pt, disparity_factor)

    LOGGER.info("Running postprocessing.")
    intrinsics = (
        torch.tensor(
            [
                [f_px, 0, width / 2, 0],
                [0, f_px, height / 2, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        .float()
        .to(device)
    )
    intrinsics_resized = intrinsics.clone()
    intrinsics_resized[0] *= internal_shape[0] / width
    intrinsics_resized[1] *= internal_shape[1] / height

    # Convert Gaussians to metrics space.
    gaussians = unproject_gaussians(
        gaussians_ndc, torch.eye(4).to(device), intrinsics_resized, internal_shape
    )

    return gaussians
