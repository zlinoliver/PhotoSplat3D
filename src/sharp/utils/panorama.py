"""Utilities for slicing equirectangular panoramas into perspective views."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Literal

import numpy as np
import torch
import torch.nn.functional as F

from .gaussians import Gaussians3D, apply_transform

PanoramaMode = Literal["180", "360"]

_RATIO_MIN = 1.8
_RATIO_MAX = 2.2


@dataclass(frozen=True)
class FaceSlice:
    """Holds a perspective face derived from a panorama."""

    name: str
    image: np.ndarray
    f_px: float
    camera_to_world: torch.Tensor
    face_size: int
    mask: np.ndarray


@dataclass(frozen=True)
class FacePose:
    """Pose definition for a panorama face."""

    name: str
    yaw_deg: float | None = None
    pitch_deg: float | None = None
    forward: tuple[float, float, float] | None = None
    up: tuple[float, float, float] | None = None
    roll_deg: float = 0.0


def detect_panorama_mode(image: np.ndarray) -> PanoramaMode | None:
    """Best-effort detection of 180/360 equirectangular panoramas."""
    height, width = image.shape[:2]
    if height == 0:
        return None

    ratio = width / float(height)
    if ratio < _RATIO_MIN or ratio > _RATIO_MAX:
        return None

    if _looks_like_half_blank(image):
        return "180"
    return "360"


def default_face_size(image: np.ndarray, max_size: int = 1536) -> int:
    """Pick a conservative default face size based on input resolution."""
    height, width = image.shape[:2]
    return int(max(256, min(height, width // 2, max_size)))


def get_default_strategy(mode: PanoramaMode) -> str:
    """Return default slicing strategy for a panorama mode."""
    return "cube6" if mode == "360" else "front4"


def get_strategy_faces(mode: PanoramaMode, strategy: str) -> list[FacePose]:
    """Return list of face poses for a strategy."""
    if mode == "360":
        if strategy == "ring8":
            return [FacePose(name=f"yaw_{yaw}", yaw_deg=yaw, pitch_deg=0.0) for yaw in range(0, 360, 45)]
        if strategy == "ring12":
            return [FacePose(name=f"yaw_{yaw}", yaw_deg=yaw, pitch_deg=0.0) for yaw in range(0, 360, 30)]
        # cube6 default
        return [
            FacePose(name="front", forward=(0.0, 0.0, 1.0), up=(0.0, 1.0, 0.0)),
            FacePose(name="right", forward=(1.0, 0.0, 0.0), up=(0.0, 1.0, 0.0)),
            FacePose(name="back", forward=(0.0, 0.0, -1.0), up=(0.0, 1.0, 0.0)),
            FacePose(name="left", forward=(-1.0, 0.0, 0.0), up=(0.0, 1.0, 0.0)),
            FacePose(name="up", forward=(0.0, 1.0, 0.0), up=(0.0, 0.0, 1.0), roll_deg=180.0),
            FacePose(name="down", forward=(0.0, -1.0, 0.0), up=(0.0, 0.0, -1.0), roll_deg=180.0),
        ]

    # 180 mode
    if strategy == "front6":
        return [
            FacePose(name="yaw_-90", yaw_deg=-90.0, pitch_deg=0.0),
            FacePose(name="yaw_-30", yaw_deg=-30.0, pitch_deg=0.0),
            FacePose(name="yaw_30", yaw_deg=30.0, pitch_deg=0.0),
            FacePose(name="yaw_90", yaw_deg=90.0, pitch_deg=0.0),
            FacePose(name="pitch_up", yaw_deg=0.0, pitch_deg=60.0),
            FacePose(name="pitch_down", yaw_deg=0.0, pitch_deg=-60.0),
        ]
    # front4 default
    return [
        FacePose(name="front", forward=(0.0, 0.0, 1.0), up=(0.0, 1.0, 0.0)),
        FacePose(name="right", forward=(1.0, 0.0, 0.0), up=(0.0, 1.0, 0.0)),
        FacePose(name="left", forward=(-1.0, 0.0, 0.0), up=(0.0, 1.0, 0.0)),
        FacePose(name="up", forward=(0.0, 1.0, 0.0), up=(0.0, 0.0, 1.0), roll_deg=180.0),
    ]


def get_strategy_fov_deg(mode: PanoramaMode, strategy: str) -> float:
    """Return a base FOV for a strategy."""
    if mode == "360":
        if strategy in {"ring8", "ring12"}:
            return 60.0
        return 90.0
    if strategy == "front6":
        return 75.0
    return 90.0


def generate_faces(
    image: np.ndarray,
    mode: PanoramaMode,
    *,
    strategy: str,
    face_size: int,
    overlap_deg: float = 0.0,
    pole_latitude_deg: float = 70.0,
    flip_poles: bool = False,
) -> list[FaceSlice]:
    """Generate perspective faces from an equirectangular panorama."""
    faces: list[FaceSlice] = []
    base_fov = get_strategy_fov_deg(mode, strategy)
    fov_deg = min(160.0, max(20.0, base_fov + overlap_deg))
    for pose in get_strategy_faces(mode, strategy):
        if flip_poles and pose.name in {"up", "down"} and pose.up is not None:
            pose = FacePose(
                name=pose.name,
                yaw_deg=pose.yaw_deg,
                pitch_deg=pose.pitch_deg,
                forward=pose.forward,
                up=(-pose.up[0], -pose.up[1], -pose.up[2]),
            )
        camera_to_world = _camera_to_world_from_pose(pose)
        image_face, f_px, mask = equirect_to_perspective(
            image=image,
            face_size=face_size,
            fov_deg=fov_deg,
            camera_to_world=camera_to_world,
            pole_latitude_deg=pole_latitude_deg,
        )
        camera_to_world_4x4 = torch.eye(4, dtype=torch.float32)
        camera_to_world_4x4[:3, :3] = camera_to_world
        faces.append(
            FaceSlice(
                name=pose.name,
                image=image_face,
                f_px=f_px,
                camera_to_world=camera_to_world_4x4,
                face_size=face_size,
                mask=mask,
            )
        )
    return faces


def merge_gaussians(
    gaussians_list: Iterable[Gaussians3D],
    *,
    opacity_threshold: float = 0.02,
) -> Gaussians3D:
    """Merge multiple Gaussians sets with a simple opacity filter."""
    merged = []
    for gaussians in gaussians_list:
        if gaussians.opacities is None:
            merged.append(gaussians)
            continue
        mask = gaussians.opacities.flatten(0, 1) >= opacity_threshold
        if mask.all():
            merged.append(gaussians)
            continue
        merged.append(
            Gaussians3D(
                mean_vectors=gaussians.mean_vectors.flatten(0, 1)[mask].unsqueeze(0),
                singular_values=gaussians.singular_values.flatten(0, 1)[mask].unsqueeze(0),
                quaternions=gaussians.quaternions.flatten(0, 1)[mask].unsqueeze(0),
                colors=gaussians.colors.flatten(0, 1)[mask].unsqueeze(0),
                opacities=gaussians.opacities.flatten(0, 1)[mask].unsqueeze(0),
            )
        )

    if not merged:
        raise ValueError("No Gaussians provided to merge.")

    mean_vectors = torch.cat([g.mean_vectors for g in merged], dim=1)
    singular_values = torch.cat([g.singular_values for g in merged], dim=1)
    quaternions = torch.cat([g.quaternions for g in merged], dim=1)
    colors = torch.cat([g.colors for g in merged], dim=1)
    opacities = torch.cat([g.opacities for g in merged], dim=1)

    return Gaussians3D(
        mean_vectors=mean_vectors,
        singular_values=singular_values,
        quaternions=quaternions,
        colors=colors,
        opacities=opacities,
    )


def filter_gaussians_by_mask(
    gaussians: Gaussians3D,
    mask: np.ndarray,
    f_px: float,
) -> Gaussians3D:
    """Filter Gaussians by a 2D boolean mask in face image space."""
    face_size = mask.shape[0]
    device = gaussians.mean_vectors.device
    mask_t = torch.from_numpy(mask.astype(np.bool_)).to(device=device)

    points = gaussians.mean_vectors.flatten(0, 1)
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    valid = z > 1e-6
    if not valid.any():
        return Gaussians3D(
            mean_vectors=gaussians.mean_vectors[:, :0, :],
            singular_values=gaussians.singular_values[:, :0, :],
            quaternions=gaussians.quaternions[:, :0, :],
            colors=gaussians.colors[:, :0, :],
            opacities=gaussians.opacities[:, :0],
        )

    cx = (face_size - 1) * 0.5
    cy = (face_size - 1) * 0.5
    u = f_px * x / z + cx
    v = f_px * y / z + cy

    u_idx = torch.round(u).long()
    v_idx = torch.round(v).long()

    in_bounds = (u_idx >= 0) & (u_idx < face_size) & (v_idx >= 0) & (v_idx < face_size)
    valid = valid & in_bounds
    if not valid.any():
        return Gaussians3D(
            mean_vectors=gaussians.mean_vectors[:, :0, :],
            singular_values=gaussians.singular_values[:, :0, :],
            quaternions=gaussians.quaternions[:, :0, :],
            colors=gaussians.colors[:, :0, :],
            opacities=gaussians.opacities[:, :0],
        )

    masked = mask_t[v_idx[valid], u_idx[valid]]
    keep = torch.zeros_like(valid)
    keep[valid] = masked

    if keep.all():
        return gaussians

    keep_idx = keep.nonzero(as_tuple=False).flatten()
    if keep_idx.numel() == 0:
        return Gaussians3D(
            mean_vectors=gaussians.mean_vectors[:, :0, :],
            singular_values=gaussians.singular_values[:, :0, :],
            quaternions=gaussians.quaternions[:, :0, :],
            colors=gaussians.colors[:, :0, :],
            opacities=gaussians.opacities[:, :0],
        )
    return Gaussians3D(
        mean_vectors=gaussians.mean_vectors[:, keep_idx, :],
        singular_values=gaussians.singular_values[:, keep_idx, :],
        quaternions=gaussians.quaternions[:, keep_idx, :],
        colors=gaussians.colors[:, keep_idx, :],
        opacities=gaussians.opacities[:, keep_idx],
    )


def transform_gaussians_to_world(
    gaussians: Gaussians3D,
    camera_to_world: torch.Tensor,
    *,
    flip_y: bool = True,
) -> Gaussians3D:
    """Apply a camera-to-world transform to Gaussians."""
    transform = camera_to_world.to(gaussians.mean_vectors.device, gaussians.mean_vectors.dtype)
    if flip_y:
        flip = torch.eye(4, device=transform.device, dtype=transform.dtype)
        flip[1, 1] = -1.0
        transform = transform @ flip
    return apply_transform(gaussians, transform[:3])


def flip_gaussians_y(gaussians: Gaussians3D) -> Gaussians3D:
    """Flip Gaussians along the Y axis in world space."""
    transform = torch.eye(4, device=gaussians.mean_vectors.device, dtype=gaussians.mean_vectors.dtype)
    transform[1, 1] = -1.0
    return apply_transform(gaussians, transform[:3])


def equirect_to_perspective(
    *,
    image: np.ndarray,
    face_size: int,
    fov_deg: float,
    camera_to_world: torch.Tensor,
    pole_latitude_deg: float = 70.0,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Project an equirectangular image into a perspective face."""
    src_height, src_width = image.shape[:2]
    image_t = torch.from_numpy(image.astype(np.float32) / 255.0)
    if image_t.ndim == 2:
        image_t = image_t[:, :, None]
    image_t = image_t.permute(2, 0, 1).unsqueeze(0)

    f_px = 0.5 * face_size / math.tan(math.radians(fov_deg) / 2.0)

    grid, lat = _build_grid(
        face_size=face_size,
        src_width=src_width,
        src_height=src_height,
        f_px=f_px,
        camera_to_world=camera_to_world,
    )
    sampled = F.grid_sample(
        image_t,
        grid.unsqueeze(0),
        mode="bilinear",
        align_corners=True,
    )
    output = (sampled[0].permute(1, 2, 0).clamp(0, 1) * 255.0).byte().cpu().numpy()
    lat_limit = math.radians(pole_latitude_deg)
    mask = (lat.abs() <= lat_limit).cpu().numpy()
    return output, float(f_px), mask


def _build_grid(
    *,
    face_size: int,
    src_width: int,
    src_height: int,
    f_px: float,
    camera_to_world: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    xs = torch.linspace(0, face_size - 1, face_size, dtype=torch.float32)
    ys = torch.linspace(0, face_size - 1, face_size, dtype=torch.float32)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing="xy")

    cx = (face_size - 1) * 0.5
    cy = (face_size - 1) * 0.5

    x = (grid_x - cx) / f_px
    y = -(grid_y - cy) / f_px
    z = torch.ones_like(x)
    dirs = torch.stack([x, y, z], dim=-1)
    dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)

    dirs_world = dirs @ camera_to_world.T

    lon = torch.atan2(dirs_world[..., 0], dirs_world[..., 2])
    lat = torch.asin(torch.clamp(dirs_world[..., 1], -1.0, 1.0))

    u = (lon / (2.0 * math.pi) + 0.5) * (src_width - 1)
    v = (0.5 - lat / math.pi) * (src_height - 1)

    grid_u = (u / (src_width - 1)) * 2.0 - 1.0
    grid_v = (v / (src_height - 1)) * 2.0 - 1.0

    return torch.stack([grid_u, grid_v], dim=-1), lat


def _rotation_matrix(yaw_deg: float, pitch_deg: float) -> torch.Tensor:
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)

    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    cos_pitch = math.cos(pitch)
    sin_pitch = math.sin(pitch)

    rot_y = torch.tensor(
        [
            [cos_yaw, 0.0, sin_yaw],
            [0.0, 1.0, 0.0],
            [-sin_yaw, 0.0, cos_yaw],
        ],
        dtype=torch.float32,
    )
    rot_x = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos_pitch, -sin_pitch],
            [0.0, sin_pitch, cos_pitch],
        ],
        dtype=torch.float32,
    )
    return rot_y @ rot_x


def _camera_to_world_from_pose(pose: FacePose) -> torch.Tensor:
    if pose.forward is not None and pose.up is not None:
        forward = torch.tensor(pose.forward, dtype=torch.float32)
        up = torch.tensor(pose.up, dtype=torch.float32)
        forward = forward / torch.norm(forward)
        up = up / torch.norm(up)
        right = torch.cross(up, forward)
        right = right / torch.norm(right)
        up = torch.cross(forward, right)
        up = up / torch.norm(up)
        rotation = torch.stack([right, up, forward], dim=1)
        if pose.roll_deg:
            roll = math.radians(pose.roll_deg)
            cos_r = math.cos(roll)
            sin_r = math.sin(roll)
            roll_matrix = torch.tensor(
                [
                    [cos_r, -sin_r, 0.0],
                    [sin_r, cos_r, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=torch.float32,
            )
            rotation = rotation @ roll_matrix
        return rotation
    if pose.yaw_deg is None or pose.pitch_deg is None:
        raise ValueError(f"Invalid pose: {pose}")
    return _rotation_matrix(pose.yaw_deg, pose.pitch_deg)


def _looks_like_half_blank(image: np.ndarray) -> bool:
    if image.ndim == 3:
        gray = image.mean(axis=2)
    else:
        gray = image
    gray = gray.astype(np.float32) / 255.0
    mid = gray.shape[1] // 2
    left = gray[:, :mid]
    right = gray[:, mid:]
    left_blank = _is_blank_region(left)
    right_blank = _is_blank_region(right)
    return (left_blank or right_blank) and not (left_blank and right_blank)


def _is_blank_region(region: np.ndarray) -> bool:
    return region.mean() < 0.05 and region.std() < 0.05
