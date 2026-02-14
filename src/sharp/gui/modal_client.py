"""Helper utilities for calling the Modal SHARP endpoint."""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import requests

logger = logging.getLogger(__name__)


class ModalConversionError(RuntimeError):
    """Raised when the Modal service returns an error."""


@dataclass
class ModalConversionResult:
    """Container for Modal responses."""

    ply_bytes: bytes
    logs: List[str]


def _format_logs(logs_field: Optional[Iterable[str]]) -> List[str]:
    if logs_field is None:
        return []
    if isinstance(logs_field, (list, tuple)):
        return [str(entry) for entry in logs_field]
    return [str(logs_field)]


def _image_to_base64(image_path: Path) -> str:
    data = image_path.read_bytes()
    return base64.b64encode(data).decode("ascii")


def convert_image_via_modal(
    endpoint: str,
    image_path: Path,
    *,
    timeout: float = 120.0,
    session: Optional[requests.Session] = None,
) -> ModalConversionResult:
    """Upload an image to Modal and return decoded PLY bytes."""
    if not endpoint:
        raise ModalConversionError("Modal endpoint is not configured.")

    payload = {
        "image": _image_to_base64(image_path),
        "filename": image_path.name,
    }

    http = session or requests
    logger.info("Uploading %s to Modal endpoint %s", image_path.name, endpoint)

    try:
        response = http.post(
            endpoint,
            json=payload,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise ModalConversionError(f"Modal request failed: {exc}") from exc

    if response.status_code >= 400:
        raise ModalConversionError(f"Modal endpoint error {response.status_code}: {response.text}")

    try:
        data = response.json()
    except ValueError as exc:
        raise ModalConversionError("Modal response is not valid JSON.") from exc

    if not data.get("success", True):
        raise ModalConversionError(data.get("error") or "Modal conversion failed.")

    ply_b64 = data.get("ply_base64")
    if not ply_b64:
        raise ModalConversionError("Modal response missing ply_base64 field.")

    try:
        ply_bytes = base64.b64decode(ply_b64)
    except (ValueError, TypeError) as exc:
        raise ModalConversionError("Failed to decode PLY payload from Modal.") from exc

    logs = _format_logs(data.get("logs"))
    return ModalConversionResult(ply_bytes=ply_bytes, logs=logs)
