"""Standalone viewer process to render PLY files using Spark in a webview."""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import threading
import time
from http.server import SimpleHTTPRequestHandler
import hashlib
import json
from pathlib import Path
from urllib.parse import quote

import webview
from plyfile import PlyData

from sharp.gui.constants import APP_SUPPORT_DIR


def _find_viewer_source() -> Path:
    """Locate the bundled viewer assets."""
    candidates = []
    module_dir = Path(__file__).resolve().parents[3]
    candidates.append(module_dir / "viewer")
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(Path(meipass) / "viewer")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Viewer assets not found.")


def _viewer_cache_root() -> Path:
    APP_SUPPORT_DIR.mkdir(parents=True, exist_ok=True)
    cache_root = APP_SUPPORT_DIR / "viewer_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    return cache_root


def _compute_signature(source: Path) -> str:
    hasher = hashlib.sha256()
    for path in sorted(source.rglob("*")):
        if path.is_file():
            rel = path.relative_to(source).as_posix().encode("utf-8")
            stat = path.stat()
            hasher.update(rel)
            hasher.update(str(stat.st_size).encode("utf-8"))
            hasher.update(str(stat.st_mtime_ns).encode("utf-8"))
    return hasher.hexdigest()


def _get_cached_viewer(source: Path) -> Path:
    cache_root = _viewer_cache_root()
    payload = cache_root / "payload"
    meta_file = cache_root / "meta.json"
    signature = _compute_signature(source)

    if payload.exists() and meta_file.exists():
        try:
            meta = json.loads(meta_file.read_text())
        except Exception:
            meta = {}
        if meta.get("signature") == signature:
            return payload

    shutil.rmtree(payload, ignore_errors=True)
    shutil.copytree(source, payload, dirs_exist_ok=True)
    meta_file.write_text(json.dumps({"signature": signature}, indent=2))
    return payload


class ViewerAPI:
    """Bridge between JavaScript and Python."""

    def __init__(self, app: "ViewerApp"):
        # store as private attribute to avoid pywebview introspection warnings
        self._app = app

    def choose_model(self):
        """Prompt user to select a new PLY file."""
        window = webview.windows[0] if webview.windows else None
        if not window:
            return {"status": "error", "message": "Viewer not ready"}

        filetypes = (("PLY files (*.ply)", "*.ply"), ("All files", "*.*"))
        directory = str(self._app.last_directory or Path.home())

        try:
            result = window.create_file_dialog(
                webview.OPEN_DIALOG,
                directory=directory,
                allow_multiple=False,
                file_types=filetypes,
            )
        except Exception as exc:
            return {"status": "error", "message": str(exc)}

        if not result:
            return {"status": "cancel"}

        rel = self._app.copy_model(Path(result[0]))
        return {"status": "ok", "model": rel}

    def close_viewer(self):
        if self._app.window:
            try:
                webview.destroy_window(self._app.window.uid)
            except Exception:
                try:
                    self._app.window.destroy()
                except Exception:
                    pass
        return True

    def _pose_path(self):
        source = self._app.current_source_path
        if not source:
            return None
        return source.with_suffix(source.suffix + ".pose.json")

    def load_pose(self):
        pose_path = self._pose_path()
        if not pose_path or not pose_path.exists():
            return {"status": "not_found"}
        try:
            data = json.loads(pose_path.read_text())
            return {"status": "ok", "pose": data}
        except Exception as exc:
            return {"status": "error", "message": str(exc)}

    def save_pose(self, pose):
        pose_path = self._pose_path()
        if not pose_path:
            return {"status": "error", "message": "No source path"}
        try:
            pose_path.write_text(json.dumps(pose, indent=2))
            return {"status": "ok"}
        except Exception as exc:
            return {"status": "error", "message": str(exc)}

    def clear_pose(self):
        pose_path = self._pose_path()
        if not pose_path or not pose_path.exists():
            return {"status": "not_found"}
        try:
            pose_path.unlink()
            return {"status": "ok"}
        except Exception as exc:
            return {"status": "error", "message": str(exc)}


class ViewerApp:
    """Controls the local web server and webview."""

    def __init__(self, model_path: Path, language: str = "zh"):
        self.model_path = model_path
        self.window = None
        self.server = None
        self.server_thread = None
        self.base_temp = Path(tempfile.mkdtemp(prefix="photosplat_viewer_"))
        self.static_dir = self.base_temp / "static"
        self.data_dir = self.static_dir / "data"
        self.port = None
        self.last_directory = model_path.parent if model_path else None
        self.current_relative_model = "data/scene.ply"
        self.language = language or "zh"
        self.current_source_path = model_path

    def prepare_assets(self):
        source = _find_viewer_source()
        cached = _get_cached_viewer(source)
        shutil.copytree(cached, self.static_dir, dirs_exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)
        if self.model_path:
            self.current_relative_model = self.copy_model(self.model_path)

    def copy_model(self, source: Path) -> str:
        self.last_directory = source.parent
        self.current_source_path = source
        target = self.data_dir / "scene.ply"
        shutil.copy2(source, target)
        self._write_model_meta(target)
        rel = f"data/{quote(target.name)}?t={int(time.time())}"
        self.current_relative_model = rel
        return rel

    def _write_model_meta(self, target: Path) -> None:
        meta_path = self.data_dir / "scene.meta.json"
        meta = {}
        try:
            ply = PlyData.read(str(target))
            for element in ply.elements:
                if element.name == "image_size":
                    values = element.data["image_size"]
                    if len(values) >= 2:
                        meta["width"] = int(values[0])
                        meta["height"] = int(values[1])
                    break
        except Exception as exc:
            meta["error"] = str(exc)
        meta_path.write_text(json.dumps(meta, indent=2))

    def start_server(self):
        handler = lambda *args, **kwargs: SimpleHTTPRequestHandler(
            *args, directory=str(self.static_dir), **kwargs
        )
        from socketserver import ThreadingTCPServer

        self.server = ThreadingTCPServer(("127.0.0.1", 0), handler)
        self.server.allow_reuse_address = True
        self.port = self.server.server_address[1]
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()

    def shutdown(self):
        if self.window:
            try:
                self.window.destroy()
            except Exception:
                pass
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        shutil.rmtree(self.base_temp, ignore_errors=True)

    def run(self):
        self.prepare_assets()
        self.start_server()

        initial_model = self.current_relative_model or "data/scene.ply"
        lang = quote(self.language)
        url = (
            f"http://127.0.0.1:{self.port}/index.html"
            f"?model={quote(initial_model)}&lang={lang}"
        )
        api = ViewerAPI(self)
        self.window = webview.create_window("PhotoSplat 3D Viewer", url, js_api=api)
        self.window.events.closed += self.shutdown
        try:
            webview.start(debug=False)
        finally:
            self.shutdown()


def launch_with_model(model_path: Path, language: str = "zh"):
    app = ViewerApp(model_path, language=language)
    app.run()


def main():
    parser = argparse.ArgumentParser(description="Launch PhotoSplat 3D viewer")
    parser.add_argument("--model", required=True, help="Path to the PLY file")
    parser.add_argument("--lang", default="zh", help="UI language code (en or zh)")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    launch_with_model(model_path, language=args.lang)


if __name__ == "__main__":
    main()
