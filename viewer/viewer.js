import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { SplatMesh } from "@sparkjsdev/spark";

const canvas = document.getElementById("viewer-canvas");
const overlay = document.getElementById("overlay");
const statusLabel = document.getElementById("status");
const urlParams = new URLSearchParams(window.location.search);
const currentLanguage = (urlParams.get("lang") || "zh").toLowerCase();

const TEXT = {
  en: {
    "toolbar.title": "PhotoSplat 3D Preview",
    "overlay.loading": "Loading…",
    "overlay.loaded": "Model ready",
    "overlay.failed": "Load failed. Please retry.",
    "overlay.none": "No model specified.",
    "panel.rotate": "Rotate",
    "panel.translate": "Move",
    "panel.scale": "Scale",
    "panel.preset": "Pose Preset",
    "pose.position": "pose.position",
    "pose.rotation": "pose.rotation",
    "pose.scale": "pose.scale",
    "camera.position": "camera.position",
    "camera.target": "camera.target",
    "model.resolution": "model.resolution",
    "pose.position": "pose.position",
    "pose.rotation": "pose.rotation",
    "pose.scale": "pose.scale",
    "camera.position": "camera.position",
    "camera.target": "camera.target",
    "button.save_pose": "Save Pose",
    "button.zoom_in": "Zoom In",
    "button.zoom_out": "Zoom Out",
    "button.reset": "Reset",
    "status.pose_saved": "Pose saved",
    "status.pose_loaded": "Pose restored",
    "status.pose_cleared": "Pose removed",
    "status.pose_missing": "No saved pose",
    "status.pose_error": "Failed to sync pose"
  },
  zh: {
    "toolbar.title": "PhotoSplat 3D 预览",
    "overlay.loading": "加载中…",
    "overlay.loaded": "模型加载完成",
    "overlay.failed": "加载失败，请重试",
    "overlay.none": "未指定模型。",
    "panel.rotate": "旋转",
    "panel.translate": "平移",
    "panel.scale": "缩放",
    "panel.preset": "姿态预设",
    "pose.position": "模型位置",
    "pose.rotation": "模型旋转",
    "pose.scale": "模型缩放",
    "camera.position": "相机位置",
    "camera.target": "相机目标",
    "model.resolution": "模型分辨率",
    "pose.position": "模型位置",
    "pose.rotation": "模型旋转",
    "pose.scale": "模型缩放",
    "camera.position": "相机位置",
    "camera.target": "相机目标",
    "button.save_pose": "保存姿态",
    "button.zoom_in": "放大",
    "button.zoom_out": "缩小",
    "button.reset": "重置",
    "status.pose_saved": "姿态已保存",
    "status.pose_loaded": "已还原姿态",
    "status.pose_cleared": "已清除姿态",
    "status.pose_missing": "尚未保存姿态",
    "status.pose_error": "同步姿态失败"
  }
};

function t(key) {
  const pack = TEXT[currentLanguage] || TEXT.zh;
  return pack[key] || TEXT.en[key] || key;
}

const renderer = new THREE.WebGLRenderer({
  canvas,
  antialias: true,
  alpha: false
});
renderer.setPixelRatio(window.devicePixelRatio);

const scene = new THREE.Scene();
scene.background = new THREE.Color("#050505");

const camera = new THREE.PerspectiveCamera(
  55,
  window.innerWidth / (window.innerHeight - 48),
  0.01,
  1000
);
const DEFAULT_CAMERA_POSITION = new THREE.Vector3(-0.08, 0.02, -1.5);
const DEFAULT_CAMERA_TARGET = new THREE.Vector3(0, 0, 0);
const DEFAULT_MODEL_ROTATION = new THREE.Euler(0, 0, -Math.PI);
camera.position.copy(DEFAULT_CAMERA_POSITION);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = false;
controls.dampingFactor = 0;
controls.target.copy(DEFAULT_CAMERA_TARGET);
controls.enablePan = true;
controls.addEventListener("change", updatePoseInfo);

const rotationStep = THREE.MathUtils.degToRad(10);
const translationStep = 0.05;
const scaleStep = 0.15;
const trackpadPanSpeed = 0.002;

const panOffset = new THREE.Vector3();
const panColumn = new THREE.Vector3();
const cameraOffset = new THREE.Vector3();

let splat = null;
let animationId = null;
let currentModelPath = null;
let pywebviewReady = Boolean(window.pywebview && window.pywebview.api);
let pendingPoseRequest = false;

function formatVector(arr, digits = 3) {
  return arr.map((v) => v.toFixed(digits)).join(", ");
}

function updatePoseInfo() {
  const posEl = document.getElementById("pose-position");
  const rotEl = document.getElementById("pose-rotation");
  const scaleEl = document.getElementById("pose-scale");
  const camPosEl = document.getElementById("camera-position");
  const camTargetEl = document.getElementById("camera-target");
  const resolutionEl = document.getElementById("model-resolution");

  if (posEl && splat) {
    posEl.textContent = formatVector(splat.position.toArray());
  }
  if (rotEl && splat) {
    rotEl.textContent = [
      THREE.MathUtils.radToDeg(splat.rotation.x),
      THREE.MathUtils.radToDeg(splat.rotation.y),
      THREE.MathUtils.radToDeg(splat.rotation.z)
    ]
      .map((v) => v.toFixed(1) + "°")
      .join(", ");
  }
  if (scaleEl && splat) {
    scaleEl.textContent = formatVector(splat.scale.toArray());
  }
  if (camPosEl) {
    camPosEl.textContent = formatVector(camera.position.toArray());
  }
  if (camTargetEl) {
    camTargetEl.textContent = formatVector(controls.target.toArray());
  }
  if (resolutionEl && resolutionEl.textContent === "") {
    resolutionEl.textContent = "—";
  }
}

function applyTranslations() {
  document.documentElement.lang = currentLanguage.startsWith("en") ? "en" : "zh-Hans";
  document.title = t("toolbar.title");
  document.querySelectorAll("[data-i18n]").forEach((el) => {
    const key = el.dataset.i18n;
    if (key) {
      el.textContent = t(key);
    }
  });
}

function resize() {
  const width = window.innerWidth;
  const height = window.innerHeight - document.getElementById("toolbar").offsetHeight;
  renderer.setSize(width, height);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
}

window.addEventListener("resize", resize);
resize();

function animate() {
  animationId = requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}
animate();

function setOverlay(text, visible = true) {
  document.getElementById("overlay-text").textContent = text;
  overlay.classList.toggle("visible", visible);
}

async function loadModel(modelPath) {
  if (!modelPath) {
    return;
  }

  setOverlay(t("overlay.loading"), true);
  statusLabel.textContent = modelPath;
  currentModelPath = modelPath;

  if (splat) {
    try {
      scene.remove(splat);
      if (typeof splat.dispose === "function") {
        splat.dispose();
      }
    } catch (err) {
      console.warn("Failed to dispose previous model", err);
    }
    splat = null;
  }

  try {
    splat = new SplatMesh({ url: modelPath });
    scene.add(splat);
    resetTransform();
    applyDefaultCameraPose();
    setOverlay(t("overlay.loaded"), false);
    await updateModelMeta(modelPath);
    await restorePose();
    updatePoseInfo();
  } catch (err) {
    console.error(err);
    setOverlay(t("overlay.failed"), true);
  }
}

async function updateModelMeta(modelPath) {
  const resolutionEl = document.getElementById("model-resolution");
  if (!resolutionEl) {
    return;
  }
  const basePath = modelPath.split("?")[0];
  const metaPath = basePath.replace(/\.ply$/i, ".meta.json");
  try {
    const response = await fetch(`${metaPath}?t=${Date.now()}`);
    if (!response.ok) {
      throw new Error("meta fetch failed");
    }
    const meta = await response.json();
    if (meta.width && meta.height) {
      resolutionEl.textContent = `${meta.width}x${meta.height}`;
    } else {
      resolutionEl.textContent = "—";
    }
  } catch (err) {
    resolutionEl.textContent = "—";
  }
}

function resetTransform() {
  if (!splat) return;
  splat.position.set(0, 0, 0);
  splat.rotation.copy(DEFAULT_MODEL_ROTATION);
  splat.scale.set(1, 1, 1);
  updatePoseInfo();
}

function applyDefaultCameraPose() {
  camera.position.copy(DEFAULT_CAMERA_POSITION);
  controls.target.copy(DEFAULT_CAMERA_TARGET);
  controls.update();
  updatePoseInfo();
}

function adjustRotation(axis, direction) {
  if (!splat) return;
  splat.rotation[axis] += direction * rotationStep;
  updatePoseInfo();
}

function adjustTranslation(axis, direction) {
  if (!splat) return;
  splat.position[axis] += direction * translationStep;
  updatePoseInfo();
}

function adjustScale(direction) {
  if (!splat) return;
  const factor = direction > 0 ? 1 + scaleStep : 1 - scaleStep;
  const minScale = 0.1;
  const maxScale = 10;
  const nextX = THREE.MathUtils.clamp(splat.scale.x * factor, minScale, maxScale);
  const nextY = THREE.MathUtils.clamp(splat.scale.y * factor, minScale, maxScale);
  const nextZ = THREE.MathUtils.clamp(splat.scale.z * factor, minScale, maxScale);
  splat.scale.set(nextX, nextY, nextZ);
  updatePoseInfo();
}

function panWithTrackpad(deltaX, deltaY) {
  const element = renderer.domElement;
  if (!element) return;

  cameraOffset.copy(camera.position).sub(controls.target);
  let targetDistance = cameraOffset.length();
  targetDistance *= Math.tan(((camera.fov / 2) * Math.PI) / 180.0);

  const panX = (-2 * deltaX * trackpadPanSpeed * targetDistance) / element.clientHeight;
  const panY = (2 * deltaY * trackpadPanSpeed * targetDistance) / element.clientHeight;

  panColumn.setFromMatrixColumn(camera.matrix, 0);
  panColumn.multiplyScalar(panX);
  panOffset.copy(panColumn);

  panColumn.setFromMatrixColumn(camera.matrix, 1);
  panColumn.multiplyScalar(panY);
  panOffset.add(panColumn);

  camera.position.add(panOffset);
  controls.target.add(panOffset);
  updatePoseInfo();
}

function refreshPywebviewState() {
  if (window.pywebview && window.pywebview.api) {
    pywebviewReady = true;
    return window.pywebview.api;
  }
  return null;
}

function getPywebviewApi() {
  const api = refreshPywebviewState();
  if (api) {
    return api;
  }
  return null;
}

function serializePose() {
  if (!splat) return null;
  return {
    position: splat.position.toArray(),
    rotation: {
      x: splat.rotation.x,
      y: splat.rotation.y,
      z: splat.rotation.z
    },
    scale: splat.scale.toArray(),
    camera: {
      position: camera.position.toArray(),
      target: controls.target.toArray()
    }
  };
}

function applyPose(pose) {
  if (!pose || !splat) return;
  try {
    if (pose.position && pose.position.length === 3) {
      splat.position.fromArray(pose.position);
    }
    if (pose.rotation) {
      splat.rotation.set(
        pose.rotation.x || 0,
        pose.rotation.y || 0,
        pose.rotation.z || 0
      );
    }
    if (pose.scale && pose.scale.length === 3) {
      splat.scale.fromArray(pose.scale);
    }
    if (pose.camera) {
      if (pose.camera.position && pose.camera.position.length === 3) {
        camera.position.fromArray(pose.camera.position);
      }
      if (pose.camera.target && pose.camera.target.length === 3) {
        controls.target.fromArray(pose.camera.target);
      }
      controls.update();
    }
  } catch (err) {
    console.warn("Failed to apply pose", err);
  }
}

async function restorePose() {
  const api = getPywebviewApi();
  if (!api || !api.load_pose) {
    pendingPoseRequest = true;
    return;
  }
  pendingPoseRequest = false;
  try {
    const res = await api.load_pose();
    if (res && res.status === "ok" && res.pose) {
      applyPose(res.pose);
      showStatusMessage("status.pose_loaded");
    } else if (res && res.status === "not_found") {
      showStatusMessage("status.pose_missing");
    }
  } catch (err) {
    console.warn("Failed to load pose", err);
    showStatusMessage("status.pose_error");
  }
}

async function handleSavePose() {
  const api = getPywebviewApi();
  if (!api || !api.save_pose || !splat) {
    showStatusMessage("status.pose_error");
    return;
  }
  const pose = serializePose();
  if (!pose) return;
  try {
    const res = await api.save_pose(pose);
    if (res && res.status === "ok") {
      showStatusMessage("status.pose_saved");
    } else {
      showStatusMessage("status.pose_error");
    }
  } catch (err) {
    console.warn("Failed to save pose", err);
    showStatusMessage("status.pose_error");
  }
}

async function handleLoadPose() {
  await restorePose();
}

async function handleClearPose() {
  const api = getPywebviewApi();
  if (!api || !api.clear_pose) {
    showStatusMessage("status.pose_error");
    return;
  }
  try {
    const res = await api.clear_pose();
    if (res && res.status === "ok") {
      showStatusMessage("status.pose_cleared");
    } else if (res && res.status === "not_found") {
      showStatusMessage("status.pose_missing");
    } else {
      showStatusMessage("status.pose_error");
    }
  } catch (err) {
    console.warn("Failed to clear pose", err);
    showStatusMessage("status.pose_error");
  }
}

function showStatusMessage(key) {
  if (!key) return;
  statusLabel.textContent = t(key);
}

function attachButtons() {
  const api = () => getPywebviewApi();

  const exitBtn = document.getElementById("btn-exit");
  if (exitBtn) {
    exitBtn.addEventListener("click", () => {
      const bridge = api();
      if (bridge && bridge.close_viewer) {
        bridge.close_viewer();
      } else {
        window.close();
      }
    });
  }
  const saveBtn = document.getElementById("btn-save-pose");
  if (saveBtn) {
    saveBtn.addEventListener("click", handleSavePose);
  }
}

function attachTransformControls() {
  document.querySelectorAll("#control-panel button").forEach((btn) => {
    const action = btn.dataset.action;
    const axis = btn.dataset.axis;
    const direction = parseFloat(btn.dataset.direction || "0");

    btn.addEventListener("click", () => {
      switch (action) {
        case "rotate":
          adjustRotation(axis, direction);
          break;
        case "translate":
          adjustTranslation(axis, direction);
          break;
        case "scale":
          adjustScale(direction);
          break;
        case "reset":
          resetTransform();
          break;
        default:
          break;
      }
    });
  });
}

function attachTrackpadPan() {
  canvas.addEventListener(
    "wheel",
    (event) => {
      if (event.ctrlKey) {
        return;
      }
      if (event.deltaX === 0 && event.deltaY === 0) {
        return;
      }
      event.preventDefault();
      panWithTrackpad(event.deltaX, event.deltaY);
    },
    { passive: false }
  );
}

(function init() {
  window.addEventListener("pywebviewready", () => {
    pywebviewReady = true;
    if (pendingPoseRequest) {
      restorePose();
    }
  });
  applyTranslations();
  attachButtons();
  attachTransformControls();
  attachTrackpadPan();
  const params = new URLSearchParams(window.location.search);
  const model = params.get("model");
  if (model) {
    loadModel(model);
  } else {
    setOverlay(t("overlay.none"), true);
  }
})();
