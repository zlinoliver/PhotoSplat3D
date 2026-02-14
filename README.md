# PhotoSplat 3D

<p align="center">
  <strong>Turn photos into 3D Gaussian Splat models — instantly and privately on-device</strong>
</p>

<p align="center">
  <a href="README_cn.md">中文文档</a> •
  <a href="#downloads">Downloads</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#features">Features</a>
</p>

---

A desktop-friendly fork of Apple's [ml-sharp](https://apple.github.io/ml-sharp/) project for single-image 3D Gaussian Splatting (3DGS) reconstruction.

## Downloads

### 🖥️ Mac App (Full Version)
**[Download PhotoSplat3D for Mac](https://r2.photosplat3d.com/PhotoSplat3D_v1.0.dmg)**

Convert 2D photos to 3D Gaussian Splat models with one click.

**Features:**

- Regular photos and 180°/360° panoramic photos
- Batch processing
- Local rendering
- AI text-to-image generation (Nano Banana Pro)

**Requirements:** macOS with Apple Silicon (M2 or later)

### 📱 Vision Pro App
**[Download on App Store](https://apps.apple.com/us/app/photosplat-3d/id6757570552)**

Create and explore 3D Gaussian Splats directly on Apple Vision Pro.

**Features:**

- On-device photo-to-3D conversion
- Import and view .ply files
- Spatial interaction with gesture controls
- Completely private — no cloud, no uploads

**Requirements:** Apple Vision Pro (Both M2 and M5 versions) with visionOS 26+


---

## Features

- 📸 **Single Image → 3D Scene**: Convert any photo to a `.ply` Gaussian Splat model
- 🖥️ **Desktop GUI**: User-friendly interface built with CustomTkinter
- ⌨️ **CLI Workflow**: Batch processing and automated rendering
- 🌐 **Panorama Support**: 180° and 360° equirectangular preprocessing
- 🎨 **Multiple Strategies**: Cube6, Ring8, Ring12, Front4, Front6 projection modes
- ⚡ **Flexible Compute**: CPU, MPS (Apple Silicon), or CUDA support

---

## Quick Start

### Prerequisites

Before you begin, ensure you have:

- **Git** (for cloning the repository)
- **Python 3.10+** (Python 3.13 recommended)
  - Must include `venv` and `tkinter` modules
- **Operating System**: macOS or Linux
- **Internet Connection**: For dependency and model downloads
- **Optional GPU**:
  - Inference: CPU / MPS / CUDA
  - Video Rendering: CUDA only

**Verify your environment:**

```bash
git --version
python3 --version
python3 -m venv --help
python3 -m tkinter
```

### Installation

**Option 1: Automated Setup (Recommended)**

```bash
git clone https://github.com/zlinoliver/PhotoSplat3D.git
cd ml-sharp
bash scripts/bootstrap.sh
```

The bootstrap script will:

- Create a virtual environment
- Validate Python version and dependencies
- Install all required packages
- Set up the project in editable mode

**Option 2: Manual Setup**

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### Usage

**Launch GUI Application:**

```bash
sharp gui
```

Or alternatively:

```bash
PYTHONPATH=src python3 -m sharp.gui.launcher
```

**Command Line Interface:**

```bash
# Generate 3D .ply from image(s)
sharp predict -i input/ -o output/

# Render trajectory video (requires CUDA)
sharp render -i output/ -o output/renderings/
```

### First Run Notes

- Model checkpoint (~300MB) downloads automatically on first inference
- Default cache location: `~/.cache/torch/hub/checkpoints/`
- GUI default output: `~/Desktop/SHARP_Output`

---

## CLI Options

### `sharp predict`

Convert images to 3D Gaussian Splat models.

```bash
sharp predict -i INPUT -o OUTPUT [OPTIONS]
```

**Common Options:**

| Option | Values | Description |
|--------|--------|-------------|
| `--device` | `default`, `cpu`, `mps`, `cuda` | Compute device |
| `--render` | - | Generate preview renders (CUDA recommended) |
| `--panorama` | `auto`, `180`, `360`, `none` | Panorama detection mode |
| `--panorama-strategy` | `cube6`, `ring8`, `ring12`, `front4`, `front6` | Projection strategy |
| `--panorama-face-size` | `<int>` | Resolution per face/view |

**Examples:**

```bash
# Auto-detect panoramas
sharp predict -i photos/ -o output/ --panorama auto

# Force 360° panorama with Ring8 strategy
sharp predict -i pano.jpg -o output/ --panorama 360 --panorama-strategy ring8

# Use CUDA and render preview
sharp predict -i input/ -o output/ --device cuda --render
```

---

## Building Desktop App

Build a standalone macOS application:

```bash
bash scripts/bootstrap.sh
source venv/bin/activate
python build_release.py --mode full
```

**Output:**

- `dist/PhotoSplat3D.app` — Standalone application
- `releases/PhotoSplat3D-*-macOS.zip` — Distributable archive

---

## Project Structure

```
ml-sharp/
├── src/sharp/
│   ├── cli/          # Command-line interface (predict, render, gui)
│   ├── gui/          # Desktop GUI application
│   ├── models/       # SHARP model definitions
│   └── utils/        # I/O, camera, Gaussian, rendering utilities
├── viewer/           # Web viewer static assets
├── scripts/          # Setup and build scripts
└── requirements.txt  # Python dependencies
```

---

## Privacy & Configuration

- **Local Config**: `~/.sharp_config.json`
- **Environment Variables**: See `.env.example` for optional settings
- **No Telemetry**: No personal data or endpoints are hardcoded

---

## Contributing

We welcome contributions! Please read:

- [CONTRIBUTING.md](CONTRIBUTING.md) — Contribution guidelines
- [.github/ISSUE_TEMPLATE/](.github/ISSUE_TEMPLATE/) — Issue templates
- [SECURITY.md](SECURITY.md) — Security vulnerability reporting

---

## Research & Credits

**Paper:**  
[Sharp Monocular View Synthesis in Less Than a Second](https://arxiv.org/abs/2512.10685)

**Upstream Project:**  
<https://apple.github.io/ml-sharp/>

**3D Rendering:**  
Powered by [MetalSplatter](https://github.com/scier/MetalSplatter) (open source on GitHub)

---

## License

This is a multi-license repository:

- **Original Contributions**: [LICENSE](LICENSE) (MIT, scoped to this fork)
- **Apple-Derived Code**: [LICENSE_UPSTREAM_APPLE](LICENSE_UPSTREAM_APPLE)
- **Model Weights**: [LICENSE_MODEL](LICENSE_MODEL)
- **Third-Party Acknowledgements**: [ACKNOWLEDGEMENTS](ACKNOWLEDGEMENTS)

Apple copyright and license headers in source files must be preserved in any redistribution.

---

<p align="center">
  Made with ❤️ for the 3D reconstruction community
</p>
