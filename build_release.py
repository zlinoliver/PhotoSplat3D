#!/usr/bin/env python3
"""
One-click release build script for SHARP macOS application.

This script:
1. Checks version consistency
2. Downloads model if needed
3. Packages model into the app
4. Creates a standalone .app bundle
5. Generates release package

Usage:
    python build_release.py
    python build_release.py --version 1.1.0  # Specify version
    python build_release.py --skip-model     # Don't bundle model (smaller app)
"""

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import hashlib
import stat


class Colors:
    """Terminal colors."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


REQUIRED_MODULES = [
    'torch',
    'torchvision',
    'timm',
    'imageio',
    'pillow_heif',
    'gsplat',
]


def print_step(message):
    """Print step message."""
    print(f"\n{Colors.OKBLUE}{'='*60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{message}{Colors.ENDC}")
    print(f"{Colors.OKBLUE}{'='*60}{Colors.ENDC}\n")


def print_success(message):
    """Print success message."""
    print(f"{Colors.OKGREEN}✓ {message}{Colors.ENDC}")


def print_error(message):
    """Print error message."""
    print(f"{Colors.FAIL}✗ {message}{Colors.ENDC}")


def print_warning(message):
    """Print warning message."""
    print(f"{Colors.WARNING}⚠ {message}{Colors.ENDC}")


def get_version():
    """Read version from VERSION file."""
    version_file = Path("VERSION")
    if not version_file.exists():
        return "1.0.0"
    return version_file.read_text().strip()


def set_version(version):
    """Update version in all relevant files."""
    # Update VERSION file
    Path("VERSION").write_text(version)
    print_success(f"Updated VERSION to {version}")

    # Update pyproject.toml
    pyproject = Path("pyproject.toml")
    content = pyproject.read_text()
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if line.startswith('version = '):
            lines[i] = f'version = "{version}"'
            break
    pyproject.write_text('\n'.join(lines))
    print_success(f"Updated pyproject.toml to {version}")


def ensure_virtualenv():
    """Re-run the script inside the repo virtual environment if available."""
    repo_root = Path(__file__).resolve().parent
    venv_dir = repo_root / 'venv'
    venv_python = venv_dir / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')

    if os.environ.get('SHARP_BUILD_IN_VENV') == '1':
        return

    try:
        if Path(sys.prefix).resolve() == venv_dir.resolve():
            return
    except FileNotFoundError:
        pass

    if not venv_python.exists():
        print_warning(
            "Project virtual environment (venv) not found. Please create it and install dependencies as described in README_BUILD.md."
        )
        return

    print_step("Switching to project virtual environment")
    print_warning(
        "Detected system Python is being used to run the build script, which can produce an .app with missing dependencies."
    )
    print(f"Re-running with virtual environment: {venv_python}")

    env = os.environ.copy()
    env['SHARP_BUILD_IN_VENV'] = '1'
    cmd = [str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]]
    result = subprocess.run(cmd, env=env)
    sys.exit(result.returncode)


def check_dependencies():
    """Check if required tools are installed."""
    print_step("Checking dependencies")

    missing_modules = []
    for module in REQUIRED_MODULES:
        try:
            importlib.import_module(module)
        except ImportError:
            missing_modules.append(module)

    if missing_modules:
        print_error("The current Python interpreter is missing the following runtime dependencies:")
        for module in missing_modules:
            print(f"  - {module}")
        print("\nPlease run the following commands to install dependencies, then run build_release.py again:")
        print("  source venv/bin/activate")
        print("  pip install -r requirements.txt")
        print("\nFor detailed steps, see README_BUILD.md / QUICK_START.md.")
        sys.exit(1)
    else:
        print_success("All runtime dependencies are installed")

    try:
        import PyInstaller
        print_success("PyInstaller is installed")
    except ImportError:
        print_error("PyInstaller not found")
        print("Installing PyInstaller...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"], check=True)
        print_success("PyInstaller installed")

    return True


def download_model():
    """Download model if not present."""
    print_step("Checking model file")

    model_url = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"
    model_name = "sharp_2572gikvuh.pt"

    # Check standard cache location
    cache_dir = Path.home() / ".cache/torch/hub/checkpoints"
    cache_model = cache_dir / model_name

    # Check local directory
    local_model = Path("models") / model_name

    if cache_model.exists():
        print_success(f"Model found in cache: {cache_model}")
        return cache_model
    elif local_model.exists():
        print_success(f"Model found locally: {local_model}")
        return local_model
    else:
        print_warning("Model not found, downloading...")
        print(f"URL: {model_url}")
        print(f"Size: ~2.7 GB (this will take a few minutes)")

        # Create directory
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Download using PyTorch hub (with progress bar)
        try:
            import torch
            print("Downloading model...")
            torch.hub.download_url_to_file(model_url, str(cache_model), progress=True)
            print_success(f"Model downloaded to {cache_model}")
            return cache_model
        except Exception as e:
            print_error(f"Failed to download model: {e}")
            print("\nManual download instructions:")
            print(f"1. Download from: {model_url}")
            print(f"2. Save to: {cache_model}")
            sys.exit(1)


def verify_model(model_path):
    """Verify model file integrity."""
    print_step("Verifying model file")

    if not model_path.exists():
        print_error(f"Model file not found: {model_path}")
        return False

    # Check file size
    size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"Model size: {size_mb:.1f} MB")

    if size_mb < 2000:  # Should be ~2.7 GB
        print_warning("Model file seems too small, might be corrupted")
        return False

    # Compute hash
    print("Computing checksum...")
    hasher = hashlib.md5()
    with open(model_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)

    file_hash = hasher.hexdigest()
    print(f"Model hash: {file_hash}")
    print_success("Model verification complete")

    return True


def build_app(version, model_path, skip_model=False):
    """Build the macOS application."""
    print_step(f"Building PhotoSplat3D v{version}")

    # Clean previous builds
    for dir_name in ['build', 'dist']:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"Cleaning {dir_name}/")
            shutil.rmtree(dir_path)

    # Get absolute path to VERSION file
    version_file_path = Path('VERSION').absolute()
    if not version_file_path.exists():
        print_error(f"VERSION file not found at {version_file_path}")
        return None

    # Prepare PyInstaller options
    options = [
        # Entry point
        'src/sharp/gui/launcher.py',

        # App name with version
        f'--name=PhotoSplat3D',

        # Create .app bundle
        '--windowed',

        # One directory mode
        '--onedir',

        # Clean build
        '--clean',
        '--noconfirm',

        # Hidden imports (for modules without data files)
        '--hidden-import=sharp.gui',
        '--hidden-import=sharp.models',
        '--hidden-import=sharp.utils',
        '--hidden-import=sharp.cli',
        '--hidden-import=PIL._tkinter_finder',

        # Collect all from packages (for complex packages with plugins/data)
        '--collect-all=torch',
        '--collect-all=torchvision',
        '--collect-all=timm',
        '--collect-all=gsplat',
        '--collect-all=imageio',
        '--collect-all=pillow_heif',
        '--collect-all=plyfile',
        '--collect-all=scipy',
        '--collect-all=customtkinter',

        # Exclude problematic modules
        '--exclude-module=torch._numpy',
        '--exclude-module=torch._dynamo',
        '--exclude-module=torch.testing',

        # Version info
        f'--osx-bundle-identifier=com.apple.ml.sharp',

        # Add VERSION file (required by sharp.version.get_version())
        f'--add-data={version_file_path}:.',

        # Output directories
        '--distpath=dist',
        '--workpath=build',
        '--specpath=build',
    ]

    # Add model file if not skipping
    if not skip_model and model_path:
        model_dest = 'checkpoints'
        options.append(f'--add-data={model_path}:{model_dest}')
        print_success(f"Model will be bundled: {model_path.name}")
    else:
        print_warning("Model NOT bundled - app will download on first run")

    # Run PyInstaller
    print("Running PyInstaller...")
    import PyInstaller.__main__
    PyInstaller.__main__.run(options)

    # Verify build
    app_path = Path('dist/PhotoSplat3D.app')
    if not app_path.exists():
        print_error("Build failed - app not created")
        return None

    print_success(f"App built successfully: {app_path}")

    # Calculate app size
    app_size = sum(f.stat().st_size for f in app_path.rglob('*') if f.is_file())
    size_mb = app_size / (1024 * 1024)
    print(f"App size: {size_mb:.1f} MB")

    return app_path


def prepare_update_bundle(app_path):
    """Prepare a folder containing PhotoSplat3D.app and updater script for lite DMG."""
    bundle_dir = Path("dist/update_bundle")
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    target_app = bundle_dir / "PhotoSplat3D.app"
    print("Copying app into update bundle...")
    shutil.copytree(app_path, target_app)

    script_src = Path("scripts/update_photosplat.command")
    if script_src.exists():
        script_dst = bundle_dir / "Update PhotoSplat3D.command"
        shutil.copy2(script_src, script_dst)
        current_mode = script_dst.stat().st_mode
        script_dst.chmod(current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        print_success("Included Update PhotoSplat3D.command in bundle")
    else:
        print_warning("Update script not found; lite DMG will only contain PhotoSplat3D.app")

    return bundle_dir


def create_version_file(app_path, version):
    """Create version info file inside app.

    DEPRECATED: This function is no longer needed.
    VERSION file is now bundled via PyInstaller --add-data flag,
    and sharp.version.get_version() reads it directly from sys._MEIPASS.

    Keeping this function for reference but it should not be called.
    """
    version_info = {
        'version': version,
        'build_date': datetime.now().isoformat(),
        'platform': 'macOS',
    }

    # Write to app Resources
    resources_dir = app_path / 'Contents' / 'Resources'
    if resources_dir.exists():
        version_file = resources_dir / 'version.json'
        version_file.write_text(json.dumps(version_info, indent=2))
        print_success(f"Version info created: {version_file}")


def create_release_package(app_path, version, build_mode='full'):
    """Create release ZIP package."""
    print_step("Creating ZIP package")

    releases_dir = Path("releases").absolute()
    releases_dir.mkdir(exist_ok=True)

    if build_mode == 'lite':
        package_name = f"PhotoSplat3D-{version}-macOS-lite.zip"
    else:
        package_name = f"PhotoSplat3D-{version}-macOS.zip"
    package_path = releases_dir / package_name

    if package_path.exists():
        package_path.unlink()

    print(f"Compressing to {package_name}...")
    print(f"This may take a few minutes (app is {sum(f.stat().st_size for f in app_path.rglob('*') if f.is_file()) / (1024**2):.1f} MB)...")

    try:
        subprocess.run(
            ['zip', '-r', '-9', str(package_path), 'PhotoSplat3D.app'],
            cwd='dist',
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print_error(f"Zip failed with exit code {e.returncode}")
        print("Trying with lower compression...")
        subprocess.run(
            ['zip', '-r', '-5', str(package_path), 'PhotoSplat3D.app'],
            cwd='dist',
            check=True
        )

    size_mb = package_path.stat().st_size / (1024 * 1024)
    print_success(f"ZIP package created: {package_path} ({size_mb:.1f} MB)")

    return package_path


def create_dmg_package(app_path, version, build_mode='full'):
    """Create DMG installer using hdiutil (macOS only)."""
    print_step("Creating DMG package")

    releases_dir = Path("releases").absolute()
    releases_dir.mkdir(exist_ok=True)

    srcfolder = app_path
    if build_mode == 'lite':
        volume_name = "PhotoSplat3D-Update"
        dmg_path = releases_dir / f"PhotoSplat3D-update-{version}.dmg"
        srcfolder = prepare_update_bundle(app_path)
    else:
        volume_name = "PhotoSplat3D"
        legacy_dmg = releases_dir / "PhotoGauss3D.dmg"
        if legacy_dmg.exists():
            legacy_dmg.unlink()
        dmg_path = releases_dir / f"{volume_name}.dmg"
    if dmg_path.exists():
        dmg_path.unlink()

    hdiutil = shutil.which("hdiutil")
    if not hdiutil:
        print_warning("hdiutil not found, skipping DMG packaging.")
        return None

    try:
        subprocess.run(
            [
                hdiutil,
                "create",
                "-volname",
                volume_name,
                "-srcfolder",
                str(srcfolder),
                "-ov",
                "-format",
                "UDZO",
                str(dmg_path),
            ],
            check=True
        )
    except subprocess.CalledProcessError as exc:
        print_error(f"hdiutil create failed: {exc}")
        return None

    size_gb = dmg_path.stat().st_size / (1024 ** 3)
    print_success(f"DMG created: {dmg_path} ({size_gb:.2f} GB)")
    return dmg_path


def create_release_notes(version, primary_package, build_mode='full'):
    """Create localized release notes."""
    print_step("Creating release notes")

    notes_file = primary_package.parent / f"RELEASE_NOTES_{version}.md"
    size_gb = primary_package.stat().st_size / (1024 ** 3)

    if build_mode == 'lite':
        package_desc = "Update package (without model)"
        install_steps = f"""1. Make sure the full PhotoSplat 3D version with model has already been installed on your Mac.
2. Download `{primary_package.name}`
3. After mounting, quit any running PhotoSplat 3D process
4. Drag `PhotoSplat3D.app` from the DMG to your install location (for example, `/Applications`) and replace the old version
5. Relaunch the app after replacement completes"""
        requirements = """- `~/Library/Application Support/PhotoSplat3D/checkpoints/` cache must already exist
- Write permission is required for the target directory
- Make sure the old version is fully closed before replacing"""
        highlights = """- Includes code/UI updates only, no need to re-download the 2.7 GB model
- Small incremental update package suitable for offline distribution"""
    else:
        package_desc = "Full installer (includes model)"
        install_steps = f"""1. Download `{primary_package.name}`
2. Mount it, then drag `PhotoSplat3D.app` to Applications or any folder you prefer
3. **First launch**: right-click `PhotoSplat3D.app` -> choose "Open" -> confirm once more
4. After that, you can launch normally by double-clicking
5. If macOS security still blocks it:
   - Open System Settings -> Privacy & Security
   - In the security prompt, click "Open Anyway" or "Allow"
   - `sudo spctl --master-disable` is not recommended; if you disable Gatekeeper temporarily, remember to restore it with `sudo spctl --master-enable`"""
        requirements = """- macOS 11.0 Big Sur or later
- Apple Silicon (M-series) is recommended; 16 GB RAM provides a better experience
- At least 4 GB of free disk space
- On first launch, the model is automatically copied to `~/Library/Application Support/PhotoSplat3D/checkpoints/`"""
        highlights = """- GUI workflow: import images, view queue, monitor real-time progress and logs
- Supports common formats such as PNG/JPG/JPEG/HEIC
- Automatically caches the 3D Gaussian model with no extra configuration
- Multi-language UI: switch between Chinese and English anytime"""

    notes = f"""# Photo-to-Gaussian 3D v{version} - macOS Release Notes

**Release Date**: {datetime.now().strftime('%Y-%m-%d')}
**Package**: {primary_package.name}
**Type**: {package_desc}
**Size**: ≈{size_gb:.2f} GB

## Installation Steps

{install_steps}

## System Requirements

{requirements}

## Highlights

{highlights}

For more details, see [CHANGELOG.md](../CHANGELOG.md) in this repository.

## Help

If you run into issues, please check:
- README.md (quick start, CLI/GUI guide)
- CHANGELOG.md (version history)
If the issue persists, please open an issue. Feedback is welcome.

---
"""

    notes_file.write_text(notes)
    print_success(f"Release notes created: {notes_file}")
    return notes_file


def create_distribution_guide():
    """Create distribution guide for users."""
    guide_path = Path("releases/DISTRIBUTION_GUIDE.md")

    guide = """# SHARP Distribution Guide

## For Distributors

### Creating a New Release

```bash
# 1. Update version
python build_release.py --version X.Y.Z

# 2. Update CHANGELOG.md
# Add new version section with changes

# 3. Build release
python build_release.py

# 4. Test the app
open dist/PhotoSplat3D.app

# 5. Distribute
# Share the file from releases/PhotoSplat3D-X.Y.Z-macOS.zip
```

### Version Numbering

- **X.0.0**: Major version (breaking changes)
- **0.Y.0**: Minor version (new features)
- **0.0.Z**: Patch version (bug fixes)

## For End Users

### Installation Steps

1. **Download** the .zip file
2. **Extract** by double-clicking
3. **Move** PhotoSplat3D.app to Applications folder (optional)
4. **First launch**:
   - Right-click on PhotoSplat3D.app
   - Select "Open"
   - Click "Open" in the dialog
5. **Subsequent launches**: Just double-click

### Troubleshooting

#### "PhotoSplat3D is damaged and can't be opened"

This is a security warning, not actual damage.

**Solution**:
```bash
xattr -cr /path/to/PhotoSplat3D.app
```

Then retry opening.

#### App won't start

1. Try launching from Terminal to see errors:
   ```bash
   /path/to/PhotoSplat3D.app/Contents/MacOS/PhotoSplat3D
   ```

2. Check system requirements (macOS 11.0+)

#### Model download fails

The app needs internet on first run to download the AI model.

1. Ensure stable internet connection
2. Retry running the app
3. Check `~/.cache/torch/hub/checkpoints/` for model file

---

For more help, contact your distributor.
"""

    guide_path.write_text(guide)
    return guide_path


def print_summary(version, package_path, notes_file, dmg_path=None, build_mode='full'):
    """Print build summary."""
    print("\n" + "="*60)
    print(f"{Colors.BOLD}{Colors.OKGREEN}BUILD COMPLETE!{Colors.ENDC}")
    print("="*60 + "\n")

    print(f"{Colors.BOLD}Version:{Colors.ENDC} {version}")
    print(f"{Colors.BOLD}Mode:{Colors.ENDC} {build_mode}")
    print(f"{Colors.BOLD}ZIP:{Colors.ENDC} {package_path}")
    print(f"{Colors.BOLD}ZIP Size:{Colors.ENDC} {package_path.stat().st_size / (1024**2):.1f} MB")
    if dmg_path:
        print(f"{Colors.BOLD}DMG:{Colors.ENDC} {dmg_path}")
        print(f"{Colors.BOLD}DMG Size:{Colors.ENDC} {dmg_path.stat().st_size / (1024**2):.1f} MB")
    print(f"{Colors.BOLD}Notes:{Colors.ENDC} {notes_file}\n")

    print(f"{Colors.BOLD}Next Steps:{Colors.ENDC}")
    print("1. Test the app:")
    print(f"   {Colors.OKCYAN}open dist/PhotoSplat3D.app{Colors.ENDC}")
    print("\n2. Review release notes:")
    print(f"   {Colors.OKCYAN}cat {notes_file}{Colors.ENDC}")
    print("\n3. Distribute to users:")
    if dmg_path:
        print(f"   {Colors.OKCYAN}Share {dmg_path}{Colors.ENDC}")
    else:
        print(f"   {Colors.OKCYAN}Share {package_path}{Colors.ENDC}")
    print("\n4. Update changelog:")
    print(f"   {Colors.OKCYAN}Edit CHANGELOG.md for v{version}{Colors.ENDC}")

    print("\n" + "="*60 + "\n")


def main():
    """Main build process."""
    ensure_virtualenv()

    parser = argparse.ArgumentParser(description='Build SHARP release package')
    parser.add_argument('--version', help='Version number (e.g., 1.0.1)')
    parser.add_argument('--mode', choices=['full', 'lite'], default='full',
                        help='full=full package with model; lite=incremental/update package without model')
    parser.add_argument('--skip-model', action='store_true',
                       help='Skip bundling model (smaller app, downloads on first run)')
    args = parser.parse_args()

    print(f"\n{Colors.HEADER}{Colors.BOLD}")
    print("╔═══════════════════════════════════════════════════════╗")
    print("║                                                       ║")
    print("║      PhotoSplat3D Release Build System v1.0          ║")
    print("║                                                       ║")
    print("╚═══════════════════════════════════════════════════════╝")
    print(f"{Colors.ENDC}\n")

    # Get version
    if args.version:
        version = args.version
        set_version(version)
    else:
        version = get_version()

    print(f"Building version: {Colors.BOLD}{version}{Colors.ENDC}\n")

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    build_mode = args.mode
    bundle_model = (build_mode == 'full') and (not args.skip_model)

    # Handle model
    model_path = None
    if bundle_model:
        model_path = download_model()
        if not verify_model(model_path):
            print_error("Model verification failed")
            response = input("Continue anyway? (y/N): ")
            if response.lower() != 'y':
                sys.exit(1)
    else:
        print_warning("Skipping model bundle (lite build or --skip-model)")

    # Build app
    app_path = build_app(version, model_path, skip_model=not bundle_model)
    if not app_path:
        sys.exit(1)

    # Note: No need to create version.json - VERSION file is bundled via --add-data
    # and sharp.version.get_version() reads it directly from the bundle

    # Create release package
    package_path = create_release_package(app_path, version, build_mode=build_mode)

    # Create DMG package (if possible)
    dmg_path = create_dmg_package(app_path, version, build_mode=build_mode)

    # Create release notes (prefer DMG info)
    notes_file = create_release_notes(version, dmg_path or package_path, build_mode=build_mode)

    # Create distribution guide
    create_distribution_guide()

    # Print summary
    print_summary(version, package_path, notes_file, dmg_path=dmg_path, build_mode=build_mode)


if __name__ == '__main__':
    main()
