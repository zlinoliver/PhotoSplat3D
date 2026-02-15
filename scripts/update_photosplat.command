#!/bin/bash
# Update script for PhotoSplat 3D lite packages.

set -euo pipefail

APP_NAME="PhotoSplat3D"
APP_SUPPORT="$HOME/Library/Application Support/PhotoSplat3D"
INSTALL_INFO="$APP_SUPPORT/install_path.json"
BACKUP_DIR="$APP_SUPPORT/backups"
DEFAULT_TARGET="/Applications/PhotoSplat3D.app"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NEW_APP="$SCRIPT_DIR/PhotoSplat3D.app"

info() {
  printf "[Updater] %s\n" "$1"
}

error_exit() {
  printf "[Updater] %s\n" "$1" >&2
  exit 1
}

if [ ! -d "$NEW_APP" ]; then
  error_exit "New PhotoSplat3D.app not found, cannot continue (path: $NEW_APP)"
fi

TARGET_PATH="$DEFAULT_TARGET"
if [ -f "$INSTALL_INFO" ]; then
  TARGET_FROM_INFO=$(/usr/bin/python3 - "$INSTALL_INFO" <<'PY'
import json
import sys
try:
    with open(sys.argv[1]) as fh:
        data = json.load(fh)
    print(data.get("path", ""))
except Exception:
    print("")
PY
  )
  if [ -n "$TARGET_FROM_INFO" ]; then
    TARGET_PATH="$TARGET_FROM_INFO"
  fi
fi

printf "\nDetected possible install location:\n  %s\n" "$TARGET_PATH"
printf "If you need a custom path, enter an absolute path (you can drag PhotoSplat3D.app into this window). Press Enter to use the path above.\n"
read -r -p "Install path: " USER_PATH
if [ -n "${USER_PATH:-}" ]; then
  TARGET_PATH="$USER_PATH"
fi

TARGET_PATH="${TARGET_PATH%/}"
case "$TARGET_PATH" in
  *.app) ;;
  *) TARGET_PATH="$TARGET_PATH/PhotoSplat3D.app" ;;
esac

TARGET_PARENT="$(dirname "$TARGET_PATH")"
mkdir -p "$APP_SUPPORT"

info "Target app location: $TARGET_PATH"
if [ -d "$TARGET_PATH" ]; then
  mkdir -p "$BACKUP_DIR"
  BACKUP_PATH="$BACKUP_DIR/PhotoSplat3D-backup.app"
  info "Backing up current version to $BACKUP_PATH (only one backup is kept)"
  rm -rf "$BACKUP_PATH"
  /usr/bin/ditto "$TARGET_PATH" "$BACKUP_PATH"
else
  info "No existing version detected; treating this as a fresh install (no backup needed)"
fi

copy_app() {
  rm -rf "$TARGET_PATH"
  /usr/bin/ditto "$NEW_APP" "$TARGET_PATH"
}

info "Replacing with the latest version..."
if [ -w "$TARGET_PARENT" ]; then
  copy_app
else
  info "Administrator permission is required to write to $TARGET_PARENT; a password prompt may appear"
  sudo rm -rf "$TARGET_PATH"
  sudo /usr/bin/ditto "$NEW_APP" "$TARGET_PATH"
fi

info "Update complete! You can now launch ${APP_NAME} from its original location."
if [ -n "${BACKUP_PATH:-}" ]; then
  printf "To roll back, restore from this backup:\n%s\n" "$BACKUP_PATH"
fi
