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
  printf "[更新器] %s\n" "$1"
}

error_exit() {
  printf "[更新器] %s\n" "$1" >&2
  exit 1
}

if [ ! -d "$NEW_APP" ]; then
  error_exit "未找到新的 PhotoSplat3D.app，无法继续（路径：$NEW_APP）"
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

printf "\n检测到可能的安装位置：\n  %s\n" "$TARGET_PATH"
printf "若需要自定义路径，请输入绝对路径（可拖拽 PhotoSplat3D.app 到此窗口），直接回车则使用上面的路径。\n"
read -r -p "安装路径: " USER_PATH
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

info "目标应用位置：$TARGET_PATH"
if [ -d "$TARGET_PATH" ]; then
  mkdir -p "$BACKUP_DIR"
  BACKUP_PATH="$BACKUP_DIR/PhotoSplat3D-backup.app"
  info "正在备份现有版本到 $BACKUP_PATH （仅保留这一份）"
  rm -rf "$BACKUP_PATH"
  /usr/bin/ditto "$TARGET_PATH" "$BACKUP_PATH"
else
  info "未检测到现有版本，将视为新安装（无需备份）"
fi

copy_app() {
  rm -rf "$TARGET_PATH"
  /usr/bin/ditto "$NEW_APP" "$TARGET_PATH"
}

info "正在覆盖为最新版本..."
if [ -w "$TARGET_PARENT" ]; then
  copy_app
else
  info "需要管理员权限写入 $TARGET_PARENT，可能会弹出密码提示"
  sudo rm -rf "$TARGET_PATH"
  sudo /usr/bin/ditto "$NEW_APP" "$TARGET_PATH"
fi

info "更新完成！现在可以直接从原来位置启动 ${APP_NAME}。"
if [ -n "${BACKUP_PATH:-}" ]; then
  printf "如需回滚，可从以下备份恢复：\n%s\n" "$BACKUP_PATH"
fi
