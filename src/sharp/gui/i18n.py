"""Simple localization helper for SHARP GUI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List

DEFAULT_LANGUAGE = "zh"

TRANSLATIONS: Dict[str, Dict[str, str]] = {
    "en": {
        "app.title": "PhotoSplat3D",
        "panel.image_queue": "Image Queue",
        "control.import": "Import Images",
        "control.start": "Start",
        "control.stop": "Stop",
        "control.clear": "Clear Queue",
        "control.open_output": "Open Output",
        "control.change_output": "Change Output",
        "control.online": "Online Convert",
        "control.panorama_settings": "Panorama Settings",
        "control.language": "Language",
        "dialog.select_images": "Select Images",
        "dialog.image_files": "Image files",
        "dialog.all_files": "All files",
        "dialog.select_output_folder": "Select Output Folder",
        "dialog.unsupported_title": "Unsupported Images",
        "dialog.unsupported_message": "{count} image(s) use unsupported formats and were skipped.",
        "dialog.no_images_title": "No Images",
        "dialog.no_images_message": "No pending images to convert.",
        "dialog.stop_confirm_title": "Stop Conversion",
        "dialog.stop_confirm_message": "Are you sure you want to stop?",
        "dialog.stopping_title": "Stopping in progress",
        "dialog.stopping_message": "Please wait for the current conversion to stop before starting a new one.",
        "dialog.output_changed_title": "Output Folder Changed",
        "dialog.output_changed_message": "Output folder set to:\n{folder}\n\nNew conversions will be saved here.",
        "dialog.error_title": "Error",
        "dialog.change_output_error": "Could not change output folder: {error}",
        "dialog.open_output_error": "Could not open output folder: {error}",
        "dialog.modal_endpoint_missing": "Modal endpoint is not configured. Please set it in settings or via SHARP_MODAL_ENDPOINT.",
        "dialog.cancel": "Cancel",
        "dialog.panorama_title": "Panorama Settings",
        "dialog.panorama_face_size": "Panorama face size",
        "dialog.panorama_auto": "Auto",
        "dialog.panorama_save": "Save",
        "dialog.panorama_saved": "Panorama settings saved.",
        "status.ready": "Ready",
        "status.running": "Status: Running",
        "status.stopping": "Status: Stopping...",
        "progress.title": "Progress",
        "progress.stats": "Statistics",
        "progress.total": "Total: {count}",
        "progress.completed": "Completed: {count}",
        "progress.failed": "Failed: {count}",
        "progress.percent": "Progress: {percent:.1f}%",
        "progress.percent_zero": "Progress: 0%",
        "progress.current": "Current: {filename}",
        "progress.current_none": "Current: None",
        "progress.logs": "Logs",
        "log.welcome": "Welcome to {name}!",
        "log.history": "History: {count} images converted previously",
        "log.output_folder": "Output folder: {path}",
        "log.imported": "Imported {count} images",
        "log.start": "Started conversion of {count} images",
        "log.stopping": "Stopping current conversion...",
        "log.stopped": "Conversion stopped",
        "log.processing": "Processing: {filename}",
        "log.completed": "Completed: {filename}",
        "log.failed": "Failed: {filename} - {error}",
        "log.all_done": "All conversions complete!",
        "log.model_loading": "Loading SHARP model...",
        "log.model_device": "Using device: {device}",
        "log.model_download": "Loading model from cache (or downloading if needed)",
        "log.model_checkpoint": "Loading checkpoint from {path}",
        "log.model_loaded": "Model loaded successfully",
        "log.inference_start": "Running inference on: {filename}",
        "log.inference_end": "Inference complete: {filename}",
        "log.saving_output": "Saving to: {path}",
        "log.online_uploading": "Uploading to Modal: {filename}",
        "log.online_received": "Received Modal result for {filename} -> {path}",
        "log.online_failed": "Modal conversion failed: {filename} - {error}",
        "log.panorama_detected": "Panorama detected ({mode}), slicing into {faces} faces...",
        "log.panorama_config": "Panorama config: strategy={strategy}, face_size={face_size}, flip_poles={flip_poles}, flip_y={flip_y}, global_flip_y={global_flip_y}",
        "app.notice": "Built on Apple's open-source ml-sharp project (https://github.com/apple/ml-sharp). Free to use and share; paid redistribution is prohibited.",
        "lang.en": "English",
        "lang.zh": "Chinese (Simplified)",
        "provider.gemini": "Gemini",
    },
    "zh": {
        "app.title": "图生高斯3D",
        "panel.image_queue": "图片队列",
        "control.import": "导入图片",
        "control.start": "开始",
        "control.stop": "停止",
        "control.clear": "清空队列",
        "control.open_output": "打开输出文件夹",
        "control.change_output": "更改输出文件夹",
        "control.online": "在线转换",
        "control.panorama_settings": "全景转换配置",
        "control.language": "语言",
        "dialog.select_images": "选择图片",
        "dialog.image_files": "图片文件",
        "dialog.all_files": "所有文件",
        "dialog.select_output_folder": "选择输出文件夹",
        "dialog.unsupported_title": "格式不支持",
        "dialog.unsupported_message": "{count} 张图片格式不支持，已跳过。",
        "dialog.no_images_title": "没有图片",
        "dialog.no_images_message": "没有待转换的图片。",
        "dialog.stop_confirm_title": "停止转换",
        "dialog.stop_confirm_message": "确定要停止当前任务吗？",
        "dialog.stopping_title": "正在停止",
        "dialog.stopping_message": "请等待当前任务停止后再开始新的转换。",
        "dialog.output_changed_title": "输出文件夹已更新",
        "dialog.output_changed_message": "输出目录已设置为：\n{folder}\n\n新的转换结果将保存到这里。",
        "dialog.error_title": "错误",
        "dialog.change_output_error": "无法修改输出文件夹：{error}",
        "dialog.open_output_error": "无法打开输出文件夹：{error}",
        "dialog.modal_endpoint_missing": "未配置 Modal Endpoint，请在设置中填写或设置 SHARP_MODAL_ENDPOINT 环境变量。",
        "dialog.cancel": "取消",
        "dialog.panorama_title": "全景转换配置",
        "dialog.panorama_face_size": "全景面尺寸",
        "dialog.panorama_auto": "自动",
        "dialog.panorama_save": "保存",
        "dialog.panorama_saved": "全景配置已保存。",
        "status.ready": "就绪",
        "status.running": "状态：运行中",
        "status.stopping": "状态：正在停止…",
        "progress.title": "进度",
        "progress.stats": "统计",
        "progress.total": "总数：{count}",
        "progress.completed": "完成：{count}",
        "progress.failed": "失败：{count}",
        "progress.percent": "进度：{percent:.1f}%",
        "progress.percent_zero": "进度：0%",
        "progress.current": "当前：{filename}",
        "progress.current_none": "当前：无",
        "progress.logs": "日志",
        "log.welcome": "欢迎使用 {name}！",
        "log.history": "历史：共转换 {count} 张图片",
        "log.output_folder": "输出文件夹：{path}",
        "log.imported": "已导入 {count} 张图片",
        "log.start": "开始转换 {count} 张图片",
        "log.stopping": "正在停止当前任务…",
        "log.stopped": "转换已停止",
        "log.processing": "处理中：{filename}",
        "log.completed": "已完成：{filename}",
        "log.failed": "失败：{filename} - {error}",
        "log.all_done": "全部转换完成！",
        "log.model_loading": "正在加载 SHARP 模型…",
        "log.model_device": "使用设备：{device}",
        "log.model_download": "正在从缓存加载模型（或根据需要下载）",
        "log.model_checkpoint": "从 {path} 加载检查点",
        "log.model_loaded": "模型加载完成",
        "log.inference_start": "开始推理：{filename}",
        "log.inference_end": "推理完成：{filename}",
        "log.saving_output": "保存至：{path}",
        "log.online_uploading": "正在上传到 Modal：{filename}",
        "log.online_received": "Modal 已返回 {filename} -> {path}",
        "log.online_failed": "Modal 转换失败：{filename} - {error}",
        "log.panorama_detected": "检测到全景图（{mode}），正在切分为 {faces} 个面…",
        "log.panorama_config": "全景参数：策略={strategy}，面尺寸={face_size}，极区翻转={flip_poles}，Y 轴翻转={flip_y}，整体翻转={global_flip_y}",
        "app.notice": "本软件基于 Apple 开源的 ml-sharp 项目 (https://github.com/apple/ml-sharp) 构建，仅供免费分享与使用，禁止任何形式的二次收费分发。",
        "lang.en": "英语",
        "lang.zh": "简体中文",
        "provider.gemini": "Gemini",
    },
}


def _fallback_text(key: str) -> str:
    return TRANSLATIONS["en"].get(key, key)


@dataclass
class Translator:
    """Simple observer-based translator."""

    language: str = DEFAULT_LANGUAGE
    _listeners: List[Callable[[], None]] = field(default_factory=list)

    def translate(self, key: str, **kwargs) -> str:
        bundle = TRANSLATIONS.get(self.language, TRANSLATIONS["en"])
        template = bundle.get(key, _fallback_text(key))
        return template.format(**kwargs)

    def set_language(self, language: str, *, notify: bool = True):
        if language not in TRANSLATIONS:
            language = DEFAULT_LANGUAGE
        if language == self.language:
            return
        self.language = language
        if notify:
            for listener in list(self._listeners):
                listener()

    def register(self, callback: Callable[[], None]):
        self._listeners.append(callback)

    def available_languages(self) -> List[str]:
        return list(TRANSLATIONS.keys())

    def language_name(self, code: str) -> str:
        bundle = TRANSLATIONS.get(self.language, TRANSLATIONS["en"])
        fallback = TRANSLATIONS["en"]
        return bundle.get(f"lang.{code}", fallback.get(f"lang.{code}", code))


_translator = Translator()


def get_translator() -> Translator:
    """Return global translator singleton."""
    return _translator
