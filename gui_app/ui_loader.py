"""UI 装配：把 ui/ 目录资源拼接进 index.html 模板。"""

from pathlib import Path

UI_DIR = Path(__file__).resolve().parent / "ui"
INDEX_PATH = UI_DIR / "index.html"
CSS_PATH = UI_DIR / "styles.css"
JS_DIR = UI_DIR / "js"

CSS_PLACEHOLDER = "<!--INJECT_CSS-->"
JS_PLACEHOLDER = "<!--INJECT_JS-->"

# JS 拼接顺序敏感：core.js 最先（顶层绑定），bootstrap.js 最后（立即执行的启动序列）
JS_MODULES = [
    "core.js",
    "grid.js",
    "detail.js",
    "sources.js",
    "actions.js",
    "rename.js",
    "settings.js",
    "prompts.js",
    "terms.js",
    "workspace.js",
    "experimental.js",
    "bootstrap.js",
]


def build_html() -> str:
    """读取 ui/index.html，注入合并后的 CSS 与 JS，返回完整 HTML 字符串。"""
    html = INDEX_PATH.read_text(encoding="utf-8")

    css = CSS_PATH.read_text(encoding="utf-8")

    parts = []
    for name in JS_MODULES:
        code = (JS_DIR / name).read_text(encoding="utf-8")
        parts.append(f"/* ===== {name} ===== */\n{code}")
    js = "\n\n".join(parts)

    if CSS_PLACEHOLDER not in html:
        raise ValueError(f"index.html 缺少 CSS 占位符 {CSS_PLACEHOLDER}")
    if JS_PLACEHOLDER not in html:
        raise ValueError(f"index.html 缺少 JS 占位符 {JS_PLACEHOLDER}")

    html = html.replace(CSS_PLACEHOLDER, css)
    html = html.replace(JS_PLACEHOLDER, js)
    return html
