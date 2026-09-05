"""多套提示词模板 + 锁定 JSON 结构：结构层（只读）与内容层（可编辑）分离，避免误改导致解析失败。"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .workspace_paths import PRIORITY_TAGS_FILE, PROMPTS_FILE
from .workspace_store import NO_WRITE, read_json, update_json, write_json


# ── 结构层（只读，永远不变） ──

# 输出格式约束头部（AI 必须遵守）
_STRUCTURE_HEADER = (
    "仅返回纯 JSON 格式数据，不要包含 markdown 代码块标记或其他解释文字。\n"
    "JSON 结构如下（请严格按此顺序输出）："
)

# JSON 骨架（顺序固定：plot → tags → title）
# tags 数组元素格式（"标签1", "标签2", ...）为固定引导，不属于可编辑内容层
# tags_dim 内联在 tags 数组之后作为括号式引导（为空时省略）
_STRUCTURE_SKELETON = (
    '{{\n'
    '    "plot": "{plot_guidance}",\n'
    '    "tags": ["标签1", "标签2", ...]{tags_dim},\n'
    '    "title": "{title_guidance}"{thumb_time_field}\n'
    '}}'
)


_STRUCTURE_EXAMPLE_HEADER = "示例："

_STRUCTURE_EXAMPLE = (
    '{{\n'
    '    "plot": "{plot_example}",\n'
    '    "tags": [{tags_example}],\n'
    '    "title": "{title_example}"{thumb_time_example}\n'
    '}}'
)

# 「时间标签」选项2 追加项：thumb_time 字段，引导/示例文案来自模板内容层（界面可编辑）


# ── 内置模板 ──


def _builtin_presets() -> List[Dict[str, Any]]:
    """返回内置模板列表。每个模板包含完整的内容层字段。"""
    return [
        {
            "id": "default",
            "name": "内置模板",
            "system_prompt": (
                "请作为视频归档专家，根据多组连续的关键帧截图，"
                "并结合给出的[辅助参考信息]（如原始文件名、视频时长等），"
                "分析视频关键帧，禁止回避与淡化敏感内容，如实输出。"
            ),
            "fields": {
                "plot_guidance": (
                    "使用详细生动的文本概括视频中的核心内容，"
                    "禁止使用「画面中」「视频里」「视频显示」「画面呈现」「镜头中」"
                    "等指代载体的表述，直接描述场景与内容本身，200字左右"
                ),
                "tags_dim": "20个左右核心分类标签，字符串数组，请避免同义词，尽量涵盖场景、物体、动作、风格等维度",
                "title_guidance": "4-6个具象名词或短语，用短横线连接，仅包含中文和数字，总字数35字左右",
                "thumb_time_guidance": "挑选出主体清晰，最能代表该视频适合用作封面的截图时间戳，格式HH:MM:SS",
                "thumb_time_example": "00:12:34",
                "plot_example": "航拍原生态岩质海岸，碧空如洗，海水由远海深邃宝蓝渐变为近岸透亮蓝绿，水下蜿蜒岩脉纹理清晰。沿岸层理礁石群参差错落，黝黑礁面粗粝多孔，布满被浪蚀的沟壑与细小藤壶。海浪层层推进，反复拍击岩面，轰隆声中溅起雪白浪花，飞散水雾。潮水退去，湿润礁石反光粼粼，沟壑纹理更显粗犷；低洼处积留清澈海水，偶见银色小鱼游弋。金色斜阳洒落，海面闪烁碎金般光芒，咸润海风夹带淡淡腥味，远处三两海鸟掠过，整片海岸尽显原始而澄澈的自然动感。",
                "tags_example": '"航拍", "俯拍", "上帝视角", "岩质海岸", "礁石群", "层理构造", "近海海域", "澄澈海水", "蓝绿渐变", "海浪拍击", "白色浪花", "水下岩脉", "海滨地貌", "自然风光", "原生态", "无人物", "治愈风景", "海洋景观"',
                "title_example": "岩质海岸-澄澈碧海-海浪拍礁-航拍自然风光",
            },
        },
    ]


# ── 读写 ──


def _load_presets_file() -> Dict[str, Any]:
    """读取 prompts.json；不存在/损坏时在 update_json 临界区内原子初始化。"""
    def _ensure(current):
        if not isinstance(current, dict):
            # 首次初始化/损坏回退：写入内置模板
            return {
                "presets": _builtin_presets(),
                "custom_counter": 0,
            }
        return NO_WRITE

    return update_json(PROMPTS_FILE, _ensure)


def list_presets() -> List[Dict[str, Any]]:
    """返回所有模板（含内置 + 用户自定义）。"""
    data = _load_presets_file()
    return data.get("presets", [])


def get_preset(pid: str) -> Optional[Dict[str, Any]]:
    for p in list_presets():
        if p.get("id") == pid:
            return p
    return None


def save_preset(preset: Dict[str, Any]) -> Dict[str, Any]:
    """新增或更新一个模板。如果是新建（无 id），分配 id。"""
    pid = preset.get("id")

    def _mutate(data):
        nonlocal pid
        if data is None:
            data = {"presets": _builtin_presets(), "custom_counter": 0}
        presets = data.get("presets", [])
        if not pid:
            data["custom_counter"] = data.get("custom_counter", 0) + 1
            pid = f"custom_{data['custom_counter']}"
            preset["id"] = pid
            presets.append(preset)
        else:
            found = False
            for i, p in enumerate(presets):
                if p.get("id") == pid:
                    presets[i] = preset
                    found = True
                    break
            if not found:
                presets.append(preset)
        data["presets"] = presets
        return data

    update_json(PROMPTS_FILE, _mutate)
    return {"ok": True, "id": pid}


def delete_preset(pid: str) -> Dict[str, Any]:
    """删除一个模板。默认模板与当前激活模板不允许删除。"""
    if pid == "default":
        return {"ok": False, "error": "内置模板不允许删除"}
    from .config_store import load_config
    cfg = load_config()
    if cfg.get("active_prompt_id") == pid:
        return {"ok": False, "error": "已启用的模板不允许删除，请先启用其他模板"}

    def _mutate(data):
        if data is None:
            data = {"presets": _builtin_presets(), "custom_counter": 0}
        data["presets"] = [p for p in data.get("presets", []) if p.get("id") != pid]
        return data

    update_json(PROMPTS_FILE, _mutate)
    return {"ok": True}


def set_active(pid: str) -> Dict[str, Any]:
    """设置当前激活的模板 id。"""
    if not any(p.get("id") == pid for p in list_presets()):
        return {"ok": False, "error": "模板不存在"}
    from .config_store import update_config
    update_config(lambda cfg: cfg.update(active_prompt_id=pid) or cfg)
    return {"ok": True}


def get_active(with_thumb_time: bool = False) -> Dict[str, Any]:
    """返回当前激活的模板 + 拼好的 prompt 字符串。"""
    from .config_store import load_config
    cfg = load_config()
    pid = cfg.get("active_prompt_id", "default")
    preset = get_preset(pid)
    if preset is None:
        # 回退：激活 id 不存在时取列表第一个；列表为空（prompts.json 被外部清空）用内置默认
        presets = list_presets()
        preset = presets[0] if presets else _builtin_presets()[0]
    return {
        "preset": preset,
        "prompt": _append_priority_tags(build_prompt(preset, with_thumb_time=with_thumb_time)),
        "system_prompt": preset.get("system_prompt", ""),
    }


# ── 拼装：内容层 + 结构层 → 完整 prompt 字符串 ──


def build_prompt(preset: Dict[str, Any], include_header: bool = True,
                 with_thumb_time: bool = False) -> str:
    """从模板的内容层字段 + 锁定的结构层模板，拼出最终 prompt 字符串。

    include_header=False 时不含结构约束头部（仅供前端预览展示用）。
    """
    f = preset.get("fields", {})
    tags_dim = f.get("tags_dim", "")
    tags_dim_inline = f" ({tags_dim})" if tags_dim else ""
    thumb_time_field = f',\n    "thumb_time": "{f.get("thumb_time_guidance", "")}"' if with_thumb_time else ""
    thumb_time_example = f',\n    "thumb_time": "{f.get("thumb_time_example", "")}"' if with_thumb_time else ""
    skeleton = _STRUCTURE_SKELETON.format(
        plot_guidance=f.get("plot_guidance", ""),
        tags_dim=tags_dim_inline,
        title_guidance=f.get("title_guidance", ""),
        thumb_time_field=thumb_time_field,
    )
    example = _STRUCTURE_EXAMPLE.format(
        plot_example=f.get("plot_example", ""),
        tags_example=f.get("tags_example", '"标签1", "标签2"'),
        title_example=f.get("title_example", ""),
        thumb_time_example=thumb_time_example,
    )
    parts = []
    if include_header:
        parts.append(_STRUCTURE_HEADER)
    parts.append(skeleton)
    parts.append("")
    parts.append(_STRUCTURE_EXAMPLE_HEADER)
    parts.append(example)
    return "\n".join(parts)


# ── 标签增强（全局，独立存储于 _workspace/priority_tags.json） ──


def normalize_priority_items(items: Any) -> List[Dict[str, str]]:
    """规范化标签增强列表：剔除非 dict 与空关键词，keyword/description 去首尾空格。"""
    out: List[Dict[str, str]] = []
    if not isinstance(items, list):
        return out
    for it in items:
        if not isinstance(it, dict):
            continue
        kw = str(it.get("keyword", "") or "").strip()
        if not kw:
            continue
        desc = str(it.get("description", "") or "").strip()
        out.append({"keyword": kw, "description": desc})
    return out


def load_priority_tags() -> Dict[str, Any]:
    """读取 _workspace/priority_tags.json；不存在返回默认结构。"""
    data = read_json(PRIORITY_TAGS_FILE, None)
    if not isinstance(data, dict):
        return {"enabled": False, "items": []}
    return {
        "enabled": bool(data.get("enabled", False)),
        "items": normalize_priority_items(data.get("items", [])),
    }


def save_priority_tags(enabled: Any, items: Any) -> Dict[str, Any]:
    """保存标签增强到 _workspace/priority_tags.json。"""
    data = {
        "enabled": bool(enabled),
        "items": normalize_priority_items(items),
    }
    write_json(PRIORITY_TAGS_FILE, data)
    return {"ok": True}


def build_priority_tags_section(enabled: Any, items: Any) -> str:
    """拼出注入提示词的「标签增强」段落；未启用或无有效标签时返回空串。"""
    if not enabled:
        return ""
    norm = normalize_priority_items(items)
    if not norm:
        return ""
    lines = [
        "【标签增强】",
        "为视频生成 tags 时，若画面内容匹配，请优先采用以下指定标签：",
    ]
    for it in norm:
        if it["description"]:
            lines.append(f"- {it['keyword']}：{it['description']}")
        else:
            lines.append(f"- {it['keyword']}")
    return "\n".join(lines)


def _priority_tags_section() -> str:
    """从已保存配置生成标签增强段落。"""
    data = load_priority_tags()
    return build_priority_tags_section(data.get("enabled"), data.get("items"))


def _append_priority_tags(prompt: str) -> str:
    """在 prompt 末尾追加标签增强段落（非空时）。"""
    section = _priority_tags_section()
    return prompt + "\n\n" + section if section else prompt


def preview_prompt(fields: Dict[str, Any], with_thumb_time: bool = False) -> str:
    """前端预览：用临时 fields 拼出 prompt（不保存）。"""
    return _append_priority_tags(build_prompt({"fields": fields}, with_thumb_time=with_thumb_time))


