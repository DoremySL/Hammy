"""两轮标签检索流程测试：首轮字面注入 / 消息拼接（前缀一致 + assistant 原文回传）/
slot_id 透传 / 第二轮失败回退首轮 / 无候选单轮结束 / on 档单轮注入布局 /
enhanced 无向量降级等同 on / recalled 召回数 / 成功路径无逐视频日志。"""
import json
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from batch_rename.ai import analyze_frames
from batch_rename.config import Config
from batch_rename.pipeline import _ai_seconds_per_video, _worker_slot_ids
from batch_rename.tag_recall import TagRecall
from batch_rename.types import Frame


def _resp(content, pt=5000, ct=300):
    usage = SimpleNamespace(prompt_tokens=pt, completion_tokens=ct)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=usage)


class _FakeCompletions:
    """按脚本依次返回内容或抛错，并捕获调用参数。"""

    def __init__(self, script, calls):
        self._script = list(script)
        self._calls = calls

    def create(self, **kwargs):
        self._calls.append(kwargs)
        if not self._script:
            raise RuntimeError("脚本耗尽")
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return _resp(item)


class _FakeClient:
    def __init__(self, script):
        self.calls = []
        self.chat = SimpleNamespace(completions=_FakeCompletions(script, self.calls))


class _FakeDense:
    """向量后端桩：预置 (keyword, score) 命中表。"""

    def __init__(self, hits=None, fail=False):
        self.hits = hits or []
        self.fail = fail
        self.last_parts = None

    def build_index(self, items):
        if self.fail:
            raise RuntimeError("boom")
        self._kw2idx = {it["keyword"]: i for i, it in enumerate(items)}

    def search(self, parts, top_k=64):
        if self.fail:
            raise RuntimeError("boom")
        self.last_parts = dict(parts)
        return [{"idx": self._kw2idx[kw], "score": sc}
                for kw, sc in self.hits if kw in self._kw2idx]


R1 = json.dumps({"plot": "她穿着泳装在海边漫步", "tags": ["沙滩"], "title": "海边-沙滩",
                 "thumb_time": "00:01:02"}, ensure_ascii=False)
R2 = json.dumps({"plot": "她穿着泳装在海边漫步", "tags": ["沙滩", "海边", "泳装"],
                 "title": "海边-沙滩"}, ensure_ascii=False)
# 与标签库（海边/泳装/制服）无词面交集的文件名
R1_CITY = json.dumps({"plot": "城市霓虹夜景", "tags": ["夜景"], "title": "城市-夜景"},
                     ensure_ascii=False)

ITEMS = [{"keyword": "海边", "description": "海边的场景"},
         {"keyword": "泳装", "description": "游泳服装"},
         {"keyword": "制服", "description": "制服"}]


def _on(**kw):
    return TagRecall(ITEMS, mode="on", **kw)


def _enh(dense=None, **kw):
    return TagRecall(ITEMS, mode="enhanced", dense=dense, **kw)


def _cfg(**kw):
    return Config(ai_timeout=5, retry_times=1, **kw)


def _frames():
    return [Frame(0.0, "fakeb64")]


class TestTwoRoundSuccess(unittest.TestCase):
    def test_messages_prefix_and_slot(self):
        client = _FakeClient([R1, R2])
        analyze_frames(client, "m", _frames(), _cfg(), threading.Event(),
                       "C:/视频/海边测试.mp4", 60.0,
                       slot_id=2, tag_recall=_enh(_FakeDense([("海边", 0.8), ("泳装", 0.7)])))
        self.assertEqual(len(client.calls), 2)

        call1, call2 = client.calls
        # 两轮都透传 id_slot=2（llama-server OAI 端点原样复制该字段）
        self.assertEqual(call1["extra_body"], {"id_slot": 2})
        self.assertEqual(call2["extra_body"], {"id_slot": 2})
        # 首轮已注入字面召回段落（文件名「海边测试」命中海边；泳装无字面命中不注入）
        r1_text = json.dumps(call1["messages"], ensure_ascii=False)
        self.assertIn("【标签检索】", r1_text)
        self.assertIn("- 海边：海边的场景", r1_text)
        self.assertNotIn("泳装", r1_text)

        msgs1 = call1["messages"]
        msgs2 = call2["messages"]
        # 第二轮 = 第一轮消息 + assistant(首轮原文) + user(候选+指令)：前缀逐条一致
        self.assertEqual(len(msgs2), len(msgs1) + 2)
        self.assertEqual(msgs2[:len(msgs1)], msgs1)
        self.assertEqual(msgs2[-2]["role"], "assistant")
        self.assertEqual(msgs2[-2]["content"], R1)
        self.assertEqual(msgs2[-1]["role"], "user")
        self.assertIn("召回候选", msgs2[-1]["content"])
        self.assertIn("海边", msgs2[-1]["content"])
        self.assertIn("泳装", msgs2[-1]["content"])

    def test_second_round_result_adopted(self):
        client = _FakeClient([R1, R2])
        result = analyze_frames(client, "m", _frames(), _cfg(), threading.Event(),
                                "C:/视频/海边测试.mp4", 60.0,
                                tag_recall=_enh(_FakeDense([("海边", 0.8), ("泳装", 0.7)])))
        # 采纳第二轮结果；thumb_time 第二轮缺失 → 回退首轮
        self.assertEqual(result.title, "海边-沙滩")
        self.assertEqual(result.tags, ["沙滩", "海边", "泳装"])
        self.assertEqual(result.thumb_time, "00:01:02")
        # 召回数 = 第二轮向量候选数（海边+泳装）
        self.assertEqual(result.recalled, 2)

    def test_dense_query_parts(self):
        dense = _FakeDense([("海边", 0.8)])
        client = _FakeClient([R1, R2])
        analyze_frames(client, "m", _frames(), _cfg(), threading.Event(),
                       "C:/视频/海边测试.mp4", 60.0,
                       tag_recall=_enh(dense))
        # 第二轮向量查询包含 plot / tags / filename / aux
        self.assertEqual(set(dense.last_parts), {"plot", "tags", "filename", "aux"})
        self.assertIn("海边", dense.last_parts["filename"])

    def test_slot_id_none_no_extra_body(self):
        client = _FakeClient([R1])
        result = analyze_frames(client, "m", _frames(), _cfg(), threading.Event(),
                                "C:/a.mp4", 60.0, tag_recall=None)
        self.assertNotIn("extra_body", client.calls[0])
        self.assertEqual(result.recalled, -1)  # 检索关闭：完成日志不追加召回数

    def test_no_dense_candidates_single_round(self):
        client = _FakeClient([R1_CITY])
        result = analyze_frames(client, "m", _frames(), _cfg(), threading.Event(),
                                "C:/城市夜景.mp4", 60.0, slot_id=0,
                                tag_recall=_enh(_FakeDense([])))
        self.assertEqual(len(client.calls), 1)  # 向量无命中 → 不再发起第二轮
        self.assertEqual(result.tags, ["夜景"])
        self.assertEqual(result.recalled, 0)


class TestTwoRoundFallback(unittest.TestCase):
    def test_round2_exception_keeps_round1(self):
        client = _FakeClient([R1, RuntimeError("boom")])
        result = analyze_frames(client, "m", _frames(), _cfg(), threading.Event(),
                                "C:/海边测试.mp4", 60.0,
                                tag_recall=_enh(_FakeDense([("海边", 0.8), ("泳装", 0.7)])))
        self.assertEqual(result.tags, ["沙滩"])
        self.assertEqual(result.err_msg, "")
        self.assertEqual(result.recalled, 2)  # 召回已发生，回退不影响计数

    def test_round2_invalid_output_keeps_round1(self):
        bad = json.dumps({"plot": "没有 title"}, ensure_ascii=False)
        client = _FakeClient([R1, bad, bad])
        result = analyze_frames(client, "m", _frames(), _cfg(), threading.Event(),
                                "C:/海边测试.mp4", 60.0,
                                tag_recall=_enh(_FakeDense([("海边", 0.8), ("泳装", 0.7)])))
        self.assertEqual(result.tags, ["沙滩"])
        self.assertEqual(result.recalled, 2)

    def test_enhanced_without_dense_degrades_to_on(self):
        # 无嵌入模型：增强档 = 开启档（单轮 + 首轮字面注入）
        client = _FakeClient([R1])
        result = analyze_frames(client, "m", _frames(), _cfg(), threading.Event(),
                                "C:/海边测试.mp4", 60.0, tag_recall=_enh())
        self.assertEqual(len(client.calls), 1)
        r1_text = json.dumps(client.calls[0]["messages"], ensure_ascii=False)
        self.assertIn("【标签检索】", r1_text)
        self.assertEqual(result.tags, ["沙滩"])
        self.assertEqual(result.recalled, -1)  # 字面注入不计入召回日志


class TestOnModeSingleRound(unittest.TestCase):
    def test_on_mode_literal_injected_layout(self):
        # 回归锁定：开启档下，字面召回段落与 prompt 拼接进同一段文本且以空行分隔
        sep = chr(10) * 2
        tr = _on()
        client = _FakeClient([R1])
        analyze_frames(client, "m", _frames(), _cfg(frame_time_tags=0), threading.Event(),
                       "C:/海边.mp4", 60.0, tag_recall=tr)
        expected, kws = tr.literal_injection({"filename": "海边"})
        self.assertEqual(kws, ["海边"])
        parts = client.calls[0]["messages"][-1]["content"]
        texts = [p["text"] for p in parts if p.get("type") == "text"]
        combined = texts[-1]
        self.assertTrue(combined.endswith(sep + expected))
        self.assertEqual(len(texts), 2)  # meta 段 + 合并后的 prompt/标签段

    def test_on_mode_no_hit_no_injection(self):
        client = _FakeClient([R1_CITY])
        result = analyze_frames(client, "m", _frames(), _cfg(), threading.Event(),
                                "C:/城市夜景.mp4", 60.0, tag_recall=_on())
        self.assertEqual(len(client.calls), 1)
        r1_text = json.dumps(client.calls[0]["messages"], ensure_ascii=False)
        self.assertNotIn("标签检索", r1_text)
        self.assertEqual(result.recalled, -1)  # 字面检索不追加召回日志

    def test_on_mode_no_recall_log_count(self):
        # 开启档有命中：注入生效但不计入 recalled（完成日志不显示「召回N个关键词」）
        client = _FakeClient([R1])
        result = analyze_frames(client, "m", _frames(), _cfg(), threading.Event(),
                                "C:/海边测试.mp4", 60.0, tag_recall=_on())
        r1_text = json.dumps(client.calls[0]["messages"], ensure_ascii=False)
        self.assertIn("- 海边：海边的场景", r1_text)
        self.assertEqual(result.recalled, -1)
        self.assertEqual(result.recalled_keywords, ())

    def test_frame_time_tags_inserts_stamp_before_each_frame(self):
        tr = _on()
        client = _FakeClient([R1])
        frames = _frames()
        analyze_frames(client, "m", frames, _cfg(frame_time_tags=2), threading.Event(),
                       "C:/海边.mp4", 60.0, tag_recall=tr)
        parts = client.calls[0]["messages"][-1]["content"]
        texts = [p["text"] for p in parts if p.get("type") == "text"]
        self.assertEqual(len(texts), len(frames) + 2)  # 每帧时间戳段 + meta 段 + prompt/标签段
        self.assertEqual(parts[0], {"type": "text", "text": "\n00:00:00:"})
        self.assertEqual(parts[1]["type"], "image_url")

    def test_literal_query_excludes_extra_meta(self):
        # 首轮字面查询只看 filename + literal_meta，不含提示词 aux（转录等噪音）
        client = _FakeClient([R1_CITY])
        result = analyze_frames(client, "m", _frames(), _cfg(), threading.Event(),
                                "C:/城市夜景.mp4", 60.0,
                                extra_meta="[辅助参考信息]\n- 海边漫步的角色",  # 不参与字面查询
                                tag_recall=_on())
        r1_text = json.dumps(client.calls[0]["messages"], ensure_ascii=False)
        self.assertNotIn("标签检索", r1_text)  # aux 命中不算字面召回
        self.assertEqual(result.recalled, -1)  # 字面检索不追加召回日志

    def test_literal_meta_participates(self):
        # literal_meta（tagger 上下文）参与首轮字面查询
        client = _FakeClient([R1_CITY])
        result = analyze_frames(client, "m", _frames(), _cfg(), threading.Event(),
                                "C:/城市夜景.mp4", 60.0,
                                literal_meta="- 角色标签: 泳装",
                                tag_recall=_on())
        r1_text = json.dumps(client.calls[0]["messages"], ensure_ascii=False)
        self.assertIn("- 泳装：游泳服装", r1_text)
        self.assertEqual(result.recalled, -1)  # 字面注入不计入召回日志


class TestPipelineHelpers(unittest.TestCase):
    def test_worker_slot_ids(self):
        self.assertEqual(_worker_slot_ids(4, 4), [0, 1, 2, 3])
        self.assertEqual(_worker_slot_ids(7, 4), [0, 1, 2, 3, 0, 1, 2])
        self.assertEqual(_worker_slot_ids(3, 0), [None] * 3)  # 远程 API：不固定槽位

    def test_ai_seconds_two_round_doubled(self):
        cfg = Config(ai_timeout=60, retry_times=2)
        self.assertEqual(_ai_seconds_per_video(cfg, False), (60 + 10) * 3)
        self.assertEqual(_ai_seconds_per_video(cfg, True), (60 + 10) * 3 * 2)


class TestQuietLogging(unittest.TestCase):
    """成功路径不再输出逐视频 RAG 日志块（恢复原版简洁显示；
    完成日志由管线统一打一行「旧名 -> 新名（召回N个关键词）」）。"""

    def test_no_logs_two_round_success(self):
        client = _FakeClient([R1, R2])
        with self.assertNoLogs("BatchRename", level="INFO"):
            analyze_frames(client, "m", _frames(), _cfg(), threading.Event(),
                           "C:/海边测试.mp4", 60.0,
                           tag_recall=_enh(_FakeDense([("海边", 0.8)])))

    def test_no_logs_no_candidate(self):
        client = _FakeClient([R1_CITY])
        with self.assertNoLogs("BatchRename", level="INFO"):
            analyze_frames(client, "m", _frames(), _cfg(), threading.Event(),
                           "C:/城市夜景.mp4", 60.0, tag_recall=_on())

    def test_no_logs_single_round_without_recall(self):
        client = _FakeClient([R1])
        with self.assertNoLogs("BatchRename", level="INFO"):
            analyze_frames(client, "m", _frames(), _cfg(), threading.Event(),
                           "C:/a.mp4", 60.0)


if __name__ == "__main__":
    unittest.main()
