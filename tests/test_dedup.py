import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import tempfile
import unittest

from batch_rename.dedup import DUPLICATES_DIR, find_duplicates, move_to_duplicates, partial_hash
from batch_rename.collector import VideoCollector


class TestDedup(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _write(self, name: str, content: bytes) -> str:
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)
        return str(p)

    # ── partial_hash ────────────────────────────────────
    def test_partial_hash_same_content_same_hash(self):
        a = self._write("a.mp4", b"hello world" * 100)
        b = self._write("b.mp4", b"hello world" * 100)
        self.assertEqual(partial_hash(a), partial_hash(b))

    def test_partial_hash_different_content_different_hash(self):
        a = self._write("a.mp4", b"hello world" * 100)
        b = self._write("b.mp4", b"goodbye world" * 100)
        self.assertNotEqual(partial_hash(a), partial_hash(b))

    def test_partial_hash_small_file_full_read(self):
        # 小于 2*CHUNK 的文件走全量哈希分支
        a = self._write("a.mp4", b"tiny")
        b = self._write("b.mp4", b"tiny")
        self.assertEqual(partial_hash(a), partial_hash(b))

    def test_partial_hash_head_tail_distinction(self):
        # 头尾不同、中间相同 → 哈希应不同（验证确实读了头尾两段）
        big = 3 * 1024 * 1024  # 3MB，触发头尾采样
        a = self._write("a.mp4", b"A" * 1024 * 1024 + b"M" * 1024 * 1024 + b"A" * 1024 * 1024)
        b = self._write("b.mp4", b"A" * 1024 * 1024 + b"M" * 1024 * 1024 + b"Z" * 1024 * 1024)
        with open(a, "rb") as fh:
            self.assertEqual(len(fh.read()), big)
        self.assertNotEqual(partial_hash(a), partial_hash(b))

    def test_partial_hash_nonexistent_returns_none(self):
        self.assertIsNone(partial_hash(str(self.root / "nope.mp4")))

    # ── find_duplicates ─────────────────────────────────
    def test_find_duplicates_detects_identical(self):
        a = self._write("a.mp4", b"same content here" * 50)
        b = self._write("sub/b.mp4", b"same content here" * 50)
        groups = find_duplicates([a, b])
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]["remove"]), 1)
        # 两个路径分别在 keep / remove 中
        all_paths = {groups[0]["keep"]} | set(groups[0]["remove"])
        self.assertEqual(all_paths, {a, b})

    def test_find_duplicates_different_size_not_grouped(self):
        a = self._write("a.mp4", b"short")
        b = self._write("b.mp4", b"this is a longer content")
        self.assertEqual(find_duplicates([a, b]), [])

    def test_find_duplicates_same_size_different_content(self):
        a = self._write("a.mp4", b"AAAA" * 100)
        b = self._write("b.mp4", b"BBBB" * 100)  # 同大小不同内容
        self.assertEqual(find_duplicates([a, b]), [])

    def test_find_duplicates_keeps_shortest_path(self):
        # 保留路径字典序最前的（通常最短/最浅）
        deep = self._write("x/y/z/deep.mp4", b"dup content" * 50)
        shallow = self._write("top.mp4", b"dup content" * 50)
        groups = find_duplicates([deep, shallow])
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["keep"], shallow)
        self.assertEqual(groups[0]["remove"], [deep])

    def test_find_duplicates_multiple_groups(self):
        a1 = self._write("a1.mp4", b"group one content" * 40)
        a2 = self._write("a2.mp4", b"group one content" * 40)
        b1 = self._write("b1.mp4", b"group two different" * 40)
        b2 = self._write("b2.mp4", b"group two different" * 40)
        groups = find_duplicates([a1, a2, b1, b2])
        self.assertEqual(len(groups), 2)

    def test_find_duplicates_triple(self):
        a = self._write("a.mp4", b"triple" * 100)
        b = self._write("b.mp4", b"triple" * 100)
        c = self._write("c.mp4", b"triple" * 100)
        groups = find_duplicates([a, b, c])
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]["remove"]), 2)

    def test_find_duplicates_empty_input(self):
        self.assertEqual(find_duplicates([]), [])

    # ── move_to_duplicates ──────────────────────────────
    def test_move_creates_duplicates_dir(self):
        a = self._write("keep.mp4", b"data" * 100)
        b = self._write("dup.mp4", b"data" * 100)
        moved, errors = move_to_duplicates([b])
        self.assertEqual(moved, [b])
        self.assertEqual(errors, [])
        self.assertFalse(Path(b).exists())
        self.assertTrue((self.root / DUPLICATES_DIR / "dup.mp4").exists())
        self.assertTrue(Path(a).exists())  # keep 未被移动

    def test_move_handles_name_collision(self):
        b1 = self._write("a/dup.mp4", b"data one" * 100)
        b2 = self._write("b/dup.mp4", b"data two" * 100)
        # 两个同名文件，移到同一个 _duplicates（此处分别在各自父目录，不冲突）
        moved, errors = move_to_duplicates([b1, b2])
        self.assertEqual(len(moved), 2)
        self.assertEqual(errors, [])

    def test_move_same_dir_collision_appends_suffix(self):
        # 同一目录下两个同名重复文件 → 第二个应追加 _1 后缀
        a = self._write("clip.mp4", b"same" * 100)
        b = self._write("clip copy.mp4", b"same" * 100)
        # 先手动制造一个已存在的 _duplicates/clip copy.mp4
        dup_dir = self.root / DUPLICATES_DIR
        dup_dir.mkdir(exist_ok=True)
        (dup_dir / "clip copy.mp4").write_bytes(b"pre-existing")
        moved, errors = move_to_duplicates([b])
        self.assertEqual(moved, [b])
        # 应生成 clip copy_1.mp4 避让冲突
        self.assertTrue((dup_dir / "clip copy_1.mp4").exists())

    def test_move_nonexistent_reported_as_error(self):
        moved, errors = move_to_duplicates([str(self.root / "ghost.mp4")])
        self.assertEqual(moved, [])
        self.assertEqual(len(errors), 1)

    # ── 集成：_duplicates 目录被扫描排除 ─────────────────
    def test_collector_skips_duplicates_dir(self):
        self._write("ok.mp4", b"video")
        self._write(f"{DUPLICATES_DIR}/dup.mp4", b"video")
        result = VideoCollector.collect([str(self.root)])
        names = [Path(p).name for p in result]
        self.assertEqual(names, ["ok.mp4"])


if __name__ == "__main__":
    unittest.main()
