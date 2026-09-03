import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import tempfile
import unittest

from gui_app.discovery import prune_excluded_dirs


class TestPruneExcludedDirs(unittest.TestCase):
    def _make(self, root, name="_failed", files=()):
        d = Path(root) / name
        d.mkdir(parents=True, exist_ok=True)
        for f in files:
            fp = d / f
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_bytes(b"x")
        return d

    def test_removes_empty_dir(self):
        with tempfile.TemporaryDirectory() as root:
            d = self._make(root)
            prune_excluded_dirs([str(d)])
            self.assertFalse(d.exists())

    def test_removes_dir_without_video(self):
        # 残留 nfo 等非视频文件也一并清掉
        with tempfile.TemporaryDirectory() as root:
            d = self._make(root, files=("a.nfo", "b.jpg"))
            prune_excluded_dirs([str(d)])
            self.assertFalse(d.exists())

    def test_keeps_dir_with_video(self):
        with tempfile.TemporaryDirectory() as root:
            d = self._make(root, files=("a.mp4",))
            prune_excluded_dirs([str(d)])
            self.assertTrue(d.exists())

    def test_keeps_dir_with_video_in_subdir(self):
        with tempfile.TemporaryDirectory() as root:
            d = self._make(root, files=("sub/a.mp4",))
            prune_excluded_dirs([str(d)])
            self.assertTrue(d.exists())

    def test_ignores_non_excluded_dir(self):
        with tempfile.TemporaryDirectory() as root:
            d = self._make(root, name="normal")
            prune_excluded_dirs([str(d)])
            self.assertTrue(d.exists())


if __name__ == "__main__":
    unittest.main()
