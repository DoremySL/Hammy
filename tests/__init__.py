"""tests 包初始化：规避本机对受限 ACL 临时目录删除的慢速过滤。

mkdtemp 用 0o700 创建目录会产生仅所有者 ACL，本机删除这类目录每个
约 1.5s（实测整套测试因此从 282s 降到 74s）。创建后立即 icacls /reset
恢复继承 ACL，删除恢复瞬时；仅 Windows 生效，失败静默降级。
"""
import os
import subprocess
import tempfile

if os.name == "nt":
    _orig_mkdtemp = tempfile.mkdtemp

    def _mkdtemp(*args, **kwargs):
        path = _orig_mkdtemp(*args, **kwargs)
        try:
            subprocess.run(["icacls", path, "/reset"], capture_output=True, timeout=10)
        except Exception:
            pass
        return path

    tempfile.mkdtemp = _mkdtemp
