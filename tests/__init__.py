"""tests 包初始化：mkdtemp 的 0o700 ACL 目录本机删除极慢（实测拖慢 3 倍），
创建后 icacls /reset 恢复继承 ACL；仅 Windows 生效，失败静默降级。"""
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
