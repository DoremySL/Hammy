"""gui_app/constants.py — GUI 层数值常量集中管理。"""

# ── 缩略图 ──
THUMB_MAX_SIDE = 320          # 缩略图最大边长（px）
THUMB_TIMEOUT = 15            # ffmpeg 抽帧超时（秒）
THUMB_LRU_MAX = 500          # 后端缩略图内存缓存上限（条）

# ── 探针 ──
PROBE_TIMEOUT = 30            # ffprobe 超时（秒）
PROBE_POOL_WORKERS = 8        # 后台探针/缩略图线程池并发数

# ── JS 推送队列 ──
PUSH_QUEUE_MAX = 200          # 队列上限，超出截断保留最新

