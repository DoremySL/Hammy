"""配置数据类。"""
from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING

# openai 延迟导入：TYPE_CHECKING 用真实类型，运行时回落为 Any
if TYPE_CHECKING:
    from openai import OpenAI as OpenAIClient
else:
    OpenAIClient = Any

@dataclass
class Config:
    """所有用户可配置参数，由 GUI config.json 填充。"""
    model: str = "gpt-4o"
    base_url: str = "http://localhost:8080/v1"
    api_key: str = "not-needed"
    sampling_points: int = 5
    frames_per_point: int = 3
    frame_max_side: int = 640
    frame_time_tags: int = 0  # 0=不添加 1=添加时间标签 2=添加并用于优化缩略图
    ai_workers: int = 4
    retry_times: int = 2
    ai_timeout: int = 60
    max_tokens: int = 500
    temperature: float = 0.6
    top_p: float = 0.8
    enforce_json_mode: bool = True
    nfo_target_dir: Optional[str] = None
    # prompt / system_prompt 由 GUI 注入激活预设
    system_prompt: str = ""
    prompt: str = ""
    include_date: bool = True
    include_original: bool = False
    rag_top_k: int = 40
    # 调试开关（GUI 无界面）：完成日志在「召回N个关键词」后列出具体关键词。
    # 由 config.json ai 段同名键控制，手动改为 true 生效
    rag_debug_keywords: bool = False
    llama_slots: int = 0
    rag_vec_enabled: bool = False
    rag_vec_device: str = "auto"      # auto / cuda / cpu
    rag_vec_threshold: float = 0.45   # 余弦相似度阈值
    rag_vec_top_n: int = 20           # 向量候选上限
    rag_vec_model: str = ""           # 模型 repo（空 = 模块默认 Qwen3-Embedding-0.6B）

    def validate(self) -> None:
        """钳制所有数值参数到有效范围。"""
        self.temperature = max(0.0, min(2.0, self.temperature))
        self.top_p = max(0.0, min(1.0, self.top_p))
        self.max_tokens = max(1, self.max_tokens)
        self.sampling_points = max(1, self.sampling_points)
        self.frames_per_point = max(1, self.frames_per_point)
        self.frame_max_side = max(64, self.frame_max_side)
        self.ai_workers = max(1, self.ai_workers)
        self.ai_timeout = max(1, self.ai_timeout)
        self.retry_times = max(0, self.retry_times)
        self.rag_top_k = max(1, self.rag_top_k)
        self.llama_slots = max(0, self.llama_slots)
        self.rag_vec_threshold = max(0.0, min(1.0, self.rag_vec_threshold))
        self.rag_vec_top_n = max(1, self.rag_vec_top_n)
