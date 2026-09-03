# Hammy

Hammy 是仓鼠党整理入库杂乱网络视频的小工具：从视频中抽取关键帧，发送给支持视觉的
AI模型（本地或云端均可）， 自动生成标题、剧情描述与标签，再据此重命名视频文件，
并可选生成 Kodi/Jellyfin/Emby 可用的 `.nfo` 媒体信息文件和经过AI挑选的缩略图。

---

## 功能特性

- **视频去重**：快速排除完全相同副本（非删除，可找回），可选基于时长预筛 + 分阶段指纹比对排除相似版本。
- **AI 分析**：抽取多组连续关键帧 → 发送给 Vision 模型 → 返回结构化  `{"plot", "tags", "title"}` JSON。
- **重命名**：根据AI生成内容进行重命名，默认文件名格式为 日期_标题（可选追加原始名）。
- **优化缩略图**：让AI从输入的截图中挑选封面，解决jellyfin默认抽取视频10%位置关键帧作为缩略图时效果不佳问题。
- **NFO 生成**：生成包含 title / plot / tags / 时长 / 分辨率 / 编码 / 音轨等信息的 `movie` XML（Kodi/Emby/Jellyfin 兼容），可根据需要导出到视频目录。
- **提示词**：结构层（只读 JSON 约束）与内容层（可编辑）分离；支持自定义。
- **本地推理集成**：可选安装 `llama.cpp` ，提供启动设置界面与模型下载功能，自动接管 AI 地址。
- **可选语音 / 标签辅助上下文**：
  - `faster-whisper` 转录语音，注入提示词辅助AI理解，也可调用 AI 翻译导出中文字幕；
  - PixAI 角色 / IP 标签识别，用于MMD等同人视频的分类。

---

## 环境要求

| 依赖 | 说明 |
| --- | --- |
| **Windows** | GUI支持平台（脚本简单修改后或许可以在其他平台使用） |
| **Python 3.8+** | 优先使用目录内 `python\python.exe`（便携版）；缺失时回退系统 `python` |
| **ffmpeg / ffprobe** | 放入 `Hammy\ffmpeg\` 或加入系统 `PATH`，用于抽帧 / 探针 / 缩略图 |
| **openai** | `pip install "openai>=1.0"`（AI 调用 SDK） |
| **pywebview** | `pip install pywebview`（GUI 渲染后端） |
| **WebView2 Runtime** | Windows 版 pywebview 所需（系统自带 Edge 通常已包含） |
| **.NET Framework 4.8** | 原生对话框（pythonnet / WinForms）所需，Windows 一般自带 |


安装 Python 依赖：

```bash
pip install openai pywebview
```

---

## 快速开始

### 方式一：丢给龙虾、WorkBuddy甚至豆包工作等Agent（推荐）

项目本质是多个python脚本组合在一起，可以丢给Agent使用某个功能或是注册成技能

- 可以让Agent自动安装依赖；
- 可以让Agent定时或检测文件夹变化后自动处理；
- ~~Agent跑不起来它会自己改bug；~~

### 方式二：双击启动GUI

安装好依赖后直接双击 `Hammy\GUI Launcher.bat`：

---

### 推荐模型

- Gemma4系列：12B、26B、31B，默认640长边单图只需要140token，生成描述文笔流畅，对整个视频的理解较强；

- Qwen3.X系列：4B即可正确按格式输出，单图物体和角色的识别能力显著强于Gemma4，但生成描述生硬，需要修改提示词引导；

- Glimmer 30B：单图物体和角色的识别能力最准确，但中文能力弱，生成描述会使用一些很奇怪的词，且在llama.cpp中无法关闭思考。

---

## 致谢 Hammy 的功能依赖各种优秀的开源项目

Hammy 源码**不包含**下列项目的源码，而是在需要时引导用户手动安装、或由内置安装器自动安装到隔离虚拟环境后加以调用。

- **ffmpeg** —— 抽帧 / 探针 / 缩略图
  https://ffmpeg.org/
- **llama.cpp** —— 本地视觉模型推理
  https://github.com/ggml-org/llama.cpp
- **pixai-tagger** —— 角色 / IP 标签识别模型
  https://huggingface.co/pixai-labs/pixai-tagger-v0.9
- **anime_real_cls** —— 真实系 / 二次元分类模型，辅助标签判定
  https://huggingface.co/deepghs/anime_real_cls
- **faster-whisper** —— 本地语音转录
  https://github.com/SYSTRAN/faster-whisper
