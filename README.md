# 本地音视频转文字

使用免费的 `faster-whisper` 在本机转写音频或视频，不调用任何付费 API。支持本地文件，也支持使用 `yt-dlp` 下载单个 YouTube 视频的音频后转写。

## 特点

- 完全本地转写，不需要 OpenAI API 或其他付费服务
- 支持 MP3、M4A、WAV、MP4、MOV、WebM 等常见格式
- 可以直接处理本地文件或单个 YouTube 地址
- 自动生成 TXT、SRT、VTT 和 JSON
- 可选的本地说话人分离，生成按人物发言合并的对话稿
- 自动检测语言，也可以手动指定中文、英文等语言

## 安装

在终端中运行：

```bash
cd /Users/lixingjuan/Documents/lxj/local-transcriber
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

首次转写某个模型时，会从 Hugging Face 免费下载模型文件。之后可以离线转写本地文件。

如果需要按说话人生成对话稿，还要使用免费 Hugging Face 账号完成一次登录：

```bash
hf auth login
```

登录前请先在
[`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1)
页面同意模型的使用条件。Token 只用于免费下载模型，不会产生 API 费用。

## 使用本地文件

```bash
source .venv/bin/activate
python transcribe.py "/完整路径/视频.mp4"
```

音频文件（如 MP3、M4A、WAV）也可以使用同样的命令。

## 按说话人生成对话稿

```bash
python transcribe.py "视频.mp4" --dialogue
```

如果已知视频中有两个人，明确指定人数通常更稳定：

```bash
python transcribe.py "视频.mp4" --dialogue --speakers 2
```

连续属于同一个人的内容会合并为一个段落，最终使用“说话人 1”、
“说话人 2”等默认名称，不需要手动填写人物姓名。对话稿正文不显示时间戳。

## 使用 YouTube 地址

```bash
source .venv/bin/activate
python transcribe.py "https://www.youtube.com/watch?v=Ts85OjXMtpc"
```

这会把音频保存在 `downloads/`，再在本机进行转写。请只下载你有权下载和使用的内容，并遵守网站条款。

如果 YouTube 要求登录，可以先在浏览器中手动下载视频，再按“使用本地文件”的方式转写。

## 输出

默认在 `output/` 目录生成四个文件：

- `.txt`：纯文字
- `.srt`：常见字幕格式
- `.vtt`：网页字幕格式
- `.json`：包含时间段和识别信息的结构化结果

使用 `--dialogue` 时，还会生成：

- `.dialogue.txt`：按说话人分段的纯文本对话稿
- `.dialogue.md`：按说话人分段的 Markdown 对话稿
- `.dialogue.json`：保留说话人和内部时间信息的结构化结果

## 常用选项

```bash
# 明确指定中文
python transcribe.py "视频.mp4" --language zh

# 更快但准确率略低
python transcribe.py "视频.mp4" --model base

# 更准确但更慢、占用更多磁盘和内存
python transcribe.py "视频.mp4" --model medium

# 自定义输出目录
python transcribe.py "视频.mp4" --output-dir "/完整路径/转写结果"
```

默认使用 `small` 模型，是速度和准确率之间较稳妥的选择。整个识别过程使用 CPU，不会产生 API 费用。

## 测试

核心功能测试不需要下载模型：

```bash
python -m unittest discover -s tests -v
```

## 许可证

本项目使用 [MIT License](LICENSE)。
