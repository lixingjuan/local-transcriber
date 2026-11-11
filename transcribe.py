#!/usr/bin/env python3
"""Transcribe local audio/video files or a single video URL locally."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse


@dataclass(frozen=True)
class TranscriptSegment:
    start: float
    end: float
    text: str


def is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def download_audio(url: str, download_dir: Path) -> Path:
    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError(
            "缺少 yt-dlp。请先运行：python -m pip install -r requirements.txt"
        ) from exc

    download_dir.mkdir(parents=True, exist_ok=True)
    options = {
        "format": "bestaudio/best",
        "outtmpl": str(download_dir / "%(title).120B [%(id)s].%(ext)s"),
        "noplaylist": True,
        "restrictfilenames": False,
    }

    print("正在下载音频…")
    with yt_dlp.YoutubeDL(options) as downloader:
        info = downloader.extract_info(url, download=True)
        requested = info.get("requested_downloads") or []
        candidates = [item.get("filepath") for item in requested if item.get("filepath")]
        candidates.append(downloader.prepare_filename(info))

    for candidate in candidates:
        path = Path(candidate)
        if path.is_file():
            print(f"已下载：{path}")
            return path

    raise RuntimeError("下载完成后没有找到音频文件。")


def transcribe(
    media_path: Path,
    model_name: str,
    language: str | None,
    task: str,
) -> tuple[list[TranscriptSegment], dict[str, object]]:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "缺少 faster-whisper。请先运行：python -m pip install -r requirements.txt"
        ) from exc

    print(f"正在加载模型：{model_name}（首次使用会免费下载模型）")
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    raw_segments, info = model.transcribe(
        str(media_path),
        language=language,
        task=task,
        vad_filter=True,
        beam_size=5,
    )

    print("正在转写…")
    segments: list[TranscriptSegment] = []
    for segment in raw_segments:
        text = segment.text.strip()
        if not text:
            continue
        item = TranscriptSegment(segment.start, segment.end, text)
        segments.append(item)
        print(f"[{format_clock(item.start)}] {item.text}")

    metadata = {
        "source": str(media_path),
        "model": model_name,
        "language": info.language,
        "language_probability": info.language_probability,
        "duration_seconds": info.duration,
        "task": task,
    }
    return segments, metadata


def format_clock(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_timestamp(seconds: float, separator: str) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


def write_outputs(
    segments: Iterable[TranscriptSegment],
    metadata: dict[str, object],
    output_dir: Path,
    stem: str,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    items = list(segments)

    txt_path = output_dir / f"{stem}.txt"
    txt_path.write_text("\n".join(item.text for item in items) + "\n", encoding="utf-8")

    srt_parts = []
    for index, item in enumerate(items, start=1):
        start = format_timestamp(item.start, ",")
        end = format_timestamp(item.end, ",")
        srt_parts.append(f"{index}\n{start} --> {end}\n{item.text}\n")
    srt_path = output_dir / f"{stem}.srt"
    srt_path.write_text("\n".join(srt_parts), encoding="utf-8")

    vtt_parts = ["WEBVTT\n"]
    for item in items:
        start = format_timestamp(item.start, ".")
        end = format_timestamp(item.end, ".")
        vtt_parts.append(f"{start} --> {end}\n{item.text}\n")
    vtt_path = output_dir / f"{stem}.vtt"
    vtt_path.write_text("\n".join(vtt_parts), encoding="utf-8")

    json_path = output_dir / f"{stem}.json"
    payload = {"metadata": metadata, "segments": [asdict(item) for item in items]}
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    return [txt_path, srt_path, vtt_path, json_path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用免费的本地 Whisper 模型转写音频、视频或单个视频网址。"
    )
    parser.add_argument("input", help="本地音视频文件路径，或 YouTube 等 yt-dlp 支持的网址")
    parser.add_argument(
        "--model",
        default="small",
        choices=["tiny", "base", "small", "medium", "large-v3", "turbo"],
        help="Whisper 模型（默认：small；越大通常越准确，但越慢）",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="语言代码，如 zh、en；不填写则自动检测",
    )
    parser.add_argument(
        "--task",
        choices=["transcribe", "translate"],
        default="transcribe",
        help="transcribe 保留原语言；translate 翻译为英文",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="输出目录（默认：当前目录下的 output）",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=Path("downloads"),
        help="网址音频下载目录（默认：当前目录下的 downloads）",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if is_url(args.input):
            media_path = download_audio(args.input, args.download_dir).resolve()
        else:
            media_path = Path(args.input).expanduser().resolve()
            if not media_path.is_file():
                raise RuntimeError(f"找不到文件：{media_path}")

        segments, metadata = transcribe(
            media_path=media_path,
            model_name=args.model,
            language=args.language,
            task=args.task,
        )
        if not segments:
            raise RuntimeError("没有识别到语音内容。")

        output_paths = write_outputs(
            segments=segments,
            metadata=metadata,
            output_dir=args.output_dir.resolve(),
            stem=media_path.stem,
        )
        print("\n转写完成：")
        for path in output_paths:
            print(f"- {path}")
        return 0
    except (RuntimeError, OSError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
