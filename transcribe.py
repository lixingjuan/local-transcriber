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


@dataclass(frozen=True)
class TranscriptWord:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class SpeakerSpan:
    start: float
    end: float
    speaker: str


@dataclass(frozen=True)
class DialogueTurn:
    start: float
    end: float
    speaker: str
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
) -> tuple[list[TranscriptSegment], list[TranscriptWord], dict[str, object]]:
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
        word_timestamps=True,
    )

    print("正在转写…")
    segments: list[TranscriptSegment] = []
    words: list[TranscriptWord] = []
    for segment in raw_segments:
        text = segment.text.strip()
        if not text:
            continue
        item = TranscriptSegment(segment.start, segment.end, text)
        segments.append(item)
        for word in segment.words or []:
            if word.start is None or word.end is None or not word.word.strip():
                continue
            words.append(TranscriptWord(word.start, word.end, word.word))
        print(f"[{format_clock(item.start)}] {item.text}")

    metadata = {
        "source": str(media_path),
        "model": model_name,
        "language": info.language,
        "language_probability": info.language_probability,
        "duration_seconds": info.duration,
        "task": task,
    }
    return segments, words, metadata


def diarize(media_path: Path, num_speakers: int | None) -> list[SpeakerSpan]:
    try:
        import torch
        from faster_whisper.audio import decode_audio
        from huggingface_hub import get_token
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise RuntimeError(
            "缺少说话人分离依赖。请运行：python -m pip install -r requirements.txt"
        ) from exc

    token = get_token()
    if not token:
        raise RuntimeError(
            "没有找到 Hugging Face Token。请先运行：hf auth login"
        )

    print("正在加载本地说话人分离模型（首次使用会免费下载模型）…")
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1", token=token
        )
    except Exception as exc:
        raise RuntimeError(
            "无法下载说话人分离模型。请确认已在模型页面同意使用条件，"
            "并且 Hugging Face Token 具有读取权限。"
        ) from exc

    print("正在识别不同说话人…")
    audio = decode_audio(str(media_path), sampling_rate=16_000)
    waveform = torch.from_numpy(audio).unsqueeze(0)
    arguments = {"num_speakers": num_speakers} if num_speakers else {}
    try:
        result = pipeline(
            {"waveform": waveform, "sample_rate": 16_000}, **arguments
        )
    except Exception as exc:
        raise RuntimeError(f"说话人分离失败：{exc}") from exc

    annotation = getattr(result, "speaker_diarization", result)
    spans = [
        SpeakerSpan(turn.start, turn.end, str(speaker))
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]
    spans.sort(key=lambda item: (item.start, item.end))
    if not spans:
        raise RuntimeError("没有识别到说话人。")
    return spans


def build_dialogue(
    words: Iterable[TranscriptWord], spans: Iterable[SpeakerSpan]
) -> list[DialogueTurn]:
    word_items = sorted(words, key=lambda item: (item.start, item.end))
    span_items = list(spans)
    if not word_items or not span_items:
        return []

    speaker_names: dict[str, str] = {}
    turns: list[DialogueTurn] = []
    current_speaker = ""
    current_start = 0.0
    current_end = 0.0
    current_words: list[str] = []

    for word in word_items:
        raw_speaker = best_speaker_for_word(word, span_items)
        if raw_speaker not in speaker_names:
            speaker_names[raw_speaker] = f"说话人 {len(speaker_names) + 1}"
        speaker = speaker_names[raw_speaker]

        if current_words and speaker != current_speaker:
            turns.append(
                DialogueTurn(
                    current_start,
                    current_end,
                    current_speaker,
                    "".join(current_words).strip(),
                )
            )
            current_words = []

        if not current_words:
            current_speaker = speaker
            current_start = word.start
        current_end = word.end
        current_words.append(word.text)

    if current_words:
        turns.append(
            DialogueTurn(
                current_start,
                current_end,
                current_speaker,
                "".join(current_words).strip(),
            )
        )
    return [turn for turn in turns if turn.text]


def best_speaker_for_word(word: TranscriptWord, spans: list[SpeakerSpan]) -> str:
    midpoint = (word.start + word.end) / 2

    def score(span: SpeakerSpan) -> tuple[float, float]:
        overlap = max(0.0, min(word.end, span.end) - max(word.start, span.start))
        distance = max(span.start - midpoint, midpoint - span.end, 0.0)
        return overlap, -distance

    return max(spans, key=score).speaker


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


def write_dialogue_outputs(
    turns: Iterable[DialogueTurn], output_dir: Path, stem: str
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    items = list(turns)

    txt_blocks = [f"{item.speaker}：\n\n{item.text}" for item in items]
    txt_path = output_dir / f"{stem}.dialogue.txt"
    txt_path.write_text("\n\n".join(txt_blocks) + "\n", encoding="utf-8")

    markdown_blocks = [f"## {item.speaker}\n\n{item.text}" for item in items]
    markdown_path = output_dir / f"{stem}.dialogue.md"
    markdown_path.write_text(
        "\n\n".join(markdown_blocks) + "\n", encoding="utf-8"
    )

    json_path = output_dir / f"{stem}.dialogue.json"
    json_path.write_text(
        json.dumps([asdict(item) for item in items], ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return [txt_path, markdown_path, json_path]


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
    parser.add_argument(
        "--dialogue",
        action="store_true",
        help="识别不同说话人，并生成按人物发言合并的对话稿",
    )
    parser.add_argument(
        "--speakers",
        type=int,
        default=None,
        help="已知的说话人数；例如两人访谈填写 2",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.speakers is not None and args.speakers < 1:
            raise RuntimeError("说话人数必须大于或等于 1。")
        if args.speakers is not None and not args.dialogue:
            raise RuntimeError("--speakers 需要和 --dialogue 一起使用。")

        if is_url(args.input):
            media_path = download_audio(args.input, args.download_dir).resolve()
        else:
            media_path = Path(args.input).expanduser().resolve()
            if not media_path.is_file():
                raise RuntimeError(f"找不到文件：{media_path}")

        segments, words, metadata = transcribe(
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
        if args.dialogue:
            if not words:
                raise RuntimeError("没有获得词级时间信息，无法生成对话稿。")
            speaker_spans = diarize(media_path, args.speakers)
            dialogue_turns = build_dialogue(words, speaker_spans)
            if not dialogue_turns:
                raise RuntimeError("无法把转写内容匹配到说话人。")
            output_paths.extend(
                write_dialogue_outputs(
                    dialogue_turns, args.output_dir.resolve(), media_path.stem
                )
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
