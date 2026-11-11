import json
import tempfile
import unittest
from pathlib import Path

from transcribe import (
    DialogueTurn,
    SpeakerSpan,
    TranscriptSegment,
    TranscriptWord,
    build_dialogue,
    format_clock,
    format_timestamp,
    is_url,
    write_dialogue_outputs,
    write_outputs,
)


class TranscribeTests(unittest.TestCase):
    def test_url_detection(self):
        self.assertTrue(is_url("https://www.youtube.com/watch?v=example"))
        self.assertFalse(is_url("/Users/example/video.mp4"))

    def test_timestamp_formatting(self):
        self.assertEqual(format_clock(3661.9), "01:01:01")
        self.assertEqual(format_timestamp(3661.234, ","), "01:01:01,234")
        self.assertEqual(format_timestamp(3661.234, "."), "01:01:01.234")

    def test_writes_all_output_formats(self):
        segments = [
            TranscriptSegment(0.0, 1.25, "第一句。"),
            TranscriptSegment(1.25, 3.5, "Second sentence."),
        ]
        metadata = {"language": "zh", "model": "tiny"}

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            paths = write_outputs(segments, metadata, output_dir, "sample")

            self.assertEqual(
                {path.suffix for path in paths}, {".txt", ".srt", ".vtt", ".json"}
            )
            self.assertEqual(
                (output_dir / "sample.txt").read_text(encoding="utf-8"),
                "第一句。\nSecond sentence.\n",
            )
            self.assertIn(
                "00:00:00,000 --> 00:00:01,250",
                (output_dir / "sample.srt").read_text(encoding="utf-8"),
            )
            payload = json.loads(
                (output_dir / "sample.json").read_text(encoding="utf-8")
            )
            self.assertEqual(payload["metadata"]["language"], "zh")
            self.assertEqual(len(payload["segments"]), 2)

    def test_builds_dialogue_by_speaker_turn(self):
        words = [
            TranscriptWord(0.0, 0.5, " Hello"),
            TranscriptWord(0.5, 1.0, " there."),
            TranscriptWord(1.1, 1.5, " Hi!"),
            TranscriptWord(1.6, 2.0, " How"),
            TranscriptWord(2.0, 2.5, " are you?"),
        ]
        spans = [
            SpeakerSpan(0.0, 1.05, "SPEAKER_03"),
            SpeakerSpan(1.05, 1.55, "SPEAKER_07"),
            SpeakerSpan(1.55, 3.0, "SPEAKER_03"),
        ]

        turns = build_dialogue(words, spans)

        self.assertEqual(
            [(turn.speaker, turn.text) for turn in turns],
            [
                ("说话人 1", "Hello there."),
                ("说话人 2", "Hi!"),
                ("说话人 1", "How are you?"),
            ],
        )

    def test_writes_dialogue_without_visible_timestamps(self):
        turns = [
            DialogueTurn(0.0, 1.0, "说话人 1", "第一段。"),
            DialogueTurn(1.0, 2.0, "说话人 2", "第二段。"),
        ]

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            paths = write_dialogue_outputs(turns, output_dir, "sample")
            text = (output_dir / "sample.dialogue.txt").read_text(encoding="utf-8")
            markdown = (output_dir / "sample.dialogue.md").read_text(
                encoding="utf-8"
            )

            self.assertEqual(len(paths), 3)
            self.assertEqual(
                text,
                "说话人 1：\n\n第一段。\n\n说话人 2：\n\n第二段。\n",
            )
            self.assertIn("## 说话人 1\n\n第一段。", markdown)
            self.assertNotIn("00:00", text)


if __name__ == "__main__":
    unittest.main()
