import json
import tempfile
import unittest
from pathlib import Path

from transcribe import (
    TranscriptSegment,
    format_clock,
    format_timestamp,
    is_url,
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


if __name__ == "__main__":
    unittest.main()
