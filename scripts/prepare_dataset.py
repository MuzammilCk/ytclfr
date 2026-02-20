"""
scripts/prepare_dataset.py

Download, transcribe, and extract frames from a labelled video list
to produce training CSVs for both classifiers.

Input CSV format (e.g., data/video_list.csv):
  youtube_id, url, label

Output:
  data/frames_dataset.csv   → frame_path, label
  data/transcripts_dataset.csv → text (title + transcript), label
"""
import argparse
import asyncio
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from services.video_processor.downloader import VideoDownloader
from services.video_processor.frame_extractor import FrameExtractor
from services.audio_processor.transcriber import AudioTranscriber
from core.config import get_settings

settings = get_settings()


async def process_video(
    row: dict,
    downloader: VideoDownloader,
    extractor: FrameExtractor,
    transcriber: AudioTranscriber,
    frames_out: list,
    transcripts_out: list,
):
    url   = row["url"]
    label = row["label"]
    print(f"Processing {url} [{label}]...")

    try:
        result = await downloader.download(url)
        frames = await extractor.extract(result.video_path, result.video_id)
        transcript = await transcriber.transcribe(result.audio_path)

        # Emit frame entries (sample up to 60 frames per video)
        for fp in frames.frame_paths[:60]:
            frames_out.append({"frame_path": fp, "label": label})

        # Emit transcript entry
        title = result.metadata.get("title", "")
        text  = f"{title}. {transcript.full_text[:1500]}"
        transcripts_out.append({"text": text, "label": label})

        print(f"  ✓ {len(frames.frame_paths)} frames, {transcript.word_count} words")
    except Exception as exc:
        print(f"  ✗ Failed: {exc}")


async def main(args):
    input_csv   = args.input
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(input_csv) as f:
        rows = list(csv.DictReader(f))

    print(f"Processing {len(rows)} videos...")

    downloader  = VideoDownloader()
    extractor   = FrameExtractor()
    transcriber = AudioTranscriber()

    frames_out      = []
    transcripts_out = []

    # Process in chunks of 5 to avoid overwhelming disk/memory
    CHUNK = 5
    for i in range(0, len(rows), CHUNK):
        batch = rows[i : i + CHUNK]
        await asyncio.gather(*[
            process_video(row, downloader, extractor, transcriber, frames_out, transcripts_out)
            for row in batch
        ])
        print(f"Progress: {min(i+CHUNK, len(rows))}/{len(rows)}")

    # Write output CSVs
    frames_csv = output_dir / "frames_dataset.csv"
    with open(frames_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_path", "label"])
        writer.writeheader()
        writer.writerows(frames_out)

    transcripts_csv = output_dir / "transcripts_dataset.csv"
    with open(transcripts_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(transcripts_out)

    print(f"\n✓ Done!")
    print(f"  Frame samples:      {len(frames_out)} → {frames_csv}")
    print(f"  Transcript samples: {len(transcripts_out)} → {transcripts_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare training dataset from video list")
    parser.add_argument("--input",      required=True, help="CSV with youtube_id,url,label columns")
    parser.add_argument("--output-dir", default="data/", help="Directory for output CSVs")
    args = parser.parse_args()
    asyncio.run(main(args))
