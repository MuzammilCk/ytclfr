"""
test_download.py
----------------
Standalone test to download a YouTube video from a URL in the terminal.
Mirrors the logic in services/video_processor/downloader.py with NO app imports.

Usage:
    python test_download.py <youtube_url> [--out ./test_downloads] [--audio] [--thumbnail]

Requirements:
    pip install yt-dlp
    ffmpeg must be on PATH (for audio extraction and mp4 merging)
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

try:
    import yt_dlp
except ImportError:
    print("ERROR: yt-dlp is not installed. Run: pip install yt-dlp")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# YouTube URL / ID helpers
# ─────────────────────────────────────────────────────────────────────────────
_YT_PATTERNS = [
    re.compile(r"youtu\.be/([a-zA-Z0-9_-]{11})"),
    re.compile(r"youtube\.com/watch\?.*v=([a-zA-Z0-9_-]{11})"),
    re.compile(r"youtube\.com/shorts/([a-zA-Z0-9_-]{11})"),
    re.compile(r"youtube\.com/embed/([a-zA-Z0-9_-]{11})"),
]


def extract_video_id(url: str) -> Optional[str]:
    for pattern in _YT_PATTERNS:
        m = pattern.search(url)
        if m:
            return m.group(1)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Common yt-dlp options (matches downloader.py)
# ─────────────────────────────────────────────────────────────────────────────
def _common_opts() -> dict:
    return {
        "quiet": False,          # show progress in terminal
        "no_warnings": False,
        "noprogress": False,
        "socket_timeout": 30,
        "retries": 3,
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            )
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 – Fetch metadata (no download)
# ─────────────────────────────────────────────────────────────────────────────
def fetch_metadata(url: str) -> dict:
    print("[metadata] Fetching video info...")
    opts = {**_common_opts(), "skip_download": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    return info or {}


def print_metadata(meta: dict):
    fields = {
        "Title"       : meta.get("title", "N/A"),
        "Channel"     : meta.get("uploader") or meta.get("channel", "N/A"),
        "Duration"    : f"{meta.get('duration', 0)} seconds",
        "Views"       : meta.get("view_count", "N/A"),
        "Upload date" : meta.get("upload_date", "N/A"),
        "Tags"        : ", ".join((meta.get("tags") or [])[:10]) or "None",
        "Live"        : meta.get("is_live", False),
    }
    print("\n─── Video Metadata ─────────────────────────────────")
    for k, v in fields.items():
        print(f"  {k:<14}: {v}")
    print("────────────────────────────────────────────────────\n")


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 – Download video (mp4)
# ─────────────────────────────────────────────────────────────────────────────
def download_video(url: str, video_id: str, out_dir: Path) -> Path:
    out_path = out_dir / f"{video_id}.mp4"
    if out_path.exists():
        print(f"[video] Already downloaded: {out_path}")
        return out_path

    print(f"[video] Downloading → {out_path}")
    opts = {
        **_common_opts(),
        # Same format fallback chain as downloader.py
        "format": (
            "bestvideo[vcodec*=avc1][ext=mp4][height<=1080]+bestaudio[ext=m4a]"
            "/bestvideo[vcodec!=av01][ext=mp4][height<=1080]+bestaudio[ext=m4a]"
            "/bestvideo[vcodec!=av01][height<=1080]+bestaudio"
            "/bestvideo[vcodec!=av01]+bestaudio"
            "/bestvideo+bestaudio"
            "/best[height<=1080]"
            "/best"
        ),
        "outtmpl": str(out_dir / f"{video_id}.%(ext)s"),
        "merge_output_format": "mp4",
        "postprocessors": [
            {"key": "FFmpegVideoConvertor", "preferedformat": "mp4"}
        ],
    }

    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])

    if not out_path.exists():
        candidates = list(out_dir.glob(f"{video_id}.*"))
        if not candidates:
            raise FileNotFoundError(f"yt-dlp did not produce a file for {video_id}")
        out_path = candidates[0]

    size_mb = out_path.stat().st_size / 1e6
    print(f"[video] Downloaded → {out_path} ({size_mb:.1f} MB)")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 – Extract audio (WAV 16kHz mono, for Whisper)
# ─────────────────────────────────────────────────────────────────────────────
def extract_audio(video_path: Path, out_dir: Path, video_id: str) -> Path:
    audio_path = out_dir / f"{video_id}.wav"
    if audio_path.exists():
        print(f"[audio] Already exists: {audio_path}")
        return audio_path

    print(f"[audio] Extracting 16kHz mono WAV → {audio_path}")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(audio_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[audio] ERROR: FFmpeg failed:\n{result.stderr}")
        return None

    print(f"[audio] Done → {audio_path}")
    return audio_path


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 – Download thumbnail
# ─────────────────────────────────────────────────────────────────────────────
def download_thumbnail(url: str, video_id: str, out_dir: Path) -> Optional[Path]:
    print(f"[thumbnail] Downloading thumbnail...")
    try:
        opts = {
            **_common_opts(),
            "skip_download": True,
            "writethumbnail": True,
            "outtmpl": str(out_dir / f"{video_id}_thumb.%(ext)s"),
            "postprocessors": [{"key": "FFmpegThumbnailsConvertor", "format": "jpg"}],
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
    except Exception as exc:
        print(f"[thumbnail] WARNING: Failed (non-fatal) — {exc}")
        return None

    candidates = list(out_dir.glob(f"{video_id}_thumb.*"))
    if candidates:
        print(f"[thumbnail] Saved → {candidates[0]}")
        return candidates[0]
    return None


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Download a YouTube video from a URL (mirrors downloader.py)"
    )
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("--out", default="./test_downloads", help="Output directory (default: ./test_downloads)")
    parser.add_argument("--audio", action="store_true", help="Also extract audio as 16kHz mono WAV (requires ffmpeg)")
    parser.add_argument("--thumbnail", action="store_true", help="Also download thumbnail as JPG")
    parser.add_argument("--meta-only", action="store_true", help="Only fetch and print metadata, do not download")
    args = parser.parse_args()

    url = args.url
    if not re.match(r"https?://(www\.)?(youtube\.com|youtu\.be)/", url):
        print(f"ERROR: Not a valid YouTube URL: {url!r}")
        sys.exit(1)

    video_id = extract_video_id(url)
    if not video_id:
        print(f"ERROR: Could not extract video ID from URL: {url!r}")
        sys.exit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nURL      : {url}")
    print(f"Video ID : {video_id}")
    print(f"Out dir  : {out_dir.resolve()}\n")

    # ── Metadata ──────────────────────────────────────────────────────────────
    meta = fetch_metadata(url)
    print_metadata(meta)

    if args.meta_only:
        print("[--meta-only] Skipping download.")
        return

    # ── Video ─────────────────────────────────────────────────────────────────
    video_path = download_video(url, video_id, out_dir)

    # ── Audio ─────────────────────────────────────────────────────────────────
    if args.audio:
        extract_audio(video_path, out_dir, video_id)

    # ── Thumbnail ─────────────────────────────────────────────────────────────
    if args.thumbnail:
        download_thumbnail(url, video_id, out_dir)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n=== Download Complete ===")
    for f in sorted(out_dir.glob(f"{video_id}*")):
        print(f"  {f.name:<40} {f.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
