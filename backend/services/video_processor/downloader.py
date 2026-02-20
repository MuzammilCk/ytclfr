"""
services/video_processor/downloader.py

Downloads YouTube videos using yt-dlp, validates constraints,
extracts rich metadata, and returns structured output.
"""
import asyncio
import hashlib
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yt_dlp
from loguru import logger

from core.config import get_settings

settings = get_settings()

# Compiled regex patterns for YouTube ID extraction
_YT_PATTERNS = [
    re.compile(r"youtu\.be/([a-zA-Z0-9_-]{11})"),
    re.compile(r"youtube\.com/watch\?.*v=([a-zA-Z0-9_-]{11})"),
    re.compile(r"youtube\.com/shorts/([a-zA-Z0-9_-]{11})"),
    re.compile(r"youtube\.com/embed/([a-zA-Z0-9_-]{11})"),
]


@dataclass
class DownloadResult:
    video_id: str
    video_path: str
    audio_path: str
    metadata: Dict[str, Any]
    thumbnail_path: Optional[str] = None


def extract_video_id(url: str) -> Optional[str]:
    """Return the 11-character YouTube video ID from any valid URL form."""
    for pattern in _YT_PATTERNS:
        m = pattern.search(url)
        if m:
            return m.group(1)
    return None


class VideoDownloader:
    """
    Wraps yt-dlp to download video + audio tracks into configured directories.
    Enforces duration limits and cleans up temp artefacts automatically.
    """

    def __init__(self):
        self.download_dir = Path(settings.DOWNLOAD_DIR)
        self.audio_dir = Path(settings.AUDIO_DIR)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    async def download(self, url: str) -> DownloadResult:
        """
        Download video & separate audio track; return paths + metadata.
        Runs yt-dlp in a thread-pool executor so it doesn't block the
        asyncio event loop.
        """
        video_id = extract_video_id(url)
        if not video_id:
            raise ValueError(f"Cannot extract YouTube video ID from: {url!r}")

        # Quick pre-flight metadata check (no download yet)
        meta = await asyncio.get_running_loop().run_in_executor(
            None, self._fetch_metadata, url
        )

        duration = meta.get("duration", 0) or 0
        if duration > settings.MAX_VIDEO_DURATION_SECS:
            raise ValueError(
                f"Video duration {duration}s exceeds max allowed "
                f"{settings.MAX_VIDEO_DURATION_SECS}s"
            )

        # Download video (mp4)
        video_path = await asyncio.get_running_loop().run_in_executor(
            None, self._download_video, url, video_id
        )

        # Separate audio track (wav, 16kHz mono — Whisper optimal format)
        audio_path = await asyncio.get_running_loop().run_in_executor(
            None, self._extract_audio, video_path, video_id
        )

        # Download thumbnail
        thumbnail_path = await asyncio.get_running_loop().run_in_executor(
            None, self._download_thumbnail, url, video_id
        )

        return DownloadResult(
            video_id=video_id,
            video_path=str(video_path),
            audio_path=str(audio_path),
            thumbnail_path=thumbnail_path,
            metadata=self._normalise_metadata(meta),
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _ydl_common_opts(self) -> Dict[str, Any]:
        """Shared yt-dlp options."""
        return {
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
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

    def _fetch_metadata(self, url: str) -> Dict[str, Any]:
        """Fetch video info dict without downloading media."""
        opts = {**self._ydl_common_opts(), "skip_download": True}
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
        return info or {}

    def _download_video(self, url: str, video_id: str) -> Path:
        out_path = self.download_dir / f"{video_id}.mp4"
        if out_path.exists():
            logger.info(f"Video {video_id} already cached, skipping download")
            return out_path

        opts = {
            **self._ydl_common_opts(),
            # Prefer best single-file mp4 ≤1080p; fall back to best available
            "format": (
                "bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]"
                "/best[ext=mp4][height<=1080]/best"
            ),
            "outtmpl": str(self.download_dir / f"{video_id}.%(ext)s"),
            "merge_output_format": "mp4",
            "postprocessors": [
                {
                    "key": "FFmpegVideoConvertor",
                    "preferedformat": "mp4",
                }
            ],
        }

        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])

        if not out_path.exists():
            # Fallback: yt-dlp may have produced a differently named file
            candidates = list(self.download_dir.glob(f"{video_id}.*"))
            if not candidates:
                raise FileNotFoundError(f"yt-dlp did not produce a file for {video_id}")
            out_path = candidates[0]

        logger.info(f"Downloaded {video_id} → {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")
        return out_path

    def _extract_audio(self, video_path: str | Path, video_id: str) -> Path:
        """
        Use FFmpeg to extract a 16kHz mono WAV — the format Whisper
        expects and provides best WER performance with.
        """
        audio_path = self.audio_dir / f"{video_id}.wav"
        if audio_path.exists():
            return audio_path

        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vn",                   # no video
            "-acodec", "pcm_s16le", # 16-bit PCM
            "-ar", "16000",          # 16 kHz
            "-ac", "1",              # mono
            str(audio_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg audio extraction failed:\n{result.stderr}")

        logger.info(f"Audio extracted → {audio_path}")
        return audio_path

    def _download_thumbnail(self, url: str, video_id: str) -> Optional[str]:
        thumb_path = self.download_dir / f"{video_id}_thumb.jpg"
        if thumb_path.exists():
            return str(thumb_path)
        try:
            opts = {
                **self._ydl_common_opts(),
                "skip_download": True,
                "writethumbnail": True,
                "outtmpl": str(self.download_dir / f"{video_id}_thumb.%(ext)s"),
                "postprocessors": [{"key": "FFmpegThumbnailsConvertor", "format": "jpg"}],
            }
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([url])
        except Exception as exc:
            logger.warning(f"Thumbnail download failed (non-fatal): {exc}")
            return None

        candidates = list(self.download_dir.glob(f"{video_id}_thumb.*"))
        return str(candidates[0]) if candidates else None

    @staticmethod
    def _normalise_metadata(info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert raw yt-dlp info dict to our clean metadata schema.
        Only keeps fields we actually use downstream.
        """
        return {
            "youtube_id": info.get("id", ""),
            "title": info.get("title", ""),
            "channel_name": info.get("uploader") or info.get("channel", ""),
            "duration_secs": info.get("duration"),
            "view_count": info.get("view_count"),
            "like_count": info.get("like_count"),
            "description": (info.get("description") or "")[:2000],
            "tags": info.get("tags", [])[:50],
            "thumbnail_url": info.get("thumbnail", ""),
            "upload_date": info.get("upload_date", ""),   # YYYYMMDD string
            "categories": info.get("categories", []),
            "subtitles_available": bool(info.get("subtitles")),
            "is_live": bool(info.get("is_live")),
        }

    async def cleanup(self, video_id: str):
        """Remove downloaded media files for a given video ID."""
        for pattern in [
            f"{video_id}.mp4",
            f"{video_id}.wav",
            f"{video_id}_thumb.*",
        ]:
            for f in list(self.download_dir.glob(pattern)) + list(self.audio_dir.glob(pattern)):
                try:
                    f.unlink()
                    logger.debug(f"Cleaned up {f}")
                except OSError as e:
                    logger.warning(f"Cleanup failed for {f}: {e}")
