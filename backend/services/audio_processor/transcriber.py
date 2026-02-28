"""
services/audio_processor/transcriber.py

Audio transcription using faster-whisper (CTranslate2 backend).

faster-whisper is a drop-in replacement for openai-whisper that:
  - Has proper Python 3.13 wheels (no C build required)
  - Is 2–4x faster on CPU thanks to CTranslate2 kernel optimisations
  - Supports word-level timestamps and language detection

Optional: pass use_source_separation=True to isolate vocals via Spleeter
(requires spleeter to be installed separately — it is commented-out in requirements.txt).
"""
import asyncio
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from faster_whisper import WhisperModel
from loguru import logger

from core.config import get_settings

settings = get_settings()


@dataclass
class Segment:
    start: float
    end: float
    text: str
    avg_logprob: float = 0.0       # confidence proxy
    no_speech_prob: float = 0.0    # higher = silent / noise


@dataclass
class TranscriptionResult:
    full_text: str
    language: str
    language_probability: float
    segments: List[Segment]
    word_count: int = field(init=False)

    def __post_init__(self):
        self.word_count = len(self.full_text.split())


# Module-level model cache (single load per process)
_whisper_model: Optional[WhisperModel] = None


def _load_model() -> WhisperModel:
    """Load and cache the WhisperModel singleton."""
    global _whisper_model
    if _whisper_model is None:
        device = settings.WHISPER_DEVICE   # "cpu" or "cuda"
        compute_type = "int8" if device == "cpu" else "float16"
        logger.info(
            f"Loading faster-whisper model '{settings.WHISPER_MODEL_SIZE}' "
            f"on {device} (compute_type={compute_type})"
        )
        _whisper_model = WhisperModel(
            settings.WHISPER_MODEL_SIZE,
            device=device,
            compute_type=compute_type,
            download_root=os.getenv("WHISPER_CACHE_DIR", None),
        )
    return _whisper_model


class AudioTranscriber:
    """
    Wraps faster-whisper for robust speech-to-text transcription.

    Workflow
    ────────
    1. Optionally run Spleeter source-separation if background music is loud.
    2. Pass the (vocals-only or original) WAV to faster-whisper.
    3. Return structured segments with timestamps.

    faster-whisper API differences from openai-whisper
    ───────────────────────────────────────────────────
    - model.transcribe() returns (segments_generator, TranscriptionInfo)
    - Segments are Segment objects (not dicts)
    - Language detection is returned in TranscriptionInfo, not a separate call
    """

    def __init__(self):
        self.audio_dir = Path(settings.AUDIO_DIR)

    # ── Public API ────────────────────────────────────────────────────────────

    async def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        use_source_separation: bool = False,
    ) -> TranscriptionResult:
        """
        Main entry-point. Offloads CPU-intensive work to a thread-pool.

        Args:
            audio_path:             Path to 16kHz mono WAV file.
            language:               BCP-47 code ('en', 'hi', …) or None for auto-detect.
            use_source_separation:  Separate vocals before transcribing (slower but
                                    more accurate for music-heavy content).
        """
        effective_path = audio_path
        if use_source_separation:
            try:
                effective_path = await asyncio.get_running_loop().run_in_executor(
                    None, self._run_source_separation, audio_path
                )
            except Exception as exc:
                logger.warning(f"Source separation failed (will use raw audio): {exc}")

        return await asyncio.get_running_loop().run_in_executor(
            None, self._transcribe_sync, effective_path, language
        )

    # ── Internal ──────────────────────────────────────────────────────────────

    def _transcribe_sync(
        self,
        audio_path: str,
        language: Optional[str],
    ) -> TranscriptionResult:
        # Prefer the model pre-loaded by the Celery worker_process_init signal;
        # fall back to the local lazy-load singleton if running outside Celery.
        try:
            from services.pipeline import _models  # noqa: PLC0415
            model = _models.get("whisper")
        except Exception:
            model = None

        if model is None:
            model = _load_model()

        # Duration warning for very long audio
        try:
            import wave, contextlib
            with contextlib.closing(wave.open(audio_path, 'r')) as wf:
                duration_secs = wf.getnframes() / wf.getframerate()
                if duration_secs > 1800:
                    logger.warning(
                        f"Audio > 30 min ({duration_secs/60:.1f} min) — transcription may be slow"
                    )
        except Exception:
            pass  # non-fatal; just skip the duration check

        logger.info(f"Transcribing: {Path(audio_path).name} (language={language or 'auto'})")

        # faster-whisper returns a generator + info object
        segments_gen, info = model.transcribe(
            audio_path,
            language=language,          # None → auto-detect
            task="transcribe",
            word_timestamps=True,
            beam_size=5,
            best_of=5,
            temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
            condition_on_previous_text=True,
            vad_filter=True,            # skip silent regions (replaces no_speech check)
        )

        logger.info(
            f"Detected language: {info.language} "
            f"({info.language_probability:.2%})"
        )

        # Consume generator — each item is a faster_whisper.Segment
        segments: List[Segment] = []
        for s in segments_gen:
            if s.no_speech_prob < 0.8:   # filter silent/noise segments
                segments.append(Segment(
                    start=s.start,
                    end=s.end,
                    text=s.text.strip(),
                    avg_logprob=s.avg_logprob,
                    no_speech_prob=s.no_speech_prob,
                ))

        full_text = " ".join(seg.text for seg in segments).strip()

        return TranscriptionResult(
            full_text=full_text,
            language=info.language,
            language_probability=info.language_probability,
            segments=segments,
        )

    def _run_source_separation(self, audio_path: str) -> str:
        """
        Use Spleeter 2-stem model to isolate vocals.
        Returns path to vocals WAV.
        Requires: pip install spleeter  (not in requirements.txt — optional heavy dep)
        """
        from spleeter.separator import Separator       # noqa: PLC0415
        from spleeter.audio.adapter import AudioAdapter  # noqa: PLC0415

        separator = Separator("spleeter:2stems")
        audio_loader = AudioAdapter.default()

        out_dir = Path(audio_path).parent / "separated"
        separator.separate_to_file(
            audio_path,
            str(out_dir),
            codec="wav",
            synchronous=True,
        )

        stem_name = Path(audio_path).stem
        vocals_path = out_dir / stem_name / "vocals.wav"
        if not vocals_path.exists():
            raise FileNotFoundError(f"Spleeter did not produce vocals at {vocals_path}")

        logger.info(f"Source separation complete → {vocals_path}")
        return str(vocals_path)
