"""
services/extraction/llm_extractor.py

CPU-optimised local LLM extraction using `llama.cpp` and `Phi-3-mini`.
Replaces fragile heuristic regex parsing by prompting a local small language model
to return structured JSON.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

from loguru import logger

# Try import gracefully so we don't crash if llama-cpp isn't installed
try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False


class LlmExtractor:
    """
    Singleton wrapper around a llama.cpp instance running Phi-3-mini.
    """
    _instance: LlmExtractor | None = None
    _llm: Any = None
    _model_path: Path = Path("models/Phi-3-mini-4k-instruct-q4.gguf")

    def __new__(cls) -> LlmExtractor:
        if cls._instance is None:
            cls._instance = super(LlmExtractor, cls).__new__(cls)
            cls._instance._init_llm()
        return cls._instance

    def _init_llm(self) -> None:
        if not LLAMA_AVAILABLE:
            logger.warning("llama-cpp-python not installed. LlmExtractor disabled.")
            return

        if not self._model_path.exists():
            logger.warning(f"Model file not found at {self._model_path}. LlmExtractor disabled.")
            return

        try:
            logger.info("ytclfr-extract: Loading Phi-3-mini into RAM via llama.cpp...")
            # n_ctx=4096 is enough for transcript + metadata
            self._llm = Llama(
                model_path=str(self._model_path),
                n_ctx=4096,
                n_threads=None,  # auto-detect CPU cores
                n_gpu_layers=0,  # run strictly on CPU memory
                verbose=False    # suppress C++ spam
            )
            logger.info("ytclfr-extract: Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load llama.cpp model: {e}")
            self._llm = None

    def is_available(self) -> bool:
        return self._llm is not None

    def extract(
        self, category: str, transcript: str, metadata: dict, ocr_text: str
    ) -> Dict[str, Any]:
        """
        Builds a category-specific prompt, runs generation, and parses the JSON response.
        """
        if not self.is_available():
            raise RuntimeError("LlmExtractor is not available (check logs)")

        prompt = self._build_prompt(category, transcript, metadata, ocr_text)
        
        try:
            response = self._llm(
                prompt,
                max_tokens=1024,
                stop=["<|end|>", "}"],
                temperature=0.1,    # We heavily penalize creativity for data extraction
                echo=False
            )
            
            raw_text = response["choices"][0]["text"].strip()
            
            # The model might omit the closing brace because we use '}' as a stop token
            # to prevent rambling. We append it back.
            if not raw_text.endswith("}"):
                raw_text += "}"
                
            # Clean up potential markdown formatting blocks the model might wrap the JSON in
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]

            parsed = json.loads(raw_text)
            
            # Enforce the baseline wrapper around the response
            return {"type": category, **parsed}
            
        except json.JSONDecodeError as jde:
            logger.error(f"LlmExtractor JSON parse failed: {jde}. Raw output: {raw_text}")
            return {"type": "error", "message": "Failed to parse LLM output"}
        except Exception as e:
            logger.error(f"LlmExtractor inference failed: {e}")
            return {"type": "error", "message": str(e)}

    def _build_prompt(self, category: str, transcript: str, metadata: dict, ocr_text: str) -> str:
        """
        Constructs a Phi-3-instruct formatted prompt asking for strictly verifiable JSON fields.
        """
        # Truncate inputs so we don't blow the 4096 context window (1 token ~= 4 chars)
        desc = (metadata.get("description") or "")[:2000]
        tx = transcript[:6000] 
        ocr = ocr_text[:1000]
        
        title = metadata.get("title", "")

        # Define category specific instructions
        if category == "music":
            instructions = (
                "Extract a list of all songs/tracks mentioned in the video. "
                "Output ONLY valid JSON containing a single key 'tracks', which is a list of objects "
                "with keys 'rank' (integer or null), 'title' (string), 'artist' (string)."
            )
        elif category == "listicle":
            instructions = (
                "Extract the ranked list of movies, games, or products discussed. "
                "Output ONLY valid JSON containing a single key 'items', which is a list of objects "
                "with keys 'rank' (integer), 'title' (string)."
            )
        elif category == "shopping":
            instructions = (
                "Extract any physical products, brands, or items being recommended or reviewed. "
                "Output ONLY valid JSON containing a single key 'products', which is a list of objects "
                "with keys 'name' (string), 'brand' (string or null)."
            )
        else:
            instructions = (
                "Extract the top 5 key concepts discussed in this video. "
                "Output ONLY valid JSON containing a single key 'key_concepts', which is a list of strings."
            )

        # Phi-3 instruct format
        return (
            f"<|system|>\n"
            f"You are an expert data extraction bot. Your job is to extract structured data from video transcripts and metadata. "
            f"You must strictly obey JSON formats. You NEVER output conversational text. You ONLY output JSON.\n<|end|>\n"
            f"<|user|>\n"
            f"Video Title: {title}\n\n"
            f"Video Description: {desc}\n\n"
            f"OCR Text: {ocr}\n\n"
            f"Transcript: {tx}\n\n"
            f"INSTRUCTIONS: {instructions}\n<|end|>\n"
            f"<|assistant|>\n"
            f"{{"  # Prime the model to start outputting the JSON dictionary immediately
        )
