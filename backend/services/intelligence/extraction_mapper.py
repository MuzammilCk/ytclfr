"""
services/intelligence/extraction_mapper.py

Maps BrainResult items into the extraction dict format that downstream
enrichment steps (Spotify, TMDb, etc.) already understand.

Kept in a separate module so it can be tested without loading pipeline.py
(which requires Celery and the full worker environment).
"""
from __future__ import annotations

from typing import Any, Dict

from services.intelligence.llm_brain import BrainResult


def brain_result_to_extraction(brain_result: BrainResult, category: str) -> Dict[str, Any]:
    """
    Convert a BrainResult into the extraction dict format that downstream
    enrichment steps (Spotify, TMDb, etc.) already understand.

    This is the translation layer between the new intelligence layer
    and the existing enrichment pipeline.
    """
    items = brain_result.items or []
    base = {
        "type": category,
        "extraction_source": brain_result.extraction_source,
        "brain_confidence": brain_result.confidence,
        "brain_model": brain_result.model_used,
    }

    if category == "music":
        tracks = []
        for item in items:
            if not isinstance(item, dict):
                continue
            title = item.get("title", "").strip()
            artist = item.get("artist", "").strip()
            if not title:
                continue
            tracks.append({
                "title": title,
                "artist": artist,
                "rank": item.get("rank"),
                "timestamp_secs": item.get("timestamp_secs"),
                "raw_ocr": item.get("raw_ocr", ""),
                "source": brain_result.extraction_source,
            })
        return {**base, "tracks": tracks}

    elif category == "listicle":
        list_items = []
        for item in items:
            if not isinstance(item, dict):
                continue
            title = item.get("title", "").strip()
            if not title:
                continue
            list_items.append({
                "rank": item.get("rank"),
                "title": title,
                "year": item.get("year"),
                "timestamp_secs": item.get("timestamp_secs"),
                "raw_ocr": item.get("raw_ocr", ""),
            })
        return {**base, "items": list_items}

    elif category == "shopping":
        products = []
        for item in items:
            if not isinstance(item, dict):
                continue
            name = item.get("name", "").strip()
            if not name:
                continue
            products.append({
                "name": name,
                "brand": item.get("brand"),
                "price": item.get("price"),
                "category": item.get("category", ""),
                "timestamp_secs": item.get("timestamp_secs"),
            })
        return {**base, "products": products}

    elif category == "recipe":
        recipe_data = items[0] if items and isinstance(items[0], dict) else {}
        return {
            **base,
            "ingredients": recipe_data.get("ingredients", []),
            "steps": recipe_data.get("steps", []),
        }

    elif category == "educational":
        edu_data = items[0] if items and isinstance(items[0], dict) else {}
        return {
            **base,
            "chapters": edu_data.get("chapters", []),
            "key_concepts": edu_data.get("key_concepts", []),
        }

    else:
        # gaming, vlog, news, review, comedy, unknown
        return {**base, "items": items}
