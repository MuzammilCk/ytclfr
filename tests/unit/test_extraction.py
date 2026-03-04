"""
tests/unit/test_extraction.py

Unit tests for information extraction modules.
Run with: pytest tests/unit/ -v --tb=short
"""
import sys
import pytest
from services.extraction.extractors import (
    _extract_ranked_list,
    _parse_music_entries,
    _parse_chapters_from_description,
    _extract_key_phrases,
    ListicleExtractor,
    MusicExtractor,
    EducationalExtractor,
)

# ── Skip guards for tests that require optional ML dependencies ─────────────
def _torchvision_available():
    try:
        import torchvision  # noqa: F401
        return True
    except ImportError:
        return False

def _spacy_model_available():
    try:
        import spacy
        spacy.load("en_core_web_sm")
        return True
    except (ImportError, OSError):
        return False

requires_torchvision = pytest.mark.skipif(
    not _torchvision_available(),
    reason="torchvision not installed (ML stack not set up)",
)
requires_spacy_model = pytest.mark.skipif(
    not _spacy_model_available(),
    reason="spaCy en_core_web_sm model not downloaded — run: python -m spacy download en_core_web_sm",
)


# ── Ranked list extraction ─────────────────────────────────────────────────────

class TestRankedListExtraction:
    def test_numbered_dot_format(self):
        text = "1. The Shawshank Redemption\n2. The Godfather\n3. The Dark Knight"
        result = _extract_ranked_list(text)
        assert len(result) == 3
        assert result[0] == (1, "The Shawshank Redemption")
        assert result[2] == (3, "The Dark Knight")

    def test_hash_format(self):
        text = "#1 Avatar\n#2 Avengers\n#3 Titanic"
        result = _extract_ranked_list(text)
        assert len(result) == 3
        assert result[0][1] == "Avatar"

    def test_mixed_format(self):
        text = "Number 1: Inception\n2 - Interstellar\n3. Tenet"
        result = _extract_ranked_list(text)
        assert len(result) >= 2

    def test_empty_text(self):
        assert _extract_ranked_list("") == []

    def test_skips_urls(self):
        text = "1. https://youtube.com/something\n2. Real Movie Title"
        result = _extract_ranked_list(text)
        assert all(not item[1].startswith("http") for item in result)


# ── Music parsing ──────────────────────────────────────────────────────────────

class TestMusicParsing:
    def test_hyphen_separator(self):
        text = "Bohemian Rhapsody - Queen\nHotel California - Eagles"
        tracks = _parse_music_entries(text)
        assert len(tracks) == 2
        assert tracks[0]["title"] == "Bohemian Rhapsody"
        assert tracks[0]["artist"] == "Queen"

    def test_dash_with_year(self):
        text = "Thriller - Michael Jackson - 1982\nBillie Jean - Michael Jackson - 1982"
        tracks = _parse_music_entries(text)
        assert len(tracks) >= 1

    def test_ft_separator(self):
        text = "Sunflower ft. Swae Lee - Post Malone"
        tracks = _parse_music_entries(text)
        assert len(tracks) >= 1

    def test_empty(self):
        assert _parse_music_entries("") == []


# ── Chapter parsing ────────────────────────────────────────────────────────────

class TestChapterParsing:
    def test_standard_format(self):
        desc = "0:00 Introduction\n1:30 Setup\n5:00 Main Content\n15:00 Conclusion"
        chapters = _parse_chapters_from_description(desc)
        assert len(chapters) == 4
        assert chapters[0]["title"] == "Introduction"
        assert chapters[0]["start_secs"] == 0
        assert chapters[1]["start_secs"] == 90

    def test_hour_format(self):
        desc = "1:00:00 Advanced Topics\n1:30:00 Summary"
        chapters = _parse_chapters_from_description(desc)
        assert chapters[0]["start_secs"] == 3600
        assert chapters[1]["start_secs"] == 5400

    def test_end_times_filled(self):
        desc = "0:00 Intro\n5:00 Main\n10:00 End"
        chapters = _parse_chapters_from_description(desc)
        assert chapters[0]["end_secs"] == chapters[1]["start_secs"]
        assert chapters[-1]["end_secs"] is None

    def test_no_chapters(self):
        desc = "This video is about Python programming"
        chapters = _parse_chapters_from_description(desc)
        assert chapters == []


# ── Extractor integration tests ────────────────────────────────────────────────

class TestListicleExtractor:
    def setup_method(self):
        self.extractor = ListicleExtractor()

    def test_extracts_items(self):
        transcript = "Today we look at the top 5 movies of all time.\n1. The Godfather\n2. Pulp Fiction\n3. Schindler's List\n4. The Dark Knight\n5. 12 Angry Men"
        metadata = {
            "title": "Top 5 Movies of All Time",
            "description": "1. The Godfather\n2. Pulp Fiction\n3. Schindler's List\n4. The Dark Knight\n5. 12 Angry Men",
            "tags": ["movies", "ranking"],
        }
        result = self.extractor.extract(transcript, [], metadata, [], [])
        assert result["type"] == "listicle"
        assert result["total_count"] >= 3
        assert all("title" in item for item in result["items"])
        assert all("rank" in item for item in result["items"])

    def test_handles_empty_transcript(self):
        result = self.extractor.extract("", [], {"title": "", "description": "", "tags": []}, [], [])
        assert result["type"] == "listicle"


class TestMusicExtractor:
    def setup_method(self):
        self.extractor = MusicExtractor()

    def test_extracts_tracks(self):
        description = "Bohemian Rhapsody - Queen\nStairway to Heaven - Led Zeppelin\nHotel California - Eagles"
        metadata = {"title": "Best Rock Songs Ever", "description": description, "tags": []}
        result = self.extractor.extract("", [], metadata, [], [])
        assert result["type"] == "music"
        assert len(result["tracks"]) >= 2

    def test_track_structure(self):
        metadata = {"title": "Playlist", "description": "Song A - Artist B\nSong C - Artist D", "tags": []}
        result = self.extractor.extract("", [], metadata, [], [])
        for track in result["tracks"]:
            assert "title" in track
            assert "artist" in track
            assert "spotify" in track


class TestEducationalExtractor:
    def setup_method(self):
        self.extractor = EducationalExtractor()

    @requires_spacy_model
    def test_extracts_chapters(self):
        metadata = {
            "title": "Python Tutorial",
            "description": "0:00 Introduction\n5:00 Variables\n10:00 Functions\n20:00 Classes",
            "tags": [],
        }
        segments = [
            {"start": 0, "end": 5, "text": "Welcome to this Python tutorial.", "no_speech_prob": 0.0},
            {"start": 5, "end": 10, "text": "Let's look at variables.", "no_speech_prob": 0.0},
        ]
        result = self.extractor.extract("Python tutorial content", segments, metadata, [], [])
        assert result["type"] == "educational"
        assert len(result["chapters"]) >= 1

    @requires_spacy_model
    def test_auto_segment_fallback(self):
        metadata = {"title": "Lecture", "description": "No chapters here.", "tags": []}
        segments = [
            {"start": i * 60, "end": (i + 1) * 60, "text": f"Segment {i}", "no_speech_prob": 0.0}
            for i in range(20)
        ]
        result = self.extractor.extract("Full lecture content.", segments, metadata, [], [])
        assert result["type"] == "educational"
        assert len(result["chapters"]) >= 1


# ── Heuristic classifier tests ─────────────────────────────────────────────────

class TestHeuristicClassifier:
    @requires_torchvision
    def test_listicle_detection(self):
        from services.classification.classifier import MultiModalClassifier
        # Test just the heuristic component to avoid loading ML models in CI
        scores = MultiModalClassifier.classify_heuristic(
            title="Top 10 Best Movies of 2024",
            description="Ranking the best films this year",
            tags=["movies", "ranking", "top10"],
        )
        from services.classification.classifier import CATEGORIES
        listicle_idx = CATEGORIES.index("listicle")
        assert scores[listicle_idx] == scores.max(), "Should predict listicle"

    @requires_torchvision
    def test_music_detection(self):
        from services.classification.classifier import MultiModalClassifier, CATEGORIES
        scores = MultiModalClassifier.classify_heuristic(
            title="Best 80s Hits Playlist Mix",
            description="80s music hits songs album playlist",
            tags=["music", "80s", "playlist"],
        )
        music_idx = CATEGORIES.index("music")
        assert scores[music_idx] == scores.max()

    @requires_torchvision
    def test_gaming_detection(self):
        from services.classification.classifier import MultiModalClassifier, CATEGORIES
        scores = MultiModalClassifier.classify_heuristic(
            title="Minecraft Speedrun Gameplay Walkthrough",
            description="speedrun world record gameplay esports",
            tags=["gaming", "minecraft"],
        )
        gaming_idx = CATEGORIES.index("gaming")
        assert scores[gaming_idx] == scores.max()
