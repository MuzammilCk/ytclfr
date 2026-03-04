from unittest.mock import MagicMock
import pytest
from services.integration.spotify_service import _clean_for_search, SpotifyService, TrackInfo

def test_clean_for_search():
    assert _clean_for_search("1. Song Title") == "Song Title"
    assert _clean_for_search("#1 - Song Title") == "- Song Title"
    assert _clean_for_search("20. Track Name (2020)") == "Track Name"
    assert _clean_for_search("Song (feat. Artist)") == "Song"
    assert _clean_for_search("Song ft. Artist") == "Song"
    assert _clean_for_search("Artist - Title [1994]") == "Artist - Title"

@pytest.mark.asyncio
async def test_search_track_cascade_logic(mocker):
    # Mock the spotipy client
    mock_client = MagicMock()
    
    # We will simulate the client returning no results for the first 2 queries,
    # and a valid result for the 3rd query (the fuzzy one).
    def mock_search(q, **kwargs):
        if ":" in q or '""' in q.replace(" ", ""):
            # First two queries use `track:` and `artist:` or exact quotes without fuzzy
            return {"tracks": {"items": []}}
        else:
            return {"tracks": {"items": [{
                "id": "123",
                "uri": "spotify:track:123",
                "name": "Found Song",
                "artists": [{"name": "Found Artist"}],
                "album": {"name": "Album", "release_date": "2020"},
                "duration_ms": 200000,
                "popularity": 50,
                "external_urls": {"spotify": "http://spotify.com/123"}
            }]}}
            
    mock_client.search.side_effect = mock_search
    
    service = SpotifyService()
    service._client = mock_client
    
    # The strategies in _search_track_sync:
    # 1. exact track & artist
    # 2. strict quotes
    # 3. quote title only (fuzzy) <- we want it to hit this one
    
    result = service._search_track_sync("Some Title", "Found Artist")
    assert result is not None
    assert result.match_confidence == "fuzzy"
    assert result.spotify_id == "123"

# Note: running this requires `pytest tests/unit/test_spotify.py`
