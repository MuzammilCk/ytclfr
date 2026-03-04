import pytest
from unittest.mock import MagicMock
from services.integration.tmdb_service import TMDbService, MovieInfo, StreamingAvailability

@pytest.mark.asyncio
async def test_enrich_list_items_tv_fallback(mocker):
    service = TMDbService()
    
    # Mock search_movie to return None
    mock_search_movie = mocker.patch.object(service, "search_movie", return_value=None)
    
    # Mock search_tv_show to return a TV show
    fake_tv_info = MovieInfo(
        tmdb_id=123,
        title="Fake TV Show",
        original_title="Fake TV Show",
        year="2020",
        vote_average=8.0,
        vote_count=100,
        overview="Overview",
        genres=[],
        poster_url=None,
        backdrop_url=None,
        imdb_id=None,
        imdb_url=None,
        homepage=None,
    )
    mock_search_tv_show = mocker.patch.object(service, "search_tv_show", return_value=fake_tv_info)
    
    # Mock get_streaming
    fake_streaming = StreamingAvailability(flatrate=["Netflix"])
    mock_get_streaming = mocker.patch.object(service, "get_streaming", return_value=fake_streaming)
    
    input_item = {"title": "Fake TV Show", "year": "2020"}
    
    results = await service.enrich_list_items([input_item])
    
    assert len(results) == 1
    assert results[0]["media_type"] == "tv"
    assert results[0]["tmdb_id"] == 123
    assert results[0]["streaming"]["flatrate"] == ["Netflix"]
    
    # Verify the calls were made
    mock_search_movie.assert_called_once_with("Fake TV Show", "2020")
    mock_search_tv_show.assert_called_once_with("Fake TV Show", "2020")
    mock_get_streaming.assert_called_once_with(123, media_type="tv")
