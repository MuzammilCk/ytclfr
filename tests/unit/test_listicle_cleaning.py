import pytest
from services.extraction.extractors import ListicleExtractor

def test_clean_listicle_title():
    extractor = ListicleExtractor()
    clean = extractor._clean_listicle_title

    # Basic
    assert clean("#1 THE SHAWSHANK REDEMPTION (1994)") == ("The Shawshank Redemption", 1994)
    
    # Rank formats
    assert clean("No. 5 Pulp Fiction") == ("Pulp Fiction", None)
    assert clean("10. Inception [2010]") == ("Inception", 2010)
    assert clean("1) The Dark Knight") == ("The Dark Knight", None)
    
    # Noise words
    assert clean("BEST MOVIE EVER The Godfather") == ("The Godfather", None)
    assert clean("TOP MOVIE: Parasite") == ("Parasite", None)
    
    # Combined rules
    assert clean("number 1 MUST WATCH: INTERSTELLAR (2014) HD") == ("Interstellar Hd", 2014) # Parentheses stripped, spaces minimized
    
    # Without rank
    assert clean("Avatar (2009)") == ("Avatar", 2009)

# run with `pytest tests/unit/test_listicle_cleaning.py`
