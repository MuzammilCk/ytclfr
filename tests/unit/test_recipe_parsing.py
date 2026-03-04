import pytest
from services.vision.ocr_service import FrameOCRData
from services.extraction.extractors import RecipeExtractor

def test_extract_ingredients():
    extractor = RecipeExtractor()
    
    # Mock OCR data
    mock_data = [
        FrameOCRData(
            frame_index=0,
            timestamp_secs=1.0,
            frame_path="/tmp/f.jpg",
            raw_text="2 cups flour\n1/2 tsp salt\n1.5 tbsp olive oil\n3 large eggs\nA pinch of black pepper\n- 2 oz shredded cheese\n1 1/2 lbs chicken breast\nSome random text",
            cleaned_text="2 cups flour\n1/2 tsp salt\n1.5 tbsp olive oil\n3 large eggs\nA pinch of black pepper\n- 2 oz shredded cheese\n1 1/2 lbs chicken breast\nSome random text",
            confidence=100.0,
            has_content=True,
            detected_items=[]
        )
    ]
    
    ingredients = extractor._extract_ingredients(mock_data)
    
    assert len(ingredients) == 7
    
    assert ingredients[0] == {"quantity": 2.0, "unit": "cup", "name": "flour"}
    assert ingredients[1] == {"quantity": 0.5, "unit": "tsp", "name": "salt"}
    assert ingredients[2] == {"quantity": 1.5, "unit": "tbsp", "name": "olive oil"}
    assert ingredients[3] == {"quantity": 3.0, "unit": None, "name": "large eggs"}
    assert ingredients[4] == {"quantity": None, "unit": "pinch", "name": "black pepper"}
    assert ingredients[5] == {"quantity": 2.0, "unit": "oz", "name": "shredded cheese"}
    assert ingredients[6] == {"quantity": 1.5, "unit": "lb", "name": "chicken breast"}

# run via `pytest tests/unit/test_recipe_parsing.py`
