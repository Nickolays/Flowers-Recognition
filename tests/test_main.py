import io
import pytest
from fastapi.testclient import TestClient
from main import app
from PIL import Image

client = TestClient(app)

@pytest.fixture
def dummy_image():
    image = Image.new("RGB", (224, 224), color="green")
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    buf.seek(0)
    return buf

def test_search_endpoint(dummy_image):
    files = {"file": ("test.jpg", dummy_image, "image/jpeg")}
    response = client.post("/search", files=files)
    assert response.status_code == 200
    result = response.json()
    assert "results" in result
    assert len(result["results"]) == 5
    for item in result["results"]:
        assert "image_path" in item
        assert "similarity_score" in item