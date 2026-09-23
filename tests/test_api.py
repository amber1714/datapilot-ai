from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    payload = response.json()
    assert payload["name"] == "DataPilot AI API"
    assert payload["status"] == "running"
