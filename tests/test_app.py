from fastapi.testclient import TestClient

from open_er.server.app import app


def test_root_redirects_to_web():
    client = TestClient(app)
    response = client.get("/", follow_redirects=False)
    assert response.status_code in {302, 307}
    assert response.headers["location"] == "/docs"


def test_admin_redirects_to_docs():
    client = TestClient(app)
    response = client.get("/admin", follow_redirects=False)
    assert response.status_code in {302, 307}
    assert response.headers["location"] == "/docs"
