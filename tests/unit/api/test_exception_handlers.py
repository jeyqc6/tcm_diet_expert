"""Non-streaming JSON endpoints: structured DietExpertError / internal_error bodies."""
from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app, get_user_profile_fetcher
from backend.exceptions import AuthorizationError, DietExpertError


def test_known_error_returns_structured_400_without_details():
    def boom(**kw):
        raise DietExpertError("internal detail must not leak")

    app.dependency_overrides[get_user_profile_fetcher] = lambda: boom
    try:
        client = TestClient(app)
        resp = client.get("/api/profile")
        assert resp.status_code == 400
        assert resp.json() == {"error": {"type": "diet_expert_error"}}
        assert "must not leak" not in resp.text
    finally:
        app.dependency_overrides.pop(get_user_profile_fetcher, None)


def test_authorization_error_returns_403():
    def boom(**kw):
        raise AuthorizationError("role mismatch")

    app.dependency_overrides[get_user_profile_fetcher] = lambda: boom
    try:
        client = TestClient(app)
        resp = client.get("/api/profile")
        assert resp.status_code == 403
        assert resp.json() == {"error": {"type": "authorization_error"}}
        assert "role mismatch" not in resp.text
    finally:
        app.dependency_overrides.pop(get_user_profile_fetcher, None)


def test_unexpected_error_returns_generic_500():
    def boom(**kw):
        raise RuntimeError("postgresql://user:secret@localhost/db")

    app.dependency_overrides[get_user_profile_fetcher] = lambda: boom
    try:
        # FastAPI installs `@app.exception_handler(Exception)` on
        # ServerErrorMiddleware, which sends the 500 body then re-raises so
        # the server can log it. TestClient defaults to letting that
        # re-raise escape the test.
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/api/profile")
        assert resp.status_code == 500
        assert resp.json() == {"error": {"type": "internal_error"}}
        assert "secret" not in resp.text
        assert "postgresql" not in resp.text
    finally:
        app.dependency_overrides.pop(get_user_profile_fetcher, None)


def test_validation_error_still_returns_422():
    """RequestValidationError must not be swallowed by the Exception handler."""
    client = TestClient(app)
    resp = client.patch("/api/profile", json={})
    assert resp.status_code == 422
    assert resp.json().get("error", {}).get("type") != "internal_error"
