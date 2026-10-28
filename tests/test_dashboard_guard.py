import pytest

from momentum_rebalancer.dashboard.app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    return app.test_client()


def test_get_status_ok(client):
    r = client.get("/api/status")
    assert r.status_code == 200
    body = r.get_json()
    assert body["regime"] == "baseline" and "warnings" in body


@pytest.mark.parametrize("path", ["/api/rebalance", "/api/refresh"])
def test_post_without_json_is_refused(client, path):
    """A cross-site HTML form can only send urlencoded/multipart/text-plain."""
    r = client.post(path, data="x=1", content_type="application/x-www-form-urlencoded")
    assert r.status_code == 415
    r = client.post(path)
    assert r.status_code == 415


def test_cross_origin_json_post_is_refused(client):
    r = client.post("/api/rebalance", json={}, headers={"Origin": "http://evil.example"})
    assert r.status_code == 403


def test_same_origin_json_post_passes_the_guard(client, monkeypatch):
    from momentum_rebalancer.dashboard import app as appmod
    monkeypatch.setattr(appmod, "_target_weights", lambda: {})   # no cluster data in this test
    r = client.post("/api/rebalance", json={}, headers={"Origin": "http://localhost"},
                    base_url="http://localhost")
    assert r.status_code not in (415, 403)


def test_index_page_renders_and_posts_json(client):
    r = client.get("/")
    assert r.status_code == 200
    html = r.get_data(as_text=True)
    assert html.count("'Content-Type':'application/json'") == 2
