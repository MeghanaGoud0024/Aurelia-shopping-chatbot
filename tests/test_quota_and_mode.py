"""Quota tracking and the manual fallback toggle.

Two properties matter here: the quota numbers shown must be honest about what
is and isn't actually known (Groq exposes per-minute windows live but the
daily ceiling only in a 429's text), and the toggle must actually change which
code path a turn takes.
"""

from __future__ import annotations

import httpx
import pytest

from app.agent.llm import QuotaTracker, _parse_duration
from app.agent.mode import AssistantMode


# ---------------------------------------------------------------- QuotaTracker

def test_fresh_tracker_reports_nothing_known():
    """No calls made yet: every field must say so honestly, not default to 0
    in a way that could be misread as 'zero quota left'."""
    snap = QuotaTracker().snapshot()
    assert snap["per_minute_requests"] is None
    assert snap["per_minute_tokens"] is None
    assert snap["daily"] is None
    assert snap["session_tokens_used"] == 0


def test_records_per_minute_headers():
    tracker = QuotaTracker()
    headers = httpx.Headers({
        "x-ratelimit-limit-requests": "1000",
        "x-ratelimit-remaining-requests": "995",
        "x-ratelimit-reset-requests": "1m26.4s",
        "x-ratelimit-limit-tokens": "8000",
        "x-ratelimit-remaining-tokens": "7646",
        "x-ratelimit-reset-tokens": "2.655s",
    })
    tracker.record_headers(headers)
    snap = tracker.snapshot()

    assert snap["per_minute_requests"]["limit"] == 1000
    assert snap["per_minute_requests"]["remaining"] == 995
    assert snap["per_minute_requests"]["reset_in_seconds"] == pytest.approx(86.4)
    assert snap["per_minute_tokens"]["remaining"] == 7646
    assert snap["per_minute_tokens"]["reset_in_seconds"] == pytest.approx(2.7, abs=0.1)


def test_missing_headers_leave_the_snapshot_unchanged():
    """A response with no rate-limit headers (e.g. a network-level failure
    path) must not clobber the last good reading with nulls."""
    tracker = QuotaTracker()
    tracker.record_headers(httpx.Headers({
        "x-ratelimit-limit-requests": "1000", "x-ratelimit-remaining-requests": "995",
        "x-ratelimit-limit-tokens": "8000", "x-ratelimit-remaining-tokens": "7646",
    }))
    tracker.record_headers(httpx.Headers({}))
    snap = tracker.snapshot()
    assert snap["per_minute_requests"]["remaining"] == 995
    assert snap["per_minute_tokens"]["remaining"] == 7646


def test_daily_limit_parsed_from_429_body():
    """This is the exact shape Groq actually returned in testing - the daily
    ceiling only ever shows up here, never in a success response header."""
    tracker = QuotaTracker()
    body = (
        'Rate limit reached for model `openai/gpt-oss-120b` in organization '
        '`org_x` service tier `on_demand` on tokens per day (TPD): '
        'Limit 200000, Used 199074, Requested 3363. '
        'Please try again in 17m32.783999999s. Need more tokens?'
    )
    tracker.record_error_body(429, body)
    daily = tracker.snapshot()["daily"]

    assert daily["limit"] == 200000
    assert daily["used"] == 199074
    assert daily["remaining"] == 926
    assert daily["reset_in_seconds"] == pytest.approx(1052.78, abs=0.1)
    assert daily["observed_at"]  # timestamped, so staleness is visible


def test_trailing_period_in_retry_hint_does_not_break_parsing():
    """The 's.' in '...783999999s. Need more tokens?' is a sentence boundary,
    not part of the duration - _parse_duration's fullmatch rejects it if the
    period leaks into the captured group."""
    assert _parse_duration("17m32.78s") is not None
    assert _parse_duration("17m32.78s.") is None  # confirms the failure mode
    tracker = QuotaTracker()
    tracker.record_error_body(
        429,
        "on tokens per day (TPD): Limit 100, Used 99. Please try again in 5s. more",
    )
    assert tracker.snapshot()["daily"]["reset_in_seconds"] == pytest.approx(5.0)


def test_non_daily_429_does_not_fabricate_a_daily_snapshot():
    """A per-minute 429 (no 'tokens per day' text) must not be misread as a
    daily-limit event."""
    tracker = QuotaTracker()
    tracker.record_error_body(429, "Rate limit reached on tokens per minute. Try again in 2s.")
    assert tracker.snapshot()["daily"] is None


def test_success_status_does_not_parse_error_body():
    tracker = QuotaTracker()
    tracker.record_error_body(200, "tokens per day (TPD): Limit 200000, Used 199074")
    assert tracker.snapshot()["daily"] is None


def test_session_usage_accumulates():
    tracker = QuotaTracker()
    tracker.record_usage(prompt_tokens=100, completion_tokens=50)
    tracker.record_usage(prompt_tokens=200, completion_tokens=10)
    assert tracker.snapshot()["session_tokens_used"] == 360


def test_negative_usage_values_are_ignored_not_subtracted():
    """A malformed usage payload must not be able to drive the counter
    negative or claw back a prior legitimate reading."""
    tracker = QuotaTracker()
    tracker.record_usage(prompt_tokens=100, completion_tokens=50)
    tracker.record_usage(prompt_tokens=-500, completion_tokens=-500)
    assert tracker.snapshot()["session_tokens_used"] == 150


# ------------------------------------------------------------------ AssistantMode

def test_mode_defaults_to_live():
    mode = AssistantMode()
    assert mode.forced_fallback is False
    assert mode.snapshot()["changed_at"] is None


def test_toggling_mode_records_a_timestamp():
    mode = AssistantMode()
    mode.set_forced_fallback(True)
    snap = mode.snapshot()
    assert snap["forced_fallback"] is True
    assert snap["changed_at"] is not None


def test_setting_the_same_value_twice_does_not_move_the_timestamp():
    mode = AssistantMode()
    mode.set_forced_fallback(True)
    first = mode.snapshot()["changed_at"]
    mode.set_forced_fallback(True)
    assert mode.snapshot()["changed_at"] == first


# --------------------------------------------------------------------- HTTP API

@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from app.main import app as fastapi_app
    with TestClient(fastapi_app) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def reset_mode():
    """Every test starts from live mode, regardless of execution order."""
    from app.agent.mode import assistant_mode as global_mode

    global_mode.set_forced_fallback(False)
    yield
    global_mode.set_forced_fallback(False)


def test_llm_status_endpoint_shape(client):
    body = client.get("/api/ops/llm-status").json()
    assert body["effective_mode"] in {"live", "fallback"}
    assert "quota" in body
    assert set(body["quota"]) == {
        "per_minute_requests", "per_minute_tokens", "daily",
        "session_tokens_used", "session_started_at",
    }


def test_llm_mode_toggle_changes_effective_mode(client, monkeypatch):
    """The test suite runs with no API key by design (see conftest.py), so
    llm_client.available is normally False and effective_mode would read
    'fallback' either way - that would make this test pass without the
    toggle doing anything. A fake key here isolates what's actually being
    tested: that the toggle, not key presence, drives effective_mode.
    """
    import app.config as config_module
    monkeypatch.setattr(config_module.settings, "llm_api_key", "fake-key-for-test", raising=False)

    before = client.get("/api/ops/llm-status").json()
    assert before["effective_mode"] == "live"

    on = client.post("/api/ops/llm-mode", json={"forced_fallback": True}).json()
    assert on["effective_mode"] == "fallback"
    assert on["forced_fallback"] is True

    off = client.post("/api/ops/llm-mode", json={"forced_fallback": False}).json()
    assert off["effective_mode"] == "live"


def test_llm_mode_requires_boolean_payload(client):
    assert client.post("/api/ops/llm-mode", json={}).status_code == 400
    assert client.post("/api/ops/llm-mode", json={"forced_fallback": "yes"}).status_code == 400


def test_forced_fallback_actually_routes_the_turn(client):
    """The point of the toggle: it must change which planner answers, not
    just flip a flag nobody reads. This runs in the suite's normal no-key
    environment, where the fallback path is already in effect regardless of
    the toggle - the property under test here is simply that a turn placed
    while forced_fallback=True actually completes via that path, not that
    the toggle is what caused it (that distinction is covered, with a faked
    key, in test_llm_mode_label_distinguishes_manual_from_no_key below)."""
    client.post("/api/ops/llm-mode", json={"forced_fallback": True})
    body = client.post("/api/chat", json={"message": "Show me Nike t-shirts"}).json()
    assert body["model"] in {"rule-based planner", "rule-based planner (manual)"}
    assert any(step["kind"] == "tool_call" for step in body["trace"])


def test_llm_mode_label_distinguishes_manual_from_no_key(client, monkeypatch):
    """With a key present, forcing fallback must say so explicitly rather
    than reading identically to 'no key configured' - an operator glancing
    at the trace needs to tell a deliberate override from a missing key."""
    import app.config as config_module
    monkeypatch.setattr(config_module.settings, "llm_api_key", "fake-key-for-test", raising=False)

    client.post("/api/ops/llm-mode", json={"forced_fallback": True})
    body = client.post("/api/chat", json={"message": "Show me Nike t-shirts"}).json()
    assert body["model"] == "rule-based planner (manual)"


def test_health_reports_forced_fallback_state(client):
    client.post("/api/ops/llm-mode", json={"forced_fallback": True})
    assert client.get("/api/ops/health").json()["forced_fallback"] is True
    client.post("/api/ops/llm-mode", json={"forced_fallback": False})
    assert client.get("/api/ops/health").json()["forced_fallback"] is False
