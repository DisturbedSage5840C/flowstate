"""Headless run of the Streamlit dashboard: it must execute end-to-end without raising."""

from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")

APP = Path(__file__).resolve().parents[1] / "src" / "app" / "streamlit_app.py"


def test_dashboard_runs_without_exceptions():
    at = st_testing.AppTest.from_file(str(APP), default_timeout=180)
    at.run()
    assert not at.exception, [e.value for e in at.exception]


def test_dashboard_shows_data_provenance_banner():
    at = st_testing.AppTest.from_file(str(APP), default_timeout=180)
    at.run()
    banners = " ".join(w.value for w in at.warning).lower()
    assert "demonstration" in banners or "real" in banners or "in-situ" in banners
