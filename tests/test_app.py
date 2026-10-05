"""Smoke test of the Streamlit dashboard: every page renders, and a short training run feeds Detect."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PRE = f"import sys; sys.path.insert(0, r'{ROOT}'); import app; app.setup_page(); "
HAS_SYNTHETIC = (ROOT / "data" / "processed" / "synthetic" / "cleaned.parquet").exists()


def page(name: str) -> AppTest:
    return AppTest.from_string(PRE + f"app.page_{name}()", default_timeout=600)


@pytest.mark.parametrize("name", ["overview", "train", "detect", "compare"])
def test_page_renders(name):
    at = page(name).run()
    assert not at.exception, [e.message for e in at.exception]


@pytest.mark.skipif(not HAS_SYNTHETIC, reason="run `python scripts/prepare_data.py --synthetic` first")
def test_train_then_detect():
    at = page("train").run()
    at.selectbox[0].set_value("synthetic")
    at.selectbox[1].set_value("Full privacy (DP + secure aggregation)")
    at.slider[1].set_value(3)
    at.button[0].click().run()
    assert not at.exception, [e.message for e in at.exception]
    run_dir = at.session_state["last_run"]
    assert (Path(run_dir) / "model" / "autoencoder.pt").exists()
    assert any(m.label == "Privacy ε" for m in at.metric)

    det = page("detect")
    det.session_state["last_run"] = run_dir
    det.run()
    det.select_slider[0].set_value(60)
    det.select_slider[1].set_value("fast")
    det.button[0].click().run()
    assert not det.exception, [e.message for e in det.exception]
    analysed = next(m for m in det.metric if m.label == "Records analysed")
    assert analysed.value == "60"
