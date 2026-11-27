"""Smoke tests: the app renders every tab and the main interactions work.

Run:  pip install pytest && pytest -q
"""
from pathlib import Path

import joblib
import numpy as np
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app.py")


def fresh() -> AppTest:
    return AppTest.from_file(APP, default_timeout=120).run()


def test_renders_without_errors():
    at = fresh()
    assert not at.exception
    assert len(at.tabs) >= 7


def test_every_quick_example_runs():
    at = fresh()
    keys = [b.key for b in at.button if b.key and b.key.startswith("qe_")]
    assert len(keys) == 10
    for k in keys:
        at.button(key=k).click().run()
        assert not at.exception


def test_clear_sentiment_direction():
    at = fresh()
    at.text_area[0].set_value("An absolute masterpiece, brilliant and moving.").run()
    html = "".join(m.value for m in at.markdown if 'class="verdict"' in m.value)
    assert "Positive" in html
    at.text_area[0].set_value("Terrible, boring and a complete waste of time.").run()
    html = "".join(m.value for m in at.markdown if 'class="verdict"' in m.value)
    assert "Negative" in html


def test_batch_paste():
    at = fresh()
    at.radio(key="batch_mode").set_value("Paste reviews").run()
    at.text_area(key="batch_paste").set_value("Loved it!\nHated it.").run()
    at.button(key="batch_run").click().run()
    assert not at.exception
    assert at.metric[0].value == "2"


def test_pipeline_loads_and_predicts():
    pipe = joblib.load(ROOT / "models" / "tfidf_lr.joblib")
    p = pipe.predict_proba(["great movie", "awful movie"])[:, 1]
    assert p[0] > 0.9 and p[1] < 0.1 and np.all((0 <= p) & (p <= 1))
