from pathlib import Path

import pytest
from shapely.geometry import Polygon

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


def test_app_plans_a_field():
    pytest.importorskip("streamlit_folium")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception
    at.session_state["field"] = Polygon(
        [(77.5940, 12.9710), (77.5952, 12.9710), (77.5953, 12.9718), (77.5941, 12.9719)]
    )
    at.session_state["home"] = (12.9708, 77.5938)
    at.run()
    next(b for b in at.button if "Plan mission" in b.label).click().run()
    assert not at.exception and not at.error
    labels = {m.label for m in at.metric}
    assert {"Field", "Coverage", "Flights"} <= labels
