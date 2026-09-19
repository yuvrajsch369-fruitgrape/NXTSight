"""NXTSight — Future Vision page.

A separate page (Streamlit's native pages/ multi-page mechanism), not a
6th feature tab — this is vision/roadmap content, not something you
interact with like Scam Shield or Call Shield. Renders FUTURE_VISION.md
verbatim: the source of truth is that file, not this script, so the page
can never drift from it.
"""

from pathlib import Path

import streamlit as st

st.set_page_config(page_title="NXTSight — Future Vision", layout="centered")

VISION_PATH = Path(__file__).resolve().parent.parent / "FUTURE_VISION.md"
st.markdown(VISION_PATH.read_text())
