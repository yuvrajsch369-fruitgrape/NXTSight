"""NXTSight — Future Vision page.

A separate page (Streamlit's native pages/ multi-page mechanism), not a
6th feature tab — this is vision/roadmap content, not something you
interact with like Scam Shield or Call Shield. Renders FUTURE_VISION.md
verbatim: the source of truth is that file, not this script, so the page
can never drift from it. Only the surrounding chrome (theme, hero
banner, typography) is this file's own.
"""

from pathlib import Path

import streamlit as st

from src.ui.theme import hero, inject_theme, sidebar_brand

st.set_page_config(page_title="NXTSight — Future Vision", page_icon="🔭", layout="centered")
inject_theme()
sidebar_brand()

hero(
    "Future Vision",
    "Where NXTSight goes from here — written to be held to, not just read.",
    icon="🔭",
    compact=True,
)

VISION_PATH = Path(__file__).resolve().parent.parent / "FUTURE_VISION.md"
vision_markdown = VISION_PATH.read_text()

st.markdown(f'<div class="nxt-vision">\n\n{vision_markdown}\n\n</div>', unsafe_allow_html=True)
