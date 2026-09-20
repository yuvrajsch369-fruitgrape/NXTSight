"""NXTSight's shared dark UI theme.

One CSS file (assets/theme.css) injected by every page, plus a handful
of small markup helpers for the bits every page repeats: a hero header,
a section label, a status badge row. These helpers only ever render
strings this codebase itself authors (titles, static labels, badge
text) — never OCR output, a transcript, or a classifier's `reason`
text, which keep rendering through ordinary st.write()/st.success()
exactly as before. Nothing here changes what any widget does; it only
changes how the page around those widgets looks.
"""

from pathlib import Path

import streamlit as st

_CSS_PATH = Path(__file__).resolve().parent.parent.parent / "assets" / "theme.css"
_CSS_CACHE: str | None = None


def inject_theme() -> None:
    """Load assets/theme.css into the page. Call right after set_page_config()."""
    global _CSS_CACHE
    if _CSS_CACHE is None:
        _CSS_CACHE = _CSS_PATH.read_text()
    st.markdown(f"<style>{_CSS_CACHE}</style>", unsafe_allow_html=True)


def hero(title: str, subtitle: str, icon: str = "\U0001F6E1️", compact: bool = False) -> None:
    """The gradient banner at the top of every page."""
    compact_class = " nxt-hero-compact" if compact else ""
    st.markdown(
        f"""
        <div class="nxt-hero{compact_class}">
          <div class="nxt-hero-icon">{icon}</div>
          <div class="nxt-hero-text">
            <p class="nxt-title">{title}</p>
            <p class="nxt-subtitle">{subtitle}</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(title: str, icon: str) -> None:
    """A styled replacement for st.subheader() — icon + gradient-friendly type."""
    st.markdown(
        f"""
        <div class="nxt-section">
          <span class="nxt-section-icon">{icon}</span>
          <p class="nxt-section-title">{title}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_brand() -> None:
    """A small branded header pinned above the auto-generated page nav."""
    st.sidebar.markdown(
        """
        <div class="nxt-sidebar-brand">
          <span class="nxt-dot"></span> NXTSight
        </div>
        """,
        unsafe_allow_html=True,
    )


def badge_row(*badges: tuple[str, str]) -> None:
    """Render a row of pill badges. Each badge is (variant, label) where
    variant is one of "npu", "cpu", "safe" — matching the CSS classes."""
    spans = "".join(f'<span class="nxt-badge nxt-badge-{variant}">{label}</span>' for variant, label in badges)
    st.markdown(f'<div class="nxt-badge-row">{spans}</div>', unsafe_allow_html=True)
