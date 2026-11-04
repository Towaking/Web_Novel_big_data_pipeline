"""
NovelLens — product interface
A Streamlit app that presents the pipeline output as a finished product rather
than an experiment console. That choice drives the design:

  * No infrastructure details (storage endpoints, per-bucket object counts,
    pipeline-phase checklists). That is the operator's concern, not the user's.
  * Backend failures surface as plain-language messages; the technical trace is
    tucked behind an expander for maintainers.
  * Model performance numbers stay visible — on the Methodology page — because
    users deserve to know how far the AI output can be trusted. What was cut is
    the trial-and-error residue, not the transparency.

Data is read from materialised Spark SQL results (Phase 3); no Spark job runs
from here, so interaction stays instant.
"""

from __future__ import annotations

import html
import io
import json
import os
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pyarrow as pa
import streamlit as st
import streamlit.components.v1 as components

from src.common.config import get_settings
from src.common.model_registry import (
    ARTIFACT_DIR,
    ModelNotTrainedError,
    artifacts_available,
    load_registry,
)
from src.common.storage import get_s3_client, list_objects, read_parquet_dir
from src.common.title_generator import TitleGenerator, description_keywords

# Arrow's default jemalloc pool keeps freed buffers: the Dataset page held ~400 MB more RAM with it.
pa.set_memory_pool(pa.system_memory_pool())

st.set_page_config(
    page_title="NovelLens: Novel Catalogue Intelligence",
    page_icon=":material/menu_book:",
    layout="wide",
    initial_sidebar_state="auto",
)

SETTINGS = get_settings()
#: NOVELLENS_PUBLIC=1 for a published deployment: operator pages (full dataset,
#: lake inventory) and raw error text, which names hosts and paths, stay hidden.
PUBLIC = os.environ.get("NOVELLENS_PUBLIC", "0") == "1"

# Visual identity: warm paper, library green for anything clickable,
# ink blue for data, terracotta for the one highlight on a screen.
PAPER = "#FBF8F3"
SURFACE = "#FFFFFF"
INK = "#1E2433"
# Secondary text: 5.62:1 on paper, 5.11:1 on the sidebar surface (WCAG AA needs 4.5).
MUTED = "#5B6475"
LINE = "#E4DED3"
ACTION = "#1F5E52"
FOCUS = ACTION
# Headline figures: 9.71:1 on paper.
DATA_INK = "#1F3F78"
# First chart series and the single-series colour.
BRAND = "#2F5FA8"
# Chart highlight; text in the same hue uses ACCENT_TEXT (5.13:1 on paper).
ACCENT = "#C8562E"
ACCENT_TEXT = "#B8431F"
GOOD_TEXT = "#1E7A4E"
WARN_TEXT = "#8A5A0B"
# Plus Jakarta Sans: an Indonesian typeface that stays legible at small label sizes.
# The system stack covers an unreachable Google Fonts.
FONT_STACK = ('"Plus Jakarta Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", '
              "Roboto, sans-serif")
# Figures only: equal-width digits keep number columns aligned.
MONO_STACK = '"IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace'
# Categorical order passes CVD and normal-vision separation on the paper surface, so scatters may use all four.
# Slot 4 is 2.63:1 against paper, so charts using it carry labels or tooltips.
SEQUENCE = ["#2F5FA8", "#C8562E", "#1B9474", "#B39A12"]
OTHER = "#8A909C"
SCALE = [[0.0, "#DCE6F5"], [0.5, "#6F93CC"], [1.0, "#1F3F78"]]

st.markdown(
    f"""
    <style>
      @import url("https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400..800&family=IBM+Plex+Mono:wght@500;600&display=swap");

      /* Streamlit sets Source Sans on p/div/span through its own classes, so a
         container rule loses. Icons (Material Symbols ligatures) and code keep
         their own fonts: overriding them turns icons into words. */
      html, body, .stApp,
      .stApp *:not([data-testid="stIconMaterial"]):not(code):not(code *):not(pre *) {{
        font-family: {FONT_STACK};
      }}

      /* --- Page rhythm ------------------------------------------------ */
      .block-container {{padding-top: 1.4rem; padding-bottom: 5rem; max-width: 1380px;}}
      header[data-testid="stHeader"] {{background: transparent; height: 0;}}
      #MainMenu, footer {{visibility: hidden;}}
      /* Streamlit's rainbow strip along the top edge carries no state. */
      [data-testid="stDecoration"] {{display: none;}}

      /* A consistent type scale, not a per-element size free-for-all. */
      h1 {{font-size: 1.9rem; letter-spacing: -.02em; font-weight: 650; color: {INK};}}
      h2 {{font-size: 1.25rem; letter-spacing: -.01em; font-weight: 620; margin-top: .3rem; color: {INK};}}
      h3 {{font-size: 1.02rem; font-weight: 620; color: {INK};}}

      /* Figures in mono ink blue: the one typographic voice repeated on every page. */
      .nl-card .val, .nl-kpis .v, .nl-lead .num, .nl-findings .fig, .nl-rec .num,
      .nl-rank .v, .nl-rank .n, .nl-bar .p, [data-testid="stMetricValue"] {{
        font-family: {MONO_STACK} !important; font-variant-numeric: tabular-nums;
        letter-spacing: -.01em; color: {DATA_INK};
      }}

      /* --- Page header: the title and its one sentence, no box ------------ */
      .nl-hero {{padding: .2rem 0 1.1rem; margin-bottom: 1.6rem; border-bottom: 1px solid {LINE};}}
      .nl-hero h1 {{margin: 0;}}
      .nl-hero p {{margin: .45rem 0 0; color: {MUTED}; font-size: .98rem; max-width: 66ch; line-height: 1.55;}}

      /* --- Metric cards: flat panels, no shadow ---------------------------- */
      .nl-card {{
        position: relative; height: 100%;
        padding: 1rem 1.1rem; border-radius: 12px;
        background: {SURFACE}; border: 1px solid {LINE};
      }}
      .nl-card .lbl {{color: {MUTED}; font-size: .8rem; font-weight: 600; margin-bottom: .4rem;}}
      .nl-card .val {{font-size: 1.6rem; font-weight: 600; line-height: 1.1;}}
      .nl-card .val.word {{font-family: {FONT_STACK} !important; font-weight: 680; letter-spacing: -.015em;}}
      .nl-card .sub {{color: {MUTED}; font-size: .8rem; font-weight: 500; margin-top: .35rem; line-height: 1.45;}}

      /* --- Genre chips: labels, not buttons, so square-ish and quiet --------- */
      .nl-chip {{
        display: inline-block; padding: .16rem .55rem; margin: .14rem .24rem .14rem 0;
        border-radius: 6px; font-size: .75rem; font-weight: 550;
        background: #F2EDE4; border: 1px solid {LINE}; color: #3A4253;
      }}
      .nl-chip.on {{background: #E4EFEC; border-color: #BCD6CF; color: {ACTION};}}

      /* --- Supporting text ------------------------------------------------- */
      .nl-note {{
        color: {MUTED}; font-size: .84rem; line-height: 1.6; margin-top: .55rem;
      }}
      .nl-syn {{color: #3A4253; font-size: .875rem; line-height: 1.6;}}

      /* --- Tabs, buttons, tables --------------------------------------------- */
      .stTabs [data-baseweb="tab-list"] {{gap: .35rem; border-bottom: 1px solid {LINE};}}
      .stTabs [data-baseweb="tab"] {{font-size: .92rem; font-weight: 580; padding: .55rem .9rem;}}
      .stTabs [aria-selected="true"] {{color: {ACTION};}}
      .stButton > button {{border-radius: 8px; font-weight: 600; padding: .5rem 1.05rem; min-height: 44px;}}
      div[data-testid="stDataFrame"] {{border-radius: 8px; overflow: hidden; border: 1px solid {LINE};}}
      [data-testid="stMetricValue"] {{font-size: 1.4rem; font-weight: 600;}}
      [data-testid="stMetricLabel"] {{color: {MUTED};}}
      hr {{border-color: {LINE};}}

      /* --- Inputs: white field, hairline border, green ring on focus -------- */
      [data-baseweb="input"], [data-baseweb="textarea"], [data-baseweb="select"] > div {{
        background: {SURFACE} !important;
        border: 1px solid #D9D2C5 !important; border-radius: 8px !important;
        transition: border-color .15s ease, box-shadow .15s ease;
      }}
      /* The inner field paints the theme's grey surface over the wrapper otherwise. */
      [data-baseweb="input"] input, [data-baseweb="textarea"] textarea,
      [data-baseweb="base-input"] {{background: transparent !important;}}
      [data-baseweb="input"]:hover, [data-baseweb="textarea"]:hover,
      [data-baseweb="select"] > div:hover {{border-color: #B9B0A0 !important;}}
      [data-baseweb="input"]:focus-within, [data-baseweb="textarea"]:focus-within,
      [data-baseweb="select"] > div:focus-within {{
        border-color: {FOCUS} !important; box-shadow: 0 0 0 2px {FOCUS}40 !important;
      }}
      .stButton > button:focus-visible, a.nl-open:focus-visible {{
        outline: 2px solid {FOCUS}; outline-offset: 2px;
      }}

      /* --- Recommendation card: one novel, so it earns a panel ------------ */
      .nl-rec {{
        display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: .4rem 1.4rem;
        padding: 1.1rem 1.2rem; margin-bottom: .75rem; border-radius: 12px;
        background: {SURFACE}; border: 1px solid {LINE};
      }}
      .nl-rec .title {{font-size: 1.02rem; font-weight: 680; line-height: 1.35; margin-bottom: .35rem; color: {INK};}}
      .nl-rec .nl-syn {{
        margin-top: .55rem; display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical;
        overflow: hidden; min-height: calc(3 * 1.6em);
      }}
      .nl-rec .stats {{display: flex; flex-direction: column; align-items: flex-end; gap: .7rem; text-align: right;}}
      .nl-rec .num {{font-size: 1.4rem; font-weight: 600; line-height: 1;}}
      .nl-rec .num-lbl {{color: {MUTED}; font-size: .75rem; font-weight: 600; margin-top: .2rem;}}
      .nl-rec .meta {{
        grid-column: 1 / -1; display: flex; justify-content: space-between; align-items: center;
        gap: .8rem; padding-top: .7rem; margin-top: .25rem; border-top: 1px solid {LINE};
        color: {MUTED}; font-size: .8rem; font-weight: 500;
      }}
      a.nl-open {{
        flex: none; padding: .45rem .95rem; min-height: 44px; display: inline-flex; align-items: center;
        border-radius: 8px; border: 1px solid #BCD6CF; color: {ACTION} !important; background: #F3F8F6;
        text-decoration: none !important; font-weight: 600; font-size: .85rem;
      }}
      a.nl-open:hover {{border-color: {ACTION}; background: #E4EFEC;}}
      .nl-rec.compact .stats {{flex-direction: row; gap: 1.6rem; align-items: flex-start;}}
      @media (max-width: 640px) {{
        .nl-rec {{grid-template-columns: 1fr;}}
        .nl-rec .stats {{flex-direction: row; justify-content: flex-start; align-items: baseline;
                        gap: 1.6rem; text-align: left;}}
      }}

      /* --- Sidebar -------------------------------------------------------- */
      section[data-testid="stSidebar"] {{border-right: 1px solid {LINE};}}
      section[data-testid="stSidebar"] .stRadio label {{font-size: .92rem;}}
      .nl-brand {{padding: .2rem .2rem 1rem; margin-bottom: .4rem; border-bottom: 1px solid #DDD5C7;}}
      .nl-brand .name {{font-size: 1.15rem; font-weight: 720; letter-spacing: -.015em; color: {INK};}}
      .nl-brand .sub {{font-size: .78rem; color: {MUTED}; margin-top: .15rem;}}
      /* Section groups: a plain header row, no card. */
      section[data-testid="stSidebar"] [data-testid="stExpander"] details {{
        border: none; background: transparent; border-radius: 0;
      }}
      section[data-testid="stSidebar"] [data-testid="stExpander"] summary {{
        padding: .45rem .25rem; border-radius: 8px;
      }}
      section[data-testid="stSidebar"] [data-testid="stExpander"] summary p {{
        font-size: .74rem; font-weight: 700; color: {MUTED}; letter-spacing: .02em;
      }}
      section[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover p {{color: {INK};}}
      section[data-testid="stSidebar"] [data-testid="stExpander"] summary:focus-visible {{
        outline: 2px solid {FOCUS}; outline-offset: 1px;
      }}
      section[data-testid="stSidebar"] [data-testid="stExpanderDetails"] {{padding: 0 0 .35rem .1rem;}}
      section[data-testid="stSidebar"] [data-testid="stExpanderDetails"] [data-testid="stVerticalBlock"] {{gap: .15rem;}}
      /* Folding animation: the links slide up into the header and back down.
         ::details-content + interpolate-size animate height:auto (Chrome 131+);
         older browsers simply open and close without motion. */
      section[data-testid="stSidebar"] {{interpolate-size: allow-keywords;}}
      section[data-testid="stSidebar"] details::details-content {{
        height: 0; opacity: 0; overflow: clip;
        transition: height .26s ease, opacity .2s ease, content-visibility .26s allow-discrete;
      }}
      section[data-testid="stSidebar"] details[open]::details-content {{height: auto; opacity: 1;}}
      @media (prefers-reduced-motion: reduce) {{
        section[data-testid="stSidebar"] details::details-content {{transition: none;}}
      }}
      /* Streamlit paints the current page's link itself; only the overview's card look is undone. */
      section[data-testid="stSidebar"] [data-testid="stPageLink"] a {{
        border-color: transparent; min-height: 36px; padding: .3rem .55rem;
      }}
      .nl-status {{display: flex; align-items: center; gap: .45rem; margin-top: 1rem; padding-top: .8rem;
                  border-top: 1px solid #DDD5C7;
                  font-size: .75rem; color: {MUTED}; font-weight: 500;}}
      .nl-status .dot {{width: 8px; height: 8px; border-radius: 50%; flex: none;}}

      /* --- Change against a reference (KPI cards) --------------------------
         Arrow + sign carry the direction; colour only repeats it. */
      .nl-card .delta {{display: inline-block; margin-top: .4rem; font-size: .78rem; font-weight: 640;}}
      .nl-card .delta.up {{color: {GOOD_TEXT};}}
      .nl-card .delta.down {{color: {ACCENT_TEXT};}}
      .nl-card .delta.flat {{color: {MUTED};}}

      /* --- Title ideas ----------------------------------------------------- */
      .nl-idea {{
        padding: .95rem 1.1rem; margin-bottom: .7rem; border-radius: 12px;
        background: {SURFACE}; border: 1px solid {LINE};
      }}
      .nl-idea .rank {{color: {MUTED}; font-size: .75rem; font-weight: 640; margin-bottom: .25rem;}}
      .nl-idea .t {{font-size: 1.06rem; font-weight: 700; line-height: 1.35; color: {INK};}}

      /* --- Lead number, facts, findings, answers ----------------------- */
      .nl-lead .num {{font-size: 3.2rem; font-weight: 600; letter-spacing: -.04em; line-height: 1;}}
      .nl-lead .lbl {{color: {MUTED}; font-size: .95rem; line-height: 1.5; margin-top: .6rem; max-width: 34ch;}}
      .nl-facts {{display: grid; grid-template-columns: repeat(auto-fit, minmax(13rem, 1fr)); gap: .9rem 1.6rem; margin: 0;}}
      .nl-facts div {{border-top: 1px solid {LINE}; padding-top: .55rem;}}
      .nl-facts dt {{color: {MUTED}; font-size: .78rem; font-weight: 600;}}
      .nl-facts dd {{margin: .15rem 0 0; font-size: 1rem; font-weight: 620; color: {INK};}}
      .nl-findings {{list-style: none; margin: 0; padding: 0; counter-reset: f;}}
      .nl-findings li {{display: grid; grid-template-columns: 7.5rem minmax(0, 1fr); gap: 1.2rem; align-items: baseline;
                       padding: .85rem 0; border-bottom: 1px solid {LINE};}}
      .nl-findings .fig {{font-size: 1.4rem; font-weight: 600;}}
      .nl-findings .txt {{font-size: .97rem; line-height: 1.55; color: #3A4253; max-width: 70ch;}}
      .nl-answer {{font-size: 1.04rem; line-height: 1.55; color: {INK}; max-width: 80ch; margin: .2rem 0 1rem;}}
      .nl-group {{font-size: .8rem; font-weight: 700; color: {MUTED}; margin-bottom: .35rem;}}
      .nl-prose {{max-width: 72ch; line-height: 1.65; color: #3A4253;}}
      .nl-prose p {{margin: 0 0 .8rem;}}
      .nl-prose.small {{font-size: .86rem; color: {MUTED};}}
      .nl-limits {{margin: 0; padding-left: 1.1rem; line-height: 1.6; color: #3A4253;}}
      .nl-limits li {{margin-bottom: .45rem;}}
      /* Governance checklist: the status word leads each row, colour repeats it. */
      .nl-pillar {{display: flex; justify-content: space-between; align-items: baseline; margin: .4rem 0 .3rem;
                  font-weight: 700; font-size: 1.02rem;}}
      .nl-pillar .tally {{color: {MUTED}; font-size: .8rem; font-weight: 500;}}
      .nl-checks {{list-style: none; margin: 0 0 1.4rem; padding: 0;}}
      .nl-checks li {{display: grid; grid-template-columns: 4.6rem minmax(0, 1fr); gap: .8rem; padding: .6rem 0;
                     border-top: 1px solid {LINE};}}
      .nl-checks .st {{font-size: .78rem; font-weight: 700; padding-top: .1rem;}}
      .nl-checks .st-pass .st {{color: {GOOD_TEXT};}}
      .nl-checks .st-info .st {{color: {MUTED};}}
      .nl-checks .st-warning .st {{color: {WARN_TEXT};}}
      .nl-checks .st-fail .st {{color: {ACCENT_TEXT};}}
      .nl-checks .nm {{font-weight: 600; font-size: .9rem;}}
      .nl-checks .nm code {{font-size: .72rem; color: {MUTED}; background: none; padding: 0;}}
      .nl-checks .ev, .nl-checks .fu {{color: {MUTED}; font-size: .82rem; line-height: 1.5; margin-top: .15rem;}}
      @media (max-width: 640px) {{
        .nl-findings li {{grid-template-columns: 1fr; gap: .2rem;}}
        .nl-lead .num {{font-size: 2.6rem;}}
      }}

      /* --- Overview: metric band and summary panels ------------------- */
      /* No box around the band: hairlines between figures, the first one leads. */
      .nl-kpis {{display: grid; grid-template-columns: 1.4fr repeat(4, minmax(0, 1fr));
                border-top: 1px solid {LINE}; border-bottom: 1px solid {LINE};}}
      .nl-kpis > div {{padding: 1rem 1.2rem; border-left: 1px solid {LINE};}}
      .nl-kpis > div:first-child {{border-left: none; padding-left: 0;}}
      .nl-kpis .k {{color: {MUTED}; font-size: .8rem; font-weight: 600;}}
      .nl-kpis .v {{font-size: 1.55rem; font-weight: 600; margin-top: .2rem;}}
      .nl-kpis > div:first-child .v {{font-size: 2.3rem;}}
      .nl-kpis .s {{color: {MUTED}; font-size: .78rem; margin-top: .15rem;}}
      .nl-kpis .up {{color: {GOOD_TEXT}; font-weight: 650;}}
      @media (max-width: 900px) {{
        .nl-kpis {{grid-template-columns: repeat(2, minmax(0, 1fr));}}
        .nl-kpis > div {{border-left: none; border-top: 1px solid {LINE}; padding-left: 0;}}
        .nl-kpis > div:first-child {{grid-column: 1 / -1; border-top: none;}}
      }}
      /* :has() also matches the page's outer block, so no background here: an
         unbordered wrapper then stays invisible and only real panels show a line. */
      [data-testid="stMain"] [data-testid="stVerticalBlockBorderWrapper"]:has(.nl-panel-h) {{
        border-color: {LINE}; border-radius: 12px;
      }}
      .nl-panel-h {{display: flex; justify-content: space-between; align-items: baseline; gap: .8rem; flex-wrap: wrap;}}
      .nl-panel-h span:first-child {{font-weight: 700; font-size: .98rem;}}
      .nl-panel-h .note {{color: {MUTED}; font-size: .76rem;}}
      /* Panel links read as a footer link, not a second card inside the card. */
      [data-testid="stVerticalBlockBorderWrapper"]:has(.nl-panel-h) [data-testid="stPageLink"] a {{
        border: none; background: none; padding: .2rem 0; min-height: 36px; width: auto;
      }}
      [data-testid="stVerticalBlockBorderWrapper"]:has(.nl-panel-h) [data-testid="stPageLink"] a p {{
        color: {ACTION}; font-weight: 600;
      }}
      .nl-rank.compact li {{grid-template-columns: 1.4rem minmax(0, 1fr) auto; padding: .55rem .1rem;}}

      /* --- Ranked list (overview) --------------------------------------- */
      .nl-rank {{list-style: none; margin: 0; padding: 0;}}
      .nl-rank li {{display: grid; grid-template-columns: 1.6rem minmax(0, 1fr) auto 5.6rem; gap: .7rem;
                   align-items: baseline; padding: .62rem .2rem; border-bottom: 1px solid {LINE};}}
      .nl-rank .r {{color: {MUTED}; font-variant-numeric: tabular-nums;}}
      .nl-rank .t {{white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-weight: 560;}}
      .nl-rank .v {{font-weight: 600;}}
      .nl-rank .n {{color: {MUTED} !important; font-size: .8rem; text-align: right;}}
      /* The top row is the screen's one highlight: a terracotta rank number. */
      .nl-rank li:first-child .r {{color: {ACCENT_TEXT}; font-weight: 700;}}

      /* --- Genre confidence bars (AI Studio) ------------------------------ */
      .nl-bars {{display: grid; gap: .45rem; margin-top: .3rem;}}
      .nl-bar {{display: grid; grid-template-columns: 8.5rem minmax(0, 1fr) 2.6rem 4.8rem; gap: .7rem;
               align-items: center; font-size: .85rem;}}
      .nl-bar .g {{white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}}
      .nl-bar .track {{position: relative; height: 10px; border-radius: 4px; background: #EFEAE1;}}
      .nl-bar .fill {{position: absolute; inset: 0 auto 0 0; border-radius: 4px; background: #B9BFC9;}}
      .nl-bar.on .fill {{background: {BRAND};}}
      .nl-bar .tick {{position: absolute; top: -3px; width: 2px; height: 16px; background: {ACCENT};}}
      .nl-bar .p {{text-align: right; color: {INK} !important;}}
      .nl-bar.on .p {{color: {DATA_INK} !important; font-weight: 600;}}
      .nl-bar .s {{color: {MUTED}; font-size: .76rem;}}
      .nl-bar.on .g, .nl-bar.on .s {{color: {DATA_INK}; font-weight: 600;}}
      @media (max-width: 640px) {{.nl-bar {{grid-template-columns: 6.5rem minmax(0, 1fr) 2.4rem;}} .nl-bar .s {{display: none;}}}}

      /* --- Page links on the overview ------------------------------------- */
      [data-testid="stMain"] [data-testid="stPageLink"] a {{
        width: 100%; box-sizing: border-box;
        border: 1px solid {LINE}; border-radius: 8px; padding: .55rem .8rem;
        background: {SURFACE}; min-height: 44px;
      }}
      [data-testid="stMain"] [data-testid="stPageLink"] a:hover {{border-color: {ACTION};}}
      [data-testid="stPageLink"] a:focus-visible {{outline: 2px solid {FOCUS}; outline-offset: 2px;}}
    </style>
    """,
    unsafe_allow_html=True,
)


class DataUnavailable(RuntimeError):
    """Product data is not ready to serve yet (the pipeline has not run)."""


@st.cache_data(ttl=60, show_spinner=False)
def lake_version() -> str:
    """
    Version stamp of the curated zone: the newest object's modification time.

    Every lake loader takes this stamp as a cache key, so all tables switch to
    a new pipeline run together (within a minute) instead of each one expiring
    on its own 15-minute clock and a page mixing old and new tables.
    """
    try:
        objects = list_objects(SETTINGS.minio.bucket_curated, "")
    except Exception:
        return ""
    return str(max((o["LastModified"] for o in objects), default=""))


@st.cache_data(show_spinner=False, max_entries=64)
def _load_lake(prefix: str, version: str) -> pd.DataFrame:
    try:
        return read_parquet_dir(SETTINGS.minio.bucket_curated, prefix)
    except Exception as exc:
        raise DataUnavailable(str(exc)) from exc


def load_lake(prefix: str) -> pd.DataFrame:
    """Read one table from the curated zone (`analytics/`, `model/`, `governance/`)."""
    return _load_lake(prefix, lake_version())


def load_table(name: str) -> pd.DataFrame:
    """Read one analytics result table from the data lake."""
    return load_lake(f"analytics/{name}")


@st.cache_data(show_spinner=False, max_entries=64)
def _table_profile(prefix: str, group_col: str | None, version: str) -> tuple[int, dict]:
    frame = _load_lake(prefix, version)
    return len(frame), (frame[group_col].value_counts().to_dict() if group_col else {})


def table_profile(prefix: str, group_col: str | None = None) -> tuple[int, dict]:
    """Row count (and optional value counts) of a lake table without copying the table on every rerun."""
    return _table_profile(prefix, group_col, lake_version())


@st.cache_data(ttl=120, show_spinner=False)
def _probe_backend() -> bool:
    # Raises on failure, so only a healthy answer is cached: an outage does not
    # keep the dashboard locked for two minutes after storage comes back.
    get_s3_client(SETTINGS).list_buckets()
    return True


def backend_online() -> tuple[bool, str]:
    """Check storage availability without leaking its details into the UI."""
    try:
        return _probe_backend(), ""
    except Exception as exc:
        return False, str(exc)


def _artifact_stamp() -> int:
    # metrics.json is written last by Phase 4, so its mtime marks a complete set.
    path = ARTIFACT_DIR / "metrics.json"
    return path.stat().st_mtime_ns if path.is_file() else 0


@st.cache_resource(show_spinner=False, max_entries=1)
def _registry(stamp: int):
    return load_registry()


def get_registry():
    """Model registry, reloaded when Phase 4 writes a new set of artefacts."""
    return _registry(_artifact_stamp())


@st.cache_data(show_spinner=False, max_entries=1)
def _model_metrics(stamp: int) -> dict:
    return json.loads((ARTIFACT_DIR / "metrics.json").read_text(encoding="utf-8"))


def model_metrics() -> dict:
    """metrics.json alone, for pages that report scores without loading ~1 GB of models."""
    return _model_metrics(_artifact_stamp())


@st.cache_data(show_spinner=False, max_entries=8)
def _load_labels(bridge: str, dim: str, key: str, label: str, version: str) -> pd.DataFrame:
    """A bridge table resolved to (series_id, label); labels as categories to keep 600k rows small."""
    pairs = _load_lake(f"model/{bridge}", version)
    names = _load_lake(f"model/{dim}", version)[[key, label]]
    out = pairs.merge(names, on=key)[["series_id", label]]
    out[label] = out[label].astype("category")
    return out


def genre_labels() -> pd.DataFrame:
    return _load_labels("bridge_novel_genre", "dim_genre", "genre_key", "genre", lake_version())


def tag_labels() -> pd.DataFrame:
    return _load_labels("bridge_novel_tag", "dim_tag", "tag_key", "tag", lake_version())


def author_labels() -> pd.DataFrame:
    return _load_labels("bridge_novel_author", "dim_author", "author_key", "author", lake_version())


@st.cache_data(show_spinner=False, max_entries=2)
def _genre_lists(version: str) -> pd.Series:
    pairs = _load_labels("bridge_novel_genre", "dim_genre", "genre_key", "genre", version)
    return pairs.assign(genre=pairs["genre"].astype(str)).groupby("series_id")["genre"].agg(list)


def genre_lists() -> pd.Series:
    """series_id -> list of genres, for cards that show genre chips."""
    return _genre_lists(lake_version())


@st.cache_resource(show_spinner=False, max_entries=1)
def _title_generator(version: str) -> TitleGenerator:
    viz = _load_lake("analytics/viz_novels", version)[["series_id", "title", "type_origin"]]
    return TitleGenerator(viz.assign(genre_list=viz["series_id"].map(_genre_lists(version))))


def title_generator() -> TitleGenerator:
    """Title pattern model over the whole lake catalogue, rebuilt when the lake changes (~3 s)."""
    return _title_generator(lake_version())


def last_full_year(years: pd.Series) -> int:
    """Latest release year that is over: the current year is still filling up."""
    return int(min(years.max(), pd.Timestamp.now(tz="UTC").year - 1))


#: Fixed colour per origin so a filter never repaints the survivors.
ORIGIN_COLORS = {"CN": SEQUENCE[0], "KR": SEQUENCE[1], "JP": SEQUENCE[2], "EN": SEQUENCE[3], "Other": OTHER}


def origin_group(origins: pd.Series) -> pd.Series:
    return origins.where(origins.isin(["CN", "KR", "JP", "EN"]), "Other")


def hero(title: str, subtitle: str) -> None:
    """Page header: title and one explanatory sentence."""
    st.markdown(
        f'<div class="nl-hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def card(container, label: str, value: str, sub: str = "", delta: tuple[float, str] | None = None) -> None:
    """Metric card; `delta` is (change, text): the sign picks the arrow, the text names the baseline."""
    change = ""
    if delta is not None and pd.notna(delta[0]):
        cls, arrow = ("up", "▲") if delta[0] > 0 else ("down", "▼") if delta[0] < 0 else ("flat", "■")
        change = f'<div class="delta {cls}">{arrow} {delta[1]}</div>'
    # Mono is for figures; a word value ("Action", "CN") keeps the text face.
    kind = "val" if re.match(r"^[\d+\-−.±]", str(value)) else "val word"
    container.markdown(
        f'<div class="nl-card"><div class="lbl">{label}</div>'
        f'<div class="{kind}">{value}</div>{change}'
        f'<div class="sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def chips(values, active: bool = False) -> str:
    cls = "nl-chip on" if active else "nl-chip"
    return "".join(f'<span class="{cls}">{html.escape(str(v))}</span>' for v in values)


def style_fig(fig, height: int | None = None):
    """Match every chart to the app theme."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#3A4253", size=13, family=FONT_STACK),
        # 4px rounded data ends, matching the card radius language; plotly only
        # rounds the outer end of a stacked bar, so segments still meet flush.
        barcornerradius=4,
        margin=dict(l=0, r=10, t=10, b=0),
        legend=dict(bgcolor="rgba(0,0,0,0)", title_text=""),
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=LINE, font=dict(color=INK)),
    )
    fig.update_xaxes(gridcolor="#EDE7DD", zerolinecolor="#D9D2C5")
    fig.update_yaxes(gridcolor="#EDE7DD", zerolinecolor="#D9D2C5")
    if height:
        fig.update_layout(height=height)
    return fig


#: Bucket labels come out of Spark in Indonesian ("3. Panjang (200-499)").
BUCKET_WORDS = [
    ("Sangat Panjang", "Very long"), ("Sangat panjang", "Very long"), ("Sangat ringkas", "Very brief"),
    ("Pendek", "Short"), ("Sedang", "Medium"), ("Panjang", "Long"), ("Epik", "Epic"),
    ("Ringkas", "Brief"), ("Harian", "Daily"), ("Mingguan", "Weekly"), ("Bulanan", "Monthly"),
    ("Lambat", "Slow"), ("Terhenti", "Stalled"), (" kata", " words"), (" hari", " days"),
    (" bulan", " months"), ("<=", "≤"),
]


def en_label(value) -> str:
    """English display label for a pipeline bucket; the order prefix is dropped."""
    text = re.sub(r"^\d+\.\s*", "", str(value))
    for old, new in BUCKET_WORDS:
        text = text.replace(old, new)
    return text


def dot_rating_chart(frame: pd.DataFrame, category: str, horizontal: bool = False, height: int = 340,
                     translate: bool = True):
    """Average rating per group as dots: the means differ by tenths, and a bar on a
    cut axis would overstate the gap where a dot does not."""
    plot = frame.assign(label=frame[category].map(en_label) if translate else frame[category])
    xy = dict(x="avg_weighted_rating", y="label") if horizontal else dict(x="label", y="avg_weighted_rating")
    fig = px.scatter(
        plot, **xy, color="n_novels", color_continuous_scale=SCALE,
        hover_data={"n_novels": ":,", "label": False},
        labels={"label": "", "avg_weighted_rating": "Average rating", "n_novels": "Titles"},
    )
    fig.update_traces(marker=dict(size=15, line=dict(width=2, color=SURFACE)))
    lo, hi = plot["avg_weighted_rating"].min(), plot["avg_weighted_rating"].max()
    pad = max(hi - lo, 0.1) * 0.35
    (fig.update_xaxes if horizontal else fig.update_yaxes)(range=[lo - pad, hi + pad])
    fig.update_coloraxes(colorbar_title_text="Titles")
    return style_fig(fig, height)


def rating_text(values: pd.Series, empty: str = "no rated titles") -> pd.Series:
    """Averages as fixed two-decimal text; Streamlit 1.39 prints NaN in number columns as None."""
    return values.map(lambda v: f"{v:.2f}" if pd.notna(v) else empty)


def as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, (list, np.ndarray)):
        return [str(v) for v in value]
    return [str(value)]


def novel_card(row: pd.Series, show_similarity: bool = True, similarity_label: str = "match",
               show_synopsis: bool = True) -> None:
    """
    Recommendation card: identity on the left (title, genres, synopsis), the two
    numbers a reader compares across cards on the right, metadata and the link
    in a footer row. One HTML block so every card has the same structure and
    the synopsis clamps to three lines, which keeps the cards an even height.
    """
    title = html.escape(str(row.get("title") or "Untitled"))
    snippet = row.get("synopsis_snippet")
    synopsis = (
        "" if not show_synopsis
        else f'<div class="nl-syn">{html.escape(snippet.strip())}</div>'
        if isinstance(snippet, str) and snippet.strip()
        else '<div class="nl-syn">No synopsis available.</div>'
    )

    rating, votes = row.get("rating_score"), row.get("rating_votes")
    stats = [
        f'<div><div class="num">{rating:.1f}</div><div class="num-lbl">'
        f'{"rating" if pd.notna(votes) and votes >= 10 else "rating, few votes"}</div></div>'
        if pd.notna(rating) else
        '<div><div class="num">n/a</div><div class="num-lbl">not rated yet</div></div>'
    ]
    if show_similarity and pd.notna(row.get("similarity")):
        stats.append(
            f'<div><div class="num">{row["similarity"]:.0%}</div>'
            f'<div class="num-lbl">{similarity_label}</div></div>'
        )

    meta = []
    if pd.notna(row.get("chapter_count")):
        meta.append(f"{int(row['chapter_count']):,} chapters")
    if pd.notna(votes):
        meta.append(f"{int(votes):,} vote{'' if int(votes) == 1 else 's'}")
    if isinstance(row.get("type_origin"), str):
        meta.append(html.escape(row["type_origin"]))
    url = row.get("url")
    link = (
        f'<a class="nl-open" href="{html.escape(url, quote=True)}" target="_blank" '
        f'rel="noopener noreferrer">Open on NovelUpdates</a>'
        if isinstance(url, str) and url.startswith("http") else ""
    )

    st.markdown(
        f'<div class="nl-rec{"" if show_synopsis else " compact"}"><div><div class="title">{title}</div>'
        f'{chips(as_list(row.get("genre_list"))[:5])}{synopsis}</div>'
        f'<div class="stats">{"".join(stats)}</div>'
        f'<div class="meta"><span>{" · ".join(meta)}</span>{link}</div></div>',
        unsafe_allow_html=True,
    )


def facts_html(items: list[tuple[str, str]]) -> str:
    """A definition list of label/value pairs, for facts that support one lead number."""
    return '<dl class="nl-facts">' + "".join(
        f"<div><dt>{html.escape(k)}</dt><dd>{v}</dd></div>" for k, v in items
    ) + "</dl>"


def findings_html(items: list[tuple[str, str]]) -> str:
    """Numbered findings: the supporting figure, then the sentence that says what it means."""
    return '<ol class="nl-findings">' + "".join(
        f'<li><span class="fig">{html.escape(fig)}</span><span class="txt">{text}</span></li>'
        for fig, text in items
    ) + "</ol>"


def answer(text: str) -> None:
    """The one-sentence answer a section gives, above the chart that backs it."""
    st.markdown(f'<p class="nl-answer">{text}</p>', unsafe_allow_html=True)


def kpi_strip_html(items: list[tuple[str, str, str]]) -> str:
    """One band of headline metrics: label, value, and what the value is measured against."""
    return '<div class="nl-kpis">' + "".join(
        f'<div><div class="k">{html.escape(k)}</div><div class="v">{v}</div><div class="s">{s}</div></div>'
        for k, v, s in items
    ) + "</div>"


def panel_head(title: str, note: str = "") -> None:
    st.markdown(f'<div class="nl-panel-h"><span>{html.escape(title)}</span>'
                f'<span class="note">{html.escape(note)}</span></div>', unsafe_allow_html=True)


def mini_fig(fig, height: int):
    """Panel-sized chart: no legend, tight margins, readable ticks."""
    fig = style_fig(fig, height)
    fig.update_layout(showlegend=False, margin=dict(l=0, r=8, t=4, b=0), font=dict(size=12))
    return fig


def page_overview() -> None:
    viz = load_table("viz_novels")
    hero("Overview", "The NovelUpdates catalogue at a glance. Every panel summarises one page; "
                     "follow its link to dig in.")
    reliable = viz[viz["is_rated_reliable"] == True]  # noqa: E712
    years = viz["year"].dropna().astype(int)
    last_full = last_full_year(years)
    per_year = years.value_counts()
    cur, prev = int(per_year.get(last_full, 0)), int(per_year.get(last_full - 1, 0))
    growth = f'<span class="up">▲ {(cur - prev) / prev:+.0%}</span> vs {last_full - 1}' if prev else ""

    st.markdown(kpi_strip_html([
        ("Titles", f"{len(viz):,}", "in the catalogue"),
        ("Rated", f"{len(reliable):,}", f"{len(reliable) / len(viz):.0%} have 10+ votes"),
        ("Average rating", f"{reliable['weighted_rating'].mean():.2f}", "weighted, out of 5"),
        ("Median length", f"{viz['chapter_count'].median():,.0f}", "chapters"),
        (f"Released in {last_full}", f"{cur:,}", growth),
    ]), unsafe_allow_html=True)

    st.write("")
    left, right = st.columns([3, 2], gap="medium")
    with left:
        with st.container(border=True):
            panel_head("New titles per release year", f"{last_full - 14} to {last_full}")
            span = per_year.sort_index().loc[last_full - 14:last_full].reset_index()
            span.columns = ["year", "titles"]
            fig = px.bar(span, x="year", y="titles", labels={"year": "", "titles": "Titles"})
            fig.update_traces(marker_color=BRAND, hovertemplate="%{x}: %{y:,} titles<extra></extra>")
            fig.update_yaxes(title_text="")
            st.plotly_chart(mini_fig(fig, 230), use_container_width=True, config={"displayModeBar": False})
            st.page_link(PAGE["trends"], label="Market trends", icon=PAGE_ICON["trends"])
    with right:
        with st.container(border=True):
            panel_head("Where titles come from", "share of the catalogue")
            share = origin_group(viz["type_origin"].dropna()).value_counts(normalize=True).mul(100)
            share = share.reindex(list(ORIGIN_COLORS)).dropna()
            share = share[share >= 0.5].reset_index()
            share.columns = ["origin", "pct"]
            fig = px.bar(share, x="pct", y="origin", orientation="h", color="origin",
                         color_discrete_map=ORIGIN_COLORS, text=share["pct"].map("{:.0f}%".format),
                         labels={"pct": "% of titles", "origin": ""})
            fig.update_traces(textposition="outside", cliponaxis=False, hovertemplate="%{y}: %{x:.1f}%<extra></extra>")
            fig.update_yaxes(categoryorder="array", categoryarray=share["origin"].tolist()[::-1])
            fig.update_xaxes(range=[0, share["pct"].max() * 1.2], visible=False)
            st.plotly_chart(mini_fig(fig, 230), use_container_width=True, config={"displayModeBar": False})
            st.page_link(PAGE["creators"], label="Authors & publishers", icon=PAGE_ICON["creators"])

    a, b, c = st.columns(3, gap="medium")
    with a:
        with st.container(border=True):
            panel_head("Most common genres", "titles carrying the genre")
            genres = load_table("genre_frequency").nlargest(8, "n_novels").sort_values("n_novels")
            fig = px.bar(genres, x="n_novels", y="genre", orientation="h",
                         text=genres["pct_of_catalog"].map("{:.0f}%".format), labels={"n_novels": "", "genre": ""})
            fig.update_traces(marker_color=BRAND, textposition="outside", cliponaxis=False,
                              hovertemplate="%{y}: %{x:,} titles<extra></extra>")
            fig.update_xaxes(range=[0, genres["n_novels"].max() * 1.25], visible=False)
            st.plotly_chart(mini_fig(fig, 260), use_container_width=True, config={"displayModeBar": False})
            st.page_link(PAGE["genre"], label="Genre explorer", icon=PAGE_ICON["genre"])
    with b:
        with st.container(border=True):
            mean = reliable["weighted_rating"].mean()
            panel_head("How readers rate", f"titles with 10+ votes · average {mean:.2f}")
            fig = px.histogram(reliable, x="weighted_rating", nbins=40, labels={"weighted_rating": "Weighted rating"})
            fig.update_traces(marker_color=BRAND, hovertemplate="%{x}: %{y:,} titles<extra></extra>")
            fig.add_vline(x=mean, line=dict(color=ACCENT, width=2))
            fig.update_yaxes(visible=False)
            st.plotly_chart(mini_fig(fig, 260), use_container_width=True, config={"displayModeBar": False})
            st.page_link(PAGE["analytics"], label="Catalogue analytics", icon=PAGE_ICON["analytics"])
    with c:
        with st.container(border=True):
            panel_head("Highest rated", "weighted rating, 10+ votes")
            top = reliable.nlargest(5, "weighted_rating")
            ranked = "".join(
                f'<li><span class="r">{i}</span><span class="t" title="{html.escape(str(r.title), quote=True)}">'
                f'{html.escape(str(r.title))}</span><span class="v">{r.weighted_rating:.2f}</span></li>'
                for i, r in enumerate(top.itertuples(index=False), 1)
            )
            st.markdown(f'<ol class="nl-rank compact">{ranked}</ol>', unsafe_allow_html=True)
            st.page_link(PAGE["lookup"], label="Title lookup", icon=PAGE_ICON["lookup"])

    left, right = st.columns([3, 2], gap="medium")
    with left:
        with st.container(border=True):
            panel_head("What stands out")
            corr = load_table("correlation_metrics").set_index("pair")
            ranking = load_table("genre_ranking").sort_values("avg_weighted_rating", ascending=False)
            richness = load_table("text_richness_vs_rating").sort_values("text_richness")
            gain = richness.iloc[-1]["avg_weighted_rating"] - richness.iloc[0]["avg_weighted_rating"]
            st.markdown(findings_html([
                (f"{corr.loc['chapter_count ~ weighted_rating', 'pearson_r']:+.2f}",
                 "correlation between length and rating: more chapters do not earn better scores."),
                (f"{ranking.iloc[0]['avg_weighted_rating']:.2f}",
                 f"for <b>{html.escape(ranking.iloc[0]['genre'])}</b>, the best-rated genre "
                 f"({int(ranking.iloc[0]['n_novels']):,} titles)."),
                (f"{gain:+.2f}", "rating gap between the richest and the shortest synopses."),
            ]), unsafe_allow_html=True)
            st.page_link(PAGE["tags"], label="Story elements", icon=PAGE_ICON["tags"])
    with right:
        with st.container(border=True):
            panel_head("Data health")
            try:
                audit = load_lake("governance/audit_report")
                tally = audit["status"].map(STATUS_LABEL).fillna(audit["status"]).value_counts()
                checks = (f"{int(tally.get('Pass', 0))} pass · {int(tally.get('Warning', 0))} warning · "
                          f"{int(tally.get('Fail', 0))} fail")
            except DataUnavailable:
                checks = "audit not run yet"
            stamp = pd.to_datetime(lake_version(), errors="coerce", utc=True)
            st.markdown(facts_html([
                ("Governance audit", checks),
                ("Synopsis coverage", f"{viz['has_synopsis'].mean():.1%} of titles"),
                ("Missing chapter count", f"{viz['chapter_count'].isna().mean():.0%} of titles"),
                ("Data refreshed", f"{stamp:%d %b %Y, %H:%M} UTC" if pd.notna(stamp) else "unknown"),
            ]), unsafe_allow_html=True)
            st.page_link(PAGE["governance"], label="Quality & governance", icon=PAGE_ICON["governance"])


def page_analytics() -> None:
    hero(
        "Catalogue analytics",
        "The structure of the translated-novel market: genre mix, how story length "
        "relates to reader reception, and the vocabulary that dominates the stories.",
    )

    viz = load_table("viz_novels")
    tab_genre, tab_length, tab_text, tab_market = st.tabs(
        ["Genres", "Length & Rating", "Story Language", "Market Map"]
    )

    with tab_genre:
        # Bar length counts every title; colour and "best liked" use titles with 10+ votes, like every ranking.
        genres = load_table("genre_frequency").drop(columns="avg_weighted_rating").merge(
            load_table("genre_ranking")[["genre", "avg_weighted_rating"]], on="genre", how="left")
        common, liked = genres.nlargest(1, "n_novels").iloc[0], genres.nlargest(1, "avg_weighted_rating").iloc[0]
        answer(f"<b>{html.escape(common['genre'])}</b> is the most common genre ({common['pct_of_catalog']:.0f}% of titles); "
               f"<b>{html.escape(liked['genre'])}</b> is the best liked ({liked['avg_weighted_rating']:.2f} on average "
               "across its titles with 10+ votes).")
        top_n = st.slider("Genres shown", 5, min(30, len(genres)), 15)
        subset = genres.nlargest(top_n, "n_novels").sort_values("n_novels")

        fig = px.bar(
            subset, x="n_novels", y="genre", orientation="h",
            color="avg_weighted_rating", color_continuous_scale=SCALE,
            labels={"n_novels": "Number of titles", "genre": "", "avg_weighted_rating": "Rating (10+ votes)"},
            hover_data={"pct_of_catalog": ":.1f", "avg_chapters": ":.0f"},
        )
        fig.update_coloraxes(colorbar_title_text="Rating")
        st.plotly_chart(style_fig(fig, max(360, 26 * top_n)), use_container_width=True)
        st.markdown(
            '<div class="nl-note">Bar length = number of titles, colour = average rating of the titles with 10+ '
            "votes. The most common genre is not necessarily the best-liked one, so compare both.</div>",
            unsafe_allow_html=True,
        )

        st.write("")
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown("**Most frequent genre combinations**")
            combos = load_table("genre_combinations").head(12)
            st.dataframe(
                combos[["genre_combo", "n_novels", "avg_weighted_rating"]].rename(
                    columns={"genre_combo": "Combination", "n_novels": "Titles",
                             "avg_weighted_rating": "Rating"}),
                hide_index=True, use_container_width=True, height=430,
                column_config={"Rating": st.column_config.NumberColumn(format="%.2f")},
            )
        with col2:
            st.markdown("**How widely ratings spread inside each genre**")
            top_genres = viz["primary_genre"].value_counts().head(8).index.tolist()
            box_df = viz[viz["primary_genre"].isin(top_genres) & viz["weighted_rating"].notna()]
            # One colour: the genre is already on the axis, a hue per box would only repeat it.
            fig = px.box(
                box_df, x="primary_genre", y="weighted_rating", points=False,
                labels={"primary_genre": "", "weighted_rating": "Rating"},
            )
            fig.update_traces(marker_color=BRAND, line_color=BRAND, fillcolor="rgba(47,95,168,.18)")
            st.plotly_chart(style_fig(fig, 430), use_container_width=True)

    with tab_length:
        lengths = load_table("length_vs_rating").sort_values("length_bucket")
        hi_b = lengths.loc[lengths["avg_weighted_rating"].idxmax()]
        lo_b = lengths.loc[lengths["avg_weighted_rating"].idxmin()]
        answer(f"Shorter runs rate a little higher: {en_label(hi_b['length_bucket']).lower()} titles average "
               f"{hi_b['avg_weighted_rating']:.2f}, {en_label(lo_b['length_bucket']).lower()} ones "
               f"{lo_b['avg_weighted_rating']:.2f}.")
        col_a, col_b = st.columns([3, 2])
        available = sorted(viz["primary_genre"].dropna().unique().tolist())
        default = load_table("genre_frequency").nlargest(4, "n_novels")["genre"].tolist()
        # Four genres at most: the chart palette has four colours that stay distinct in a scatter.
        picked = col_a.multiselect("Genres", available, default=[g for g in default if g in available],
                                   max_selections=4, help="Up to four genres, so each keeps its own colour.")
        only_reliable = col_b.checkbox("Only titles with an established score", value=True)

        scatter = viz.dropna(subset=["chapter_count", "weighted_rating"])
        if picked:
            scatter = scatter[scatter["primary_genre"].isin(picked)]
        if only_reliable:
            scatter = scatter[scatter["is_rated_reliable"] == True]  # noqa: E712

        if scatter.empty:
            st.info("No titles match these filters. Pick more genres or include titles without an established score.")
        else:
            # render_mode="webgl": 20k+ points stay responsive to hover and zoom.
            fig = px.scatter(
                scatter, x="chapter_count", y="weighted_rating", color="primary_genre",
                size="rating_votes", size_max=26, opacity=0.62, log_x=True, render_mode="webgl",
                color_discrete_sequence=SEQUENCE, hover_name="title",
                hover_data={"rating_score": True, "rating_votes": True, "type_origin": True},
                labels={"chapter_count": "Chapter count (log scale)",
                        "weighted_rating": "Rating", "primary_genre": ""},
            )
            buckets = load_table("length_vs_rating").sort_values("length_bucket")
            fig.add_scatter(
                x=buckets["avg_chapters"], y=buckets["avg_weighted_rating"],
                mode="lines+markers", name="Average per length bucket",
                line=dict(color=INK, width=2, dash="dash"),
            )
            st.plotly_chart(style_fig(fig, 500), use_container_width=True)
            st.caption(f"{len(scatter):,} titles plotted.")

            corr = load_table("correlation_metrics")
            r = corr.loc[corr["pair"] == "chapter_count ~ weighted_rating", "pearson_r"].iloc[0]
            best = buckets.loc[buckets["avg_weighted_rating"].idxmax()]
            worst = buckets.loc[buckets["avg_weighted_rating"].idxmin()]
            st.markdown(
                f'<div class="nl-note">The orange line is the average rating per length bucket. '
                f"The correlation is r = {r}. Ratings are highest for "
                f"{en_label(best['length_bucket']).lower()} chapters ({best['avg_weighted_rating']:.2f}) and "
                f"lowest for {en_label(worst['length_bucket']).lower()} ({worst['avg_weighted_rating']:.2f})"
                + (": stories that run long tend to lose satisfied readers." if worst["avg_chapters"]
                   > best["avg_chapters"] else ".")
                + "</div>",
                unsafe_allow_html=True,
            )

        st.write("")
        rate_col, pace_col = st.columns(2, gap="large")
        with rate_col:
            st.markdown("**Rating by length bucket**")
            buckets = load_table("length_vs_rating").sort_values("length_bucket")
            st.plotly_chart(dot_rating_chart(buckets, "length_bucket"), use_container_width=True)
        with pace_col:
            st.markdown("**Rating by release pace**")
            pace = load_table("release_pace_vs_rating").sort_values("release_pace")
            st.plotly_chart(dot_rating_chart(pace, "release_pace"), use_container_width=True)
        pace_r = load_table("correlation_metrics")
        pace_r = pace_r.loc[pace_r["pair"] == "log_days_per_chapter ~ weighted_rating", "pearson_r"]
        st.markdown(
            '<div class="nl-note">Release pace is parsed from the catalogue&apos;s '
            '"Every N Day(s)" field, which had never been used before. Its correlation '
            f"with rating (r = {pace_r.iloc[0] if len(pace_r) else 'n/a'}) is about as strong as "
            "the chapter-count one: slow-updating titles score slightly lower.</div>",
            unsafe_allow_html=True,
        )

    with tab_text:
        tokens = load_table("top_tokens")
        richness = load_table("text_richness_vs_rating").sort_values("text_richness")
        gain = richness.iloc[-1]["avg_weighted_rating"] - richness.iloc[0]["avg_weighted_rating"]
        answer(f"Longer synopses go with better ratings ({gain:+.2f} from the shortest to the longest group), "
               "and a handful of words appear far more often in the best-rated stories.")
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown("**Words that set top-rated synopses apart**")
            try:
                disc = load_table("synopsis_discriminative_tokens")
                disc = disc.assign(**{c: pd.to_numeric(disc[c], errors="coerce")
                                      for c in ("top_to_bottom_ratio", "n_top", "n_bottom")})
                disc = disc.nlargest(15, "top_to_bottom_ratio")
            except DataUnavailable:
                disc = pd.DataFrame()
            if disc.empty:
                st.info("The word comparison has not been computed yet. Run Phase 3 of the pipeline.")
            else:
                disc = disc.sort_values("top_to_bottom_ratio")
                fig = px.bar(disc, x="top_to_bottom_ratio", y="token", orientation="h",
                             text=disc["top_to_bottom_ratio"].map("×{:.1f}".format),
                             hover_data={"n_top": ":,", "n_bottom": ":,"},
                             labels={"top_to_bottom_ratio": "Times more common in the top-rated quarter",
                                     "token": "", "n_top": "Top-rated synopses", "n_bottom": "Bottom-rated synopses"})
                fig.update_traces(marker_color=BRAND, textposition="outside", cliponaxis=False)
                fig.update_xaxes(range=[0, disc["top_to_bottom_ratio"].max() * 1.2])
                st.plotly_chart(style_fig(fig, 440), use_container_width=True)
        with col2:
            st.markdown("**The most common words in synopses**")
            common = tokens.head(15).sort_values("n_novels")
            fig = px.bar(common, x="n_novels", y="token", orientation="h",
                         labels={"n_novels": "Synopses using the word", "token": ""})
            fig.update_traces(marker_color=BRAND)
            st.plotly_chart(style_fig(fig, 440), use_container_width=True)
        st.caption("Words after stopword removal, from the synopses rather than the catalogue tags.")

        st.write("")
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown("**Does a longer synopsis go with a better rating?**")
            fig = dot_rating_chart(richness, "text_richness", height=380)
            fig.update_layout(xaxis_tickangle=-18)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown("**Which story elements readers rate highest**")
            tags = load_table("best_rated_tags").head(14).sort_values("avg_weighted_rating")
            st.plotly_chart(dot_rating_chart(tags, "tag", horizontal=True, height=380, translate=False),
                            use_container_width=True)

    with tab_market:
        answer("The same genre can rate differently by origin: read across a row to compare countries.")
        st.markdown("**Average rating by genre and country of origin**")
        matrix = load_table("genre_origin_matrix")
        pivot = matrix.pivot(index="genre", columns="type_origin", values="avg_weighted_rating")
        pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
        fig = px.imshow(
            pivot, color_continuous_scale=SCALE, aspect="auto", text_auto=".2f",
            labels=dict(x="Country of origin", y="", color="Rating"),
        )
        st.plotly_chart(style_fig(fig, max(380, 30 * len(pivot))), use_container_width=True)
        st.markdown(
            '<div class="nl-note">Empty cells = combinations with fewer than 25 titles, '
            "deliberately left uncomputed so no average rests on a handful of titles.</div>",
            unsafe_allow_html=True,
        )

        st.write("")
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown("**Average rating by decade**")
            # Decades with a handful of titles (12 from the 1960s) would anchor the line on noise.
            decade = load_table("decade_trend").query("n_novels >= 50")
            fig = px.line(
                decade, x="decade", y="avg_weighted_rating", markers=True,
                hover_data={"n_novels": ":,"},
                labels={"decade": "", "avg_weighted_rating": "Rating", "n_novels": "Titles"},
            )
            fig.update_traces(line=dict(color=BRAND, width=2), marker=dict(size=9, color=BRAND))
            st.plotly_chart(style_fig(fig, 340), use_container_width=True)
        with col2:
            st.markdown("**Longest titles in the catalogue**")
            outliers = load_table("stat_outlier_novels").head(10)
            st.dataframe(
                outliers[["title", "chapter_count", "rating_score"]].rename(
                    columns={"title": "Title", "chapter_count": "Chapters", "rating_score": "Rating"}),
                hide_index=True, use_container_width=True, height=340,
            )


@st.cache_data(ttl=900, show_spinner=False)
def render_wordcloud(tokens: pd.DataFrame) -> bytes:
    """Render the word cloud on a transparent ground so it blends with the theme."""
    from wordcloud import WordCloud

    cloud = WordCloud(
        width=1400, height=560, mode="RGBA", background_color=None,
        colormap="cool", max_words=180, prefer_horizontal=0.92,
    ).generate_from_frequencies(dict(zip(tokens["token"], tokens["freq"])))

    buffer = io.BytesIO()
    cloud.to_image().save(buffer, format="PNG")
    return buffer.getvalue()


SAMPLE_SYNOPSIS = (
    "After being betrayed by their allies and dying at the top of the tower, a "
    "swordsman wakes up in their own body ten years younger. This time they intend "
    "to master the forbidden sword techniques before their old enemy rises again."
)
SAMPLE_TAGS = "Time Travel, Weak to Strong, Cultivation, Male Protagonist, Revenge, Sword Wielder"


def page_studio() -> None:
    hero(
        "AI Studio",
        "Describe a story once and get its likely genres, reader reception, narrative family, "
        "closest titles in the catalogue and working-title ideas, all from the same brief.",
    )
    if not artifacts_available():
        st.info("AI features are still being prepared. They appear once model training has finished.")
        return
    try:
        registry = get_registry()
    except ModelNotTrainedError as exc:
        st.info("AI features are not available yet. They appear once the models finish training.")
        technical_details(str(exc))
        return

    tab_story, tab_similar = st.tabs(["Analyse a story", "Similar to a catalogue title"])
    with tab_story:
        _studio_story(registry)
    with tab_similar:
        _studio_similar(registry)


TITLE_STYLES = {
    "Any": "Any origin", "CN": "Chinese web novel", "KR": "Korean web novel",
    "JP": "Japanese light novel", "EN": "English original",
}

#: English copy for each genre model; metrics.json carries the pipeline's
#: Indonesian wording, which stays there for the written report.
MODEL_TEXT = {
    "logreg": ("Linear · sparse TF-IDF",
               "A linear model over TF-IDF features. The fastest, with the best-calibrated probabilities."),
    "svm": ("Margin-based · sparse TF-IDF",
            "Maximises the margin between classes; probabilities come from Platt calibration."),
    "rf": ("Tree ensemble · chi-square selected TF-IDF",
           "One forest of decision trees votes on every genre at once, using only the terms most tied to "
           "a genre. It can pick up word combinations a linear model cannot, but scores lower than the "
           "linear models on this catalogue."),
    "ensemble": ("Weighted log-odds average · linear + trees",
                 "Combines the Linear SVM and Random Forest in log-odds space, with weights picked on "
                 "cross-validated predictions. It gets the most genres right and ranks the top three "
                 "about as well as the SVM alone."),
}


#: Short names for running text; the radio labels come from metrics.json.
MODEL_NAMES = {
    "logreg": "Logistic Regression", "svm": "Linear SVM", "rf": "Random Forest",
    "ensemble": "SVM + Random Forest ensemble",
}


def model_text(name: str, info: dict) -> tuple[str, str]:
    """(family, description) in English, falling back to the stored label for a new model."""
    return MODEL_TEXT.get(name, (info.get("label", name), ""))


def reference_options(registry) -> tuple[list[int], dict[int, str]]:
    """Selectbox options keyed by series_id; duplicate titles get their year so both stay pickable."""
    catalog = registry.catalog.sort_values("rating_votes", ascending=False)
    duplicated = catalog["title"].duplicated(keep=False)
    year = pd.to_numeric(catalog["year"], errors="coerce")
    suffix = np.where(duplicated & year.notna(), " (" + year.fillna(0).astype(int).astype(str) + ")", "")
    labels = dict(zip(catalog["series_id"].astype(int), catalog["title"] + suffix))
    return list(labels), labels


def trained_origins(registry) -> list[str]:
    """Origins the rating model has a feature for, most common first."""
    learned = {
        name[len("origin_"):] for name in registry.metrics.get("numeric_features", [])
        if name.startswith("origin_")
    }
    ranked = registry.catalog["type_origin"].value_counts().index.tolist()
    ordered = [o for o in ranked if o in learned] + sorted(learned - set(ranked))
    return ordered or ["CN", "KR", "JP"]


def genre_bars_html(result: pd.DataFrame) -> str:
    """Confidence per genre as bars; the tick is that genre's own threshold, the label says predicted or not."""
    rows = []
    for r in result.itertuples(index=False):
        pct, tick = min(r.probability, 1) * 100, min(r.threshold, 1) * 100
        rows.append(
            f'<div class="nl-bar{" on" if r.predicted else ""}">'
            f'<span class="g">{html.escape(str(r.genre))}</span>'
            f'<span class="track"><span class="fill" style="width:{pct:.1f}%"></span>'
            f'<span class="tick" style="left:{tick:.1f}%"></span></span>'
            f'<span class="p">{pct:.0f}%</span>'
            f'<span class="s">{"predicted" if r.predicted else ""}</span></div>'
        )
    return '<div class="nl-bars">' + "".join(rows) + "</div>"


def _studio_story(registry) -> None:
    names = registry.genre_model_names
    info = {name: registry.genre_model_info(name) for name in names}
    head_d = registry.metrics.get("head_d_rating_estimator", {})
    origins = trained_origins(registry)

    with st.form("story_brief", border=True):
        st.markdown("**Story brief**")
        left, right = st.columns([3, 2], gap="large")
        with left:
            synopsis = st.text_area("Story idea or synopsis", SAMPLE_SYNOPSIS, height=168)
            tags = st.text_input("Tags (optional, comma-separated)", SAMPLE_TAGS)
        with right:
            a, b = st.columns(2)
            origin = a.selectbox("Country of origin", origins)
            novel_type = b.selectbox("Format", ["Web Novel", "Light Novel", "Published Novel"])
            a, b = st.columns(2)
            chapters = a.number_input("Planned chapters", 1, 12000, 300, 10)
            year = b.number_input("Release year", 1990, 2030, 2025)
            a, b = st.columns(2)
            pace = a.number_input("Days between chapters", 0.5, 365.0, 7.0, 0.5, format="%.1f")
            status = b.selectbox("Status", ["Ongoing", "Completed"])
            completed = status == "Completed"
        with st.expander("Model settings"):
            m1, m2 = st.columns([3, 2])
            choice = m1.radio(
                "Genre model", names, horizontal=True,
                format_func=lambda n: f"{info[n].get('label', n)} · F1 {info[n].get('f1_macro', 0):.2f}",
            )
            sensitivity = m2.slider(
                "Genre strictness", 0.6, 1.6, 1.0, 0.05,
                help="Below 1 surfaces more genres; above 1 keeps only the most confident ones.",
            )
            family, description = model_text(choice, info[choice])
            st.caption(f"{family}: {description}")
        submitted = st.form_submit_button("Analyse story", type="primary")

    if submitted:
        text = f"{synopsis} {tags}".strip()
        try:
            with st.spinner("Reading the story…"):
                genres = registry.predict_genres(
                    synopsis, tags=tags, model=choice, top_k=10,
                    type_origin=origin, novel_type=novel_type, sensitivity=sensitivity,
                )
                similar = registry.recommend_by_text(text, k=12)
                archetype = registry.assign_cluster(text)
                rating = registry.estimate_rating(
                    synopsis, chapter_count=float(chapters), year=int(year), is_completed=completed,
                    type_origin=origin, novel_type=novel_type, days_per_chapter=float(pace), tags=tags,
                ) if head_d.get("trained") else None
        except ValueError:
            st.warning("Not enough story detail to work from. Add a sentence or two about the plot.")
            return
        st.session_state["story"] = {
            "synopsis": synopsis, "tags": tags, "origin": origin, "genres": genres, "similar": similar,
            "archetype": archetype, "rating": rating, "model": choice,
        }
        st.session_state.pop("title_ideas", None)
        st.session_state["title_seen"] = set()

    story = st.session_state.get("story")
    if story is None:
        st.caption("Fill in the brief and press Analyse story. Only the story idea is required.")
        return
    _story_report(registry, story, info)


def _story_report(registry, story: dict, info: dict) -> None:
    genres, archetype = story["genres"], story["archetype"]
    predicted = genres[genres["predicted"]]["genre"].tolist()
    profile = archetype["profile"]
    catalog = registry.catalog
    average = catalog.loc[catalog["rating_votes"] >= 10, "rating_score"].mean()
    head_d = registry.metrics.get("head_d_rating_estimator", {})

    st.write("")
    st.subheader("What the models see")
    cards = st.columns(3)
    card(cards[0], "Likely genre", predicted[0] if predicted else "None confident",
         (f"also {', '.join(predicted[1:3])} · " if len(predicted) > 1 else "")
         + f"{len(predicted)} pass their threshold")
    if story["rating"] is not None:
        diff = story["rating"] - average
        card(cards[1], "Expected rating", f"{story['rating']:.2f}",
             f"±{head_d.get('mae', 0):.2f} typical error",
             delta=(diff, f"{diff:+.2f} vs catalogue"))
    else:
        card(cards[1], "Expected rating", "n/a", "the rating model is not trained on this data")
    card(cards[2], "Narrative family", f"Archetype {archetype['cluster_id']}",
         f"{profile.get('size', 0):,} titles · average rating {profile.get('avg_weighted_rating', 0):.2f}")

    st.write("")
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("**Genre confidence**")
        st.markdown(genre_bars_html(genres), unsafe_allow_html=True)
        st.caption("The bar is the model's confidence; the tick is that genre's own threshold, tuned so "
                   "rare genres can still be detected.")
    with right:
        st.markdown(f"**Archetype {archetype['cluster_id']}: signature terms**")
        st.markdown(chips(profile.get("top_terms", [])[:10], active=True), unsafe_allow_html=True)
        top = profile.get("top_genres", {})
        if top:
            st.caption("Dominant genres: " + ", ".join(list(top)[:4]))
        examples = profile.get("example_titles", [])[:4]
        if examples:
            st.caption("Typical titles: " + " · ".join(examples))
        st.caption(f"Typical length {profile.get('avg_chapters', 0):,.0f} chapters.")

    st.write("")
    st.subheader("Working-title ideas")
    _title_ideas(registry, story, predicted)

    st.write("")
    st.subheader("Closest titles in the catalogue")
    similar = story["similar"]
    shown = st.radio("Show", [6, 12], horizontal=True, format_func=lambda n: f"{n} titles", key="similar_n")
    for _, row in similar.head(shown).iterrows():
        novel_card(row)


def _title_ideas(registry, story: dict, predicted: list[str]) -> None:
    with st.spinner("Learning title patterns from the catalogue…"):
        generator = title_generator()
    origin = story["origin"] if story["origin"] in TITLE_STYLES else "Any"
    c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
    style = c1.selectbox("Title style", list(TITLE_STYLES), index=list(TITLE_STYLES).index(origin),
                         format_func=TITLE_STYLES.get, key="title_style",
                         help="Chinese titles run short and formulaic; Japanese light-novel titles are often whole sentences.")
    more = c2.button("New ideas", use_container_width=True, key="title_go")

    signature = (story["synopsis"], story["tags"], style)
    if more or st.session_state.get("title_signature") != signature or "title_ideas" not in st.session_state:
        if st.session_state.get("title_signature") != signature:
            st.session_state["title_signature"], st.session_state["title_seen"] = signature, set()
        st.session_state["title_batch"] = st.session_state.get("title_batch", 0) + 1
        st.session_state["title_ideas"] = _make_title_ideas(registry, generator, story, predicted, style)

    ideas = st.session_state["title_ideas"]
    if not ideas:
        st.info("No new title survived the filters. Add more story detail or switch the title style.")
        return
    cols = st.columns(2, gap="medium")
    for i, title in enumerate(ideas):
        cols[i % 2].markdown(f'<div class="nl-idea"><div class="t">{html.escape(title)}</div></div>',
                             unsafe_allow_html=True)
    st.caption("Built from title patterns that recur in the catalogue, filled with words from your story that "
               "play the same part in real titles. Titles that already exist are removed. Check that a title is "
               "free before you use it.")


def _make_title_ideas(registry, generator, story: dict, predicted: list[str], style: str) -> list[str]:
    seen = st.session_state.get("title_seen", set())
    similarity = pd.Series(registry.embeddings @ registry.embed(f"{story['synopsis']} {story['tags']}"),
                           index=registry.catalog["series_id"].to_numpy())
    weights = generator.title_weights(
        generator.catalog["series_id"].map(similarity).to_numpy(dtype=float),
        predicted[:3], None if style == "Any" else style,
    )
    keywords = (description_keywords(story["synopsis"]) | description_keywords(story["tags"])
                | description_keywords(" ".join(predicted[:3])))
    titles = [i.title for i in generator.generate(
        weights, keywords, n=8, seed=st.session_state["title_batch"], exclude=seen,
        story=f"{story['synopsis']}. {story['tags']}",
    )]
    st.session_state["title_seen"] = seen | {t.lower() for t in titles}
    return titles


def _studio_similar(registry) -> None:
    st.markdown("Pick a title you already like to find the ones whose story content sits closest to it.")
    options, labels = reference_options(registry)
    c1, c2 = st.columns([4, 1], vertical_alignment="bottom")
    series_id = int(c1.selectbox("Reference title", options, format_func=labels.get, key="ref_title"))
    k = c2.selectbox("Results", [6, 9, 12], key="ref_k")
    try:
        results = registry.recommend_by_series(series_id, k=k)
    except KeyError:
        st.warning("That title is not in the model's catalogue yet.")
        return
    st.write("")
    for _, row in results.iterrows():
        novel_card(row)


def page_method() -> None:
    hero(
        "Methodology & limitations",
        "How the numbers in this app are computed, how accurate the models are, and where "
        "the output should not be taken at face value.",
    )
    viz = load_table("viz_novels")
    reliable_share = (viz["is_rated_reliable"] == True).mean()  # noqa: E712
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader("The data")
        st.markdown(
            f'<div class="nl-prose"><p>The catalogue holds <b>{len(viz):,} translated web novels</b> '
            "scraped from NovelUpdates: metadata, reader ratings and synopses, consolidated into a "
            "single source file before any cleaning.</p>"
            "<p>Cleaning restores missing values that the site disguises as <code>0</code>, "
            "<code>Unknown</code> or <code>N/A</code> to genuinely empty. "
            f"{viz['chapter_count'].isna().mean():.0%} of titles have no known chapter count; "
            "counting them as zero would pull the average length down by the same share.</p>"
            "<p>Ratings appear as a <b>Bayesian weighted rating</b>, so a title with 2 votes is not "
            "treated like one with 500, and every ranking uses only titles with at least 10 votes "
            f"({reliable_share:.0%} of the catalogue).</p></div>",
            unsafe_allow_html=True,
        )
    with right:
        st.subheader("Limitations")
        cn_share = (viz["type_origin"] == "CN").mean()
        st.markdown(
            '<ul class="nl-limits">'
            "<li>The models read the description, not the manuscript: a weak synopsis is judged weakly.</li>"
            "<li>Recommendations rest on content similarity, not writing quality.</li>"
            f"<li>Chinese titles make up {cn_share:.0%} of the catalogue, so other origins are modelled less well.</li>"
            "<li>Ratings come from one reader community, not the general public.</li>"
            "<li>Archetypes overlap; they are not hard categories.</li></ul>",
            unsafe_allow_html=True,
        )

    if not artifacts_available():
        return
    metrics = model_metrics()
    head_a, head_b = metrics["head_a_genre_tagger"], metrics["head_b_recommender"]
    head_c, head_d = metrics["head_c_clustering"], metrics["head_d_rating_estimator"]

    st.write("")
    st.subheader("How well the genre models tag")
    answer(
        f"{head_a['n_train']:,} titles for development and {head_a['n_test']:,} held out for testing, the same "
        "split for every model. Settings and each genre's threshold come from cross-validation inside the "
        "development set; the test set is used once, at the end."
    )
    best = head_a.get("best_model")
    rows = [{
        "Model": entry.get("label", name) + (" (selected)" if name == best else ""),
        "Macro F1, CV": entry.get("f1_macro_oof"),
        "Macro F1, test": entry["f1_macro"],
        "Micro F1, test": entry["f1_micro"],
        "Top-3 precision": entry.get("precision_at_3"),
        "Macro F1 at 0.5": entry["f1_macro_fixed_threshold"],
    } for name, entry in head_a["models"].items()]
    st.dataframe(
        pd.DataFrame(rows).sort_values("Macro F1, CV", ascending=False),
        hide_index=True, use_container_width=True,
        column_config={
            **{c: st.column_config.NumberColumn(format="%.3f") for c in
               ("Macro F1, test", "Micro F1, test", "Top-3 precision", "Macro F1 at 0.5")},
            "Macro F1, CV": st.column_config.NumberColumn(
                format="%.3f", help="3-fold cross-validation inside the development set"),
        },
    )
    st.markdown(
        '<div class="nl-prose small"><p><b>Macro F1</b> weights every genre equally, so it falls when rare '
        "genres are missed; the model is chosen on its cross-validated value. <b>Micro F1</b> is dominated by "
        "common genres. <b>Top-3 precision</b> scores the three most likely genres without any threshold, "
        "closest to what a user sees. The last column uses a flat 0.5 threshold; the gap to the test column "
        "is what per-genre thresholds add.</p>"
        "<p>Every model here is classical machine learning (TF-IDF features, no neural network). The "
        "ensemble averages the Linear SVM and Random Forest predictions in log-odds space: the forest is "
        "weaker alone but wrong in different places, so the blend gets more genres right. A plain average "
        "of probabilities lowered top-3 precision; the log-odds blend keeps it level with the SVM. "
        "Words that NovelUpdates censors "
        "(for example mu*der) are restored before training, so a synopsis typed in full reads the same way "
        "as the catalogue. Regularisation and the weight of each text block were tuned with Bayesian "
        "search (TPE). "
        "References: TF-IDF + linear baselines from genre-from-synopsis research; "
        "per-genre thresholds after Hoang (2018); multi-output random forests after Madjarov et al. (2012); "
        "TPE after Bergstra et al. (2011); logarithmic pooling after Genest et al. (1984).</p></div>",
        unsafe_allow_html=True,
    )

    st.write("")
    st.subheader("The other three models")
    items = [
        ("Recommendations", f"{head_b['genre_overlap_at_k']:.0%} of the top {head_b['k']} suggestions share "
                            "a genre with the title they were asked about"),
        ("Archetypes", f"{head_c['best_k']} clusters, the count chosen by silhouette score"),
        ("Rating estimate",
         f"typical error ±{head_d['mae']:.2f} on a 1 to 5 scale, "
         f"{head_d['improvement_over_baseline_pct']:.0f}% better than always guessing the average"
         if head_d.get("trained") else "not trained on this data"),
    ]
    st.markdown(facts_html(items), unsafe_allow_html=True)


# Data platform pages: the star schema, property graph and governance audit, read from the Parquet tables
# the pipeline materialises, so a number here matches the pipeline log.

#: Groups smaller than this are hidden for non-additive measures, the same
#: HAVING COUNT(*) >= 10 rule data_model.py uses for its OLAP roll-up.
MIN_GROUP = 10

#: Status labels come out of governance_audit.py in Indonesian.
STATUS_LABEL = {"LULUS": "Pass", "PERINGATAN": "Warning", "GAGAL": "Fail", "INFO": "Info"}
STATUS_COLOR = {"Pass": GOOD_TEXT, "Warning": WARN_TEXT, "Fail": ACCENT_TEXT, "Info": MUTED}

DIMENSIONS: dict[str, dict] = {
    "Genre": {"kind": "bridge", "bridge": "bridge_novel_genre", "dim": "dim_genre",
              "key": "genre_key", "label": "genre", "alias": "g", "bridge_alias": "bg"},
    "Tag (top 25)": {"kind": "bridge", "bridge": "bridge_novel_tag", "dim": "dim_tag",
                     "key": "tag_key", "label": "tag", "alias": "t", "bridge_alias": "bt",
                     "top": 25},
    "Country of origin": {"kind": "column", "col": "type_origin", "sql": "n.type_origin", "table": "n"},
    "Format": {"kind": "column", "col": "type", "sql": "n.type", "table": "n"},
    "Status": {"kind": "column", "col": "status", "table": "n",
               "sql": "CASE WHEN n.is_completed = 1 THEN 'Completed' ELSE 'Ongoing' END"},
    "Decade": {"kind": "column", "col": "decade", "sql": "f.decade", "ordinal": True},
    "Year": {"kind": "column", "col": "year", "sql": "f.year", "ordinal": True},
    "Length": {"kind": "column", "col": "length_bucket", "sql": "n.length_bucket",
               "table": "n", "ordinal": True},
    "Release pace": {"kind": "column", "col": "release_pace", "sql": "n.release_pace",
                     "table": "n", "ordinal": True},
    "Original publisher (top 15)": {"kind": "column", "col": "publisher", "sql": "p.publisher",
                                    "table": "p", "top": 15},
}

MEASURES: dict[str, dict] = {
    "Titles": {"col": "series_id", "agg": "nunique", "sql": "COUNT(DISTINCT f.series_id)",
               "additive": True, "fmt": ",.0f"},
    "Total votes": {"col": "rating_votes", "agg": "sum", "sql": "SUM(f.rating_votes)",
                    "additive": True, "fmt": ",.0f"},
    "Total reading-list adds": {"col": "reading_list_count", "agg": "sum",
                                "sql": "SUM(f.reading_list_count)", "additive": True, "fmt": ",.0f"},
    "Average weighted rating": {"col": "weighted_rating", "agg": "mean",
                                "sql": "ROUND(AVG(f.weighted_rating), 3)", "additive": False,
                                "fmt": ".3f"},
    "Average chapters": {"col": "chapter_count", "agg": "mean",
                         "sql": "ROUND(AVG(f.chapter_count), 1)", "additive": False, "fmt": ",.1f"},
    "Median days between chapters": {"col": "days_per_chapter", "agg": "median",
                                     "sql": "PERCENTILE_APPROX(f.days_per_chapter, 0.5)",
                                     "additive": False, "fmt": ",.1f"},
}


def load_star() -> pd.DataFrame:
    return _load_star(lake_version())


@st.cache_data(show_spinner=False, max_entries=2)
def _load_star(version: str) -> pd.DataFrame:
    """Fact table joined to its single-valued dimensions: one row per title."""
    fact = load_lake("model/fact_novel")
    novel = load_lake("model/dim_novel")[
        ["series_id", "title", "type", "type_origin", "is_completed", "length_bucket", "release_pace"]
    ]
    publisher = load_lake("model/dim_publisher")[["publisher_key", "publisher"]]
    star = fact.merge(novel, on="series_id", how="left").merge(publisher, on="publisher_key", how="left")
    star["status"] = np.where(star["is_completed"] == 1, "Completed", "Ongoing")
    # publisher_key is ranked by title count, so key <= N is exactly "top N".
    star["publisher_rank"] = star["publisher_key"]
    return star


def load_bridge(name: str) -> pd.DataFrame:
    return _load_bridge(name, lake_version())


@st.cache_data(show_spinner=False, max_entries=8)
def _load_bridge(name: str, version: str) -> pd.DataFrame:
    """Bridge table resolved to labels: (series_id, label), one row per pair."""
    spec = DIMENSIONS[name]
    bridge = load_lake(f"model/{spec['bridge']}")
    dim = load_lake(f"model/{spec['dim']}")
    if "top" in spec:
        dim = dim[dim[spec["key"]] <= spec["top"]]
    return bridge.merge(dim[[spec["key"], spec["label"]]], on=spec["key"])[
        ["series_id", spec["label"]]
    ].rename(columns={spec["label"]: name})


def dimension_frame(star: pd.DataFrame, name: str) -> pd.DataFrame:
    """(series_id, <name>) for the titles in `star`, exploded for bridge dimensions."""
    spec = DIMENSIONS[name]
    if spec["kind"] == "bridge":
        return load_bridge(name).merge(star[["series_id"]], on="series_id")
    frame = star[["series_id", spec["col"]]].rename(columns={spec["col"]: name})
    if "top" in spec:
        frame = frame[star["publisher_rank"] <= spec["top"]]
    frame = frame.dropna(subset=[name])
    if spec.get("ordinal") and name in ("Decade", "Year"):
        frame[name] = frame[name].astype(int)
    return frame


def sql_literal(value) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def build_sql(rows: str, cols: str | None, measure: str, filters: dict) -> str:
    """The Spark SQL that computes the current view over the star schema tables."""
    dims = [d for d in (rows, cols) if d]
    joins = ["FROM        fact_novel f"]
    needs_n = any(DIMENSIONS[d].get("table") == "n" for d in dims) or any(
        filters.get(k) for k in ("origin", "format", "status")
    )
    if needs_n:
        joins.append("JOIN        dim_novel n       ON n.series_id = f.series_id")
    if any(DIMENSIONS[d].get("table") == "p" for d in dims):
        joins.append("JOIN        dim_publisher p   ON p.publisher_key = f.publisher_key")
    for d in dims:
        spec = DIMENSIONS[d]
        if spec["kind"] == "bridge":
            b, a = spec["bridge_alias"], spec["alias"]
            joins.append(f"JOIN        {spec['bridge']} {b} ON {b}.series_id = f.series_id")
            joins.append(f"JOIN        {spec['dim']} {a} ON {a}.{spec['key']} = {b}.{spec['key']}")

    def select_expr(d: str) -> str:
        spec = DIMENSIONS[d]
        return f"{spec['alias']}.{spec['label']}" if spec["kind"] == "bridge" else spec["sql"]

    where = []
    if filters.get("origin"):
        where.append("n.type_origin IN (" + ", ".join(map(sql_literal, filters["origin"])) + ")")
    if filters.get("format"):
        where.append("n.type IN (" + ", ".join(map(sql_literal, filters["format"])) + ")")
    if filters.get("status"):
        where.append("n.is_completed = 1" if filters["status"] == "Completed"
                     else "COALESCE(n.is_completed, 0) = 0")
    if filters.get("years"):
        lo, hi = filters["years"]
        where.append(f"f.year BETWEEN {lo} AND {hi}")
    if filters.get("reliable"):
        where.append("f.is_rated_reliable")
    if filters.get("decade") is not None:
        where.append(f"f.decade = {filters['decade']}")
    for d in dims:
        spec = DIMENSIONS[d]
        if spec["kind"] == "bridge" and "top" in spec:
            where.append(f"{spec['alias']}.{spec['key']} <= {spec['top']}")
        elif spec.get("table") == "p" and "top" in spec:
            where.append(f"f.publisher_key <= {spec['top']}")
        elif spec["kind"] == "column":
            where.append(f"{spec['sql']} IS NOT NULL")

    m = MEASURES[measure]
    select = [f"{select_expr(d):<40} AS {d.split(' (')[0].lower().replace(' ', '_')}" for d in dims]
    select.append(f"{m['sql']:<40} AS measure")
    select.append(f"{'COUNT(DISTINCT f.series_id)':<40} AS n_titles")
    lines = ["SELECT      " + ",\n            ".join(select), *joins]
    if where:
        lines.append("WHERE       " + "\n  AND       ".join(where))
    lines.append("GROUP BY    " + ", ".join(select_expr(d) for d in dims))
    if not m["additive"]:
        lines.append(f"HAVING      COUNT(f.{m['col']}) >= {MIN_GROUP}")
    lines.append("ORDER BY    measure DESC")
    return "\n".join(lines)


def aggregate(frame: pd.DataFrame, star: pd.DataFrame, dims: list[str], measure: str) -> pd.DataFrame:
    """Group `frame` (series_id + dims) by dims and compute `measure` from the fact columns."""
    m = MEASURES[measure]
    data = frame.merge(star[["series_id", m["col"]]] if m["col"] != "series_id" else star[["series_id"]],
                       on="series_id")
    grouped = data.groupby(dims, observed=True)
    out = grouped[m["col"]].agg(m["agg"]).rename("measure").to_frame()
    out["n_titles"] = grouped["series_id"].nunique()
    if not m["additive"]:
        out = out[grouped[m["col"]].count() >= MIN_GROUP]
    return out.reset_index()


def fmt_value(value: float, measure: str) -> str:
    return format(value, MEASURES[measure]["fmt"])


def olap_operations(rows: str, cols: str | None, filters: dict, drilled: bool) -> list[str]:
    """Name the OLAP operations the current controls perform, in plain words."""
    ops = []
    fixed = []
    if filters.get("origin"):
        fixed.append("origin in " + ", ".join(filters["origin"]))
    if filters.get("format"):
        fixed.append("format in " + ", ".join(filters["format"]))
    if filters.get("status"):
        fixed.append(f"status = {filters['status']}")
    if filters.get("years"):
        fixed.append(f"year {filters['years'][0]} to {filters['years'][1]}")
    if filters.get("reliable"):
        fixed.append("10+ votes only")
    if len(fixed) == 1:
        ops.append(f"**Slice** ({fixed[0]})")
    elif fixed:
        ops.append("**Dice** (" + "; ".join(fixed) + ")")
    if rows == "Decade" or cols == "Decade":
        ops.append("**Roll-up** (year to decade)")
    if drilled:
        ops.append("**Drill-down** (decade to year)")
    if cols:
        ops.append(f"**Pivot** ({rows} by {cols})")
    return ops


#: Power BI model-view layout, (x, y, width, role) in SVG units: single-valued
#: dimensions left, fact centre, bridges and many-valued dimensions right.
SCHEMA_LAYOUT = {
    "dim_novel": (20, 24, 220, "dim"),
    "dim_time": (20, 300, 220, "dim"),
    "dim_publisher": (20, 452, 220, "dim"),
    "fact_novel": (345, 110, 245, "fact"),
    "bridge_novel_genre": (690, 46, 200, "bridge"),
    "bridge_novel_tag": (690, 236, 200, "bridge"),
    "bridge_novel_author": (690, 426, 200, "bridge"),
    "dim_genre": (985, 24, 215, "dim"),
    "dim_tag": (985, 214, 215, "dim"),
    "dim_author": (985, 404, 215, "dim"),
}
#: (one side, its key, many side, its key, cardinality at the many end, filters both ways)
SCHEMA_RELATIONS = [
    ("dim_novel", "series_id", "fact_novel", "series_id", "1", False),
    ("dim_time", "year", "fact_novel", "year", "*", False),
    ("dim_publisher", "publisher_key", "fact_novel", "publisher_key", "*", False),
    ("fact_novel", "series_id", "bridge_novel_genre", "series_id", "*", True),
    ("fact_novel", "series_id", "bridge_novel_tag", "series_id", "*", True),
    ("fact_novel", "series_id", "bridge_novel_author", "series_id", "*", True),
    ("dim_genre", "genre_key", "bridge_novel_genre", "genre_key", "*", False),
    ("dim_tag", "tag_key", "bridge_novel_tag", "tag_key", "*", False),
    ("dim_author", "author_key", "bridge_novel_author", "author_key", "*", False),
]
SCHEMA_PK = {"dim_novel": "series_id", "dim_time": "year", "dim_publisher": "publisher_key",
             "fact_novel": "series_id", "dim_genre": "genre_key", "dim_tag": "tag_key",
             "dim_author": "author_key"}
#: Same split as the measures table under the diagram.
SUMMABLE = {"chapter_count", "rating_votes", "reading_list_count"}
SCHEMA_HEADER, SCHEMA_ROW, SCHEMA_MAX_ROWS = 46, 21, 7


def _schema_rows(name: str, columns: list[str]) -> list[tuple[str, str]]:
    """(column, marker) in model-view order: keys first, then the rest."""
    keys = {r[1] for r in SCHEMA_RELATIONS if r[0] == name} | {r[3] for r in SCHEMA_RELATIONS if r[2] == name}

    def marker(col: str) -> str:
        if SCHEMA_PK.get(name) == col:
            return "PK"
        if col in keys:
            return "FK"
        return "Σ" if name == "fact_novel" and col in SUMMABLE else ""

    ordered = sorted(columns, key=lambda c: (marker(c) not in ("PK", "FK"), columns.index(c)))
    return [(c, marker(c)) for c in ordered]


def star_schema_svg(tables: dict[str, list[str]], counts: dict[str, int]) -> str:
    """Star schema as inline SVG in Power BI's model-view style: table cards with
    PK / FK / Σ markers, cardinality at each line end, filter direction mid-line."""
    surface, line, ink, muted = SURFACE, "#8A909C", INK, MUTED
    header_fill = {"fact": "#E6ECF6", "dim": "#F4F0E8", "bridge": "#F8F6F1"}
    border = {"fact": BRAND, "dim": "#D9D2C5", "bridge": "#C9C0B0"}
    anchors: dict[tuple[str, str], float] = {}
    boxes: dict[str, tuple[float, float, float, float]] = {}
    cards = []

    for name, (x, y, w, role) in SCHEMA_LAYOUT.items():
        rows = _schema_rows(name, tables.get(name, []))
        limit = len(rows) if role == "fact" else SCHEMA_MAX_ROWS
        shown, hidden = rows[:limit], len(rows) - min(limit, len(rows))
        h = SCHEMA_HEADER + SCHEMA_ROW * (len(shown) + (1 if hidden else 0)) + 10
        boxes[name] = (x, y, w, h)
        dash = ' stroke-dasharray="5 4"' if role == "bridge" else ""
        kind = {"fact": "fact table", "dim": "dimension", "bridge": "bridge table"}[role]
        parts = [
            f"<g><title>{name}: {counts.get(name, 0):,} rows, {len(rows)} columns ({kind})</title>",
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{surface}" '
            f'stroke="{border[role]}" stroke-width="{1.6 if role == "fact" else 1.1}"{dash}/>',
            f'<path d="M{x + 1},{y + SCHEMA_HEADER} V{y + 8} Q{x + 1},{y + 1} {x + 8},{y + 1} '
            f'H{x + w - 8} Q{x + w - 1},{y + 1} {x + w - 1},{y + 8} V{y + SCHEMA_HEADER} Z" '
            f'fill="{header_fill[role]}"/>',
            f'<line x1="{x}" y1="{y + SCHEMA_HEADER}" x2="{x + w}" y2="{y + SCHEMA_HEADER}" '
            f'stroke="{border[role]}" stroke-width="1"/>',
            f'<text x="{x + 12}" y="{y + 20}" fill="{ink}" font-size="13.5" font-weight="700">{name}</text>',
            f'<text x="{x + 12}" y="{y + 37}" fill="{muted}" font-size="11">'
            f"{counts.get(name, 0):,} rows · {kind}</text>",
        ]
        for i, (col, mark) in enumerate(shown):
            cy = y + SCHEMA_HEADER + 8 + SCHEMA_ROW * i + SCHEMA_ROW / 2
            anchors[(name, col)] = cy
            weight = "650" if mark in ("PK", "FK") else "450"
            parts.append(f'<text x="{x + 12}" y="{cy + 4}" fill="{ink if mark else "#3A4253"}" '
                         f'font-size="12" font-weight="{weight}">{col}</text>')
            if mark:
                parts.append(f'<text x="{x + w - 12}" y="{cy + 4}" fill="{muted}" font-size="10.5" '
                             f'font-weight="650" text-anchor="end">{mark}</text>')
        if hidden:
            cy = y + SCHEMA_HEADER + 8 + SCHEMA_ROW * len(shown) + SCHEMA_ROW / 2
            parts.append(f'<text x="{x + 12}" y="{cy + 4}" fill="{muted}" font-size="11">'
                         f'+{hidden} more column{"s" if hidden > 1 else ""}</text>')
        parts.append("</g>")
        cards.append("".join(parts))

    links = []
    for one, one_key, many, many_key, many_card, both in SCHEMA_RELATIONS:
        (ax, ay, aw, _), (bx, by, bw, _) = boxes[one], boxes[many]
        y1 = anchors.get((one, one_key), ay + SCHEMA_HEADER / 2)
        y2 = anchors.get((many, many_key), by + SCHEMA_HEADER / 2)
        rightward = ax < bx
        x1, x2 = (ax + aw, bx) if rightward else (ax, bx + bw)
        bend = max(40.0, abs(x2 - x1) * 0.45) * (1 if rightward else -1)
        c1, c2 = (x1 + bend, y1), (x2 - bend, y2)
        links.append(f'<path d="M{x1},{y1} C{c1[0]},{c1[1]} {c2[0]},{c2[1]} {x2},{y2}" '
                     f'fill="none" stroke="{line}" stroke-width="1.4"/>')
        # Cardinality just outside each card, as in Power BI.
        step = 7 if rightward else -7
        anchor_a, anchor_b = ("start", "end") if rightward else ("end", "start")
        links.append(f'<text x="{x1 + step}" y="{y1 - 6}" fill="{ink}" font-size="12" '
                     f'font-weight="700" text-anchor="{anchor_a}">1</text>')
        links.append(f'<text x="{x2 - step}" y="{y2 - 6}" fill="{ink}" font-size="12" '
                     f'font-weight="700" text-anchor="{anchor_b}">{many_card}</text>')
        # Filter direction at the curve's midpoint, from the "1" side to the many side.
        mx = (x1 + 3 * c1[0] + 3 * c2[0] + x2) / 8
        my = (y1 + 3 * c1[1] + 3 * c2[1] + y2) / 8
        dx = 0.75 * (c1[0] - x1) + 1.5 * (c2[0] - c1[0]) + 0.75 * (x2 - c2[0])
        dy = 0.75 * (c1[1] - y1) + 1.5 * (c2[1] - c1[1]) + 0.75 * (y2 - c2[1])
        angle = float(np.degrees(np.arctan2(dy, dx)))
        for offset, flip in ([(0, 0)] if not both else [(7, 0), (-7, 180)]):
            links.append(
                f'<g transform="translate({mx:.1f},{my:.1f}) rotate({angle:.1f}) translate({offset},0) '
                f'rotate({flip})"><polygon points="-5,-4.5 5,0 -5,4.5" fill="{muted}"/></g>'
            )

    width = max(x + w for x, _, w, _ in boxes.values()) + 20
    height = max(y + h for _, y, _, h in boxes.values()) + 20
    font = FONT_STACK.replace('"', "&quot;")
    return (
        '<div style="overflow-x:auto;padding-bottom:.3rem">'
        f'<svg viewBox="0 0 {width:.0f} {height:.0f}" role="img" '
        'aria-label="Star schema: fact_novel joined to dim_novel, dim_time and dim_publisher, and '
        'through bridge tables to dim_genre, dim_tag and dim_author" '
        f'style="width:100%;min-width:900px;height:auto;font-family:{font}">'
        + "".join(links) + "".join(cards) + "</svg></div>"
    )


def page_warehouse() -> None:
    hero(
        "Data warehouse",
        "The cleaned catalogue as a star schema. Ask it a question by choosing what to break "
        "down by and what to measure; the model behind it is one tab over.",
    )

    with st.spinner("Loading the star schema tables…"):
        star = load_star()
        counts = {
            name: table_profile(f"model/{name}")[0]
            for name in ("fact_novel", "dim_novel", "dim_time", "dim_publisher", "dim_genre",
                         "dim_tag", "dim_author", "bridge_novel_genre", "bridge_novel_tag",
                         "bridge_novel_author")
        }

    tab_query, tab_model = st.tabs(["Ask the warehouse", "How it is modelled"])
    # The model tab is drawn first: the query tab returns early on empty results.
    with tab_model:
        columns = {name: list(load_lake(f"model/{name}").columns) for name in counts}
        st.markdown(star_schema_svg(columns, counts), unsafe_allow_html=True)
        st.caption("Read it like Power BI's model view: 1 and * mark each side of a relationship, the "
                   "arrow is the filter direction, and a double arrow means filters cross both ways. "
                   "The fact-to-bridge links need that, or a genre, tag or author filter could not "
                   "reach the fact table. Σ marks the summable measures; dashed cards are bridge tables.")
        notes, measures = st.columns([2, 3], gap="large")
        with notes:
            st.markdown("**Why this shape**")
            st.markdown(
                f"The fact table holds **one row per title** ({counts['fact_novel']:,} rows). "
                "Genres, tags and authors are many-valued, so they hang off bridge tables "
                "instead of being copied into the fact. A title with five genres keeps its "
                "vote count once, not five times. Single-valued dimensions sit on the left, "
                "many-valued ones reach the fact through a bridge on the right."
            )
        with measures:
            st.dataframe(
                pd.DataFrame([
                    ("chapter_count", "Additive", "SUM, AVG"),
                    ("rating_votes", "Additive", "SUM"),
                    ("reading_list_count", "Additive", "SUM"),
                    ("weighted_rating", "Non-additive", "AVG only"),
                    ("rating_score", "Non-additive", "AVG only"),
                    ("days_per_chapter", "Non-additive", "AVG, median"),
                ], columns=["Measure in fact_novel", "Type", "Valid aggregation"]),
                hide_index=True, use_container_width=True,
            )

    with tab_query:

        c1, c2, c3 = st.columns(3)
        rows = c1.selectbox("Break down by", list(DIMENSIONS), index=0)
        col_options = ["Nothing (single breakdown)"] + [d for d in DIMENSIONS if d != rows]
        cols_pick = c2.selectbox("Compare across", col_options, index=0)
        cols = None if cols_pick == col_options[0] else cols_pick
        measure = c3.selectbox("Measure", list(MEASURES), index=3)

        origins = star["type_origin"].value_counts().index.tolist()
        formats = star["type"].value_counts().index.tolist()
        with st.expander("Slice and dice filters", expanded=False):
            f1, f2, f3 = st.columns(3)
            pick_origin = f1.multiselect("Country of origin", origins, placeholder="All origins")
            pick_format = f2.multiselect("Format", formats, placeholder="All formats")
            pick_status = f3.radio("Status", ["All", "Completed", "Ongoing"], horizontal=True)
            g1, g2 = st.columns([1, 2])
            use_years = g1.checkbox("Limit release year", help="Titles with no known year drop out when this is on.")
            valid_years = star["year"].dropna().astype(int)
            years = None
            if use_years:
                years = g2.slider("Release year", int(valid_years.min()), int(valid_years.max()),
                                  (2015, int(valid_years.max())))
            reliable = g1.checkbox("Only titles with 10+ votes")

        filters = {
            "origin": pick_origin, "format": pick_format,
            "status": None if pick_status == "All" else pick_status,
            "years": years, "reliable": reliable,
        }
        view = star
        if pick_origin:
            view = view[view["type_origin"].isin(pick_origin)]
        if pick_format:
            view = view[view["type"].isin(pick_format)]
        if filters["status"]:
            view = view[view["status"] == filters["status"]]
        if years:
            view = view[view["year"].between(*years)]
        if reliable:
            view = view[view["is_rated_reliable"] == True]  # noqa: E712

        if cols and DIMENSIONS[rows]["kind"] == "bridge" and DIMENSIONS[cols]["kind"] == "bridge":
            st.info("Genre and tag are both many-valued. Compare one of them against a "
                    "single-valued dimension such as origin or decade.")
            return
        if view.empty:
            st.info("No titles match these filters. Clear a filter or widen the year range.")
            return

        frame = dimension_frame(view, rows)
        if cols:
            frame = frame.merge(dimension_frame(view, cols), on="series_id")
        dims = [rows] + ([cols] if cols else [])
        result = aggregate(frame, view, dims, measure)
        if result.empty:
            st.info(f"No group reaches {MIN_GROUP} titles with a value for this measure. "
                    "Loosen the filters or choose a coarser breakdown.")
            return

        ordinal = DIMENSIONS[rows].get("ordinal", False)
        drill = None
        if rows == "Decade" and not cols:
            decades = sorted(result["Decade"].unique().tolist(), reverse=True)
            drill_pick = st.selectbox(
                "Drill down into a decade", ["Stay at decade level"] + [str(d) for d in decades]
            )
            drill = None if drill_pick == "Stay at decade level" else int(drill_pick)

        ops = olap_operations(rows, cols, filters, drill is not None)
        # .at keeps each column's dtype; a row Series would turn decade 1970 into 1970.0.
        best_idx = result["measure"].idxmax()
        label = " / ".join(str(result.at[best_idx, d]) for d in dims)
        st.markdown(
            f"#### Highest {measure.lower()}: {label} "
            f"({fmt_value(result.at[best_idx, 'measure'], measure)})"
        )
        st.caption(
            f"{view['series_id'].nunique():,} titles in view"
            + (" · operations: " + ", ".join(ops) if ops else "")
        )

        if cols:
            pivot = result.pivot(index=rows, columns=cols, values="measure")
            if ordinal:
                pivot = pivot.sort_index()
            else:
                pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
            if DIMENSIONS[cols].get("ordinal"):
                pivot = pivot[sorted(pivot.columns)]
            pivot.columns = [str(c) for c in pivot.columns]
            pivot.index = [str(i) for i in pivot.index]
            text_fmt = ".2f" if measure == "Average weighted rating" else ".3s"
            fig = px.imshow(pivot, color_continuous_scale=SCALE, aspect="auto", text_auto=text_fmt,
                            labels=dict(x=cols, y="", color=measure))
            st.plotly_chart(style_fig(fig, max(360, 28 * len(pivot))), use_container_width=True)
            st.caption("Empty cells: the combination does not exist or, for averages, has fewer "
                       f"than {MIN_GROUP} titles.")
        else:
            plot = result.copy()
            plot[rows] = plot[rows].astype(str)
            top_label = str(result.at[best_idx, rows])
            plot["lead"] = np.where(plot[rows] == top_label, "Highest", "Other")
            # Averages differ by tenths, so they get dots on a zoomed axis (see
            # dot_rating_chart); additive measures keep bars that start at zero.
            chart = px.bar if MEASURES[measure]["additive"] else px.scatter
            if ordinal:
                plot = plot.sort_values(rows)
                fig = chart(plot, x=rows, y="measure", color="lead",
                             color_discrete_map={"Highest": ACCENT, "Other": BRAND},
                             hover_data={"n_titles": ":,", "lead": False},
                             labels={"measure": measure, rows: "", "n_titles": "Titles"})
                height = 380
            else:
                plot = plot.sort_values("measure")
                fig = chart(plot, x="measure", y=rows, color="lead",
                            color_discrete_map={"Highest": ACCENT, "Other": BRAND},
                            hover_data={"n_titles": ":,", "lead": False},
                            labels={"measure": measure, rows: "", "n_titles": "Titles"},
                            **({"orientation": "h"} if MEASURES[measure]["additive"] else {}))
                height = max(340, 24 * len(plot))
            fig.update_layout(showlegend=False)
            if not MEASURES[measure]["additive"]:
                fig.update_traces(marker=dict(size=12, line=dict(width=2, color=SURFACE)))
                lo, hi = plot["measure"].min(), plot["measure"].max()
                pad = max(hi - lo, 0.1) * 0.1
                (fig.update_yaxes if ordinal else fig.update_xaxes)(range=[lo - pad, hi + pad])
            st.plotly_chart(style_fig(fig, height), use_container_width=True)

        multi = [d for d in dims if DIMENSIONS[d]["kind"] == "bridge"]
        if multi and MEASURES[measure]["additive"]:
            total = result["measure"].sum()
            distinct = frame["series_id"].nunique()
            st.warning(
                f"Summing this column gives {fmt_value(total, measure)}, but a title with several "
                f"{multi[0].split(' (')[0].lower()}s is counted once in each. That is why the fact "
                f"table keeps one row per title: the view itself holds {distinct:,} distinct titles. "
                "Read the bars as reach per group, not as shares of one total."
            )

        if drill is not None:
            st.markdown(f"**Drill-down: {drill}s by year**")
            in_decade = view[view["decade"] == drill]
            yearly = aggregate(dimension_frame(in_decade, "Year"), in_decade, ["Year"], measure)
            if yearly.empty:
                st.info(f"No year in the {drill}s reaches {MIN_GROUP} titles for this measure.")
            else:
                yearly["Year"] = yearly["Year"].astype(str)
                fig = px.bar(yearly, x="Year", y="measure", hover_data={"n_titles": ":,"},
                             labels={"measure": measure, "Year": "", "n_titles": "Titles"})
                fig.update_traces(marker_color=BRAND)
                st.plotly_chart(style_fig(fig, 320), use_container_width=True)

        table, sql = st.tabs(["Result table", "SQL for this view"])
        with table:
            shown = result.sort_values(dims if ordinal else "measure", ascending=ordinal)
            if measure == "Titles":
                shown = shown.drop(columns="n_titles")
            shown = shown.rename(columns={"measure": measure, "n_titles": "Titles"})
            st.dataframe(shown, hide_index=True, use_container_width=True, height=320, column_config={
                measure: st.column_config.NumberColumn(format="%.3f" if not MEASURES[measure]["additive"] else "%d"),
                "Titles": st.column_config.NumberColumn(format="%d")})
            st.download_button(
                "Download this result as CSV", shown.to_csv(index=False).encode("utf-8"),
                file_name=f"warehouse_{rows}_{measure}.csv".replace(" ", "_").lower(),
                mime="text/csv",
            )
        with sql:
            st.caption("The same aggregation written against the star schema tables in "
                       "curated-data/model. The dashboard computes it from those Parquet files "
                       "with pandas; the pipeline runs this form in Spark SQL.")
            st.code(build_sql(rows, cols, measure, filters), language="sql")


@st.cache_data(ttl=900, show_spinner=False)
def genre_graph_layout(edges: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Fixed positions so the map does not jump when the edge filter changes."""
    import networkx as nx

    graph = nx.Graph()
    for row in edges.itertuples(index=False):
        # Distance = 1 - Jaccard: genres that share many titles sit close together.
        graph.add_edge(row.source, row.target, dist=1.0 - row.jaccard)
    return {k: (float(v[0]), float(v[1])) for k, v in
            nx.kamada_kawai_layout(graph, weight="dist").items()}


NODE_NAMES = {"Novel": "titles", "Author": "authors", "Tag": "tags", "Genre": "genres"}
EDGE_NAMES = {"HAS_TAG": "title-tag", "HAS_GENRE": "title-genre", "WRITTEN_BY": "title-author"}


def page_graph() -> None:
    hero(
        "Relationship graph",
        "Titles, genres, tags and authors as one property graph. The questions here are "
        "about connections, which the star schema cannot answer with a GROUP BY.",
    )

    import networkx as nx

    with st.spinner("Loading graph tables…"):
        n_nodes, node_counts = table_profile("model/graph_nodes", "node_type")
        n_edges, edge_counts = table_profile("model/graph_edges", "rel_type")
        summary = load_table("graph_summary").set_index("label")
        pagerank = load_table("graph_genre_pagerank")
        genre_dim = load_lake("model/dim_genre")[["genre", "n_novels"]]
        try:
            genre_edges = load_table("graph_genre_edges")
        except DataUnavailable:
            genre_edges = None

    st.markdown(facts_html([
        ("Nodes", f"{n_nodes:,}: " + ", ".join(f"{v:,} {NODE_NAMES.get(k, k)}" for k, v in node_counts.items())),
        ("Edges", f"{n_edges:,}: " + ", ".join(f"{v:,} {EDGE_NAMES.get(k, k)}" for k, v in edge_counts.items())),
        ("Stored as", "graph_nodes / graph_edges, ready for Neo4j or GraphFrames"),
    ]), unsafe_allow_html=True)
    genre_s, author_s = summary.loc["genre_cooccurrence"], summary.loc["author_collaboration"]
    compare = pd.DataFrame({
        "Genres": [genre_s["n_nodes"], genre_s["n_edges"], genre_s["n_components"],
                                genre_s["largest_component_size"], genre_s["n_communities"],
                                genre_s["modularity"]],
        "Authors": [author_s["n_nodes"], author_s["n_edges"], author_s["n_components"],
                                 author_s["largest_component_size"], author_s["n_communities"],
                                 author_s["modularity"]],
    }, index=["Nodes", "Edges", "Connected components", "Largest component",
              "Communities", "Modularity"])
    tab_map, tab_route, tab_auth, tab_shape = st.tabs(
        ["Genre map", "Route between genres", "Author teams", "Shape of the graph"])
    # Tabs without an early return are drawn first; the map tab stops when its edges are missing.
    with tab_shape:
        left, right = st.columns([2, 3], gap="large")
        with left:
            st.markdown("**Two projections, opposite shapes**")
            st.dataframe(compare.map(lambda v: f"{v:,.4f}" if isinstance(v, float) and v < 1 else f"{v:,.0f}"),
                         use_container_width=True)
        with right:
            st.markdown("**What the numbers say**")
            st.markdown(
                f"- Genres form **one component** with modularity {genre_s['modularity']:.3f}: they "
                "mix freely, so a genre label alone separates titles poorly.\n"
                f"- Authors split into **{int(author_s['n_components']):,} separate teams**, the "
                f"largest only {int(author_s['largest_component_size'])} people, modularity "
                f"{author_s['modularity']:.3f}. Co-writing teams almost never cross over.\n"
                "- The warehouse can count pairs; ranking influence (PageRank), finding teams "
                "(components) and routing between genres (shortest path) need graph algorithms."
            )

    with tab_auth:
        reach = load_lake("model/op_graph_author_reach")
        authors_pr = load_table("graph_author_pagerank")
        col_a, col_b = st.columns(2, gap="large")
        with col_a:
            st.markdown("**Widest genre range** (Author, Novel, Genre: two hops)")
            min_titles = st.slider("Minimum titles per author", 3, 20, 8)
            subset = reach[reach["n_novels"] >= min_titles].nlargest(12, "n_distinct_genres")
            if subset.empty:
                st.info("No author has that many titles. Lower the minimum.")
            else:
                st.dataframe(
                    subset[["author", "n_novels", "n_distinct_genres", "avg_weighted_rating"]].rename(
                        columns={"author": "Author", "n_novels": "Titles",
                                 "n_distinct_genres": "Genres covered", "avg_weighted_rating": "Rating"}),
                    hide_index=True, use_container_width=True, height=380,
                )
        with col_b:
            st.markdown("**Most connected co-authors** (collaboration graph)")
            top_auth = authors_pr.nlargest(12, "degree_centrality")[["node", "degree_centrality", "pagerank"]]
            n_auth = int(author_s["n_nodes"])
            top_auth = top_auth.assign(collaborators=(top_auth["degree_centrality"] * (n_auth - 1)).round().astype(int))
            st.dataframe(
                top_auth[["node", "collaborators", "pagerank"]].rename(
                    columns={"node": "Author", "collaborators": "Co-authors", "pagerank": "PageRank"}),
                hide_index=True, use_container_width=True, height=380,
                column_config={"PageRank": st.column_config.NumberColumn(format="%.5f")},
            )



    with tab_map:
        if genre_edges is None:
            st.info("The genre edge list has not been written yet. Run "
                    "`python -m src.phases.graph_analysis` in the pipeline container.")
            return

        sizes = dict(zip(genre_dim["genre"], genre_dim["n_novels"]))
        genre_edges = genre_edges.assign(
            jaccard=lambda d: d["weight"] / (d["source"].map(sizes) + d["target"].map(sizes) - d["weight"])
        )
        positions = genre_graph_layout(genre_edges)

        ctrl, _ = st.columns([2, 3])
        threshold = ctrl.slider(
            "Show links with Jaccard similarity of at least", 0.05, 0.40, 0.15, 0.01,
            help="Jaccard = titles sharing both genres / titles with either. Raw counts would "
                 "make big genres look linked to everything.",
        )
        shown = genre_edges[genre_edges["jaccard"] >= threshold]
        has_community = "community" in pagerank.columns
        info = pagerank.set_index("node")
        palette = SEQUENCE

        map_col, rank_col = st.columns([3, 2], gap="large")
        with map_col:
            import plotly.graph_objects as go

            fig = go.Figure()
            for lo, hi, width in ((0, 0.2, 0.8), (0.2, 0.3, 1.8), (0.3, 1.01, 3.2)):
                band = shown[(shown["jaccard"] >= lo) & (shown["jaccard"] < hi)]
                xs, ys = [], []
                for row in band.itertuples(index=False):
                    (x0, y0), (x1, y1) = positions[row.source], positions[row.target]
                    xs += [x0, x1, None]
                    ys += [y0, y1, None]
                fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", hoverinfo="skip", showlegend=False,
                                         line=dict(width=width, color="rgba(91,100,117,.30)")))
            names = [n for n in positions]
            pr = info["pagerank"].reindex(names).fillna(0)
            community = info["community"].reindex(names).fillna(-1).astype(int) if has_community else pd.Series(0, index=names)
            fig.add_trace(go.Scatter(
                x=[positions[n][0] for n in names], y=[positions[n][1] for n in names],
                mode="markers+text", text=names, textposition="top center",
                textfont=dict(size=11, color=INK),
                marker=dict(size=(12 + 260 * pr).clip(upper=46),
                            color=[palette[c] if 0 <= c < len(palette) else OTHER for c in community],
                            line=dict(width=1, color=SURFACE)),
                customdata=np.stack([pr.values, community.values, [sizes.get(n, 0) for n in names]], axis=1),
                hovertemplate="<b>%{text}</b><br>PageRank %{customdata[0]:.4f}<br>"
                              "Community %{customdata[1]}<br>%{customdata[2]:,} titles<extra></extra>",
                showlegend=False,
            ))
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
            st.plotly_chart(style_fig(fig, 560), use_container_width=True)
            st.caption(
                f"{len(shown)} of {len(genre_edges)} links shown. Dot size = PageRank, colour = "
                "community found by greedy modularity in graph_analysis.py."
            )
        with rank_col:
            st.markdown("**Most central genres (PageRank)**")
            top = pagerank.nlargest(12, "pagerank").sort_values("pagerank")
            fig = px.bar(top, x="pagerank", y="node", orientation="h",
                         labels={"pagerank": "PageRank", "node": ""})
            fig.update_traces(marker_color=BRAND)
            st.plotly_chart(style_fig(fig, 380), use_container_width=True)
            if has_community:
                groups = pagerank.sort_values("pagerank", ascending=False).groupby("community")["node"]
                for cid, members in groups:
                    st.markdown(
                        # Text stays in ink; the square beside it carries the community colour.
                        f'<span style="display:inline-block;width:.7em;height:.7em;border-radius:2px;'
                        f'margin-right:.35em;background:{palette[int(cid)] if int(cid) < len(palette) else OTHER}">'
                        f'</span><span style="font-weight:600">Community {int(cid)}</span> · ' + ", ".join(members.head(8))
                        + (f" and {len(members) - 8} more" if len(members) > 8 else ""),
                        unsafe_allow_html=True,
                    )

    with tab_route:
        graph = nx.Graph()
        graph.add_nodes_from(positions)
        for row in shown.itertuples(index=False):
            graph.add_edge(row.source, row.target, weight=row.weight, jaccard=row.jaccard)
        genres = sorted(positions)
        a, b = st.columns(2)
        src = a.selectbox("From", genres, index=genres.index("Yaoi") if "Yaoi" in genres else 0)
        dst = b.selectbox("To", genres, index=genres.index("Martial Arts") if "Martial Arts" in genres else 1)
        if src == dst:
            st.info("Pick two different genres.")
        elif not nx.has_path(graph, src, dst):
            st.info(f"No route between {src} and {dst} at Jaccard {threshold:.2f}. Lower the "
                    "threshold on the map to let weaker links through.")
        else:
            path = nx.shortest_path(graph, src, dst)
            hops = [
                f"{u} to {v}: {graph[u][v]['weight']:,} shared titles (Jaccard {graph[u][v]['jaccard']:.2f})"
                for u, v in zip(path, path[1:])
            ]
            st.markdown(f"**{len(path) - 1} hop{'s' if len(path) > 2 else ''}:** " + chips(path, active=True),
                        unsafe_allow_html=True)
            st.markdown("\n".join(f"- {h}" for h in hops))


#: Amber marks genres below the curator threshold; paired with a text column, never colour alone.
REVIEW = ACCENT
#: Genres labelled on the scatter: the ends of the trend and the two outliers. The rest overlap; hover shows them.
LANDMARK_GENRES = {"Romance", "Fantasy", "Action", "Gender Bender", "Sports", "Mecha", "Comedy",
                   "Psychological", "Seinen", "Shoujo Ai"}


@st.cache_data(show_spinner=False, max_entries=2)
def _community_layout(edges: pd.DataFrame, keep: tuple[int, ...]) -> dict[int, tuple[float, float]]:
    """Communities on a circle in Fiedler order, so heavily linked communities sit side by side."""
    import networkx as nx

    graph = nx.Graph()
    graph.add_nodes_from(keep)
    for row in edges.itertuples(index=False):
        graph.add_edge(int(row.lo), int(row.hi), weight=float(row.n_edges))
    order = nx.spectral_ordering(graph, weight="weight", seed=7) if graph.number_of_edges() else list(keep)
    step = 2 * np.pi / len(order)
    return {int(n): (float(np.cos(step * (i + .5))), float(np.sin(step * (i + .5)))) for i, n in enumerate(order)}


def _neighbourhood(edges: pd.DataFrame, sid: int) -> pd.DataFrame:
    """Direct neighbours of one title with cosine similarity, strongest first."""
    hit = edges[(edges["src"] == sid) | (edges["dst"] == sid)]
    other = np.where(hit["src"].to_numpy() == sid, hit["dst"].to_numpy(), hit["src"].to_numpy())
    return (pd.DataFrame({"series_id": other, "similarity": hit["weight"].to_numpy()})
            .sort_values("similarity", ascending=False).reset_index(drop=True))


def page_content_graph() -> None:
    hero(
        "Content graph",
        "Every title linked to its 10 closest titles by synopsis and tags, in the same text space the "
        "recommender uses. The graph shows which genres the tagging model can be trusted with and which "
        "story communities compete for the same readers.",
    )
    try:
        with st.spinner("Loading the content graph…"):
            summary = load_table("graph_similarity_summary").iloc[0]
            homophily = load_table("graph_similarity_homophily")
            communities = load_table("graph_similarity_communities").sort_values("community")
            community_edges = load_table("graph_similarity_community_edges")
            nodes = load_table("graph_similarity_nodes")
            edges = load_table("graph_similarity_edges")
    except DataUnavailable as exc:
        st.info("The content graph has not been built yet." if PUBLIC else
                "The content graph has not been built yet. Run "
                "`python -m src.phases.similarity_graph` in the pipeline container, then reload.")
        technical_details(str(exc))
        return

    st.markdown(facts_html([
        ("Graph", f"{int(summary['n_nodes']):,} titles, {int(summary['n_edges']):,} links"),
        ("Communities", f"{int(summary['n_communities'])} (Louvain), modularity {summary['modularity']:.3f}"),
        ("Built from", "TF-IDF of title, synopsis and tags, reduced to 200 SVD dimensions"),
    ]), unsafe_allow_html=True)

    tab_rel, tab_comm, tab_title, tab_hub = st.tabs(
        ["Genre reliability", "Story communities", "Title neighbourhood", "Hubs"])

    with tab_rel:
        answer(
            "Genres whose titles sit close together in the text space are the genres the model tags well. "
            f"Across {len(homophily)} genres, the share of same-genre neighbours and the model's F1 correlate at "
            f"Spearman ρ = {summary['spearman_f1_neighbour_share']:.3f}."
        )
        ctrl, _ = st.columns([2, 3])
        reference = summary.get("homophily_reference_model", "svm")
        cutoff = ctrl.slider(
            "Ask a curator to confirm genres with F1 below", 0.30, 0.80, 0.50, 0.05,
            help=f"F1 of the {MODEL_NAMES.get(reference, reference)} genre tagger on the 7,932 held-out titles.",
        )
        rel = homophily.assign(handling=np.where(homophily["f1"] < cutoff, "Confirm by curator", "Auto-tag, spot-check"))
        chart_col, list_col = st.columns([3, 2], gap="large")
        with chart_col:
            fig = go.Figure()
            for label, color in (("Auto-tag, spot-check", BRAND), ("Confirm by curator", REVIEW)):
                part = rel[rel["handling"] == label]
                fig.add_trace(go.Scatter(
                    x=part["neighbour_share"], y=part["f1"], mode="markers+text", name=label,
                    text=part["genre"].where(part["genre"].isin(LANDMARK_GENRES), ""),
                    hovertext=part["genre"], textfont=dict(size=11, color=MUTED),
                    textposition=np.where(part["neighbour_share"] < .12, "top right", "top center"),
                    marker=dict(size=8 + 60 * part["prevalence"], color=color, line=dict(width=1, color=SURFACE)),
                    customdata=np.stack([part["n_novels"], part["prevalence"] * 100], axis=1),
                    hovertemplate="<b>%{hovertext}</b><br>Same-genre neighbours %{x:.0%}<br>F1 %{y:.3f}"
                                  "<br>%{customdata[0]:,} titles (%{customdata[1]:.1f}%)<extra></extra>",
                ))
            fig.add_hline(y=cutoff, line=dict(color=REVIEW, width=1, dash="dot"))
            fig.update_layout(legend=dict(orientation="h", y=1.08, x=0))
            fig.update_xaxes(title="Same-genre neighbours", tickformat=".0%")
            fig.update_yaxes(title="F1 of the genre tagger")
            st.plotly_chart(style_fig(fig, 470), use_container_width=True)
            st.caption("Dot size = share of the catalogue carrying the genre. Sports and Gender Bender sit above "
                       "the trend because a very specific tag gives them away.")
        with list_col:
            flagged = rel[rel["handling"] == "Confirm by curator"].sort_values("f1")
            st.markdown(f"**{len(flagged)} genre{' needs' if len(flagged) == 1 else 's need'} a curator** at F1 below {cutoff:.2f}")
            if flagged.empty:
                st.caption("No genre falls below this threshold. Raise it to see the weakest ones.")
            else:
                st.markdown(chips(flagged["genre"]), unsafe_allow_html=True)
            st.dataframe(
                rel.sort_values("f1")[["genre", "f1", "neighbour_share"]].rename(columns={
                    "genre": "Genre", "f1": "F1", "neighbour_share": "Neighbours"}),
                hide_index=True, use_container_width=True, height=330,
                column_config={"F1": st.column_config.NumberColumn(format="%.2f", width="small"),
                               "Neighbours": st.column_config.NumberColumn(
                                   format="%.2f", width="small", help="Share of neighbours with the same genre")},
            )

    with tab_comm:
        answer(
            f"Titles split into {int(summary['n_communities'])} communities along origin and setting rather than "
            f"primary genre: agreement with the primary-genre label is only NMI {summary['nmi_vs_primary_genre']:.3f}. "
            "Titles in one community compete for the same readers."
        )
        top_n = 14
        keep = tuple(int(c) for c in communities.nlargest(top_n, "size")["community"])
        links = community_edges[community_edges["lo"].isin(keep) & community_edges["hi"].isin(keep)]
        links = links[links["n_edges"] >= links["n_edges"].quantile(.5)] if len(links) else links
        pos = _community_layout(links, keep)
        info = communities.set_index("community")
        map_col, detail_col = st.columns([3, 2], gap="large")
        with map_col:
            fig = go.Figure()
            peak = links["n_edges"].max() if len(links) else 1
            for row in links.itertuples(index=False):
                (x0, y0), (x1, y1) = pos[int(row.lo)], pos[int(row.hi)]
                fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines", hoverinfo="skip", showlegend=False,
                                         line=dict(width=.6 + 5 * row.n_edges / peak, color="rgba(91,100,117,.30)")))
            ids = list(pos)
            fig.add_trace(go.Scatter(
                x=[pos[c][0] for c in ids], y=[pos[c][1] for c in ids], mode="markers+text", showlegend=False,
                # Short labels pointing away from the centre stay inside a phone-width chart;
                # the full label is in the hover and the community picker.
                text=[f"K{c} {str(info.loc[c, 'label']).split(' + ')[0]}" for c in ids],
                hovertext=[f"K{c} {info.loc[c, 'label']}" for c in ids],
                textposition=[("top center" if pos[c][1] >= 0 else "bottom center") if abs(pos[c][0]) < .35
                              else ("middle right" if pos[c][0] > 0 else "middle left") for c in ids],
                textfont=dict(size=10.5, color=INK),
                marker=dict(size=[10 + info.loc[c, "size"] / 140 for c in ids], color=BRAND,
                            line=dict(width=1, color=SURFACE)),
                customdata=np.stack([[info.loc[c, "size"] for c in ids], [info.loc[c, "top_origin"] for c in ids],
                                     [info.loc[c, "origin_share"] * 100 for c in ids]], axis=1),
                hovertemplate="<b>%{hovertext}</b><br>%{customdata[0]:,} titles<br>"
                              "%{customdata[1]} %{customdata[2]:.0f}%<extra></extra>",
            ))
            fig.update_xaxes(visible=False, range=[-1.95, 1.95])
            fig.update_yaxes(visible=False, range=[-1.3, 1.3])
            st.plotly_chart(style_fig(fig, 470), use_container_width=True)
            st.caption(f"The {top_n} largest of {int(summary['n_communities'])} communities. Line width = number of "
                       "kNN links between two communities. Labels name the leading genre after Romance, Fantasy "
                       "and Drama, which appear in nearly every community; hover a dot for the second one.")
        with detail_col:
            options = list(info.index)
            cid = st.selectbox(
                "Community", options,
                format_func=lambda c: f"K{c} · {info.loc[c, 'label']} ({info.loc[c, 'top_origin']}, {int(info.loc[c, 'size']):,})",
            )
            row = info.loc[cid]
            st.markdown(facts_html([
                ("Titles", f"{int(row['size']):,}"),
                ("Dominant origin", f"{row['top_origin']} ({row['origin_share']:.0%})"),
                ("Average rating", f"{row['avg_weighted_rating']:.2f}" if pd.notna(row["avg_weighted_rating"]) else "no rated titles"),
            ]), unsafe_allow_html=True)
            st.markdown(chips(str(row["top_genres"]).split(", ")), unsafe_allow_html=True)
            members = nodes[nodes["community"] == cid]
            st.markdown("**Best-known titles**")
            st.dataframe(
                members.nlargest(8, "rating_votes")[["title", "rating_votes", "weighted_rating"]].rename(columns={
                    "title": "Title", "rating_votes": "Votes", "weighted_rating": "Rating"}),
                hide_index=True, use_container_width=True,
                column_config={"Votes": st.column_config.NumberColumn(format="%d"),
                               "Rating": st.column_config.NumberColumn(format="%.2f")},
            )

    with tab_title:
        c1, c2 = st.columns([2, 3], gap="large")
        query = c1.text_input("Search by title", placeholder="Part of a title, e.g. martial god", key="cg_query")
        pool = nodes
        if query.strip():
            pool = nodes[nodes["title"].str.contains(re.escape(query.strip()), case=False, na=False)]
        if pool.empty:
            st.info("No title matches that search. Try a shorter part of the name.")
        else:
            pool = pool.sort_values("rating_votes", ascending=False, na_position="last").head(60)
            labels = {int(r.series_id): str(r.title) for r in pool.itertuples(index=False)}
            sid = c2.selectbox(f"{len(pool)} best-known matches", list(labels), format_func=labels.get, key="cg_title")
            me = nodes.loc[nodes["series_id"] == sid].iloc[0]
            near = _neighbourhood(edges, sid).merge(nodes, on="series_id", how="left")
            rank = (nodes["pagerank"] > me["pagerank"]).mean()
            cinfo = info.loc[int(me["community"])]
            st.markdown(facts_html([
                ("Community", f"K{int(me['community'])} · {html.escape(str(cinfo['label']))}"),
                ("Linked titles", f"{int(me['degree'])} (10 chosen by this title, the rest chose it)"),
                ("Centrality", f"PageRank higher than {1 - rank:.0%} of titles"),
            ]), unsafe_allow_html=True)

            import networkx as nx

            ego = nx.Graph()
            ego.add_node(sid)
            ids = set(near["series_id"]) | {sid}
            for r in near.itertuples(index=False):
                ego.add_edge(sid, int(r.series_id), weight=float(r.similarity))
            among = edges[edges["src"].isin(ids) & edges["dst"].isin(ids)]
            for r in among.itertuples(index=False):
                ego.add_edge(int(r.src), int(r.dst), weight=float(r.weight))
            layout = nx.spring_layout(ego, weight="weight", seed=7)
            names = nodes.set_index("series_id").loc[list(ego.nodes)]
            graph_col, list_col = st.columns([3, 2], gap="large")
            with graph_col:
                fig = go.Figure()
                xs, ys = [], []
                for u, v in ego.edges():
                    xs += [layout[u][0], layout[v][0], None]
                    ys += [layout[u][1], layout[v][1], None]
                fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", hoverinfo="skip", showlegend=False,
                                         line=dict(width=1, color="rgba(91,100,117,.30)")))
                order = list(ego.nodes)
                same = [int(names.loc[n, "community"]) == int(me["community"]) for n in order]
                fig.add_trace(go.Scatter(
                    x=[layout[n][0] for n in order], y=[layout[n][1] for n in order], mode="markers",
                    showlegend=False, text=[str(names.loc[n, "title"]) for n in order],
                    marker=dict(size=[22 if n == sid else 12 for n in order],
                                color=[ACCENT if n == sid else BRAND if s else MUTED for n, s in zip(order, same)],
                                line=dict(width=1, color=SURFACE)),
                    hovertemplate="%{text}<extra></extra>",
                ))
                fig.update_xaxes(visible=False)
                fig.update_yaxes(visible=False)
                st.plotly_chart(style_fig(fig, 420), use_container_width=True)
                st.caption("Teal = this title, purple = neighbour in the same community, grey = neighbour from "
                           "another community. Lines between neighbours are links they share with each other.")
            with list_col:
                st.markdown(f"**{len(near)} closest titles**")
                st.dataframe(
                    near[["title", "similarity", "community", "weighted_rating"]].assign(
                        community=lambda d: "K" + d["community"].astype("Int64").astype(str)).rename(columns={
                        "title": "Title", "similarity": "Similarity", "community": "Community",
                        "weighted_rating": "Rating"}),
                    hide_index=True, use_container_width=True, height=380,
                    column_config={"Title": st.column_config.TextColumn(width="medium"),
                                   "Similarity": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1),
                                   "Rating": st.column_config.NumberColumn(format="%.2f")},
                )

    with tab_hub:
        answer(
            "The most central titles are not the popular ones. PageRank and vote count correlate at only "
            f"ρ = {summary['pagerank_vs_votes_spearman']:.3f}: the centre of the graph holds titles written to a "
            "template, which a plain nearest-neighbour recommender would suggest far too often."
        )
        left, right = st.columns([3, 2], gap="large")
        with left:
            hubs = nodes.nlargest(15, "pagerank").assign(
                community=lambda d: "K" + d["community"].astype(str))
            st.dataframe(
                hubs[["title", "community", "degree", "rating_votes", "pagerank"]].rename(columns={
                    "title": "Title", "community": "Community",
                    "degree": "Linked titles", "rating_votes": "Votes", "pagerank": "PageRank"}),
                hide_index=True, use_container_width=True, height=560,
                column_config={"Title": st.column_config.TextColumn(width="medium"),
                               "Votes": st.column_config.NumberColumn(format="%d"),
                               "PageRank": st.column_config.NumberColumn(format="%.6f")},
            )
        with right:
            fig = px.histogram(nodes, x="degree", nbins=60, log_y=True,
                               labels={"degree": "Linked titles per title"})
            fig.update_traces(marker_color=BRAND)
            fig.update_yaxes(title="Titles (log scale)")
            st.plotly_chart(style_fig(fig, 300), use_container_width=True)
            st.markdown(
                f'<p class="nl-note">Every title chooses 10 neighbours, so no title has fewer than 10 links. '
                f'The median is {summary["degree_median"]:.0f}, but one title is linked to '
                f'{int(summary["degree_max"])} others. Re-ranking recommendations by weighted rating, or keeping '
                "only mutual neighbours, stops these hubs from crowding out better-rated titles.</p>",
                unsafe_allow_html=True,
            )



#: English, publish-safe audit text rebuilt from the numbers in the pipeline's evidence.
#: Security lines keep their status but never name credentials, accounts, endpoints or ports.
def _num(pattern: str, text: str, default: str = "") -> str:
    match = re.search(pattern, text or "")
    return match.group(1) if match else default


def _int(text: str) -> str:
    try:
        return f"{int(str(text).replace(',', '').replace('.', '')):,}"
    except ValueError:
        return str(text)


def _zone(check_id: str) -> str:
    bucket = check_id.split("/", 1)[-1]
    return {"raw-data": "raw", "clean-data": "clean", "curated-data": "curated"}.get(bucket, "storage")


def _cmp01(ev: str) -> str:
    files = re.findall(r"([\w.]+\.csv): md5=\w+\.*, (\d+) baris", ev or "")
    if not files:
        return "Every source file carries a checksum and ingestion time."
    listed = ", ".join(f"{name} ({_int(rows)} rows)" for name, rows in files)
    noun = "source file carries" if len(files) == 1 else "source files carry"
    return f"{len(files)} {noun} a checksum, row count and ingestion time: {listed}."


def _cmp03(ev: str) -> str:
    pairs = re.findall(r"([\w-]+)=(\w+)", ev or "")
    on = [_zone("x/" + b) for b, state in pairs if state.lower() == "enabled"]
    off = [_zone("x/" + b) for b, state in pairs if state.lower() != "enabled"]
    if pairs and not off:
        names = on[0] if len(on) == 1 else ", ".join(on[:-1]) + " and " + on[-1]
        return f"Versioning is on for the {names} zones."
    return f"Versioning on: {', '.join(on) or 'none'}; off: {', '.join(off) or 'none'}."


AUDIT_PUBLIC: dict[str, tuple[str, object, str]] = {
    # check_id: (name, evidence(ev, status) -> str, follow-up shown on the page)
    "SEC-01": ("Storage credentials changed from the defaults",
               lambda ev, s: "Every storage account uses its own long secret, none of them a default "
               "or example value." if s == "LULUS" else
               "The storage still runs with its default credentials. That is acceptable only while it "
               "cannot be reached from any network.", ""),
    "SEC-02": ("Encryption in transit (TLS)",
               lambda ev, s: ("Storage traffic is encrypted: " + _num(r"^(TLSv[\d.]+)", ev, "TLS")
                              + ", with the server certificate verified against the stack's own CA.")
               if s == "LULUS" else
               "Storage traffic is not encrypted. It stays on a private container network, which is "
               "acceptable only while that traffic never leaves the machine.", ""),
    "SEC-03": ("Storage zone closed to anonymous access",
               lambda ev, s: "No anonymous access policy." if s == "LULUS" else "Anonymous access is possible.", ""),
    "SEC-04": ("Separate read and write permissions",
               lambda ev, s: "This dashboard reads through a read-only account; only the pipeline can write.", ""),
    "SEC-05": ("Services isolated on a private network",
               lambda ev, s: "All services share one private container network; only user interfaces are "
               "published to the host.", ""),
    "CMP-01": ("Provenance recorded (checksum and time)", lambda ev, s: _cmp01(ev), ""),
    "CMP-02": ("Raw zone never overwritten by transformations",
               lambda ev, s: "The raw zone holds only registered sources and their manifests ("
               + _num(r"(\d+) objek", ev, "?") + " objects), each still the size recorded at ingestion.", ""),
    "CMP-03": ("Bucket versioning (recovery and audit)", lambda ev, s: _cmp03(ev), ""),
    "CMP-04": ("Data source documented",
               lambda ev, s: "The data comes from the public NovelUpdates catalogue: work metadata plus scraped "
               "synopses. The metadata is factual; synopses remain the copyright of their owners and are used "
               "here for non-commercial analytics only.",
               "Redistribution rights must be confirmed before the dataset itself is republished."),
    "PRV-01": ("Direct PII scan (email and phone)",
               lambda ev, s: "No match in any text column of the source files." if s == "LULUS" else
               "Possible matches found; they are reviewed before release.", ""),
    "PRV-02": ("Identity-like number patterns (manual review)",
               lambda ev, s: "None found." if (ev or "").strip() in ("tidak ada", "") else
               "Some number runs look like identifiers; in prose they are usually ordinary numbers.", ""),
    "PRV-03": ("No person-level data",
               lambda ev, s: "No column has one row per reader. Reader signals exist only as aggregates "
               "(vote counts, reading-list counts), so no individual can be re-identified.", ""),
    "PRV-04": ("Personal names that are processed",
               lambda ev, s: "The authors column holds pen names the platform publishes as credits, not private "
               "identities. The artists column is dropped during cleaning (" + _num(r"([\d,]+%)", ev, "mostly").replace(",", ".")
               + " Unknown).",
               "Pen names can still be personal data when linked to a real person; the report states the basis "
               "for processing them."),
    "QLT-01": ("Clean catalogue keeps every source row (at least 99.9%)",
               lambda ev, s: _num(r"^([\d,]+) dari", ev, "?") + " of " + _num(r"dari ([\d,]+)", ev, "?")
               + " source rows kept (" + _num(r"\(([\d.]+%)\)", ev, "?") + "); only duplicate series are removed. "
               + _int(_num(r"full=([\d,]+)", ev, "0")) + " titles have full metadata, "
               + _int(_num(r"listing=([\d,]+)", ev, "0")) + " listing-only.", ""),
    "QLT-01b": ("Listing-only titles are flagged, not hidden",
                lambda ev, s: _int(_num(r"^([\d,]+)", ev, "0")) + " titles carry only text and genres; their "
                "rating, chapters, year and popularity are empty in every row (checked, not assumed). Metrics "
                "from the detail pages filter them out.", ""),
    "QLT-02": ("series_id is unique",
               lambda ev, s: _num(r"^([\d,]+)", ev, "?") + " unique values across "
               + _num(r"dari ([\d,]+)", ev, "?") + " rows.", ""),
    "QLT-03": ("Placeholder values turned into empty values",
               lambda ev, s: "No placeholder value (0 chapters, \"N/A\" and the like) is left in the clean zone."
               if s == "LULUS" else "Some placeholder values remain in the clean zone.", ""),
    "QLT-04": ("Synopsis coverage of the full-metadata core ≥ 98%",
               lambda ev, s: _num(r"^([\d.]+%)", ev, "?") + " (" + _num(r"\(([\d,/]+)\)", ev, "?")
               + ") of full-metadata titles have a synopsis.", ""),
    "QLT-04b": ("Synopsis coverage of listing-only titles",
                lambda ev, s: _num(r"^([\d.]+%)", ev, "?") + " (" + _num(r"\(([\d,/]+)\)", ev, "?") + ").", ""),
    "QLT-05": ("Reliable-rating population reported",
               lambda ev, s: _num(r"^([\d,]+)", ev, "?") + " titles have 10+ votes out of "
               + _num(r"dari ([\d,]+)", ev, "?") + ". Rating averages use this population or the weighted "
               "rating, never the raw mean.", ""),
    "QLT-06": ("Ratings within the valid range (0 to 5)",
               lambda ev, s: "All values are in range." if s == "LULUS" else "Some ratings fall outside 0 to 5.",
               ""),
}


def public_audit(audit: pd.DataFrame) -> pd.DataFrame:
    """English, publish-safe name / evidence / follow-up for every audit row."""
    def one(row) -> tuple[str, str, str]:
        base = row.check_id.split("/", 1)[0]
        spec = AUDIT_PUBLIC.get(row.check_id) or AUDIT_PUBLIC.get(base)
        if spec is None:
            # A check added to the pipeline later: show its id, never its raw text.
            return row.check_id, "Details are in the pipeline's audit log.", ""
        name, evidence, follow = spec
        if base == "SEC-03":
            name = f"{_zone(row.check_id).capitalize()} zone closed to anonymous access"
        return name, evidence(row.evidence, row.status), follow
    text = [one(r) for r in audit.itertuples(index=False)]
    return audit.assign(name_en=[t[0] for t in text], evidence_en=[t[1] for t in text],
                        follow_en=[t[2] for t in text])


#: Funnel notes, by step id.
FUNNEL_NOTES = {
    "1_raw_loaded": lambda n: "novel_catalogue.csv read from the raw zone",
    "2_identified": lambda n: "Rows with a URL and a title",
    "3_sentinels_to_null": lambda n: "Placeholder values emptied: " + ", ".join(
        f"{col} {_int(v)}" for col, v in re.findall(r"(\w+)=([\d,]+)", n)),
    "4_deduplicated": lambda n: "Duplicates removed on series_id",
    "5_with_synopsis": lambda n: "Synopsis of at least 10 words (" + _num(r"([\d.]+%)", n, "?") + " coverage)",
    "6_enriched": lambda n: "Weighted rating added (catalogue prior C = " + _num(r"=([\d.]+)", n, "?") + ")",
    "7_with_text_signal": lambda n: "At least 3 words of text to model",
    "8_reliable_rating": lambda n: "10+ votes: enters rating averages",
}

ELEMENT_TERMS = {
    "Terstruktur": "Structured", "Semi-terstruktur": "Semi-structured", "Tidak terstruktur": "Unstructured",
    "Identitas": "Identity", "Label": "Label", "Deskripsi": "Description", "Dimensi": "Dimension",
    "Temporal": "Temporal", "Metrik": "Metric", "Popularitas": "Popularity", "Flag": "Flag",
    "Tidak ada": "None", "Publik": "Public", "Nama orang (publik)": "Person name (published credit)",
    "Nama entitas (publik)": "Organisation name (public)", "Agregat (bukan PII)": "Aggregate (not PII)",
    "Agregat": "Aggregate",
}
#: Why each source column sits in its format class. Percentages live in the
#: "% placeholder" column next to it, so they are not repeated here.
ELEMENT_RATIONALE = {
    "series_id": "Natural primary key: a unique, atomic integer. Used for de-duplication because titles are not unique.",
    "url": "Hierarchical path (/series/<slug>/) that can be split into segments. The key that unifies the scrape, not an attribute.",
    "title": "Free natural-language text with no internal schema; tokenised before it enters the model corpus.",
    "genres": "Comma-separated list in one cell: fixed schema, non-atomic content. Split into an array before counting; the genre model's target.",
    "tags": "Like genres but much denser (median 12 tags): community-curated semantic labels.",
    "synopsis": "Free prose synopsis, the core unstructured data here; useful only after tokenisation and TF-IDF or embeddings.",
    "type": "Closed, low-cardinality category, one value per row.",
    "type_origin": "Country of origin: an atomic category and an aggregation axis.",
    "language": "Atomic category with very low cardinality.",
    "status_coo": "Mixes status text with number-and-unit patterns (\"182 Chapters (Completed)\"); the chapter count is recovered from it with a regex.",
    "authors": "Comma-separated list, as a title can have several authors. Holds published pen names, not private identities.",
    "original_publisher": "Organisation name, one per row; an unknown publisher becomes empty during cleaning.",
    "artists": "Dropped during cleaning: almost entirely Unknown.",
    "english_publisher": "Dropped during cleaning: almost entirely Unknown.",
    "year": "Release year, an ordinal scalar. The 0 placeholder cannot be recovered from other columns, so it becomes empty and leaves time analyses.",
    "release_frequency": "Sentence pattern (\"Every 0.9 Day(s)\") with number and unit inside the text; every row parses into days per chapter.",
    "chapter_count": "Additive numeric measure. The 0 placeholder becomes empty first, or the average length would drop falsely.",
    "rating_score": "Continuous 0 to 5 score. 0 means never rated, not rated badly, so it becomes empty.",
    "rating_votes": "Vote count, used as the confidence weight (v) in the Bayesian weighted rating.",
    "reading_list_count": "Aggregate count of readers who saved the title; no per-user rows, so no PII.",
    "activity_week_rank": "Weekly activity rank: ordinal, lower is more popular.",
    "activity_month_rank": "Monthly activity rank.",
    "activity_all_rank": "All-time activity rank.",
    "reading_list_month_rank": "Monthly reading-list rank.",
    "reading_list_all_rank": "All-time reading-list rank.",
    "is_completed": "Boolean stored as a number.",
    "licensed": "Boolean stored as a number; marks titles with an official English licence (compliance).",
    "completely_translated": "Boolean stored as a number.",
}


PILLAR_EN = {"3.1 Security": "Security", "3.2 Compliance": "Compliance",
             "3.3 Data Privacy": "Privacy", "3.4 Data Quality": "Data quality"}


def checklist_html(rows: pd.DataFrame) -> str:
    """One pillar's checks: status word first (never colour alone), then the check and its evidence."""
    items = []
    for r in rows.itertuples(index=False):
        follow = f'<div class="fu">Follow-up: {html.escape(r.follow_en)}</div>' if r.follow_en else ""
        items.append(
            f'<li class="st-{r.status_en.lower()}"><span class="st">{r.status_en}</span>'
            f'<div><div class="nm">{html.escape(r.name_en)} <code>{html.escape(r.check_id)}</code></div>'
            f'<div class="ev">{html.escape(r.evidence_en)}</div>{follow}</div></li>'
        )
    return '<ul class="nl-checks">' + "".join(items) + "</ul>"


def page_governance() -> None:
    hero(
        "Data quality & governance",
        "Every check the pipeline runs before a number reaches a chart: security, compliance, "
        "privacy and quality, then the cleaning steps and the format of every source column.",
    )
    with st.spinner("Loading audit tables…"):
        audit = load_lake("governance/audit_report")
        funnel = load_table("quality_report")
        elements = load_lake("governance/data_elements")
        formats = load_lake("governance/data_element_summary")
    audit = public_audit(audit.assign(status_en=audit["status"].map(STATUS_LABEL).fillna(audit["status"])))
    counts = audit["status_en"].value_counts()
    open_items = int(counts.get("Fail", 0) + counts.get("Warning", 0))

    if open_items:
        answer(f"<b>{open_items} of {len(audit)} checks need attention</b>; they are listed first in their pillar.")
    else:
        answer(f"<b>All {len(audit)} checks pass</b> or are informational: "
               f"{int(counts.get('Pass', 0))} pass, {int(counts.get('Info', 0))} record a fact without a verdict.")

    order = {"Fail": 0, "Warning": 1, "Pass": 2, "Info": 3}
    audit = audit.assign(_o=audit["status_en"].map(order)).sort_values(["pillar", "_o", "check_id"])
    pillars = list(dict.fromkeys(audit["pillar"]))
    for pair in (pillars[:2], pillars[2:]):
        for col, pillar in zip(st.columns(2, gap="large"), pair):
            rows = audit[audit["pillar"] == pillar]
            tally = rows["status_en"].value_counts()
            col.markdown(
                f'<div class="nl-pillar"><span>{PILLAR_EN.get(pillar, pillar)}</span>'
                f'<span class="tally">{" · ".join(f"{n} {s.lower()}" for s, n in tally.items())}</span></div>'
                + checklist_html(rows),
                unsafe_allow_html=True,
            )

    st.divider()
    st.subheader("What cleaning does to the row count")
    main_line = ["1_raw_loaded", "2_identified", "3_sentinels_to_null", "4_deduplicated", "6_enriched"]
    steps = funnel.assign(order=funnel["step"].map(lambda x: main_line.index(x) if x in main_line else 10 + int(x[0])))
    steps = steps.sort_values("order").drop(columns="order").reset_index(drop=True)
    steps["step_label"] = steps["step"].str.replace(r"^\d+[a-z]?_", "", regex=True).str.replace("_", " ")
    steps["note"] = [FUNNEL_NOTES.get(step, lambda n: "")(note) for step, note in zip(steps["step"], steps["note"])]
    raw = int(steps.iloc[0]["rows"])
    final = int(steps.loc[steps["step"] == "6_enriched", "rows"].iloc[0]) if (steps["step"] == "6_enriched").any() else raw
    answer(f"{raw:,} rows go in and {final:,} come out: only {raw - final:,} duplicate series are removed. "
           "The last three bars are subsets that specific analyses use.")
    steps["kind"] = np.where(steps["step"].isin(main_line), "Catalogue", "Subset used by an analysis")
    fig = px.bar(steps, x="rows", y="step_label", orientation="h", text="rows", color="kind",
                 color_discrete_map={"Catalogue": BRAND, "Subset used by an analysis": OTHER},
                 hover_data={"note": True, "kind": False},
                 labels={"rows": "Rows", "step_label": "", "kind": ""})
    fig.update_traces(texttemplate="%{text:,}", textposition="outside", cliponaxis=False)
    fig.update_yaxes(categoryorder="array", categoryarray=steps["step_label"].tolist()[::-1])
    fig.update_layout(legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(style_fig(fig, 40 * len(steps) + 90), use_container_width=True)

    st.divider()
    st.subheader("What kind of data each source column holds")
    formats = formats.assign(format_class=formats["format_class"].map(ELEMENT_TERMS).fillna(formats["format_class"]))
    elements = elements.assign(
        **{col: elements[col].map(ELEMENT_TERMS).fillna(elements[col])
           for col in ("format_class", "domain", "sensitivity")},
        rationale=elements["name"].map(ELEMENT_RATIONALE).fillna(""),
    )
    answer(" · ".join(f"<b>{r.n_elements}</b> {r.format_class.lower()}" for r in formats.itertuples(index=False))
           + " columns in novel_catalogue.csv, profiled before cleaning.")
    pick_fmt = st.radio("Format class", ["All"] + formats["format_class"].tolist(), horizontal=True)
    shown = elements if pick_fmt == "All" else elements[elements["format_class"] == pick_fmt]
    st.dataframe(
        shown[["name", "format_class", "domain", "physical_type", "pct_sentinel",
               "sensitivity", "rationale"]].rename(columns={
                   "name": "Column", "format_class": "Format", "domain": "Role",
                   "physical_type": "Type", "pct_sentinel": "% placeholder",
                   "sensitivity": "Sensitivity", "rationale": "Why this class"}),
        hide_index=True, use_container_width=True, height=380,
        column_config={"% placeholder": st.column_config.NumberColumn(format="%.2f"),
                       "Why this class": st.column_config.TextColumn(width="large")},
    )


def page_trends() -> None:
    hero(
        "Market trends",
        "How the catalogue changes by release year: where new titles come from, which genres "
        "are gaining ground, and whether newer stories run shorter or score differently.",
    )
    viz = load_table("viz_novels")
    dated = viz.dropna(subset=["year"]).assign(year=lambda d: d["year"].astype(int))
    if dated.empty:
        st.info("No title in the catalogue has a release year yet.")
        return
    last_full = last_full_year(dated["year"])
    # A handful of classics (one from 1740) would stretch the slider across
    # centuries with nothing in them; start where releases become regular.
    per_year_all = dated["year"].value_counts()
    first = int(per_year_all[per_year_all >= 5].index.min())
    early = int((dated["year"] < first).sum())

    c1, c2 = st.columns([3, 2], gap="large")
    span = c1.slider("Release years", first, last_full, (max(first, last_full - 19), last_full))
    origins = viz["type_origin"].value_counts().index.tolist()
    picked = c2.multiselect("Country of origin", origins, placeholder="All origins")

    data = dated[dated["year"].between(*span)]
    if picked:
        data = data[data["type_origin"].isin(picked)]
    if data.empty:
        st.info("No titles match these filters. Widen the year range or clear the origin filter.")
        return
    st.caption(
        f"{len(data):,} titles in view. {viz['year'].isna().mean():.0%} of the catalogue has no "
        f"release year and is left out; {last_full + 1} onward is excluded while the year is still running"
        + (f", and {early} titles from before {first} fall outside the slider." if early else ".")
    )

    end = span[1]
    yearly = data.groupby("year").size().reindex(range(span[0], end + 1), fill_value=0)
    recent, older = data[data["year"] > end - 5], data[data["year"] <= end - 5]

    kpi = st.columns(4)
    cur, prev = int(yearly.get(end, 0)), int(yearly.get(end - 1, 0))
    card(kpi[0], f"Released in {end}", f"{cur:,}", "titles now in the catalogue",
         delta=((cur - prev) / prev, f"{(cur - prev) / prev:+.0%} vs {end - 1}") if prev else None)
    card(kpi[1], "Busiest year", str(int(yearly.idxmax())), f"{int(yearly.max()):,} titles released")
    lead = recent["type_origin"].value_counts(normalize=True)
    if len(lead):
        before = older["type_origin"].value_counts(normalize=True).get(lead.index[0], np.nan)
        card(kpi[2], f"Leading origin {end - 4} to {end}", lead.index[0], f"{lead.iloc[0]:.0%} of new titles",
             delta=(lead.iloc[0] - before, f"{(lead.iloc[0] - before) * 100:+.1f} pts vs earlier years")
             if pd.notna(before) else None)
    med_now, med_before = recent["chapter_count"].median(), older["chapter_count"].median()
    card(kpi[3], "Median length, last 5 years", f"{med_now:,.0f} ch" if pd.notna(med_now) else "n/a",
         "recent titles are partly still running",
         delta=(med_now - med_before, f"{med_now - med_before:+,.0f} vs earlier years")
         if pd.notna(med_now) and pd.notna(med_before) else None)

    st.write("")
    st.subheader("New titles per release year")
    by_origin = (
        data.assign(origin=origin_group(data["type_origin"]))
        .groupby(["year", "origin"]).size().reset_index(name="titles")
    )
    fig = px.bar(
        by_origin, x="year", y="titles", color="origin", color_discrete_map=ORIGIN_COLORS,
        category_orders={"origin": list(ORIGIN_COLORS)},
        labels={"year": "", "titles": "Titles", "origin": ""},
    )
    fig.update_layout(hovermode="x unified", bargap=0.18)
    st.plotly_chart(style_fig(fig, 380), use_container_width=True)
    st.markdown(
        '<div class="nl-note">The catalogue lists what has been translated or picked up so far, '
        "so the newest years keep growing as translations catch up. Read the last bars as a floor.</div>",
        unsafe_allow_html=True,
    )

    st.write("")
    st.subheader("Which genres are gaining ground")
    if end - span[0] < 9:
        st.info("Widen the year range to at least ten years to compare two five-year windows.")
    else:
        genres = genre_labels().merge(data[["series_id", "year"]], on="series_id")
        n_now = int(data["year"].between(end - 4, end).sum())
        n_before = int(data["year"].between(end - 9, end - 5).sum())
        share_now = genres[genres["year"].between(end - 4, end)].groupby("genre", observed=True).size() / n_now
        share_before = (genres[genres["year"].between(end - 9, end - 5)]
                        .groupby("genre", observed=True).size() / n_before)
        momentum = pd.DataFrame({"now": share_now, "before": share_before}).fillna(0)
        momentum = momentum[(momentum["now"] + momentum["before"]) >= 0.02]
        momentum["change"] = (momentum["now"] - momentum["before"]) * 100
        top = (momentum.reindex(momentum["change"].abs().sort_values(ascending=False).index)
               .head(14).sort_values("change").reset_index(names="genre"))
        top["direction"] = np.where(top["change"] > 0, "Gaining share", "Losing share")
        top["now_pct"], top["before_pct"] = top["now"] * 100, top["before"] * 100

        left, right = st.columns([2, 3], gap="large")
        with left:
            fig = px.bar(
                top, x="change", y="genre", orientation="h", color="direction",
                color_discrete_map={"Gaining share": BRAND, "Losing share": ACCENT},
                text=top["change"].map("{:+.1f}".format),
                hover_data={"now_pct": ":.1f", "before_pct": ":.1f", "direction": False, "change": False},
                labels={"change": "Change in share of new titles (pts)", "genre": "",
                        "direction": "", "now_pct": f"% of {end - 4} to {end} titles",
                        "before_pct": f"% of {end - 9} to {end - 5} titles"},
            )
            # Symmetric room on both sides so the outside labels of the longest
            # bars stay inside the plot instead of running into the genre names.
            reach = top["change"].abs().max() * 1.45
            fig.update_xaxes(range=[-reach, reach])
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_layout(legend=dict(orientation="h", y=-0.18))
            st.plotly_chart(style_fig(fig, 470), use_container_width=True)
            st.caption(f"Share of titles released {end - 4} to {end} ({n_now:,}) against "
                       f"{end - 9} to {end - 5} ({n_before:,}). Genres under 2% in both windows are hidden.")
        with right:
            years_shown = list(range(max(span[0], end - 14), end + 1))
            window = genres[genres["year"].isin(years_shown)]
            per_year = data[data["year"].isin(years_shown)].groupby("year").size()
            top_genres = window["genre"].astype(str).value_counts().head(12).index
            heat = (window[window["genre"].isin(top_genres)]
                    .groupby(["genre", "year"], observed=True).size().unstack(fill_value=0))
            heat = (heat.div(per_year, axis=1) * 100).reindex(top_genres)
            heat.columns = [str(c) for c in heat.columns]
            fig = px.imshow(heat, color_continuous_scale=SCALE, aspect="auto", text_auto=".0f",
                            labels=dict(x="", y="", color="% of titles"))
            st.plotly_chart(style_fig(fig, 470), use_container_width=True)
            st.caption("Share of each year's new titles that carry the genre, for the 12 largest genres. "
                       "A title with several genres counts in each row.")

    st.write("")
    st.subheader("Length, completion and reception by release year")
    per_year = data.groupby("year").agg(
        titles=("series_id", "size"), median_chapters=("chapter_count", "median"),
        completed=("is_completed", "mean"),
    ).reset_index()
    per_year = per_year[per_year["titles"] >= 20]
    rated = (data[data["is_rated_reliable"] == True]  # noqa: E712
             .groupby("year")["weighted_rating"].agg(["mean", "size"]).reset_index())
    rated = rated[rated["size"] >= 30]

    a, b, c = st.columns(3, gap="large")
    for col, frame, y, title, fmt in (
        (a, per_year, "median_chapters", "Median chapters", ",.0f"),
        (b, per_year.assign(completed=per_year["completed"] * 100), "completed", "Completed (% of titles)", ".0f"),
        (c, rated, "mean", "Average rating (10+ votes)", ".2f"),
    ):
        with col:
            st.markdown(f"**{title}**")
            if frame.empty:
                st.info("Too few titles per year in this view. Widen the years or clear the origin filter.")
                continue
            fig = px.line(frame, x="year", y=y, markers=True, labels={"year": "", y: title})
            fig.update_traces(line=dict(color=BRAND, width=2), marker=dict(size=8, color=BRAND),
                              hovertemplate=f"%{{x}}: %{{y:{fmt}}}<extra></extra>")
            st.plotly_chart(style_fig(fig, 260), use_container_width=True)
    st.markdown(
        '<div class="nl-note">Years with fewer than 20 titles (30 rated titles for the rating line) '
        "are hidden. Newer titles are still being written, which pulls both their chapter count and "
        "their completion rate down; older titles that are still listed are the ones readers kept, "
        "which can lift their average rating.</div>",
        unsafe_allow_html=True,
    )


def page_genre() -> None:
    hero(
        "Genre explorer",
        "One genre at a time: how it performs against the whole catalogue, which story elements "
        "define it, what it is paired with, and who writes it.",
    )
    viz = load_table("viz_novels")
    genres = genre_labels()
    counts = genres["genre"].astype(str).value_counts()
    pick = st.selectbox("Genre", counts.index.tolist(), format_func=lambda g: f"{g} ({counts[g]:,} titles)")

    ids = genres.loc[genres["genre"] == pick, "series_id"]
    sub = viz[viz["series_id"].isin(ids)]
    reliable = viz["is_rated_reliable"] == True  # noqa: E712
    rel_all, rel = viz[reliable], sub[sub["is_rated_reliable"] == True]  # noqa: E712
    n, rank = len(sub), counts.index.get_loc(pick) + 1

    kpi = st.columns(4)
    card(kpi[0], "Titles", f"{n:,}", f"{n / len(viz):.1%} of the catalogue · #{rank} of {len(counts)} genres")
    if len(rel):
        diff = rel["weighted_rating"].mean() - rel_all["weighted_rating"].mean()
        card(kpi[1], "Average rating", f"{rel['weighted_rating'].mean():.2f}",
             f"{len(rel):,} titles with 10+ votes", delta=(diff, f"{diff:+.2f} vs catalogue"))
    else:
        card(kpi[1], "Average rating", "n/a", "no title with 10+ votes")
    done, done_all = sub["is_completed"].mean(), viz["is_completed"].mean()
    card(kpi[2], "Completed", f"{done:.0%}", f"catalogue {done_all:.0%}",
         delta=(done - done_all, f"{(done - done_all) * 100:+.0f} pts vs catalogue"))
    med, med_all = sub["chapter_count"].median(), viz["chapter_count"].median()
    card(kpi[3], "Median length", f"{med:,.0f} ch" if pd.notna(med) else "n/a", f"catalogue {med_all:,.0f} chapters",
         delta=(med - med_all, f"{med - med_all:+,.0f} vs catalogue") if pd.notna(med) else None)

    st.write("")
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**Rating distribution** (titles with 10+ votes)")
        if rel.empty:
            st.info("No title in this genre has 10+ votes yet.")
        else:
            edges = np.round(np.arange(1.0, 5.05, 0.1), 2)
            mine = np.histogram(rel["weighted_rating"], edges)[0] / len(rel) * 100
            every = np.histogram(rel_all["weighted_rating"], edges)[0] / len(rel_all) * 100
            used = np.flatnonzero((mine + every) > 0.05)
            lo, hi = used.min(), used.max() + 1
            centers = (edges[:-1] + 0.05)[lo:hi]
            fig = go.Figure()
            fig.add_bar(x=centers, y=mine[lo:hi], name=pick, marker_color=BRAND,
                        hovertemplate="%{y:.1f}% of titles<extra>" + html.escape(pick) + "</extra>")
            fig.add_scatter(x=centers, y=every[lo:hi], name="Whole catalogue", mode="lines",
                            line=dict(color=ACCENT, width=2, shape="hvh"),
                            hovertemplate="%{y:.1f}% of titles<extra>Catalogue</extra>")
            fig.update_layout(bargap=0.08, hovermode="x unified", xaxis_title="Weighted rating",
                              yaxis_title="% of titles", legend=dict(orientation="h", y=1.12))
            st.plotly_chart(style_fig(fig, 340), use_container_width=True)
    with right:
        st.markdown("**Where the titles come from**")
        mix = pd.DataFrame({
            pick: origin_group(sub["type_origin"]).value_counts(normalize=True) * 100,
            "Whole catalogue": origin_group(viz["type_origin"]).value_counts(normalize=True) * 100,
        }).reindex(list(ORIGIN_COLORS)).fillna(0)
        mix = mix[mix.max(axis=1) >= 0.5].reset_index(names="origin")
        long = mix.melt(id_vars="origin", var_name="group", value_name="share")
        fig = px.bar(long, x="share", y="origin", color="group", barmode="group", orientation="h",
                     color_discrete_map={pick: BRAND, "Whole catalogue": OTHER},
                     text=long["share"].map("{:.0f}%".format),
                     labels={"share": "% of titles", "origin": "", "group": ""})
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_xaxes(range=[0, long["share"].max() * 1.2])
        fig.update_yaxes(categoryorder="array", categoryarray=list(ORIGIN_COLORS)[::-1])
        fig.update_layout(legend=dict(orientation="h", y=1.12))
        st.plotly_chart(style_fig(fig, 340), use_container_width=True)

    st.write("")
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**Signature story elements**")
        tags = tag_labels()
        in_genre = tags[tags["series_id"].isin(ids)].groupby("tag", observed=True).size()
        overall = tags.groupby("tag", observed=True).size()
        sig = pd.DataFrame({"titles": in_genre, "share": in_genre / n * 100,
                            "lift": (in_genre / n) / (overall.reindex(in_genre.index) / len(viz))})
        sig = sig[sig["titles"] >= max(15, 0.02 * n)].nlargest(15, "lift").sort_values("lift")
        if sig.empty:
            st.info("Too few titles to find tags that set this genre apart.")
        else:
            sig = sig.reset_index(names="tag")
            fig = px.bar(sig, x="lift", y="tag", orientation="h", text=sig["lift"].map("×{:.1f}".format),
                         hover_data={"titles": ":,", "share": ":.1f"},
                         labels={"lift": "Lift vs. the whole catalogue", "tag": "",
                                 "titles": "Titles", "share": "% of genre"})
            fig.update_traces(marker_color=BRAND, textposition="outside", cliponaxis=False)
            fig.update_xaxes(range=[0, sig["lift"].max() * 1.2])
            st.plotly_chart(style_fig(fig, 460), use_container_width=True)
            st.caption("Lift = share of this genre's titles with the tag ÷ share of all titles with it. "
                       "Tags on fewer than 2% of the genre are skipped.")
    with right:
        st.markdown("**Most often paired with**")
        pair = (genres[genres["series_id"].isin(ids) & (genres["genre"] != pick)]
                .groupby("genre", observed=True).size().nlargest(10) / n * 100)
        pair = pair.sort_values().reset_index(name="share")
        fig = px.bar(pair, x="share", y="genre", orientation="h", text=pair["share"].map("{:.0f}%".format),
                     labels={"share": f"% of {pick} titles that also carry the genre", "genre": ""})
        fig.update_traces(marker_color=BRAND, textposition="outside", cliponaxis=False)
        fig.update_xaxes(range=[0, pair["share"].max() * 1.2])
        st.plotly_chart(style_fig(fig, 460), use_container_width=True)

    st.write("")
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("**Highest-rated titles** (10+ votes)")
        best = rel.nlargest(15, "weighted_rating")[["title", "weighted_rating", "rating_votes", "type_origin", "url"]]
        st.dataframe(
            best.rename(columns={"title": "Title", "weighted_rating": "Rating", "rating_votes": "Votes",
                                 "type_origin": "Origin", "url": "Link"}),
            hide_index=True, use_container_width=True, height=420,
            column_config={
                "Title": st.column_config.TextColumn(width="medium"),
                "Rating": st.column_config.NumberColumn(format="%.2f", width="small"),
                "Votes": st.column_config.NumberColumn(format="%d", width="small"),
                "Origin": st.column_config.TextColumn(width="small"),
                "Link": st.column_config.LinkColumn("Link", display_text="Open", width="small"),
            },
        )
    with right:
        st.markdown("**Most prolific authors in this genre**")
        authors = author_labels()
        written = authors[authors["series_id"].isin(ids)].merge(
            viz[["series_id", "weighted_rating", "is_rated_reliable"]], on="series_id")
        written["rated"] = written["weighted_rating"].where(written["is_rated_reliable"] == True)  # noqa: E712
        top_auth = (written.groupby("author", observed=True)
                    .agg(titles=("series_id", "nunique"), rating=("rated", "mean"))
                    .nlargest(12, "titles").reset_index())
        st.dataframe(
            top_auth.assign(rating=rating_text(top_auth["rating"]))
            .rename(columns={"author": "Author", "titles": "Titles", "rating": "Rating"}),
            hide_index=True, use_container_width=True, height=420,
        )


@st.cache_data(show_spinner=False, max_entries=16)
def tag_stats(genre: str, origins: tuple, recent_only: bool, version: str) -> tuple[pd.DataFrame, float, int]:
    """Per-tag title count, mean rating and median reading-list adds over titles with 10+ votes."""
    viz = _load_lake("analytics/viz_novels", version)
    base = viz[viz["is_rated_reliable"] == True]  # noqa: E712
    if origins:
        base = base[base["type_origin"].isin(origins)]
    if recent_only:
        base = base[base["year"] >= last_full_year(viz["year"].dropna()) - 4]
    if genre:
        genres = _load_labels("bridge_novel_genre", "dim_genre", "genre_key", "genre", version)
        base = base[base["series_id"].isin(genres.loc[genres["genre"] == genre, "series_id"])]
    tags = _load_labels("bridge_novel_tag", "dim_tag", "tag_key", "tag", version)
    joined = tags.merge(base[["series_id", "weighted_rating", "reading_list_count"]], on="series_id")
    stats = (joined.groupby("tag", observed=True)
             .agg(titles=("series_id", "size"), rating=("weighted_rating", "mean"),
                  readers=("reading_list_count", "median"))
             .reset_index())
    stats["tag"] = stats["tag"].astype(str)
    return stats, float(base["weighted_rating"].mean()), len(base)


def spaced_labels(stats: pd.DataFrame, limit: int = 9) -> pd.Series:
    """Labels for the best underserved and proven tags, skipping points too close
    (in log10 titles and lift) to one already labelled."""
    candidates = pd.concat([
        stats[stats["quadrant"] == "Underserved"].nlargest(20, "lift"),
        stats[stats["quadrant"] == "Proven"].nlargest(10, "lift"),
    ])
    x_span = np.log10(stats["titles"].max()) - np.log10(stats["titles"].min()) or 1.0
    y_span = (stats["lift"].max() - stats["lift"].min()) or 1.0
    placed: list[tuple[float, float]] = []
    chosen = set()
    for row in candidates.itertuples(index=False):
        x, y = np.log10(row.titles) / x_span, row.lift / y_span
        if all(abs(x - px_) > 0.12 or abs(y - py_) > 0.07 for px_, py_ in placed):
            placed.append((x, y))
            chosen.add(row.tag)
        if len(chosen) == limit:
            break
    return stats["tag"].where(stats["tag"].isin(chosen), "")


QUADRANT_COLORS = {"Underserved": ACCENT, "Proven": BRAND, "Crowded": SEQUENCE[3], "Rare and weaker": OTHER}


def page_tags() -> None:
    hero(
        "Story element opportunities",
        "Every tag placed on two axes: how many rated titles already use it, and how readers "
        "score those titles against the average. Elements that score well but are still rare "
        "are the openings for a new story.",
    )
    viz = load_table("viz_novels")
    genre_counts = genre_labels()["genre"].astype(str).value_counts()

    c1, c2, c3, c4 = st.columns([2, 2, 2, 2])
    genre = c1.selectbox("Within genre", ["All genres"] + genre_counts.index.tolist())
    origins = c2.multiselect("Country of origin", viz["type_origin"].value_counts().index.tolist(),
                             placeholder="All origins")
    period = c3.radio("Release years", ["All years", "Last 5 years"], horizontal=True)
    min_titles = c4.slider("Minimum titles per tag", 10, 300, 40, 10)

    stats, average, n_base = tag_stats(
        "" if genre == "All genres" else genre, tuple(sorted(origins)), period == "Last 5 years", lake_version()
    )
    stats = stats[stats["titles"] >= min_titles].copy()
    if len(stats) < 5:
        st.info(f"Only {len(stats)} tags reach {min_titles} rated titles in this view. "
                "Lower the minimum or widen the filters.")
        return

    stats["lift"] = stats["rating"] - average
    supply = stats["titles"].median()
    stats["quadrant"] = np.select(
        [(stats["lift"] > 0) & (stats["titles"] < supply), stats["lift"] > 0, stats["titles"] >= supply],
        ["Underserved", "Proven", "Crowded"], "Rare and weaker",
    )
    stats["label"] = spaced_labels(stats)

    kpi = st.columns(4)
    card(kpi[0], "Rated titles in view", f"{n_base:,}", f"average rating {average:.2f}")
    card(kpi[1], "Tags compared", f"{len(stats):,}", f"each on {min_titles}+ rated titles")
    for col, name, text in ((kpi[2], "Underserved", "rated above average, used less than the median tag"),
                            (kpi[3], "Proven", "rated above average and already common")):
        card(col, name, f"{(stats['quadrant'] == name).sum():,} tags", text)

    st.write("")
    fig = px.scatter(
        stats, x="titles", y="lift", color="quadrant", text="label", log_x=True,
        color_discrete_map=QUADRANT_COLORS, category_orders={"quadrant": list(QUADRANT_COLORS)},
        hover_name="tag", hover_data={"titles": ":,", "rating": ":.2f", "readers": ":,.0f",
                                      "lift": ":+.2f", "label": False, "quadrant": False},
        labels={"titles": "Rated titles using the tag (log scale)", "lift": "Rating vs average",
                "quadrant": "", "rating": "Average rating", "readers": "Median reading-list adds"},
    )
    fig.update_traces(marker=dict(size=10, line=dict(width=1.5, color=SURFACE)),
                      textposition="top center", textfont=dict(size=11, color=INK))
    fig.add_hline(y=0, line=dict(color="rgba(30,36,51,.35)", width=1))
    fig.add_vline(x=supply, line=dict(color="rgba(30,36,51,.35)", width=1, dash="dot"))
    fig.update_layout(legend=dict(orientation="h", y=1.08))
    st.plotly_chart(style_fig(fig, 560), use_container_width=True)
    st.markdown(
        f'<div class="nl-note">Horizontal line: the average rating of the titles in view. Dotted line: '
        f"the median tag ({supply:,.0f} titles). A high score on a rare tag rests on fewer titles, so "
        "check the title count before reading much into the top-left corner.</div>",
        unsafe_allow_html=True,
    )

    st.write("")
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("**Underserved elements, best first**")
        under = stats[stats["quadrant"] == "Underserved"].nlargest(15, "lift")
        st.dataframe(
            under[["tag", "titles", "rating", "lift", "readers"]].rename(columns={
                "tag": "Tag", "titles": "Rated titles", "rating": "Rating", "lift": "vs average",
                "readers": "Median readers"}),
            hide_index=True, use_container_width=True, height=420,
            column_config={"Rating": st.column_config.NumberColumn(format="%.2f"),
                           "vs average": st.column_config.NumberColumn(format="%+.2f"),
                           "Median readers": st.column_config.NumberColumn(format="%d")},
        )
    with right:
        st.markdown("**Check a combination**")
        combo = st.multiselect("Tags that must all be present",
                               stats.sort_values("titles", ascending=False)["tag"].tolist(),
                               max_selections=3, placeholder="Pick up to three tags")
        if combo:
            tags = tag_labels()
            have = tags[tags["tag"].isin(combo)].groupby("series_id")["tag"].nunique()
            match_ids = have[have == len(combo)].index
            hits = viz[viz["series_id"].isin(match_ids)]
            rated = hits[hits["is_rated_reliable"] == True]  # noqa: E712
            a, b = st.columns(2)
            card(a, "Titles with all", f"{len(hits):,}", f"{len(rated):,} with 10+ votes")
            if len(rated):
                diff = rated["weighted_rating"].mean() - average
                card(b, "Their rating", f"{rated['weighted_rating'].mean():.2f}", "",
                     delta=(diff, f"{diff:+.2f} vs average"))
            st.write("")
            st.dataframe(
                rated.nlargest(8, "weighted_rating")[["title", "weighted_rating", "url"]].rename(
                    columns={"title": "Title", "weighted_rating": "Rating", "url": "Link"}),
                hide_index=True, use_container_width=True,
                column_config={"Rating": st.column_config.NumberColumn(format="%.2f"),
                               "Link": st.column_config.LinkColumn("Link", display_text="Open")},
            )
        else:
            st.caption("Pick tags to see how many titles already combine them and how those score.")


@st.cache_data(show_spinner=False, max_entries=2)
def _tag_counts(version: str) -> pd.Series:
    return _load_lake("model/bridge_novel_tag", version).groupby("series_id").size()


@st.cache_data(show_spinner=False, max_entries=2)
def _tag_rarity(version: str) -> pd.Series:
    """Titles per tag, so the lookup page orders a title's tags without recounting 600k rows."""
    return _load_labels("bridge_novel_tag", "dim_tag", "tag_key", "tag", version)["tag"].astype(str).value_counts()


def tag_neighbours(series_id: int, k: int = 6) -> pd.DataFrame:
    """Titles sharing the most tags with `series_id`, by Jaccard overlap of their tag sets."""
    version = lake_version()
    bridge = _load_lake("model/bridge_novel_tag", version)
    mine = bridge.loc[bridge["series_id"] == series_id, "tag_key"]
    if mine.empty:
        return pd.DataFrame()
    shared = bridge[bridge["tag_key"].isin(mine)].groupby("series_id").size().drop(series_id, errors="ignore")
    shared = shared[shared >= 3]
    sizes = _tag_counts(version).reindex(shared.index)
    viz = load_table("viz_novels").set_index("series_id")
    out = pd.DataFrame({"similarity": shared / (len(mine) + sizes - shared)}).join(viz, how="inner")
    out = out.sort_values(["similarity", "rating_votes"], ascending=False).head(k).reset_index()
    return out.assign(genre_list=out["series_id"].map(genre_lists()))


def page_lookup() -> None:
    viz = load_table("viz_novels")
    hero(
        "Title lookup",
        f"Search any of the {len(viz):,} titles to see how it compares with its genre and which "
        "titles share the most story elements with it.",
    )
    c1, c2 = st.columns([2, 3], gap="large")
    query = c1.text_input("Search by title", placeholder="Part of a title, e.g. martial god")
    pool = viz
    if query.strip():
        pool = viz[viz["title"].str.contains(re.escape(query.strip()), case=False, na=False)]
    if pool.empty:
        st.info("No title matches that search. Try a shorter part of the name.")
        return
    pool = pool.sort_values("rating_votes", ascending=False).head(60)
    labels = {
        int(r.series_id): f"{r.title} ({int(r.year) if pd.notna(r.year) else 'year n/a'}, {r.type_origin})"
        for r in pool.itertuples(index=False)
    }
    sid = c2.selectbox(f"{len(pool)} best-known matches", list(labels), format_func=labels.get)
    row = viz.loc[viz["series_id"] == sid].iloc[0]

    tags = tag_labels()
    title_tags = tags.loc[tags["series_id"] == sid, "tag"].astype(str).tolist()
    title_genres = genre_lists().get(sid, [])
    authors = author_labels()
    names = authors.loc[authors["series_id"] == sid, "author"].astype(str).tolist()

    st.markdown(f"### {html.escape(str(row['title']))}", unsafe_allow_html=True)
    st.markdown(chips(title_genres, active=True), unsafe_allow_html=True)
    meta = [f"by {', '.join(names)}" if names else "author unknown",
            f"released {int(row['year'])}" if pd.notna(row["year"]) else "release year unknown",
            "completed" if row["is_completed"] == 1 else "ongoing",
            f"release pace {en_label(row['release_pace']).lower()}" if isinstance(row["release_pace"], str) else ""]
    st.caption(" · ".join(m for m in meta if m))
    if isinstance(row["url"], str) and row["url"].startswith("http"):
        st.markdown(f'<a class="nl-open" href="{html.escape(row["url"], quote=True)}" target="_blank" '
                    'rel="noopener noreferrer">Open on NovelUpdates</a>', unsafe_allow_html=True)

    st.write("")
    kpi = st.columns(4)
    reliable = bool(row["is_rated_reliable"])
    card(kpi[0], "Rating", f"{row['weighted_rating']:.2f}" if pd.notna(row["weighted_rating"]) else "n/a",
         f"{int(row['rating_votes']):,} votes" + ("" if reliable else ", too few for rankings"))
    card(kpi[1], "On reading lists", f"{row['reading_list_count']:,.0f}" if pd.notna(row["reading_list_count"]) else "n/a",
         "readers who saved it")
    card(kpi[2], "Chapters", f"{row['chapter_count']:,.0f}" if pd.notna(row["chapter_count"]) else "n/a",
         en_label(row["length_bucket"]) if isinstance(row["length_bucket"], str) else "length unknown")
    card(kpi[3], "Story elements", f"{len(title_tags)}", "tags on NovelUpdates")

    st.write("")
    left, right = st.columns([2, 3], gap="large")
    peers = viz[viz["primary_genre"] == row["primary_genre"]]
    with left:
        st.markdown(f"**Against other {html.escape(str(row['primary_genre']))} titles**")
        checks = [("Rating", "weighted_rating", True), ("Votes", "rating_votes", False),
                  ("Reading-list adds", "reading_list_count", False), ("Chapters", "chapter_count", False)]
        rows = []
        for name, col, need_reliable in checks:
            if pd.isna(row[col]) or (need_reliable and not reliable):
                continue
            pool_vals = peers[peers["is_rated_reliable"] == True][col] if need_reliable else peers[col]  # noqa: E712
            pool_vals = pool_vals.dropna()
            if len(pool_vals):
                # Share at or above this title: the title itself counts, so the
                # best title reads "top 0.1%", never "higher than 100%".
                top = (pool_vals >= row[col]).mean() * 100
                rows.append({"measure": name, "pct": 100 - top, "n": len(pool_vals),
                             "text": f"top {max(top, 0.1):.1f}%" if top < 10 else f"top {top:.0f}%"})
        if not rows:
            st.info("Not enough data to compare this title with its genre.")
        else:
            pct = pd.DataFrame(rows)
            fig = px.bar(pct, x="pct", y="measure", orientation="h", range_x=[0, 100],
                         text="text",
                         hover_data={"n": ":,"}, labels={"pct": "Percentile within genre", "measure": "",
                                                         "n": "Titles compared"})
            fig.update_traces(marker_color=BRAND, textposition="inside", insidetextanchor="end")
            fig.update_yaxes(categoryorder="array", categoryarray=[c[0] for c in checks][::-1])
            st.plotly_chart(style_fig(fig, 260), use_container_width=True)
            st.caption(f"Compared with {len(peers):,} titles whose primary genre is {row['primary_genre']}; "
                       "the rating only against titles with 10+ votes.")
    with right:
        st.markdown("**Story elements**")
        if title_tags:
            rarity = _tag_rarity(lake_version())
            ordered = sorted(title_tags, key=lambda t: rarity.get(t, 0))
            st.markdown(chips(ordered[:24]), unsafe_allow_html=True)
            st.caption("Rarest first: the tags near the start set this title apart most.")
        else:
            st.info("This title has no tags on NovelUpdates.")

    st.write("")
    st.subheader("Titles built from the same story elements")
    near = tag_neighbours(sid)
    if near.empty:
        st.info("No other title shares at least three tags with this one.")
    else:
        for _, other in near.iterrows():
            novel_card(other, similarity_label="tags shared", show_synopsis=False)
        st.caption("Overlap = shared tags ÷ tags on either title (Jaccard). Unlike AI Studio, this "
                   "works for every title, including those without a synopsis.")


def page_creators() -> None:
    hero(
        "Authors & publishers",
        "Who writes and publishes the catalogue, measured by output, reader reception and reach.",
    )
    viz = load_table("viz_novels")
    cols = ["series_id", "weighted_rating", "is_rated_reliable", "rating_votes", "reading_list_count", "type_origin"]
    tab_a, tab_p = st.tabs(["Authors", "Original publishers"])

    with tab_a:
        written = author_labels().merge(viz[cols], on="series_id")
        written["rated"] = written["weighted_rating"].where(written["is_rated_reliable"] == True)  # noqa: E712
        per = written.groupby("author", observed=True).agg(
            titles=("series_id", "nunique"), rated_titles=("rated", "count"), rating=("rated", "mean"),
            readers=("reading_list_count", "sum"), votes=("rating_votes", "sum"))
        origin = (written.groupby(["author", "type_origin"], observed=True).size().reset_index(name="n")
                  .sort_values("n").drop_duplicates("author", keep="last").set_index("author")["type_origin"])
        per = per.join(origin.rename("origin")).reset_index()
        per["author"] = per["author"].astype(str)

        kpi = st.columns(4)
        card(kpi[0], "Authors", f"{len(per):,}", "pen names merged where aliases match")
        card(kpi[1], "One title only", f"{(per['titles'] == 1).mean():.0%}", "of authors in the catalogue")
        top = per.nlargest(1, "titles").iloc[0]
        card(kpi[2], "Most titles", top["author"], f"{int(top['titles'])} titles")
        card(kpi[3], "Median output", f"{per['titles'].median():.0f}", "titles per author")

        st.write("")
        c1, c2 = st.columns(2)
        rank_by = c1.radio("Rank by", ["Titles", "Average rating", "Reading-list adds"], horizontal=True)
        min_titles = c2.slider("Minimum titles", 1, 20, 3, key="author_min")
        pool = per[per["titles"] >= min_titles]
        metric = {"Titles": "titles", "Average rating": "rating", "Reading-list adds": "readers"}[rank_by]
        if metric == "rating":
            pool = pool[pool["rated_titles"] >= 3]
        if pool.empty:
            st.info("No author meets the minimum. Lower it.")
        else:
            board = pool.nlargest(15, metric).sort_values(metric)
            board = board.assign(origin_group=origin_group(board["origin"]))
            # Ratings differ by hundredths, so they get dots on a zoomed axis
            # (see dot_rating_chart); counts keep bars from zero.
            chart = px.scatter if metric == "rating" else px.bar
            fig = chart(board, x=metric, y="author", color="origin_group", color_discrete_map=ORIGIN_COLORS,
                        category_orders={"origin_group": list(ORIGIN_COLORS)},
                        hover_data={"titles": True, "rating": ":.2f", "readers": ":,.0f", "origin_group": False},
                        labels={metric: rank_by, "author": "", "origin_group": "", "titles": "Titles",
                                "rating": "Rating", "readers": "Reading-list adds"},
                        **({} if metric == "rating" else {"orientation": "h"}))
            fig.update_yaxes(categoryorder="array", categoryarray=board["author"].tolist())
            if metric == "rating":
                fig.update_traces(marker=dict(size=13, line=dict(width=2, color=SURFACE)))
            st.plotly_chart(style_fig(fig, 480), use_container_width=True)
            if metric == "rating":
                st.caption("Authors need 3+ titles with 10+ votes to be ranked by rating.")
            st.dataframe(
                pool.sort_values(metric, ascending=False).head(100)
                .assign(rating=lambda d: rating_text(d["rating"]))[
                    ["author", "titles", "rated_titles", "rating", "readers", "origin"]].rename(columns={
                    "author": "Author", "titles": "Titles", "rated_titles": "Rated titles",
                    "rating": "Rating", "readers": "Reading-list adds", "origin": "Origin"}),
                hide_index=True, use_container_width=True, height=360,
                column_config={"Reading-list adds": st.column_config.NumberColumn(format="%d")},
            )

    with tab_p:
        fact = load_lake("model/fact_novel")[["series_id", "publisher_key"]]
        names = load_lake("model/dim_publisher")[["publisher_key", "publisher"]]
        books = fact.merge(names, on="publisher_key").merge(viz[cols], on="series_id")
        books = books[books["publisher"].notna() & (books["publisher"].str.strip() != "")]
        books["rated"] = books["weighted_rating"].where(books["is_rated_reliable"] == True)  # noqa: E712
        per = books.groupby("publisher").agg(
            titles=("series_id", "nunique"), rated_titles=("rated", "count"), rating=("rated", "mean"),
            readers=("reading_list_count", "sum"))
        origin = (books.groupby(["publisher", "type_origin"]).size().reset_index(name="n")
                  .sort_values("n").drop_duplicates("publisher", keep="last").set_index("publisher")["type_origin"])
        per = per.join(origin.rename("origin")).reset_index()

        min_pub = st.slider("Minimum titles", 5, 200, 30, 5, key="pub_min")
        pool = per[(per["titles"] >= min_pub) & (per["rated_titles"] >= 10)].copy()
        if len(pool) < 2:
            st.info("Too few publishers at this minimum. Lower it.")
        else:
            pool["origin_group"] = origin_group(pool["origin"])
            pool["label"] = np.where(pool["publisher"].isin(pool.nlargest(8, "titles")["publisher"]),
                                     pool["publisher"], "")
            fig = px.scatter(
                pool, x="titles", y="rating", size="readers", size_max=38, log_x=True,
                color="origin_group", color_discrete_map=ORIGIN_COLORS, text="label",
                hover_name="publisher", hover_data={"label": False, "origin_group": False, "readers": ":,.0f"},
                labels={"titles": "Titles published (log scale)", "rating": "Average rating (10+ votes)",
                        "origin_group": "", "readers": "Reading-list adds"},
            )
            fig.update_traces(textposition="top center", textfont=dict(size=11, color=INK))
            st.plotly_chart(style_fig(fig, 480), use_container_width=True)
            st.caption(f"{len(pool):,} publishers with {min_pub}+ titles and 10+ rated titles. "
                       "Bubble size = total reading-list adds.")
            st.dataframe(
                pool.sort_values("titles", ascending=False)
                .assign(rating=lambda d: rating_text(d["rating"]))[
                    ["publisher", "titles", "rated_titles", "rating", "readers", "origin"]].rename(columns={
                    "publisher": "Publisher", "titles": "Titles", "rated_titles": "Rated titles",
                    "rating": "Rating", "readers": "Reading-list adds", "origin": "Origin"}),
                hide_index=True, use_container_width=True, height=360,
                column_config={"Reading-list adds": st.column_config.NumberColumn(format="%d")},
            )


ZONE_COLORS = {"Raw": SEQUENCE[3], "Clean": SEQUENCE[2], "Curated": SEQUENCE[0]}


@st.cache_data(ttl=60, show_spinner=False)
def lake_inventory() -> pd.DataFrame:
    """Every object in the three lake zones, grouped into datasets (the first two path levels)."""
    zones = {"Raw": SETTINGS.minio.bucket_raw, "Clean": SETTINGS.minio.bucket_clean,
             "Curated": SETTINGS.minio.bucket_curated}
    rows = []
    for zone, bucket in zones.items():
        try:
            objects = list_objects(bucket, "")
        except Exception as exc:
            raise DataUnavailable(str(exc)) from exc
        for obj in objects:
            key = obj["Key"]
            if "_temporary" in key:
                continue
            parts = key.split("/")
            rows.append({"zone": zone, "bucket": bucket, "key": key,
                         "dataset": "/".join(parts[:2]) if len(parts) > 2 else parts[0],
                         "size": obj["Size"], "modified": pd.Timestamp(obj["LastModified"])})
    return pd.DataFrame(rows)


def page_lake() -> None:
    hero(
        "Lake operations",
        "Where the data physically sits: the raw, clean and curated zones of the object store, "
        "how fresh each dataset is, and what the streaming job has ingested.",
    )
    with st.spinner("Listing the lake…"):
        inventory = lake_inventory()
    if inventory.empty:
        st.info("The lake is empty. Run the pipeline first.")
        return

    now = pd.Timestamp.now(tz="UTC")
    kpi = st.columns(3)
    for col, zone in zip(kpi, ZONE_COLORS):
        part = inventory[inventory["zone"] == zone]
        if part.empty:
            card(col, f"{zone} zone", "empty", "no objects yet")
            continue
        last = part["modified"].max()
        days = (now - last).days
        card(col, f"{zone} zone · {part['bucket'].iloc[0]}", f"{part['size'].sum() / 1e6:,.1f} MB",
             f"{len(part):,} objects in {part['dataset'].nunique()} datasets · last write "
             f"{last:%d %b %H:%M} UTC ({f'{days} d ago' if days else 'today'})")

    st.write("")
    datasets = (inventory.groupby(["zone", "dataset"])
                .agg(files=("key", "size"), size=("size", "sum"), modified=("modified", "max"))
                .reset_index())
    datasets["size_mb"] = datasets["size"] / 1e6
    datasets["age_days"] = (now - datasets["modified"]).dt.total_seconds() / 86400
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("**Largest datasets**")
        top = datasets.nlargest(15, "size").sort_values("size")
        fig = px.bar(top, x="size_mb", y="dataset", color="zone", orientation="h",
                     color_discrete_map=ZONE_COLORS, hover_data={"files": ":,", "zone": False},
                     labels={"size_mb": "Size (MB)", "dataset": "", "zone": "", "files": "Files"})
        fig.update_yaxes(categoryorder="array", categoryarray=top["dataset"].tolist())
        st.plotly_chart(style_fig(fig, 470), use_container_width=True)
    with right:
        st.markdown("**Size per zone**")
        zones = inventory.groupby("zone")["size"].sum().reindex(list(ZONE_COLORS)).fillna(0) / 1e6
        fig = px.bar(x=zones.index, y=zones.values, color=zones.index, color_discrete_map=ZONE_COLORS,
                     text=[f"{v:,.0f} MB" for v in zones.values], labels={"x": "", "y": "MB", "color": ""})
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(showlegend=False)
        st.plotly_chart(style_fig(fig, 470), use_container_width=True)

    st.markdown("**Freshness of every dataset**")
    st.dataframe(
        datasets.sort_values("modified", ascending=False)[["dataset", "zone", "files", "size_mb", "modified", "age_days"]]
        .rename(columns={"dataset": "Dataset", "zone": "Zone", "files": "Files", "size_mb": "MB",
                         "modified": "Last write (UTC)", "age_days": "Days since write"}),
        hide_index=True, use_container_width=True, height=360,
        column_config={"MB": st.column_config.NumberColumn(format="%.2f"),
                       "Days since write": st.column_config.NumberColumn(format="%.1f"),
                       "Last write (UTC)": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm")},
    )

    st.divider()
    st.subheader("Streaming ingest")
    try:
        stream = load_lake("streaming/novel_metrics_incremental")
    except DataUnavailable:
        st.info("The streaming job has not written any output yet.")
        return
    stream = stream.assign(
        scraped=pd.to_datetime(stream["scraped_at_utc"], utc=True, errors="coerce"),
        ingested=pd.to_datetime(stream["ingested_at_utc"], utc=True, errors="coerce"),
    )
    stream["latency"] = (stream["ingested"] - stream["scraped"]).dt.total_seconds()
    batches = (stream.groupby("ingested")
               .agg(rows=("series_id", "size"), latency=("latency", "median"),
                    full=("metadata_tier", lambda s: (s == "full").mean()))
               .reset_index().sort_values("ingested"))
    batches["batch"] = [f"Batch {i + 1}" for i in range(len(batches))]
    commits = inventory["key"].str.contains(r"streaming/_checkpoint/commits/\d+$", regex=True).sum()

    kpi = st.columns(4)
    card(kpi[0], "Rows ingested", f"{len(stream):,}", "into curated-data/streaming")
    card(kpi[1], "Micro-batches", f"{len(batches):,}", f"{commits} committed in the checkpoint")
    card(kpi[2], "Median latency", f"{stream['latency'].median():,.0f} s", "from scrape to ingest")
    card(kpi[3], "Last batch", f"{batches['ingested'].max():%d %b %H:%M}", "UTC")

    st.write("")
    left, right = st.columns(2, gap="large")
    with left:
        fig = px.bar(batches, x="batch", y="rows", text="rows",
                     hover_data={"ingested": True, "latency": ":.0f"},
                     labels={"batch": "", "rows": "Rows", "ingested": "Ingested (UTC)", "latency": "Median latency (s)"})
        fig.update_traces(marker_color=BRAND, textposition="outside", cliponaxis=False)
        st.plotly_chart(style_fig(fig, 300), use_container_width=True)
    with right:
        st.dataframe(
            batches[["batch", "ingested", "rows", "latency", "full"]].rename(columns={
                "batch": "Batch", "ingested": "Ingested (UTC)", "rows": "Rows",
                "latency": "Latency (s)", "full": "Full detail"}),
            hide_index=True, use_container_width=True,
            column_config={"Latency (s)": st.column_config.NumberColumn(
                               format="%.1f", help="Median seconds from scrape to ingest"),
                           "Full detail": st.column_config.NumberColumn(
                               format="%.2f", help="Share of rows with full metadata"),
                           "Ingested (UTC)": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm:ss")},
        )
    st.caption("Each micro-batch is one trigger of the Structured Streaming job in "
               "src/phases/streaming_ingest.py; latency is ingest time minus the scrape timestamp "
               "carried in each record.")


DATASET_PREFIX = "novels_clean"
PAGE_SIZES = [10, 25, 50, 100, 250]
#: Model-input columns derived from the text columns; search skips them so it matches what a reader sees.
DERIVED_TEXT = {"corpus_raw", "corpus_clean", "synopsis_clean", "tokens", "synopsis_tokens"}


def _read_clean_tier(tier: str) -> tuple[pd.DataFrame, list[str]]:
    """One metadata tier, with list-of-string columns joined by ", " in Arrow, exactly as _cell shows them.

    Kept as arrays, `tokens` and `synopsis_tokens` alone become ~8 million Python strings (~0.5 GB).
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    bucket = SETTINGS.minio.bucket_clean
    prefix = f"{DATASET_PREFIX}/metadata_tier={tier}/"
    keys = sorted(o["Key"] for o in list_objects(bucket, prefix) if o["Key"].endswith(".parquet"))
    if not keys:
        raise FileNotFoundError(prefix)
    client = get_s3_client(SETTINGS)
    table = pa.concat_tables(
        [pq.read_table(io.BytesIO(client.get_object(Bucket=bucket, Key=key)["Body"].read())) for key in keys],
        promote_options="default",
    )
    joined = []
    for i, field in enumerate(table.schema):
        if (pa.types.is_list(field.type) or pa.types.is_large_list(field.type)) and (
                pa.types.is_string(field.type.value_type) or pa.types.is_large_string(field.type.value_type)):
            # Spark writes `string not null` elements, which binary_join has no kernel for.
            column = table.column(i).cast(pa.list_(pa.string()))
            table = table.set_column(i, field.name, pc.binary_join(column, ", "))
            joined.append(field.name)
    return table.to_pandas(), joined


@st.cache_resource(show_spinner=False, max_entries=1)
def _dataset(version: str) -> pd.DataFrame:
    """The whole clean table, both metadata tiers. A resource, not data: 300 MB is never copied per rerun."""
    parts, joined = [], set()
    for tier in ("full", "listing"):
        try:
            part, lists = _read_clean_tier(tier)
        except FileNotFoundError:
            continue
        except Exception as exc:
            raise DataUnavailable(str(exc)) from exc
        parts.append(part.assign(metadata_tier=tier))
        joined.update(lists)
    if not parts:
        raise DataUnavailable(f"No parquet under {SETTINGS.minio.bucket_clean}/{DATASET_PREFIX}")
    frame = pd.concat(parts, ignore_index=True)
    # Joined list columns stay unsortable and unsearched, as they were while still arrays.
    frame.attrs["list_columns"] = sorted(joined)
    return frame


@st.cache_resource(show_spinner=False, max_entries=1)
def _dataset_haystack(version: str) -> pd.Series:
    """Lowercase text of every reader-facing text column per row, built on the first search only."""
    frame = _dataset(version)
    skip = DERIVED_TEXT | set(frame.attrs.get("list_columns", ()))
    text_cols = [c for c in frame.columns if frame[c].dtype == object and c not in skip
                 and isinstance(frame[c].dropna().iloc[0] if frame[c].notna().any() else "", str)]
    hay = frame[text_cols[0]].fillna("").astype(str).str.lower()
    for col in text_cols[1:]:
        hay = hay + " \x1f " + frame[col].fillna("").astype(str).str.lower()
    return hay


def _sortable(frame: pd.DataFrame) -> list[str]:
    """Columns with an order: list-valued cells (genre_list, tokens) have none."""
    joined = set(frame.attrs.get("list_columns", ()))

    def scalar(col: str) -> bool:
        values = frame[col].dropna()
        return col not in joined and (values.empty or not isinstance(values.iloc[0], (list, tuple, np.ndarray)))
    return [c for c in frame.columns if scalar(c)]


def column_picker(container, columns: list[str]) -> list[str]:
    """Include/exclude columns from a popover; every column starts included."""
    for col in columns:
        st.session_state.setdefault(f"dataset_col_{col}", True)

    def set_all(value: bool) -> None:
        for col in columns:
            st.session_state[f"dataset_col_{col}"] = value

    included = [c for c in columns if st.session_state[f"dataset_col_{c}"]]
    with container.popover(f"Columns · {len(included)}/{len(columns)}", use_container_width=True):
        a, b = st.columns(2)
        a.button("Include all", on_click=set_all, args=(True,), use_container_width=True)
        b.button("Exclude all", on_click=set_all, args=(False,), use_container_width=True)
        grid = st.columns(3)
        for i, col in enumerate(columns):
            grid[i % 3].checkbox(col, key=f"dataset_col_{col}")
    return [c for c in columns if st.session_state[f"dataset_col_{c}"]]


def _set_page(value: int) -> None:
    st.session_state["dataset_page"] = value


#: Header-sortable table: sorting runs in Python over every row, so a click on a
#: column name orders the whole dataset, not just the page on screen.
_datatable = components.declare_component(
    "nl_datatable", path=str(SETTINGS.paths.project_root / "app" / "components" / "datatable")
)
#: Longest text sent per cell; the cell shows an ellipsis and the full value on hover.
CELL_CHARS = 400
DATASET_LEAD = ["series_id", "title"]


#: Identifiers and calendar years read wrong with thousands separators.
PLAIN_NUMBER = ("series_id", "year", "decade")


def _cell(value, plain: bool = False) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        value = ", ".join(map(str, value))
    elif isinstance(value, float):
        value = (f"{value:.0f}" if plain else f"{value:,.4f}".rstrip("0").rstrip("."))
    elif isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        value = str(value) if plain else f"{value:,}"
    text = str(value)
    return text if len(text) <= CELL_CHARS else text[:CELL_CHARS - 1] + "…"


def page_dataset() -> None:
    hero(
        "Dataset",
        "Every row and every column of the cleaned catalogue, the table all other pages are built "
        "from. Click a column name to sort every row by it.",
    )
    version = lake_version()
    with st.spinner("Loading the full dataset (about 2 seconds, once)…"):
        frame = _dataset(version)
    all_columns = DATASET_LEAD + [c for c in frame.columns if c not in DATASET_LEAD]
    sortable = set(_sortable(frame))

    c1, c2, c3 = st.columns([6, 1.3, 1.7], vertical_alignment="bottom")
    query = c1.text_input("Search", placeholder="Text anywhere in the row, e.g. cultivation")
    size = c2.selectbox("Rows per page", PAGE_SIZES, index=1)
    shown_cols = column_picker(c3, all_columns)
    if not shown_cols:
        st.info("Every column is excluded. Include at least one under Columns.")
        return

    # Default order: the leftmost shown column, ascending. A header click replaces it.
    clicked = st.session_state.get("dataset_sort")
    default_col = next((c for c in shown_cols if c in sortable), None)
    sort_col, sort_dir = (clicked["col"], clicked["dir"]) if clicked and clicked["col"] in shown_cols \
        else (default_col, "asc")

    view = frame
    needle = query.strip().lower()
    if needle:
        with st.spinner("Searching every row…"):
            view = frame[_dataset_haystack(version).str.contains(needle, regex=False).to_numpy()]
    if sort_col:
        view = view.sort_values(sort_col, ascending=sort_dir == "asc", na_position="last", kind="stable")

    total = len(view)
    pages = max(1, -(-total // size))
    # A new search, sort or page size starts again at page 1.
    signature = (needle, sort_col, sort_dir, size)
    if st.session_state.get("dataset_signature") != signature:
        st.session_state["dataset_signature"] = signature
        st.session_state["dataset_page"] = 1
    page = min(max(1, st.session_state.get("dataset_page", 1)), pages)
    start = (page - 1) * size
    rows = view.iloc[start:start + size]
    if rows.empty:
        st.info(f"No row contains “{query.strip()}”. Try a shorter word.")
        return

    clicked = _datatable(
        columns=shown_cols,
        numeric=[bool(pd.api.types.is_numeric_dtype(frame[c])) and frame[c].dtype != bool for c in shown_cols],
        sortable=[c in sortable for c in shown_cols],
        rows=[[_cell(v, c in PLAIN_NUMBER) for v, c in zip(row, shown_cols)]
              for row in rows[shown_cols].itertuples(index=False, name=None)],
        sort_col=sort_col, sort_dir=sort_dir, first_row=start + 1,
        # Capped so 250 rows scroll inside the table instead of stretching the page.
        height=min(42 + 33 * len(rows), 560),
        key="dataset_table", default=None,
    )
    if clicked and clicked != st.session_state.get("dataset_sort"):
        st.session_state["dataset_sort"] = clicked
        st.rerun()

    nav = st.columns([1, 1, 1, 4], vertical_alignment="center")
    nav[0].button("Previous", disabled=page <= 1, on_click=_set_page, args=(page - 1,), use_container_width=True)
    nav[1].button("Next", disabled=page >= pages, on_click=_set_page, args=(page + 1,), use_container_width=True)
    picked = int(nav[2].number_input("Page", 1, pages, page, label_visibility="collapsed"))
    if picked != page:
        _set_page(picked)
        st.rerun()
    nav[3].caption(
        f"Rows {start + 1:,} to {start + len(rows):,} of {total:,}"
        + (f" matching “{query.strip()}”" if needle else "")
        + f" · page {page:,} of {pages:,} · {len(shown_cols)} of {len(all_columns)} columns"
        + (f" · sorted by {sort_col} {'ascending' if sort_dir == 'asc' else 'descending'}" if sort_col else "")
    )


def technical_details(text: str) -> None:
    """Raw error text for maintainers; never on a public deployment, where it would name hosts and paths."""
    if PUBLIC:
        return
    with st.expander("Technical details"):
        st.code(text)


def guarded(render):
    """Show a plain message instead of a traceback when a page's tables are missing."""
    def run() -> None:
        try:
            render()
        except DataUnavailable as exc:
            st.warning("The data for this page is not available yet. Please try again later."
                       if PUBLIC else "The data for this page is not available yet. Run the pipeline "
                       "(`python scripts/run_pipeline.py`) and reload.")
            technical_details(str(exc))
    return run


#: url_path -> page, so pages can link to each other with st.page_link.
PAGE = {
    "overview": st.Page(guarded(page_overview), title="Overview", url_path="overview", default=True),
    "lookup": st.Page(guarded(page_lookup), title="Title Lookup", url_path="lookup"),
    "studio": st.Page(guarded(page_studio), title="AI Studio", url_path="studio"),
    "analytics": st.Page(guarded(page_analytics), title="Catalogue Analytics", url_path="analytics"),
    "trends": st.Page(guarded(page_trends), title="Market Trends", url_path="trends"),
    "genre": st.Page(guarded(page_genre), title="Genre Explorer", url_path="genre"),
    "tags": st.Page(guarded(page_tags), title="Story Elements", url_path="tags"),
    "creators": st.Page(guarded(page_creators), title="Authors & Publishers", url_path="creators"),
    "warehouse": st.Page(guarded(page_warehouse), title="Data Warehouse", url_path="warehouse"),
    "graph": st.Page(guarded(page_graph), title="Relationship Graph", url_path="graph"),
    "content_graph": st.Page(guarded(page_content_graph), title="Content Graph", url_path="similarity"),
    "lake": st.Page(guarded(page_lake), title="Lake Operations", url_path="lake"),
    "dataset": st.Page(guarded(page_dataset), title="Dataset", url_path="dataset"),
    "governance": st.Page(guarded(page_governance), title="Quality & Governance", url_path="governance"),
    "methodology": st.Page(guarded(page_method), title="Methodology", url_path="methodology"),
}
#: Material Symbols per page, picked for what the page holds.
PAGE_ICON = {
    "overview": ":material/space_dashboard:", "lookup": ":material/search:", "studio": ":material/model_training:",
    "analytics": ":material/bar_chart:", "trends": ":material/trending_up:", "genre": ":material/category:",
    "tags": ":material/sell:", "creators": ":material/groups:", "dataset": ":material/table_rows:",
    "lake": ":material/database:", "warehouse": ":material/schema:", "graph": ":material/account_tree:", "content_graph": ":material/hub:",
    "governance": ":material/verified_user:", "methodology": ":material/menu_book:",
}
PAGES = {
    "Discover": [PAGE[k] for k in ("overview", "lookup", "studio")],
    "Market insight": [PAGE[k] for k in ("analytics", "trends", "genre", "tags", "creators")],
    "Data platform": [PAGE[k] for k in (("warehouse", "graph", "content_graph", "governance") if PUBLIC else
                                       ("dataset", "lake", "warehouse", "graph", "content_graph", "governance"))],
    "About": [PAGE["methodology"]],
}


def catalogue_size() -> int | None:
    try:
        return table_profile("analytics/viz_novels")[0]
    except DataUnavailable:
        return None


def lake_status_html() -> str:
    """
    Data-lake status from the same probe main() uses, plus when the curated
    zone last changed. Static dot plus text: the state is real, refreshed on
    each rerun, and never carried by colour alone.
    """
    online, _ = backend_online()
    if not online:
        return (f'<div class="nl-status" role="status"><span class="dot" style="background:{ACCENT_TEXT}"></span>'
                "Data lake unreachable</div>")
    stamp = pd.to_datetime(lake_version(), errors="coerce", utc=True)
    when = f" · updated {stamp:%d %b %Y, %H:%M} UTC" if pd.notna(stamp) else ""
    return (f'<div class="nl-status" role="status"><span class="dot" style="background:{GOOD_TEXT}"></span>'
            f"Data lake connected{when}</div>")


MOBILE_NAV_CLOSE = """<script>
const doc = window.parent.document;
if (!window.parent.__nlNavClose) {
  window.parent.__nlNavClose = true;
  doc.addEventListener("click", (event) => {
    const link = event.target.closest('section[data-testid="stSidebar"] [data-testid="stPageLink"] a');
    if (!link || !window.parent.matchMedia("(max-width: 767px)").matches) return;
    setTimeout(() => doc.querySelector(
      'section[data-testid="stSidebar"] [data-testid="stBaseButton-headerNoPadding"]')?.click(), 60);
  }, true);
}
</script>"""


def sidebar_nav() -> None:
    """Brand, one collapsible group per section, and the data-lake status."""
    keys = {id(page): key for key, page in PAGE.items()}
    size = catalogue_size()
    st.markdown(
        '<div class="nl-brand"><div class="name">NovelLens</div>'
        f'<div class="sub">{f"{size:,} titles" if size else "Catalogue"} · NovelUpdates</div></div>',
        unsafe_allow_html=True,
    )
    for section, pages in PAGES.items():
        with st.expander(section, expanded=True):
            for page in pages:
                st.page_link(page, icon=PAGE_ICON[keys[id(page)]])
    st.markdown(lake_status_html(), unsafe_allow_html=True)
    # Streamlit closes its own navigation on phones after a tap; page links in a
    # custom sidebar do not, so the open sidebar would keep covering the page.
    components.html(MOBILE_NAV_CLOSE, height=0)


def main() -> None:
    # The built-in list cannot fold a section, so the sidebar draws its own.
    page = st.navigation(PAGES, position="hidden")
    with st.sidebar:
        sidebar_nav()

    online, detail = backend_online()
    if not online:
        st.error("The data service is unreachable right now. Please try again shortly.")
        technical_details(detail)
        st.stop()

    page.run()


main()
