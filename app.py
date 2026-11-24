"""Local dashboard for the penalties-per-game tables.

The story uses the whole file. The slice controls sit on one chart lower down. Edit this file to change layout,
colors, or which charts appear. Streamlit reloads the page on save.
"""

from __future__ import annotations

import html
import json
from dataclasses import replace

import pandas
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from analysis import (
    club_opening_flags,
    game_board,
    game_pace,
    games_involving,
    limit_reading,
    matchup_reading,
    month_totals,
    opening_games,
    quarter_breakdown,
    story_facts,
    quarter_matchups,
    SLOT_ORDER,
    schedule_slot,
    run_study,
    team_week_totals,
    type_reading,
    volume_reading,
    week_totals,
    week_window,
)
from nba_page import nba_page
from contact import contact_page
from penalties import (
    TEAM_NAMES,
    Filters,
    game_log,
    load_frames,
    per_game_rate,
    rates_by_season,
    rates_by_team_season,
    subset,
    team_games_count,
    team_summary,
    type_summary,
    TABLES,
    build_tables,
)

ROOT_NOTE = TABLES.parent.parent / "notes.json"
STORY_PATH = TABLES.parent.parent / "story.json"
OPENING_DEFAULT = (
    "Through the first few weeks of this season, a flag seems to fly on every play. "
    "It feels worse than it has ever been. "
    "Too often a team has a game-breaking play called back on a wishy washy flag "
    "or something that had no effect on the outcome of the play. "
    "Flags have always been part of the game. They feel more prominent now, "
    "and week after week they decide more of what you are watching. "
    "I am going through the numbers to see whether that feeling holds up."
)
TIMELINE_DEFAULT = (
    "Average penalties and penalty yards per game, for every regular-season game since 1999. "
    "Dotted lines mark a point of emphasis or a rule change. "
    "The 2020 line is the Covid season."
)
METRIC_LABELS = {
    "penalties_per_game": "Penalties per team",
    "yards_per_game": "Yards per team",
}
LINE_COLORS = [
    "#f3efe4",
    "#e6b325",
    "#e04b45",
    "#2aafc4",
    "#e07a28",
    "#3cba6a",
    "#e05a86",
    "#a06ad4",
]

FIELD = "#17191e"
PANEL = "#21242b"
PLOT = "#1c1f26"
INK = "#f3efe4"
MUTED = "#d9d3c5"
GRID = "#3a4150"
GOLD = "#e6b325"
RULE = "#4c5566"

PAGE_STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600;700&family=Fraunces:opsz,wght@9..144,560;9..144,640&display=swap');

:root {
  --ink: #f3efe4;
  --muted: #d9d3c5;
  --gold: #e6b325;
  --field: #17191e;
  --panel: #21242b;
  --plot: #1c1f26;
  --line: #3a4150;
  --rule: #4c5566;
}
html, body, [class*="css"] {
  font-family: "Barlow", sans-serif;
}
.stApp {
  background: var(--field);
  border-top: 3px solid var(--gold);
}
.block-container {
  max-width: 76rem;
  padding-top: 3.4rem;
  padding-bottom: 4rem;
}
[data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"] {
  display: none !important;
}
.mast {
  margin: 0.4rem auto 1.6rem;
  max-width: 68rem;
  text-align: left;
}
.kicker {
  margin: 0 0 0.55rem;
  color: var(--gold);
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  text-align: left;
}
.mast h1 {
  margin: 0;
  color: var(--ink);
  font-family: "Fraunces", serif;
  font-size: clamp(2.3rem, 4.6vw, 3.8rem);
  font-weight: 560;
  letter-spacing: -0.035em;
  line-height: 1.02;
  text-align: left;
}
div[data-testid="stMarkdownContainer"]:has(.reading),
div[data-testid="stMarkdownContainer"]:has(.chapter),
div[data-testid="stMarkdownContainer"]:has(.deck),
div[data-testid="stMarkdownContainer"]:has(.section-label),
div[data-testid="stMarkdownContainer"]:has(.chart-title),
div[data-testid="stMarkdownContainer"]:has(.hint),
div[data-testid="stMarkdownContainer"]:has(.mast) {
  display: flex;
  flex-direction: column;
  align-items: center;
}
.deck {
  margin: 0.9rem auto 0;
  color: var(--muted);
  font-size: 1.12rem;
  line-height: 1.55;
  width: min(68rem, 100%);
  text-align: left;
}
.section-label {
  margin: 0.15rem auto 0.35rem;
  color: var(--gold);
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  width: min(68rem, 100%);
  text-align: center;
}
.reading {
  margin: 2.4rem auto 0.15rem !important;
  color: var(--ink);
  font-family: "Fraunces", serif;
  font-size: clamp(1.7rem, 3vw, 2.25rem);
  font-weight: 560;
  letter-spacing: -0.03em;
  line-height: 1.15;
  width: min(68rem, 100%);
  text-align: center;
}
.section-label.when {
  margin: 0.15rem auto 0.7rem !important;
  width: min(68rem, 100%);
  color: var(--muted);
  text-align: center;
}
.chart-title {
  margin: 1.35rem auto 0.45rem;
  color: var(--ink);
  font-family: "Fraunces", serif;
  font-size: clamp(1.25rem, 2vw, 1.55rem);
  font-weight: 560;
  letter-spacing: -0.02em;
  line-height: 1.25;
  width: min(68rem, 100%);
  text-align: center;
}
[data-testid="stElementContainer"]:has(.part-break) {
  margin-top: 5.75rem;
}
.part-break {
  width: min(68rem, 100%);
  margin: 0 auto;
  border: 0;
  border-top: 2px solid var(--gold);
}
.part-title {
  margin: 1.7rem auto 0.45rem !important;
  color: var(--ink);
  font-family: "Fraunces", serif;
  font-size: clamp(2.7rem, 6vw, 4.1rem);
  font-weight: 560;
  letter-spacing: -0.045em;
  line-height: 0.98;
  width: min(68rem, 100%);
  text-align: center;
}
[data-testid="stMarkdownContainer"] p.part-title {
  margin: 1.7rem auto 0.45rem !important;
  color: var(--ink) !important;
  font-family: "Fraunces", serif !important;
  font-size: clamp(2.7rem, 6vw, 4.1rem) !important;
  font-weight: 560 !important;
  letter-spacing: -0.045em !important;
  line-height: 0.98 !important;
  text-align: center !important;
}
[data-testid="stMarkdownContainer"] hr.part-break {
  height: 2px !important;
  border: 0 !important;
  border-top: 2px solid var(--gold) !important;
  background: var(--gold) !important;
  margin: 0 auto !important;
}
[data-testid="stElementContainer"]:has(.section-label):not(:has(.reading)) {
  margin-top: 3.4rem;
}
[data-testid="stElementContainer"]:has(.section-label):not(:has(.reading)) + [data-testid="stElementContainer"] .reading {
  margin-top: 0.35rem !important;
}
.hint {
  margin: 0.45rem auto 0.9rem;
  width: min(68rem, 100%);
  padding: 0.65rem 0.9rem;
  border: 1px solid var(--line);
  border-left: 3px solid var(--gold);
  border-radius: 10px;
  background: var(--panel);
  color: var(--muted);
  font-size: 0.98rem;
  line-height: 1.45;
  text-align: left;
}
.gloss {
  display: flex;
  align-items: center;
  justify-content: flex-start;
  gap: 0.4rem;
  margin: 0.55rem 0;
  color: var(--muted);
  font-size: 0.78rem;
  line-height: 1.25;
}
.gloss-row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: flex-end;
  gap: 0.35rem 1.15rem;
  min-height: 2.4rem;
}
.gloss-row .gloss {
  margin: 0;
}
.gloss-row .gloss > span:first-child {
  width: auto;
}
.gloss-row .qmark .tip {
  top: 1.25rem;
  right: 0;
  transform: none;
}
.qmark {
  position: relative;
  flex: 0 0 auto;
  width: 0.95rem;
  height: 0.95rem;
  border: 1px solid var(--gold);
  border-radius: 50%;
  color: var(--gold);
  font-size: 0.62rem;
  line-height: 0.85rem;
  text-align: center;
  cursor: help;
}
.qmark input {
  appearance: none;
  -webkit-appearance: none;
  opacity: 0;
  position: absolute;
  inset: 0;
  margin: 0;
  cursor: help;
}
.qmark .tip {
  display: none;
  position: absolute;
  right: 1.35rem;
  top: 50%;
  transform: translateY(-40%);
  width: 16rem;
  padding: 0.55rem 0.7rem;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #2a303a;
  color: var(--ink);
  font-size: 0.75rem;
  line-height: 1.4;
  text-align: left;
  z-index: 30;
}
.qmark:hover .tip,
.qmark:focus-within .tip,
.qmark input:checked ~ .tip {
  display: block;
}
[data-testid="stElementContainer"]:has(.gloss),
[data-testid="stVerticalBlock"]:has(.gloss) {
  overflow: visible;
}
[data-testid="stColumn"]:has(.gloss),
[data-testid="stHorizontalBlock"]:has(.gloss) {
  position: relative;
  z-index: 4;
  overflow: visible;
}
.chapter.lead {
  margin-top: 1.85rem !important;
}
.chapter {
  margin: 0.15rem auto 0.6rem !important;
  color: var(--muted);
  font-family: "Barlow", sans-serif;
  font-size: 1.05rem;
  font-weight: 400;
  line-height: 1.65;
  width: min(68rem, 100%);
  text-align: left;
}
[data-testid="stCaptionContainer"] {
  max-width: 68rem;
  margin-left: auto;
  margin-right: auto;
  text-align: left;
}
[data-testid="stCaptionContainer"] p {
  color: var(--muted);
  text-align: left;
}
[data-testid="stMetric"] {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 0.9rem 1rem 0.75rem;
}
[data-testid="stElementContainer"]:has([data-testid="stPlotlyChart"]) {
  overflow: visible !important;
}
[data-testid="stPlotlyChart"] {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 16px;
  box-sizing: border-box;
  padding: 0;
  margin: 0.15rem auto 0.35rem;
  max-width: 68rem;
  overflow: hidden;
}
.modebar-container, .modebar {
  display: none !important;
}
[data-testid="stExpander"] {
  max-width: 68rem;
  margin-left: auto;
  margin-right: auto;
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 14px;
}
[data-testid="stDataFrame"] {
  max-width: 68rem;
  margin-left: auto;
  margin-right: auto;
}
[data-testid="stMetricLabel"] {
  color: #d9d3c5;
  font-size: 0.75rem;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
[data-testid="stMetricValue"] {
  color: #f3efe4;
  font-family: "Fraunces", serif;
  font-variant-numeric: tabular-nums;
  font-weight: 560;
}
[data-testid="stMetricDelta"] {
  font-variant-numeric: tabular-nums;
}
button[data-testid="stBaseButton-primary"] {
  background-color: #e6b325;
  border-color: #e6b325;
  color: #1a1608;
}
button[data-testid="stBaseButton-primary"]:hover {
  background-color: #f0c84a;
  border-color: #f0c84a;
  color: #1a1608;
}
button:focus-visible {
  outline: 2px solid #f3efe4;
  outline-offset: 2px;
}
.index {
  max-width: 44rem;
  margin: 0.2rem auto 2.5rem;
}
.subjects {
  margin-top: 1.75rem;
}
.subject {
  border-top: 1px solid #4c5566;
}
.subject:not([open]) > :not(summary) {
  display: none !important;
}
.subject:last-of-type {
  border-bottom: 1px solid #4c5566;
}
.subject[open] {
  border-top-color: #e6b325;
}
.subject summary {
  display: flex;
  align-items: center;
  gap: 1rem;
  min-height: 56px;
  padding: 0.9rem 0.1rem;
  cursor: pointer;
  list-style: none;
  color: #f3efe4;
}
.subject summary::-webkit-details-marker,
.subject summary::marker {
  display: none;
  content: "";
}
.subject summary::after {
  content: "▾";
  width: auto;
  height: auto;
  border: 0;
  transform: none;
  flex: none;
  color: #e6b325;
  font-family: "Barlow", sans-serif;
  font-size: 0.95rem;
  line-height: 1;
}
.subject[open] summary::after {
  content: "▴";
  transform: none;
}
.subject summary:hover {
  color: #e6b325;
}
.subject summary:focus-visible,
.project:focus-visible,
.back a:focus-visible,
.colophon a:focus-visible {
  outline: 2px solid #f3efe4;
  outline-offset: 3px;
}
.subject-name {
  margin: 0;
  flex: 1;
  font-family: "Fraunces", serif;
  font-size: clamp(1.6rem, 3vw, 2.05rem);
  font-weight: 560;
  letter-spacing: -0.03em;
  line-height: 1.15;
}
.subject-count {
  flex: none;
  color: #d9d3c5;
  font-family: "Barlow", sans-serif;
  font-size: 0.78rem;
  font-weight: 600;
  letter-spacing: 0.14em;
}
.project {
  display: block;
  padding: 0.15rem 0.1rem 1.15rem;
  color: inherit;
  text-decoration: none;
}
.project + .project {
  border-top: 1px solid #3a4150;
  padding-top: 0.95rem;
}
.project-title {
  margin: 0;
  color: #f3efe4;
  font-family: "Fraunces", serif;
  font-size: 1.28rem;
  font-weight: 560;
  letter-spacing: -0.02em;
  line-height: 1.3;
}
.project-dek,
.empty-subject {
  display: block;
  margin: 0.35rem 0 0;
  color: #d9d3c5;
  font-family: "Barlow", sans-serif;
  font-size: 1.05rem;
  font-weight: 400;
  letter-spacing: 0;
  line-height: 1.45;
  text-transform: none;
}
.empty-subject {
  margin: 0;
  padding: 0.1rem 0.1rem 1.15rem;
}
.project:hover .project-title {
  color: #e6b325;
}
.back {
  margin: 0 0 0.85rem;
  text-align: left;
}
.colophon a {
  color: #d9d3c5;
  font-family: "Barlow", sans-serif;
  font-size: 0.78rem;
  font-weight: 600;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  text-decoration: none;
}
.back a {
  display: inline-flex;
  align-items: center;
  gap: 0.45rem;
  min-height: 44px;
  padding: 0.35rem 0.95rem 0.35rem 0.8rem;
  border: 1px solid #4c5566;
  background: #2a303a;
  color: #f3efe4;
  font-family: "Barlow", sans-serif;
  font-size: 0.95rem;
  font-weight: 600;
  letter-spacing: 0.02em;
  text-decoration: none;
}
.back a::before {
  content: "←";
  color: #e6b325;
  font-size: 1.05rem;
  line-height: 1;
}
.back a:hover {
  border-color: #e6b325;
  color: #f3efe4;
}
.colophon a:hover,
.subject summary:hover .subject-count {
  color: #e6b325;
}
.colophon {
  margin: 1.7rem 0 0;
  text-align: center;
}
footer {visibility: hidden;}
@media (prefers-reduced-motion: reduce) {
  * { animation: none !important; transition: none !important; }
}
</style>
"""


def load_note() -> dict[str, str]:
    if not ROOT_NOTE.exists():
        return {"title": "", "body": ""}
    try:
        saved = json.loads(ROOT_NOTE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"title": "", "body": ""}
    return {
        "title": str(saved.get("title", "")),
        "body": str(saved.get("body", "")),
    }


def load_story() -> dict[str, str]:
    if not STORY_PATH.exists():
        return {}
    try:
        saved = json.loads(STORY_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(saved, dict):
        return {}
    return {str(key): str(value) for key, value in saved.items()}


def save_story(blocks: dict[str, str]) -> None:
    STORY_PATH.write_text(json.dumps(blocks, indent=2), encoding="utf-8")


def personal_copy() -> bool:
    """Edit controls stay on this machine. The live site does not show them."""
    try:
        url = str(st.context.url or "")
        headers = st.context.headers
        host = ""
        if headers is not None:
            host = str(headers.get("Host") or headers.get("host") or "")
    except Exception:
        return False
    blob = f"{url} {host}".lower()
    if "streamlit.app" in blob:
        return False
    return "localhost" in blob or "127.0.0.1" in blob


def story_line(key: str, default: str) -> str:
    saved = load_story().get(key, "")
    if saved.strip():
        return saved
    return default


def story_html(text: str) -> str:
    return "<br>".join(html.escape(line) for line in text.split("\n"))


def story_paragraph(key: str, default: str, css_class: str) -> None:
    """Show one story paragraph. On localhost, Edit underneath opens that paragraph only."""
    saved = load_story().get(key, "")
    if saved.strip().lower() in {"delete", "deleted"}:
        return
    state_key = f"story_edit_{key}"
    editing = personal_copy() and st.session_state.get("story_editing") == key
    if editing:
        if state_key not in st.session_state:
            st.session_state[state_key] = story_line(key, default)
        _left, mid, _right = st.columns([0.2, 2.6, 0.2])
        with mid:
            with st.form(f"story_form_{key}"):
                st.text_area("Paragraph", key=state_key, height=220, label_visibility="collapsed")
                st.caption("Clear the box and save to bring back the original.")
                save_col, cancel_col = st.columns(2)
                with save_col:
                    do_save = st.form_submit_button("Save", type="primary")
                with cancel_col:
                    do_cancel = st.form_submit_button("Cancel")
        if do_save:
            text = st.session_state.get(state_key, "").strip()
            saved = load_story()
            if text and text != default.strip():
                saved[key] = text
            else:
                saved.pop(key, None)
            save_story(saved)
            st.session_state["story_editing"] = None
            st.session_state.pop(state_key, None)
            st.toast("Saved")
            st.rerun()
        if do_cancel:
            st.session_state["story_editing"] = None
            st.session_state.pop(state_key, None)
            st.rerun()
        return
    st.markdown(
        f'<p class="{css_class}">{story_html(story_line(key, default))}</p>',
        unsafe_allow_html=True,
    )
    if not personal_copy():
        return
    _left, mid, _right = st.columns([1, 0.28, 1])
    with mid:
        if st.button("Edit", key=f"story_edit_btn_{key}"):
            st.session_state["story_editing"] = key
            st.rerun()


def save_note(title: str, body: str) -> None:
    ROOT_NOTE.write_text(
        json.dumps({"title": title, "body": body}, indent=2),
        encoding="utf-8",
    )


@st.cache_data(show_spinner="Loading tables...")
def cached_frames(_games_mtime: float, _penalties_mtime: float, _schema: int = 3):
    return load_frames()


def frames():
    games_path = TABLES / "games.parquet"
    penalties_path = TABLES / "penalties.parquet"
    if not games_path.exists() or not penalties_path.exists():
        build_tables()
    return cached_frames(games_path.stat().st_mtime, penalties_path.stat().st_mtime)


def season_axis(seasons) -> dict:
    values = pandas.to_numeric(pandas.Series(seasons), errors="coerce").dropna()
    axis = dict(
        gridcolor="#3a4150",
        tickmode="linear",
        dtick=2,
        tickfont=dict(color="#d9d3c5"),
        title=None,
    )
    if values.empty:
        return axis
    first = int(values.min())
    last = int(values.max())
    axis["tick0"] = first
    axis["range"] = [first - 0.5, last + 0.5]
    return axis


def chart_layout(height: int) -> dict:
    return dict(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=PLOT,
        font=dict(family="Barlow, sans-serif", color=INK, size=14),
        margin=dict(l=8, r=56, t=12, b=8),
        hoverlabel=dict(
            bgcolor=PANEL,
            bordercolor=RULE,
            font_size=13,
            font_family="Barlow, sans-serif",
            font_color=INK,
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(color=INK)),
        dragmode=False,
    )


def smooth_line(color: str, width: float = 3, dash: str | None = None) -> dict:
    """A curve through the points, so a season line bends instead of kinking."""
    style = dict(color=color, width=width, shape="spline", smoothing=0.85)
    if dash:
        style["dash"] = dash
    return style


def lift_markers(figure: go.Figure) -> None:
    """Draw the season dots on top of the curve. A spline on the same trace hides them."""
    additions = []
    for trace in figure.data:
        if "lines" not in str(trace.mode or ""):
            continue
        color = trace.line.color if trace.line and trace.line.color else "#e6b325"
        marker = go.Scatter(
            x=trace.x,
            y=trace.y,
            mode="markers",
            marker=dict(size=8, color=color, line=dict(color="#2a303a", width=1.5)),
            hovertemplate=trace.hovertemplate or "%{x}<br>%{y}<extra></extra>",
            name=trace.name,
            showlegend=False,
            legendgroup=trace.legendgroup or trace.name,
        )
        if trace.customdata is not None:
            marker.customdata = trace.customdata
        additions.append(marker)
        trace.update(mode="lines", hoverinfo="skip")
    for extra in additions:
        figure.add_trace(extra)


def rank_shades(count: int) -> list[str]:
    """Gold into sage, so a long chart stays in one family instead of a rainbow."""
    start = (232, 196, 106)
    end = (112, 148, 128)
    if count <= 1:
        return ["#e6b325"]
    shades = []
    for index in range(count):
        mix = index / (count - 1)
        red = round(start[0] + (end[0] - start[0]) * mix)
        green = round(start[1] + (end[1] - start[1]) * mix)
        blue = round(start[2] + (end[2] - start[2]) * mix)
        shades.append(f"#{red:02x}{green:02x}{blue:02x}")
    return shades


def ranking_figure(summary: pandas.DataFrame, metric: str, league: float | None, low_first: bool):
    frame = summary.sort_values([metric, "team"], ascending=[low_first, True])
    names = frame["name"].tolist()
    value_label = METRIC_LABELS[metric]
    colors = [TEAM_COLORS.get(team, "#f3efe4") for team in frame["team"]]
    figure = go.Figure(
        go.Bar(
            x=frame[metric],
            y=names,
            orientation="h",
            marker_color=colors,
            text=[f"{value:.2f}" for value in frame[metric]],
            textposition="outside",
            textfont=dict(color="#f3efe4", family="Barlow, sans-serif", size=13),
            cliponaxis=False,
            customdata=frame.loc[:, ["team", "games", "penalties", "yards"]].to_numpy(),
            hovertemplate=(
                "<b>%{customdata[0]}</b> %{y}<br>"
                + value_label
                + ": %{x:.2f}<br>"
                + "Games: %{customdata[1]}<br>"
                + "Penalties: %{customdata[2]}<br>"
                + "Yards: %{customdata[3]:.0f}<extra></extra>"
            ),
        )
    )
    if league is not None:
        figure.add_vline(
            x=league,
            line_dash="dash",
            line_color="#e6b325",
            line_width=2,
        )
        figure.add_annotation(
            x=league,
            y=1,
            yref="paper",
            text="League",
            showarrow=False,
            yanchor="bottom",
            yshift=6,
            font=dict(color="#e6b325", family="Barlow, sans-serif", size=13),
        )
    layout = chart_layout(max(460, 34 * len(frame) + 110))
    layout["margin"] = dict(l=8, r=56, t=36, b=8)
    layout["showlegend"] = False
    figure.update_layout(
        **layout,
        yaxis=dict(
            categoryorder="array",
            categoryarray=list(reversed(names)),
            tickfont=dict(color="#f3efe4", size=14),
            showgrid=False,
        ),
        xaxis=dict(
            gridcolor="#3a4150",
            zeroline=False,
            title=None,
            range=[0, float(frame[metric].max()) * 1.18] if not frame.empty else None,
            tickfont=dict(color="#d9d3c5"),
        ),
    )
    return figure


def trend_figure(league_trend: pandas.DataFrame, team_trend: pandas.DataFrame, metric: str):
    value_label = METRIC_LABELS[metric]
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=league_trend["season"],
            y=league_trend[metric],
            mode="lines+markers",
            name="League",
            line=smooth_line("#e6b325", dash="dash"),
            marker=dict(size=8, color="#e6b325"),
            hovertemplate="League %{x}<br>" + value_label + ": %{y:.2f}<extra></extra>",
        )
    )
    if not team_trend.empty:
        for index, (team, group) in enumerate(team_trend.groupby("team", sort=True)):
            group = group.sort_values("season")
            color = LINE_COLORS[index % len(LINE_COLORS)]
            figure.add_trace(
                go.Scatter(
                    x=group["season"],
                    y=group[metric],
                    mode="lines+markers",
                    name=f"{team} {group['name'].iloc[0]}",
                    line=smooth_line(color, width=2),
                    marker=dict(size=7, color=color),
                    hovertemplate=team + " %{x}<br>" + value_label + ": %{y:.2f}<extra></extra>",
                )
            )
    figure.update_layout(
        **chart_layout(420),
        xaxis=dict(
            gridcolor="#3a4150",
            tickmode="linear",
            dtick=1,
            tickfont=dict(color="#d9d3c5"),
            title=None,
        ),
        yaxis=dict(gridcolor="#3a4150", zeroline=False, tickfont=dict(color="#d9d3c5"), title=None),
    )
    lift_markers(figure)
    return figure


def type_figure(summary: pandas.DataFrame, metric: str):
    frame = summary.sort_values(metric, ascending=False).head(12)
    frame = frame.sort_values(metric, ascending=True)
    value_label = METRIC_LABELS[metric]
    figure = go.Figure(
        go.Bar(
            x=frame[metric],
            y=frame["penalty_type"],
            orientation="h",
            marker_color="#e07a28",
            text=[f"{value:.2f}" for value in frame[metric]],
            textposition="outside",
            textfont=dict(color="#f3efe4", family="Barlow, sans-serif", size=13),
            cliponaxis=False,
            customdata=frame.loc[:, ["penalties", "yards"]].to_numpy(),
            hovertemplate=(
                "%{y}<br>"
                + value_label
                + ": %{x:.2f}<br>"
                + "Penalties: %{customdata[0]}<br>"
                + "Yards: %{customdata[1]:.0f}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        **chart_layout(max(380, 32 * len(frame) + 60)),
        yaxis=dict(tickfont=dict(color="#f3efe4", size=13), showgrid=False),
        xaxis=dict(gridcolor="#3a4150", zeroline=False, tickfont=dict(color="#d9d3c5"), title=None),
    )
    return figure


def game_pace_figure(by_scope: pandas.DataFrame, current_season: int) -> go.Figure:
    figure = go.Figure()
    styles = (
        ("REG", "Regular season", "#e6b325"),
        ("POST", "Playoffs", "#2aafc4"),
    )
    for key, name, color in styles:
        group = by_scope.loc[by_scope["season_type"].eq(key)].sort_values("season")
        if group.empty:
            continue
        figure.add_trace(
            go.Scatter(
                x=group["season"],
                y=group["flags_per_game"],
                mode="lines+markers",
                name=name,
                customdata=group["games"],
                line=smooth_line(color),
                marker=dict(size=7, color=color),
                hovertemplate=name + " %{x}<br>%{y:.2f} flags in the game<br>%{customdata} games<extra></extra>",
            )
        )
    figure.add_vline(x=current_season - 0.5, line_dash="dot", line_color="#d9d3c5", line_width=1)
    figure.update_layout(
        **chart_layout(420),
        xaxis=season_axis(by_scope["season"]),
        yaxis=dict(
            gridcolor="#3a4150",
            zeroline=False,
            tickfont=dict(color="#d9d3c5"),
            title=dict(text="Flags in the game, both teams", font=dict(color="#d9d3c5", size=13)),
        ),
    )
    lift_markers(figure)
    return figure


PLAYOFF_ROUNDS = ("Wild card", "Divisional", "Conference", "Super Bowl")


def playoff_round_figure(games: pandas.DataFrame, penalties: pandas.DataFrame) -> go.Figure:
    """Average flags in the game for each playoff round, wild card through the Super Bowl."""
    board = game_board(games, penalties)
    post = board.loc[board["season_type"].eq("POST")].copy()
    post["round"] = [
        schedule_slot(season, week, season_type)
        for season, week, season_type in zip(post["season"], post["week"], post["season_type"], strict=False)
    ]
    post = post.loc[post["round"].isin(PLAYOFF_ROUNDS)]
    grouped = post.groupby("round", as_index=False).agg(games=("game_id", "nunique"), flags=("flags", "mean"))
    grouped["round"] = pandas.Categorical(grouped["round"], categories=PLAYOFF_ROUNDS, ordered=True)
    grouped = grouped.sort_values("round")
    rounds = grouped["round"].astype(str)
    flags = grouped["flags"]
    last = len(grouped) - 1
    colors = ["#2aafc4"] * len(grouped)
    sizes = [11] * len(grouped)
    if last >= 0:
        colors[last] = "#e6b325"
        sizes[last] = 16
    figure = go.Figure(
        go.Scatter(
            x=rounds,
            y=flags,
            mode="lines+text",
            line=smooth_line("#2aafc4"),
            text=[f"{value:.1f}" for value in flags],
            textposition="top center",
            textfont=dict(color="#f3efe4", family="Barlow, sans-serif", size=15),
            cliponaxis=False,
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=rounds,
            y=flags,
            mode="markers",
            marker=dict(size=sizes, color=colors, line=dict(color="#2a303a", width=1.5)),
            cliponaxis=False,
            customdata=grouped["games"],
            hovertemplate="%{x}<br>%{y:.2f} flags in the game<br>%{customdata} games<extra></extra>",
            showlegend=False,
        )
    )
    layout = chart_layout(400)
    layout["margin"] = dict(l=8, r=16, t=8, b=8)
    layout["showlegend"] = False
    low = float(flags.min()) if not flags.empty else 0.0
    high = float(flags.max()) if not flags.empty else 1.0
    span = max(high - low, 0.4)
    figure.update_layout(
        **layout,
        xaxis=dict(tickfont=dict(color="#f3efe4", size=14), showgrid=False),
        yaxis=dict(
            gridcolor="#3a4150",
            zeroline=False,
            range=[low - span * 0.15, high + span * 0.55],
            dtick=0.5,
            tickfont=dict(color="#d9d3c5"),
            title=dict(text="Flags in the game, both teams", font=dict(color="#d9d3c5", size=13)),
        ),
    )
    return figure


def week_figure(league: pandas.DataFrame, team_lines: list[tuple[str, pandas.DataFrame]]) -> go.Figure:
    figure = go.Figure()
    if not league.empty:
        figure.add_trace(
            go.Scatter(
                x=league["slot"].astype(str),
                y=league["flags"],
                mode="lines+markers",
                name="Flags in the game",
                customdata=league["games"],
                line=smooth_line("#e6b325"),
                marker=dict(size=8, color="#e6b325"),
                hovertemplate="%{x}<br>%{y:.2f} flags in the game<br>%{customdata} games<extra></extra>",
            )
        )
    for index, (name, frame) in enumerate(team_lines):
        color = LINE_COLORS[(index + 1) % len(LINE_COLORS)]
        figure.add_trace(
            go.Scatter(
                x=frame["slot"].astype(str),
                y=frame["team_flags"],
                mode="lines+markers",
                name=name,
                customdata=frame["games"],
                line=smooth_line(color, width=2),
                marker=dict(size=7, color=color),
                hovertemplate=name + "<br>%{x}: %{y:.2f} flags charged to them<br>%{customdata} games<extra></extra>",
            )
        )
    figure.update_layout(
        **chart_layout(380),
        xaxis=dict(
            gridcolor="#3a4150",
            type="category",
            categoryorder="array",
            categoryarray=SLOT_ORDER,
            tickangle=-40,
            tickfont=dict(color="#d9d3c5"),
            title=None,
        ),
        yaxis=dict(
            gridcolor="#3a4150",
            zeroline=False,
            tickfont=dict(color="#d9d3c5"),
            title=dict(text="Flags in the game, both teams", font=dict(color="#d9d3c5", size=13)),
        ),
    )
    lift_markers(figure)
    return figure


def month_figure(frame: pandas.DataFrame) -> go.Figure:
    figure = go.Figure(
        go.Bar(
            x=frame["month"].astype(str),
            y=frame["flags"],
            marker_color="#e6b325",
            text=[f"{value:.1f}" for value in frame["flags"]],
            textposition="outside",
            textfont=dict(color="#f3efe4", family="Barlow, sans-serif", size=13),
            cliponaxis=False,
            customdata=frame["games"],
            hovertemplate="%{x}<br>%{y:.2f} flags in the game<br>%{customdata} games<extra></extra>",
        )
    )
    figure.update_layout(
        **chart_layout(360),
        xaxis=dict(tickfont=dict(color="#f3efe4"), showgrid=False),
        yaxis=dict(gridcolor="#3a4150", zeroline=False, tickfont=dict(color="#d9d3c5"), title=None),
    )
    return figure


def quarter_figure(frame: pandas.DataFrame) -> go.Figure:
    shown = frame.dropna(subset=["per_game"])
    figure = go.Figure(
        go.Bar(
            x=shown["quarter"],
            y=shown["per_game"],
            marker_color=["#e6b325", "#e07a28", "#2aafc4", "#e04b45", "#3cba6a"][: len(shown)],
            text=[f"{value:.1f}" for value in shown["per_game"]],
            textposition="outside",
            textfont=dict(color="#f3efe4", family="Barlow, sans-serif", size=13),
            cliponaxis=False,
            customdata=shown.loc[:, ["flags", "games"]].to_numpy(),
            hovertemplate=(
                "%{x}<br>%{y:.2f} per game<br>%{customdata[0]} flags<br>%{customdata[1]} games<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        **chart_layout(360),
        xaxis=dict(tickfont=dict(color="#f3efe4"), showgrid=False),
        yaxis=dict(gridcolor="#3a4150", zeroline=False, tickfont=dict(color="#d9d3c5"), title=None),
    )
    return figure


TYPE_LINE_COLORS = {
    "Defensive Holding": "#e6b325",
    "Illegal Contact": "#e04b45",
    "Taunting": "#e07a28",
}
MEMO_SEASONS = (2004, 2021)


def format_p(pvalue: float) -> str:
    if pvalue < 0.001:
        return "<0.001"
    return f"{pvalue:.3f}"


def add_memo_lines(figure: go.Figure) -> None:
    for season in MEMO_SEASONS:
        figure.add_vline(x=season - 0.5, line_dash="dot", line_color="#2aafc4", line_width=1)
    figure.add_vline(x=2020 - 0.5, line_dash="dot", line_color="#e04b45", line_width=1)


def center_hover(lines: list[str]) -> str:
    """Pad shorter hover lines so the words sit in the middle of the box."""
    width = max(len(line) for line in lines)
    padded = []
    for line in lines:
        gap = width - len(line)
        left = gap // 2
        right = gap - left
        padded.append(("\u00a0" * left) + line + ("\u00a0" * right))
    return "<br>".join(padded)


def hover_lines(text: str, width: int = 38, limit: int = 2) -> str:
    """Wrap a hover note into a few short lines so the box stays on the point."""
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = word if not current else f"{current} {word}"
        if len(trial) <= width:
            current = trial
            continue
        if current:
            lines.append(current)
        current = word
    if current:
        lines.append(current)
    if len(lines) > limit:
        lines = lines[:limit]
        lines[-1] = lines[-1].rstrip(".,;:") + "…"
    return "<br>".join(lines)


def add_story_marks(figure: go.Figure, labeled: bool) -> None:
    for season, label, shift, color, detail in STORY_MARKS:
        figure.add_vline(x=season, line_dash="dot", line_color=color, line_width=1)
        note = detail if "<br>" in detail else hover_lines(detail)
        boxed = center_hover([label, *note.split("<br>")])
        if not labeled:
            continue
        if season == 2014:
            xanchor, xshift = "right", -6
        elif season == 2024:
            xanchor, xshift = "left", 8
        else:
            xanchor, xshift = "center", 0
        figure.add_annotation(
            x=season,
            y=1,
            yref="paper",
            text=label,
            hovertext=boxed,
            hoverlabel=dict(
                bgcolor="#2a303a",
                bordercolor=color,
                font=dict(size=13, color="#f3efe4", family="Consolas, ui-monospace, monospace"),
            ),
            showarrow=False,
            xanchor=xanchor,
            xshift=xshift,
            yanchor="bottom",
            yshift=shift,
            font=dict(size=12, color=color),
        )


def study_rate_figure(season: pandas.DataFrame, rates: pandas.DataFrame) -> go.Figure:
    frame = season.loc[:, ["season", "actual"]].copy()
    frame["flags"] = frame["actual"] * 2
    frame = frame.merge(rates.loc[:, ["season", "yards", "games"]], on="season", how="left")
    figure = go.Figure(
        go.Scatter(
            x=frame["season"],
            y=frame["flags"],
            mode="lines+markers",
            name="Flags in the game",
            customdata=frame.loc[:, ["yards", "games"]].to_numpy(),
            line=smooth_line("#e6b325"),
            marker=dict(size=7, color="#e6b325"),
            hovertemplate=(
                "%{x}<br>%{y:.2f} flags in the game"
                "<br>%{customdata[0]:.0f} yards in the game"
                "<br>%{customdata[1]:.0f} games<extra></extra>"
            ),
        )
    )
    lift_markers(figure)
    add_story_marks(figure, labeled=True)
    layout = chart_layout(500)
    layout["margin"] = dict(l=8, r=148, t=78, b=8)
    layout["showlegend"] = False
    figure.update_layout(
        **layout,
        xaxis=season_axis(frame["season"]),
        yaxis=dict(
            gridcolor="#3a4150",
            zeroline=False,
            tickfont=dict(color="#d9d3c5"),
            title=dict(text="Flags in the game, both teams", font=dict(color="#d9d3c5", size=13)),
        ),
    )
    return figure


def study_type_figure(type_season: pandas.DataFrame) -> go.Figure:
    figure = go.Figure()
    for penalty_type, color in TYPE_LINE_COLORS.items():
        group = type_season.loc[type_season["penalty_type"].eq(penalty_type)].sort_values("season")
        if group.empty:
            continue
        figure.add_trace(
            go.Scatter(
                x=group["season"],
                y=group["rate"] * 2,
                mode="lines",
                name=penalty_type,
                line=smooth_line(color, width=2.5),
                hovertemplate=penalty_type + "<br>%{x}: %{y:.2f} in the game<extra></extra>",
            )
        )
    add_memo_lines(figure)
    figure.update_layout(
        **chart_layout(420),
        xaxis=season_axis(type_season["season"]),
        yaxis=dict(
            gridcolor="#3a4150",
            zeroline=False,
            tickfont=dict(color="#d9d3c5"),
            title=dict(text="Calls of this type in the game", font=dict(color="#d9d3c5", size=13)),
        ),
    )
    lift_markers(figure)
    return figure


def volume_table(study) -> pandas.DataFrame:
    rows = []
    for term in study.volume.terms:
        rows.append(
            {
                "Term": term.label,
                "Flags in the game": round(term.estimate * 2, 2),
                "95% interval": f"{term.ci_low * 2:.2f} to {term.ci_high * 2:.2f}",
                "p": format_p(term.pvalue),
            }
        )
    return pandas.DataFrame(rows)


def target_table(study) -> pandas.DataFrame:
    rows = []
    for target in study.targets:
        term = target.fit.term(target.focus_key)
        rows.append(
            {
                "Type": target.penalty_type,
                "Emphasis": target.emphasis_season,
                "Prior 3 seasons": round(target.prior_rate * 2, 2),
                "Emphasis season": round(target.emphasis_rate * 2, 2),
                "Permanent shift": round(term.estimate * 2, 2),
                "95% interval": f"{term.ci_low * 2:.2f} to {term.ci_high * 2:.2f}",
                "p": format_p(term.pvalue),
            }
        )
    return pandas.DataFrame(rows)


@st.cache_data(show_spinner="Fitting the season model...")
def cached_study(games_mtime: float, penalties_mtime: float):
    del games_mtime, penalties_mtime
    games, penalties = load_frames()
    return run_study(games, penalties)


SIDE_COLORS = {"Offense": "#e6b325", "Defense": "#2aafc4", "Other": "#b7b2a4"}


def call_rates(games: pandas.DataFrame, penalties: pandas.DataFrame, season_min: int, season_max: int) -> pandas.DataFrame:
    played = games.loc[
        games["season"].between(season_min, season_max) & games["season_type"].eq("REG")
    ]
    flagged = penalties.loc[
        penalties["season"].between(season_min, season_max) & penalties["season_type"].eq("REG")
    ]
    team_games = team_games_count(played)
    if flagged.empty or team_games == 0:
        return pandas.DataFrame(columns=["penalty_type", "side", "penalties", "per_team", "per_game"])
    counted = flagged.groupby("penalty_type", as_index=False).agg(penalties=("penalty_type", "size"))
    side = flagged.groupby("penalty_type")["side"].agg(lambda values: values.value_counts().index[0])
    counted["side"] = counted["penalty_type"].map(side)
    counted["per_team"] = counted["penalties"] / team_games
    counted["per_game"] = counted["per_team"] * 2
    counted["team_games"] = team_games
    return counted


def common_call_figure(recent: pandas.DataFrame) -> go.Figure:
    frame = recent.sort_values("per_game", ascending=False).head(12)
    frame = frame.sort_values("per_game", ascending=True)
    figure = go.Figure(
        go.Bar(
            x=frame["per_game"],
            y=frame["penalty_type"],
            orientation="h",
            marker_color=[SIDE_COLORS.get(side, "#b7b2a4") for side in frame["side"]],
            text=[f"{value:.2f}" for value in frame["per_game"]],
            textposition="outside",
            textfont=dict(color="#f3efe4", family="Barlow, sans-serif", size=13),
            cliponaxis=False,
            customdata=frame.loc[:, ["side", "penalties"]].to_numpy(),
            hovertemplate="%{y}<br>%{x:.2f} in the game<br>%{customdata[0]}<br>%{customdata[1]} flags<extra></extra>",
        )
    )
    figure.update_layout(
        **chart_layout(max(380, 32 * len(frame) + 60)),
        yaxis=dict(tickfont=dict(color="#f3efe4", size=13), showgrid=False),
        xaxis=dict(gridcolor="#3a4150", zeroline=False, tickfont=dict(color="#d9d3c5"), title=None),
    )
    return figure


def common_reading(recent: pandas.DataFrame, current: pandas.DataFrame) -> str:
    if recent.empty:
        return "No regular-season flags in 2023-2025."
    ordered = recent.sort_values("per_game", ascending=False)
    leaders = ordered.head(3)
    leader_text = ", ".join(
        f"{row.penalty_type} {row.per_game:.2f}" for row in leaders.itertuples()
    )
    per_game = float(recent["penalties"].sum() / recent["team_games"].iloc[0] * 2)
    text = (
        f"In 2023-2025 the common calls in a game were {leader_text}. "
        f"Every label in the file adds up to {per_game:.1f} flags in the game. "
        f"Gold bars are calls usually charged to the offense. Blue bars are usually charged to the defense."
    )
    if current.empty:
        return text
    merged = recent.set_index("penalty_type")["per_game"].to_frame("then")
    merged["now"] = current.set_index("penalty_type")["per_game"]
    merged["jump"] = merged["now"].fillna(0) - merged["then"]
    jumps = merged.sort_values("jump", ascending=False).head(2)
    jump_text = " and ".join(
        f"{name} from {row.then:.2f} to {row.now:.2f}" for name, row in jumps.iterrows()
    )
    added = float(jumps["jump"].sum())
    text += (
        f" So far in 2026 the largest increases are {jump_text}. "
        f"Those two add {added:.1f} flags in the game."
    )
    return text


PASS_INTERFERENCE = "Pass interference, both sides"


def type_history_frame(
    games: pandas.DataFrame,
    penalties: pandas.DataFrame,
    filters: Filters,
    choice: str,
) -> pandas.DataFrame:
    if choice == PASS_INTERFERENCE:
        types = ("Defensive Pass Interference", "Offensive Pass Interference")
    else:
        types = (choice,)
    played = games.loc[
        games["season"].between(filters.season_min, filters.season_max)
        & games["season_type"].isin(filters.season_types)
    ].drop_duplicates("game_id")
    flagged = penalties.loc[
        penalties["season"].between(filters.season_min, filters.season_max)
        & penalties["season_type"].isin(filters.season_types)
        & penalties["penalty_type"].isin(types)
    ]
    if played.empty:
        return pandas.DataFrame(columns=["season", "games", "calls", "per_game"])
    frame = played.groupby("season").size().rename("games").to_frame()
    frame = frame.join(flagged.groupby("season").size().rename("calls"), how="left")
    frame["calls"] = frame["calls"].fillna(0)
    frame["per_game"] = frame["calls"] / frame["games"]
    return frame.reset_index()


def type_history_figure(frame: pandas.DataFrame, name: str) -> go.Figure:
    figure = go.Figure(
        go.Scatter(
            x=frame["season"],
            y=frame["calls"],
            mode="lines+markers",
            name=name,
            customdata=frame["games"],
            line=smooth_line("#e6b325"),
            marker=dict(size=7, color="#e6b325"),
            hovertemplate="%{x}<br>%{y:.0f} calls<br>%{customdata} games<extra></extra>",
        )
    )
    figure.update_layout(
        **chart_layout(420),
        xaxis=season_axis(frame["season"]),
        yaxis=dict(
            gridcolor="#3a4150",
            zeroline=False,
            tickfont=dict(color="#d9d3c5"),
            title=dict(text="Calls this season", font=dict(color="#d9d3c5", size=13)),
        ),
        showlegend=False,
    )
    lift_markers(figure)
    return figure


# Club colors for a dark page. A few official shades are nearly black here, so those use the same hue, brightened.
TEAM_COLORS = {
    "ARI": "#97233F",
    "ATL": "#A71930",
    "BAL": "#5B3A9E",
    "BUF": "#1F4EAE",
    "CAR": "#0085CA",
    "CHI": "#C83803",
    "CIN": "#FB4F14",
    "CLE": "#FF3C00",
    "DAL": "#869397",
    "DEN": "#FB4F14",
    "DET": "#0076B6",
    "GB": "#FFB612",
    "HOU": "#A71930",
    "IND": "#1F5AA8",
    "JAX": "#0E8A86",
    "KC": "#E31837",
    "LA": "#2A59C4",
    "LAC": "#0080C6",
    "LV": "#A5ACAF",
    "MIA": "#008E97",
    "MIN": "#4F2683",
    "NE": "#C60C30",
    "NO": "#D3BC8D",
    "NYG": "#1F4FA3",
    "NYJ": "#1C8A55",
    "PHI": "#148F7A",
    "PIT": "#FFB612",
    "SEA": "#69BE28",
    "SF": "#AA0000",
    "TB": "#D50A0B",
    "TEN": "#4B92DB",
    "WAS": "#8C1D2C",
}


def early_years_figure(games: pandas.DataFrame, penalties: pandas.DataFrame) -> go.Figure:
    """2002 club rates: the three most penalized teams above the three least."""
    reg_games = games.loc[games["season"].eq(2002) & games["season_type"].eq("REG")]
    reg_penalties = penalties.loc[penalties["season"].eq(2002) & penalties["season_type"].eq("REG")]
    clubs = team_summary(reg_games, reg_penalties)
    league = float(clubs["penalties"].sum() / clubs["games"].sum())
    # Jacksonville's 2002 home games in this file charge every flag to the opponent.
    ranked = clubs.loc[clubs["team"].ne("JAX")].sort_values(
        ["penalties_per_game", "name"], ascending=[False, True]
    )
    most = ranked.head(3).sort_values(["penalties_per_game", "name"], ascending=[True, False])
    least = ranked.tail(3).sort_values(["penalties_per_game", "name"], ascending=[True, True])
    ordered = pandas.concat([least, most], ignore_index=True)
    colors = [TEAM_COLORS.get(team, "#f3efe4") for team in ordered["team"]]
    figure = go.Figure(
        go.Bar(
            x=ordered["penalties_per_game"],
            y=ordered["name"],
            orientation="h",
            marker_color=colors,
            text=[f"{value:.2f}" for value in ordered["penalties_per_game"]],
            textposition="outside",
            textfont=dict(color="#f3efe4", family="Barlow, sans-serif", size=13),
            cliponaxis=False,
            customdata=ordered.loc[:, ["penalties", "games"]].to_numpy(),
            hovertemplate=(
                "%{y}<br>%{x:.2f} penalties a game"
                "<br>%{customdata[0]:.0f} penalties"
                "<br>%{customdata[1]:.0f} games<extra></extra>"
            ),
        )
    )
    figure.add_vline(
        x=league,
        line_dash="dash",
        line_color="#d9d3c5",
        line_width=2,
        annotation_text=f"Typical club {league:.1f}",
        annotation_position="top",
        annotation_font_color="#d9d3c5",
        annotation_font_family="Barlow, sans-serif",
    )
    layout = chart_layout(360)
    layout["margin"] = dict(l=8, r=48, t=28, b=8)
    figure.update_layout(
        **layout,
        showlegend=False,
        yaxis=dict(
            categoryorder="array",
            categoryarray=ordered["name"].tolist(),
            tickfont=dict(color="#f3efe4", size=14),
            showgrid=False,
        ),
        xaxis=dict(
            gridcolor="#3a4150",
            zeroline=False,
            title=dict(text="Penalties a game", font=dict(color="#d9d3c5", size=13)),
            tickfont=dict(color="#d9d3c5"),
            range=[0, float(ordered["penalties_per_game"].max()) + 1.8],
        ),
    )
    return figure


def contact_years_figure(games: pandas.DataFrame, penalties: pandas.DataFrame) -> go.Figure:
    """2003-2005 panels for the calls and the game total in the contact chapter."""
    seasons = [2003, 2004, 2005]
    reg_penalties = penalties.loc[
        penalties["season"].isin(seasons) & penalties["season_type"].eq("REG")
    ]
    played = (
        games.loc[games["season"].isin(seasons) & games["season_type"].eq("REG")]
        .groupby("season")["game_id"]
        .nunique()
    )
    flagged = reg_penalties.groupby("season").size()
    rates = [float(flagged.get(season, 0)) / float(played.get(season, 1)) for season in seasons]

    def counts(name: str) -> list[int]:
        grouped = reg_penalties.loc[reg_penalties["penalty_type"].eq(name)].groupby("season").size()
        return [int(grouped.get(season, 0)) for season in seasons]

    panels = (
        ("Illegal contact", counts("Illegal Contact"), "#e6b325", False),
        ("Defensive holding", counts("Defensive Holding"), "#2aafc4", False),
        ("False start", counts("False Start"), "#e07a28", False),
        ("Flags in the game", rates, "#f3efe4", True),
    )
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=tuple(title for title, *_ in panels),
        horizontal_spacing=0.12,
        vertical_spacing=0.18,
    )
    labels = [str(season) for season in seasons]
    for index, (_, values, color, is_rate) in enumerate(panels):
        row, col = divmod(index, 2)
        low = min(values)
        high = max(values)
        spread = high - low or high * 0.05
        # Keep the axis close to the three years so a move of ~30 calls reads as a tall step.
        floor = max(0.0, low - spread * 0.2)
        ceiling = high + spread * 0.5
        figure.add_trace(
            go.Bar(
                x=labels,
                y=values,
                marker_color=color,
                text=[f"{value:.1f}" if is_rate else str(int(value)) for value in values],
                textposition="outside",
                textfont=dict(color="#f3efe4", family="Barlow, sans-serif", size=14),
                cliponaxis=False,
                hovertemplate=(
                    "%{x}<br>%{y:.2f} flags in the game<extra></extra>"
                    if is_rate
                    else "%{x}<br>%{y:.0f} calls<extra></extra>"
                ),
                showlegend=False,
            ),
            row=row + 1,
            col=col + 1,
        )
        figure.update_yaxes(range=[floor, ceiling], row=row + 1, col=col + 1)
    layout = chart_layout(620)
    layout["margin"] = dict(l=36, r=16, t=40, b=12)
    figure.update_layout(**layout)
    figure.update_annotations(font=dict(family="Barlow, sans-serif", color="#f3efe4", size=15))
    figure.update_xaxes(tickfont=dict(color="#d9d3c5", size=13), showgrid=False, type="category")
    figure.update_yaxes(
        gridcolor="#3a4150",
        tickfont=dict(color="#d9d3c5", size=12),
        zeroline=False,
    )
    return figure


def drop_years_figure(games: pandas.DataFrame, penalties: pandas.DataFrame) -> go.Figure:
    """Percent decline from 2005 to 2006. The calls sit on different scales, so percent is the axis."""
    seasons = [2005, 2006]
    reg = penalties.loc[penalties["season"].isin(seasons) & penalties["season_type"].eq("REG")]
    played = (
        games.loc[games["season"].isin(seasons) & games["season_type"].eq("REG")]
        .groupby("season")["game_id"]
        .nunique()
    )

    def pair(name: str) -> tuple[float, float]:
        grouped = reg.loc[reg["penalty_type"].eq(name)].groupby("season").size()
        return float(grouped.get(2005, 0)), float(grouped.get(2006, 0))

    flagged = reg.groupby("season").size()
    rows = [
        ("Defensive holding", *pair("Defensive Holding"), False),
        ("Offensive holding", *pair("Offensive Holding"), False),
        ("Defensive offside", *pair("Defensive Offside"), False),
        ("Illegal contact", *pair("Illegal Contact"), False),
        ("False start", *pair("False Start"), False),
        (
            "Flags in the game",
            float(flagged.get(2005, 0)) / float(played.get(2005, 1)),
            float(flagged.get(2006, 0)) / float(played.get(2006, 1)),
            True,
        ),
    ]
    rows.sort(key=lambda row: (row[1] - row[2]) / row[1])
    labels = [row[0] for row in rows]
    percents = [100 * (before - after) / before for _, before, after, _ in rows]

    def shown(value: float, is_rate: bool) -> str:
        return f"{value:.1f}" if is_rate else str(int(value))

    figure = go.Figure(
        go.Bar(
            x=percents,
            y=labels,
            orientation="h",
            marker_color=["#f3efe4" if label == "Flags in the game" else "#e04b45" for label in labels],
            text=[f"-{value:.0f}%" for value in percents],
            textposition="outside",
            textfont=dict(color="#f3efe4", family="Barlow, sans-serif", size=14),
            cliponaxis=False,
            customdata=[
                [shown(before, is_rate), shown(after, is_rate)]
                for _, before, after, is_rate in rows
            ],
            hovertemplate=(
                "%{y}<br>%{customdata[0]} in 2005<br>%{customdata[1]} in 2006"
                "<br>%{x:.0f}% fewer<extra></extra>"
            ),
        )
    )
    layout = chart_layout(420)
    layout["margin"] = dict(l=8, r=72, t=8, b=36)
    layout["showlegend"] = False
    figure.update_layout(**layout)
    figure.update_xaxes(
        title=dict(text="Percent fewer than 2005", font=dict(color="#d9d3c5", size=13)),
        ticksuffix="%",
        range=[0, max(percents) * 1.28],
        gridcolor="#3a4150",
        tickfont=dict(color="#d9d3c5"),
        zeroline=False,
    )
    figure.update_yaxes(tickfont=dict(color="#f3efe4", size=14), automargin=True)
    return figure


def safety_years_figure(penalties: pandas.DataFrame) -> go.Figure:
    """2010-2013 player-safety calls. Each panel has its own scale so a rare call can show a rise."""
    seasons = [2010, 2011, 2012, 2013]
    reg = penalties.loc[penalties["season"].isin(seasons) & penalties["season_type"].eq("REG")]

    def counts(name: str) -> list[int]:
        grouped = reg.loc[reg["penalty_type"].eq(name)].groupby("season").size()
        return [int(grouped.get(season, 0)) for season in seasons]

    panels = (
        ("Unnecessary roughness", counts("Unnecessary Roughness"), "#e04b45"),
        ("Roughing the passer", counts("Roughing the Passer"), "#e6b325"),
        ("Horse-collar tackle", counts("Horse Collar Tackle"), "#2aafc4"),
        ("Taunting", counts("Taunting"), "#a06ad4"),
    )
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=tuple(title for title, *_ in panels),
        horizontal_spacing=0.12,
        vertical_spacing=0.18,
    )
    labels = [str(season) for season in seasons]
    for index, (_, values, color) in enumerate(panels):
        row, col = divmod(index, 2)
        low = min(values)
        high = max(values)
        spread = high - low or high * 0.05
        # A small step stays visible. A call that actually doubles still starts near zero.
        floor = max(0.0, low - spread * 1.5)
        ceiling = high + spread * 0.7
        figure.add_trace(
            go.Bar(
                x=labels,
                y=values,
                marker_color=color,
                text=[str(int(value)) for value in values],
                textposition="outside",
                textfont=dict(color="#f3efe4", family="Barlow, sans-serif", size=14),
                cliponaxis=False,
                hovertemplate="%{x}<br>%{y:.0f} calls<extra></extra>",
                showlegend=False,
            ),
            row=row + 1,
            col=col + 1,
        )
        figure.update_yaxes(range=[floor, ceiling], row=row + 1, col=col + 1)
    layout = chart_layout(620)
    layout["margin"] = dict(l=36, r=16, t=40, b=12)
    figure.update_layout(**layout)
    figure.update_annotations(font=dict(family="Barlow, sans-serif", color="#f3efe4", size=15))
    figure.update_xaxes(tickfont=dict(color="#d9d3c5", size=13), showgrid=False, type="category")
    figure.update_yaxes(
        gridcolor="#3a4150",
        tickfont=dict(color="#d9d3c5", size=12),
        zeroline=False,
    )
    return figure


def plateau_level_figure(games: pandas.DataFrame, penalties: pandas.DataFrame) -> go.Figure:
    """2013-2019 flags and yards, on a tight scale so the step and the flat years both read."""
    seasons = list(range(2013, 2020))
    reg = penalties.loc[penalties["season"].isin(seasons) & penalties["season_type"].eq("REG")]
    played = (
        games.loc[games["season"].isin(seasons) & games["season_type"].eq("REG")]
        .groupby("season")["game_id"]
        .nunique()
    )
    flagged = reg.groupby("season").size()
    gained = reg.groupby("season")["penalty_yards"].sum()
    rates = [float(flagged.get(season, 0)) / float(played.get(season, 1)) for season in seasons]
    yards = [float(gained.get(season, 0)) / float(played.get(season, 1)) for season in seasons]
    panels = (
        ("Flags in the game", rates, "#f3efe4", True),
        ("Penalty yards", yards, "#e6b325", False),
    )
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=tuple(title for title, *_ in panels),
        horizontal_spacing=0.12,
    )
    labels = [str(season) for season in seasons]
    for index, (_, values, color, is_rate) in enumerate(panels):
        low = min(values)
        high = max(values)
        spread = high - low or high * 0.05
        floor = low - spread * 0.45
        ceiling = high + spread * 0.55
        figure.add_trace(
            go.Bar(
                x=labels,
                y=values,
                marker_color=color,
                text=[f"{value:.1f}" if is_rate else f"{value:.0f}" for value in values],
                textposition="outside",
                textfont=dict(color="#f3efe4", family="Barlow, sans-serif", size=13),
                cliponaxis=False,
                hovertemplate="%{x}<br>%{y:.1f}<extra></extra>",
                showlegend=False,
            ),
            row=1,
            col=index + 1,
        )
        figure.update_yaxes(range=[floor, ceiling], row=1, col=index + 1)
    layout = chart_layout(460)
    layout["margin"] = dict(l=36, r=16, t=40, b=12)
    figure.update_layout(**layout)
    figure.update_annotations(font=dict(family="Barlow, sans-serif", color="#f3efe4", size=15))
    figure.update_xaxes(tickfont=dict(color="#d9d3c5", size=12), showgrid=False, type="category")
    figure.update_yaxes(gridcolor="#3a4150", tickfont=dict(color="#d9d3c5", size=12), zeroline=False)
    return figure


def plateau_calls_figure(penalties: pandas.DataFrame) -> go.Figure:
    """2013-2019 calls behind the plateau. Each panel has its own scale."""
    seasons = list(range(2013, 2020))
    reg = penalties.loc[penalties["season"].isin(seasons) & penalties["season_type"].eq("REG")]

    def counts(name: str) -> list[int]:
        grouped = reg.loc[reg["penalty_type"].eq(name)].groupby("season").size()
        return [int(grouped.get(season, 0)) for season in seasons]

    panels = (
        ("Offensive holding", counts("Offensive Holding"), "#e07a28"),
        ("Defensive holding", counts("Defensive Holding"), "#2aafc4"),
    )
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=tuple(title for title, *_ in panels),
        horizontal_spacing=0.12,
    )
    labels = [str(season) for season in seasons]
    for index, (_, values, color) in enumerate(panels):
        low = min(values)
        high = max(values)
        spread = high - low or high * 0.05
        floor = max(0.0, low - spread * 1.5)
        ceiling = high + spread * 0.85
        figure.add_trace(
            go.Bar(
                x=labels,
                y=values,
                marker_color=color,
                text=[str(int(value)) for value in values],
                textposition="outside",
                textfont=dict(color="#f3efe4", family="Barlow, sans-serif", size=13),
                cliponaxis=False,
                hovertemplate="%{x}<br>%{y:.0f} calls<extra></extra>",
                showlegend=False,
            ),
            row=1,
            col=index + 1,
        )
        figure.update_yaxes(range=[floor, ceiling], row=1, col=index + 1)
    layout = chart_layout(460)
    layout["margin"] = dict(l=36, r=16, t=40, b=12)
    figure.update_layout(**layout)
    figure.update_annotations(font=dict(family="Barlow, sans-serif", color="#f3efe4", size=15))
    figure.update_xaxes(tickfont=dict(color="#d9d3c5", size=11), showgrid=False, type="category")
    figure.update_yaxes(gridcolor="#3a4150", tickfont=dict(color="#d9d3c5", size=12), zeroline=False)
    return figure


def covid_diverging_figure(penalties: pandas.DataFrame) -> go.Figure:
    """Percent change from 2019 to 2020. Drops sit left of zero, two rises sit right."""
    rises = {"Illegal Contact", "Defensive Pass Interference"}
    reg = penalties.loc[penalties["season"].isin([2019, 2020]) & penalties["season_type"].eq("REG")]
    before = reg.loc[reg["season"].eq(2019)].groupby("penalty_type").size()
    after = reg.loc[reg["season"].eq(2020)].groupby("penalty_type").size()
    rows: list[tuple[str, int, int]] = []
    for name in set(before.index) | set(after.index):
        then = int(before.get(name, 0))
        now = int(after.get(name, 0))
        lost = then - now
        if name in rises or lost >= 10:
            rows.append((name[:1] + name[1:].lower(), then, now))
    rows = sorted(rows, key=lambda row: (row[2] - row[1]) / row[1], reverse=True)
    labels = [label for label, _, _ in rows]
    percents = [100 * (now - then) / then for _, then, now in rows]
    colors = ["#e04b45" if value < -2 else "#e6b325" if value > 2 else "#2aafc4" for value in percents]
    figure = go.Figure(
        go.Bar(
            x=percents,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[f"{value:+.0f}%" for value in percents],
            textposition="outside",
            textfont=dict(color="#f3efe4", family="Barlow, sans-serif", size=14),
            cliponaxis=False,
            customdata=[[then, now] for _, then, now in rows],
            hovertemplate=(
                "%{y}<br>%{customdata[0]} in 2019<br>%{customdata[1]} in 2020"
                "<br>%{x:+.0f}%<extra></extra>"
            ),
            showlegend=False,
        )
    )
    layout = chart_layout(max(460, 34 * len(rows) + 80))
    layout["margin"] = dict(l=8, r=72, t=8, b=36)
    layout["showlegend"] = False
    figure.update_layout(**layout)
    low = min(percents)
    high = max(percents)
    figure.update_xaxes(
        title=dict(text="Percent change from 2019", font=dict(color="#d9d3c5", size=13)),
        ticksuffix="%",
        range=[low - 18, high + 18],
        zeroline=True,
        zerolinecolor="#f3efe4",
        zerolinewidth=1,
        gridcolor="#3a4150",
        tickfont=dict(color="#d9d3c5"),
    )
    figure.update_yaxes(tickfont=dict(color="#f3efe4", size=14), automargin=True)
    return figure


def added_2024_figure(penalties: pandas.DataFrame) -> go.Figure:
    """2023 to 2024 for the calls that jumped. The line is the flags added."""
    reg = penalties.loc[penalties["season"].isin([2023, 2024]) & penalties["season_type"].eq("REG")]
    specs = (
        ("Offensive holding", ("Offensive Holding",), GOLD),
        ("Illegal formation", ("Illegal Formation",), "#e04b45"),
        ("False start", ("False Start",), "#2aafc4"),
        ("New kickoff fouls", ("Kickoff Short of Landing Zone", "Kickoff Out of Bounds"), "#3cba6a"),
        ("Illegal shift", ("Illegal Shift",), "#e07a28"),
    )
    rows: list[tuple[str, int, int, str]] = []
    for label, names, color in specs:
        then = int(reg.loc[reg["season"].eq(2023) & reg["penalty_type"].isin(names)].shape[0])
        now = int(reg.loc[reg["season"].eq(2024) & reg["penalty_type"].isin(names)].shape[0])
        rows.append((label, then, now, color))
    rows.sort(key=lambda row: row[2] - row[1])
    figure = go.Figure()
    for index, (label, then, now, color) in enumerate(rows):
        if index % 2 == 0:
            figure.add_hrect(
                y0=index - 0.5,
                y1=index + 0.5,
                fillcolor="rgba(243,239,228,0.035)",
                line_width=0,
                layer="below",
            )
        figure.add_trace(
            go.Scatter(
                x=[then, now],
                y=[index, index],
                mode="lines+markers",
                line=dict(color=color, width=3),
                marker=dict(
                    size=[12, 16],
                    color=[PLOT, color],
                    line=dict(color=color, width=2),
                ),
                hovertemplate=(
                    f"{label}<br>{then} in 2023<br>{now} in 2024"
                    f"<br>+{now - then} flags<extra></extra>"
                ),
                showlegend=False,
            )
        )
        figure.add_annotation(
            x=now,
            y=index,
            text=f"+{now - then}",
            showarrow=False,
            xanchor="left",
            xshift=14,
            font=dict(family="Barlow, sans-serif", size=14, color=INK),
        )
        if now - then >= 50:
            figure.add_annotation(
                x=then,
                y=index,
                text=str(then),
                showarrow=False,
                xanchor="right",
                xshift=-14,
                font=dict(family="Barlow, sans-serif", size=14, color=INK),
            )
    layout = chart_layout(390)
    layout["margin"] = dict(l=8, r=28, t=36, b=44)
    layout["showlegend"] = False
    figure.update_layout(**layout)
    figure.add_annotation(
        xref="paper",
        yref="paper",
        x=0,
        y=1.08,
        text="Open is 2023. Solid is 2024.",
        showarrow=False,
        xanchor="left",
        font=dict(family="Barlow, sans-serif", size=12, color=MUTED),
    )
    figure.update_xaxes(
        title=dict(text="Flags in the season", font=dict(color=MUTED, size=13)),
        range=[-70, 860],
        gridcolor=GRID,
        zeroline=False,
        tickfont=dict(color=MUTED),
    )
    figure.update_yaxes(
        range=[-0.55, len(rows) - 0.45],
        tickmode="array",
        tickvals=list(range(len(rows))),
        ticktext=[label for label, *_ in rows],
        showgrid=False,
        tickfont=dict(color=INK, size=14),
        automargin=True,
        zeroline=False,
    )
    return figure


def opening_strip_figure(games: pandas.DataFrame, penalties: pandas.DataFrame) -> go.Figure:
    """Weeks 1 to 6 of every season. Across is flags in the game. Up is yards in the game."""
    home = games.loc[games["season_type"].eq("REG") & games["site"].eq("Home") & games["week"].le(6)]
    played = home.groupby("season")["game_id"].nunique()
    flagged = penalties.loc[penalties["season_type"].eq("REG") & penalties["week"].le(6)]
    counts = flagged.groupby("season").size()
    gained = flagged.groupby("season")["penalty_yards"].sum()
    seasons = [
        int(season)
        for season in played.index
        if season in counts.index and season in gained.index and int(played[season]) > 0
    ]
    rates = [float(counts[season]) / float(played[season]) for season in seasons]
    yards = [float(gained[season]) / float(played[season]) for season in seasons]
    current = max(seasons)
    record_at = max(range(len(seasons)), key=lambda index: rates[index])
    ordered = sorted(rates)
    span = len(ordered) - 1

    def _quantile(portion: float) -> float:
        cursor = span * portion
        low = int(cursor)
        high = min(low + 1, span)
        weight = cursor - low
        return ordered[low] * (1 - weight) + ordered[high] * weight

    typical_low = _quantile(0.25)
    typical_high = _quantile(0.75)
    typical = _quantile(0.5)
    colors = ["#6a7564"] * len(seasons)
    sizes = [13] * len(seasons)
    colors[seasons.index(current)] = GOLD
    sizes[seasons.index(current)] = 18
    if record_at != seasons.index(current):
        colors[record_at] = INK
        sizes[record_at] = 16
    figure = go.Figure()
    figure.add_vrect(
        x0=typical_low,
        x1=typical_high,
        fillcolor="rgba(142,184,196,0.22)",
        line_width=0,
        layer="below",
    )
    figure.add_vline(
        x=typical,
        line=dict(color="#2aafc4", width=1.5, dash="dot"),
        layer="below",
    )
    figure.add_trace(
        go.Scatter(
            x=rates,
            y=yards,
            mode="markers",
            marker=dict(
                size=sizes,
                color=colors,
                line=dict(color=PLOT, width=1.5),
            ),
            customdata=seasons,
            hovertemplate=(
                "%{customdata}<br>%{x:.2f} flags in the game"
                "<br>%{y:.1f} yards in the game<extra></extra>"
            ),
            showlegend=False,
        )
    )
    figure.add_annotation(
        x=(typical_low + typical_high) / 2,
        y=0.96,
        yref="paper",
        text="Most seasons",
        showarrow=False,
        yanchor="top",
        font=dict(family="Barlow, sans-serif", size=12, color="#2aafc4"),
    )
    for index, color in ((seasons.index(current), GOLD), (record_at, INK)):
        figure.add_annotation(
            x=rates[index],
            y=yards[index],
            text=f"{seasons[index]}  {rates[index]:.2f}",
            showarrow=False,
            yshift=16,
            font=dict(family="Barlow, sans-serif", size=14, color=color),
        )
    yard_pad = max(4.0, (max(yards) - min(yards)) * 0.12)
    layout = chart_layout(420)
    layout["margin"] = dict(l=8, r=20, t=28, b=48)
    layout["showlegend"] = False
    figure.update_layout(**layout)
    figure.update_xaxes(
        title=dict(text="Flags in the game, weeks 1 to 6", font=dict(color=MUTED, size=13)),
        range=[min(rates) - 0.4, max(rates) + 0.55],
        dtick=1,
        gridcolor=GRID,
        zeroline=False,
        tickfont=dict(color=MUTED),
        showline=True,
        linecolor=GRID,
        mirror=True,
    )
    figure.update_yaxes(
        title=dict(text="Yards in the game, weeks 1 to 6", font=dict(color=MUTED, size=13)),
        range=[min(yards) - yard_pad, max(yards) + yard_pad * 1.8],
        gridcolor=GRID,
        zeroline=False,
        tickfont=dict(color=MUTED),
        showline=True,
        linecolor=GRID,
        mirror=True,
        automargin=True,
    )
    return figure


def chart_heading(title: str) -> None:
    st.markdown(f'<p class="chart-title">{html.escape(title)}</p>', unsafe_allow_html=True)


def part_heading(title: str, line: str) -> None:
    st.markdown(
        f'<p class="reading">{html.escape(title)}</p>'
        f'<p class="section-label when">{html.escape(line)}</p>',
        unsafe_allow_html=True,
    )


def isolate_hint(emphasis: bool = False) -> None:
    """A short box under a chart whose legend can isolate one call."""
    if emphasis:
        text = (
            "Double-click a call under the chart to see only that line. "
            "One click hides that call. "
            "The gold rings are seasons the league pointed crews at it."
        )
    else:
        text = (
            "Double-click a call under the chart to see only that line. "
            "One click hides that call. "
            "The gold ring on holding is 2019, when crews were told to watch it."
        )
    st.markdown(f'<p class="hint">{html.escape(text)}</p>', unsafe_allow_html=True)


def chapter(key: str, heading: str, when: str, body: str) -> None:
    st.markdown(
        f'<p class="reading">{html.escape(heading)}</p>'
        f'<p class="section-label when">{html.escape(when)}</p>',
        unsafe_allow_html=True,
    )
    story_paragraph(key, body, "chapter")


def side_prominence(games: pandas.DataFrame, penalties: pandas.DataFrame) -> tuple[pandas.DataFrame, float, float]:
    played = games.loc[games["season_type"].eq("REG"), "game_id"].nunique()
    flagged = penalties.loc[
        penalties["season_type"].eq("REG") & penalties["side"].isin(["Offense", "Defense"])
    ]
    if played == 0 or flagged.empty:
        return pandas.DataFrame(columns=["side", "penalty_type", "penalties", "per_game", "per_season"]), 0.0, 0.0
    counted = (
        flagged.groupby(["side", "penalty_type"], as_index=False)
        .size()
        .rename(columns={"size": "penalties"})
    )
    counted["per_game"] = counted["penalties"] / played
    played_by_season = (
        games.loc[games["season_type"].eq("REG")].groupby("season")["game_id"].nunique()
    )
    full_seasons = played_by_season.loc[played_by_season.ge(200)].index
    season_counts = (
        flagged.loc[flagged["season"].isin(full_seasons)]
        .groupby(["season", "side", "penalty_type"])
        .size()
        .rename("calls")
        .reset_index()
    )
    averages = season_counts.groupby(["side", "penalty_type"], as_index=False)["calls"].mean()
    counted = counted.merge(averages, on=["side", "penalty_type"], how="left")
    counted = counted.rename(columns={"calls": "per_season"})
    counted["per_season"] = counted["per_season"].fillna(0)
    totals = flagged.groupby("side").size() / played
    pieces = []
    for side in ("Offense", "Defense"):
        pieces.append(
            counted.loc[counted["side"].eq(side)].sort_values("per_game", ascending=False).head(6)
        )
    return pandas.concat(pieces, ignore_index=True), float(totals.get("Offense", 0.0)), float(totals.get("Defense", 0.0))


def side_prominence_figure(frame: pandas.DataFrame, side: str) -> go.Figure:
    group = frame.loc[frame["side"].eq(side)].sort_values("per_season", ascending=True)
    figure = go.Figure(
        go.Bar(
            x=group["per_season"],
            y=group["penalty_type"],
            orientation="h",
            marker_color=SIDE_COLORS.get(side, "#b7b2a4"),
            text=[f"{value:.0f}" for value in group["per_season"]],
            textposition="outside",
            textfont=dict(color="#f3efe4", family="Barlow, sans-serif", size=13),
            cliponaxis=False,
            hovertemplate="%{y}<br>%{x:.0f} calls in a season<extra></extra>",
        )
    )
    layout = chart_layout(max(320, 36 * len(group) + 40))
    layout["margin"] = dict(l=8, r=64, t=8, b=8)
    layout["showlegend"] = False
    high = float(group["per_season"].max()) if not group.empty else 1.0
    figure.update_layout(
        **layout,
        yaxis=dict(tickfont=dict(color="#f3efe4", size=13), showgrid=False),
        xaxis=dict(
            gridcolor="#3a4150",
            zeroline=False,
            range=[0, high * 1.22],
            tickfont=dict(color="#d9d3c5"),
            title=dict(text="Calls in a season", font=dict(color="#d9d3c5", size=13)),
        ),
    )
    return figure


def side_season_frame(games: pandas.DataFrame, penalties: pandas.DataFrame) -> pandas.DataFrame:
    regular = games.loc[games["season_type"].eq("REG")]
    played = regular.groupby("season")["game_id"].nunique().rename("games")
    flagged = penalties.loc[
        penalties["season_type"].eq("REG") & penalties["side"].isin(["Offense", "Defense"])
    ]
    counts = flagged.groupby(["season", "side"]).size().unstack(fill_value=0)
    yards = flagged.groupby(["season", "side"])["penalty_yards"].sum().unstack(fill_value=0)
    for side in ("Offense", "Defense"):
        if side not in counts.columns:
            counts[side] = 0
        if side not in yards.columns:
            yards[side] = 0
    frame = counts.loc[:, ["Offense", "Defense"]].join(played, how="inner")
    frame = frame.join(yards.loc[:, ["Offense", "Defense"]].add_suffix(" yards"))
    frame = frame.loc[frame["games"].gt(0)].copy()
    for side in ("Offense", "Defense"):
        flags = frame[side]
        total_yards = frame[f"{side} yards"]
        frame[f"{side} each"] = total_yards / flags.replace(0, pandas.NA)
        frame[side] = flags / frame["games"]
        frame[f"{side} yards"] = total_yards / frame["games"]
    return frame.reset_index().sort_values("season")


def side_season_figure(frame: pandas.DataFrame) -> go.Figure:
    figure = go.Figure()
    for side in ("Offense", "Defense"):
        color = SIDE_COLORS.get(side, "#b7b2a4")
        custom = frame.loc[:, ["games", f"{side} yards", f"{side} each"]]
        figure.add_trace(
            go.Scatter(
                x=frame["season"],
                y=frame[side],
                mode="lines+markers",
                name=side,
                customdata=custom.to_numpy(),
                line=smooth_line(color),
                marker=dict(size=7, color=color),
                hovertemplate=(
                    "%{x}"
                    "<br>%{y:.2f} flags a game"
                    "<br>%{customdata[1]:.1f} yards a game<extra></extra>"
                ),
            )
        )
    figure.update_layout(
        **chart_layout(420),
        xaxis=season_axis(frame["season"]),
        yaxis=dict(
            gridcolor="#3a4150",
            zeroline=False,
            tickfont=dict(color="#d9d3c5"),
            title=dict(text="Flags a game", font=dict(color="#d9d3c5", size=13)),
        ),
    )
    lift_markers(figure)
    return figure


def site_season_frame(games: pandas.DataFrame, penalties: pandas.DataFrame) -> pandas.DataFrame:
    regular = games.loc[games["season_type"].eq("REG") & games["site"].eq("Home")]
    played = regular.groupby("season")["game_id"].nunique().rename("games")
    flagged = penalties.loc[penalties["season_type"].eq("REG") & penalties["site"].isin(["Home", "Away"])]
    counts = flagged.groupby(["season", "site"]).size().unstack(fill_value=0)
    for site in ("Home", "Away"):
        if site not in counts.columns:
            counts[site] = 0
    frame = counts.loc[:, ["Home", "Away"]].join(played, how="inner")
    frame = frame.loc[frame["games"].gt(0)].copy()
    frame["Home"] = frame["Home"] / frame["games"]
    frame["Away"] = frame["Away"] / frame["games"]
    return frame.reset_index().sort_values("season")


def site_season_figure(frame: pandas.DataFrame) -> go.Figure:
    figure = go.Figure()
    for site, color in (("Home", "#e6b325"), ("Away", "#2aafc4")):
        figure.add_trace(
            go.Scatter(
                x=frame["season"],
                y=frame[site],
                mode="lines+markers",
                name=site,
                customdata=frame["games"],
                line=smooth_line(color),
                marker=dict(size=7, color=color),
                hovertemplate=site + "<br>%{x}: %{y:.2f} flags a game<br>%{customdata} games<extra></extra>",
            )
        )
    figure.update_layout(
        **chart_layout(420),
        xaxis=season_axis(frame["season"]),
        yaxis=dict(
            gridcolor="#3a4150",
            zeroline=False,
            tickfont=dict(color="#d9d3c5"),
            title=dict(text="Flags a game", font=dict(color="#d9d3c5", size=13)),
        ),
    )
    lift_markers(figure)
    return figure


STORY_MARKS = (
    (
        2004,
        "Illegal contact",
        34,
        "#2aafc4",
        "Dating back to the 1978 point of emphasis,<br>this was a reevaluation of that rule.",
    ),
    (
        2014,
        "Illegal hands",
        34,
        "#2aafc4",
        "Grabbing, hands to the face, and other<br>forms of illegal contact all came back into focus.",
    ),
    (
        2020,
        "Covid year",
        34,
        "#e04b45",
        "The year the world, and apparently<br>the refs, shut down.",
    ),
    (
        2024,
        "Illegal formation",
        34,
        "#e6b325",
        "Tackles lined up off ball and<br>presnap shifts were the main focuses.",
    ),
)
EMPHASIS_CALLS = (
    ("Defensive Pass Interference", "#e6b325"),
    ("Defensive Holding", "#f6d35a"),
    ("Illegal Contact", "#2aafc4"),
    ("Illegal Use of Hands", "#3cba6a"),
    ("Illegal Formation", "#e07a28"),
    ("Offensive Pass Interference", "#e05a86"),
    ("Taunting", "#a06ad4"),
)
VOLUME_CALLS = (
    ("Offensive Holding", "#e6b325"),
    ("False Start", "#2aafc4"),
    (PASS_INTERFERENCE, "#e05a86"),
)
CALL_EMPHASIS = {
    "Illegal Contact": (
        (2004, "Back to the 1978 rule after the 2003 title game."),
        (2014, "Crews were told to throw contact past five yards."),
        (2022, "A point of clarification. Crews were told to throw it again."),
    ),
    "Defensive Holding": (
        (2004, "Pointed at in the same push as illegal contact."),
        (2014, "A jersey grab counted even when it did not slow the route."),
    ),
    "Illegal Use of Hands": (
        (2014, "Hands to the face got a tighter look."),
    ),
    "Illegal Formation": (
        (2024, "Tackles lined up off the ball."),
    ),
    "Offensive Pass Interference": (
        (2014, "Pushing off at the top of a route."),
    ),
    "Taunting": (
        (2021, "Two taunts meant an ejection. The count fell back the next year."),
    ),
    "Offensive Holding": (
        (2019, "Crews were told to watch it, especially on the backside of a run."),
    ),
}


ERA_SPANS = (
    ("1999-03", 1999, 2003),
    ("2004-05", 2004, 2005),
    ("2006-13", 2006, 2013),
    ("2014-19", 2014, 2019),
    ("2020-23", 2020, 2023),
    ("2024-25", 2024, 2025),
)
ERA_CALLS = (
    "Offensive Holding",
    "False Start",
    "Defensive Holding",
    "Illegal Contact",
    "Illegal Use of Hands",
    "Roughing the Passer",
    "Defensive Pass Interference",
    "Illegal Formation",
    "Taunting",
    "Unsportsmanlike Conduct",
)


def call_season_with_interference(call_season: pandas.DataFrame) -> pandas.DataFrame:
    """Add one series that adds offensive and defensive pass interference together."""
    names = ("Defensive Pass Interference", "Offensive Pass Interference")
    both = call_season.loc[call_season["penalty_type"].isin(names)]
    if both.empty:
        return call_season
    combined = both.groupby("season", as_index=False).agg(calls=("calls", "sum"), yards=("yards", "sum"))
    combined["penalty_type"] = PASS_INTERFERENCE
    return pandas.concat([call_season, combined], ignore_index=True)


def season_calls(call_season: pandas.DataFrame, name: str, season: int) -> int:
    hit = call_season.loc[call_season["penalty_type"].eq(name) & call_season["season"].eq(season), "calls"]
    if hit.empty:
        return 0
    return int(hit.iloc[0])


def era_flags(season_rates: pandas.DataFrame, start: int, end: int) -> float:
    window = season_rates.loc[season_rates["season"].between(start, end)]
    games = float(window["games"].sum())
    if games == 0:
        return 0.0
    return float((window["flags"] * window["games"]).sum() / games)


def era_call_rate(call_season: pandas.DataFrame, season_rates: pandas.DataFrame, name: str, start: int, end: int) -> float:
    games = float(season_rates.loc[season_rates["season"].between(start, end), "games"].sum())
    calls = call_season.loc[
        call_season["penalty_type"].eq(name) & call_season["season"].between(start, end),
        "calls",
    ]
    if games == 0:
        return 0.0
    return float(calls.sum()) / games * 256


def era_table(call_season: pandas.DataFrame, season_rates: pandas.DataFrame) -> pandas.DataFrame:
    rows = []
    for name in ERA_CALLS:
        row = {"Call": name}
        for label, start, end in ERA_SPANS:
            row[label] = round(era_call_rate(call_season, season_rates, name, start, end))
        rows.append(row)
    return pandas.DataFrame(rows)


def add_emphasis_rings(
    figure: go.Figure,
    finished: pandas.DataFrame,
    calls: tuple[tuple[str, str], ...],
) -> None:
    """Gold rings on the seasons the league pointed crews at that call."""
    for name, color in calls:
        notes = CALL_EMPHASIS.get(name, ())
        if not notes:
            continue
        group = finished.loc[finished["penalty_type"].eq(name)]
        seasons: list[int] = []
        counts: list[int] = []
        labels: list[str] = []
        for season, note in notes:
            hit = group.loc[group["season"].eq(season), "calls"]
            if hit.empty:
                continue
            count = int(hit.iloc[0])
            seasons.append(int(season))
            counts.append(count)
            labels.append(f"{season}<br>{note}<br>{count} calls")
        if not seasons:
            continue
        figure.add_trace(
            go.Scatter(
                x=seasons,
                y=counts,
                mode="markers",
                name=name,
                legendgroup=name,
                showlegend=False,
                marker=dict(
                    size=12,
                    color="rgba(0,0,0,0)",
                    symbol="circle",
                    line=dict(color=GOLD, width=1.5),
                ),
                text=labels,
                hovertemplate="%{text}<extra></extra>",
            )
        )


def call_path_figure(
    call_season: pandas.DataFrame,
    current_season: int,
    calls: tuple[tuple[str, str], ...],
    emphasis: bool = False,
) -> go.Figure:
    figure = go.Figure()
    finished = call_season.loc[call_season["season"].lt(current_season)]
    for name, color in calls:
        group = finished.loc[finished["penalty_type"].eq(name)].sort_values("season")
        figure.add_trace(
            go.Scatter(
                x=group["season"],
                y=group["calls"],
                mode="lines",
                name=name,
                legendgroup=name,
                line=smooth_line(color, width=2.5),
                hovertemplate=name + "<br>%{x}: %{y:.0f} calls<extra></extra>",
            )
        )
    lift_markers(figure)
    if emphasis:
        add_emphasis_rings(figure, finished, calls)
    else:
        add_story_marks(figure, labeled=True)
        add_emphasis_rings(figure, finished, calls)
    layout = chart_layout(500)
    layout["margin"] = dict(l=8, r=148, t=16 if emphasis else 78, b=88)
    layout["legend"] = dict(
        orientation="h",
        yanchor="top",
        y=-0.22,
        x=0,
        font=dict(color=INK),
        itemclick="toggle",
        itemdoubleclick="toggleothers",
    )
    figure.update_layout(
        **layout,
        xaxis=season_axis(finished["season"]),
        yaxis=dict(
            gridcolor=GRID,
            zeroline=False,
            tickfont=dict(color=MUTED),
            title=dict(text="Calls this season", font=dict(color=MUTED, size=13)),
        ),
    )
    return figure


def mover_figure(movers: pandas.DataFrame) -> go.Figure:
    ups = movers.loc[movers["Change"] > 0].head(8)
    downs = movers.loc[movers["Change"] < 0].tail(6)
    picked = pandas.concat([ups, downs]).drop_duplicates("Call").sort_values("Change")
    colors = ["#e6b325" if value > 0 else "#2aafc4" for value in picked["Change"]]
    figure = go.Figure(
        go.Bar(
            x=picked["Change"],
            y=picked["Call"],
            orientation="h",
            marker_color=colors,
            customdata=picked.loc[:, ["2000-2003", "2023-2025"]].to_numpy(),
            hovertemplate=(
                "%{y}<br>%{x:+.0f} calls in a full season"
                "<br>2000-2003: %{customdata[0]:.0f}"
                "<br>2023-2025: %{customdata[1]:.0f}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        **chart_layout(520),
        xaxis=dict(gridcolor="#3a4150", zeroline=True, zerolinecolor="#d9d3c5", tickfont=dict(color="#d9d3c5"), title=None),
        yaxis=dict(gridcolor="#3a4150", tickfont=dict(color="#d9d3c5"), title=None),
        showlegend=False,
    )
    return figure


@st.cache_data(show_spinner=False)
def cached_story(_games_mtime: float, _penalties_mtime: float, _version: int = 5):
    return story_facts(*load_frames())


def analysis_section(pace) -> None:
    games_path = TABLES / "games.parquet"
    penalties_path = TABLES / "penalties.parquet"
    study = cached_study(games_path.stat().st_mtime, penalties_path.stat().st_mtime)
    facts = cached_story(games_path.stat().st_mtime, penalties_path.stat().st_mtime)
    st.markdown('<p class="section-label">The timeline</p>', unsafe_allow_html=True)
    story_paragraph("timeline", TIMELINE_DEFAULT, "reading")
    chart_heading("Flags in the game")
    st.plotly_chart(
        study_rate_figure(study.season, facts.season_rates),
        width="stretch",
        config={"displayModeBar": False, "scrollZoom": False, "doubleClick": False},
    )
    ic_2003 = season_calls(facts.call_season, "Illegal Contact", 2003)
    ic_2004 = season_calls(facts.call_season, "Illegal Contact", 2004)
    ic_2005 = season_calls(facts.call_season, "Illegal Contact", 2005)
    ic_2006 = season_calls(facts.call_season, "Illegal Contact", 2006)
    ic_2013 = season_calls(facts.call_season, "Illegal Contact", 2013)
    ic_2014 = season_calls(facts.call_season, "Illegal Contact", 2014)
    ic_2015 = season_calls(facts.call_season, "Illegal Contact", 2015)
    hold_2005 = season_calls(facts.call_season, "Offensive Holding", 2005)
    hold_2006 = season_calls(facts.call_season, "Offensive Holding", 2006)
    hold_2019 = season_calls(facts.call_season, "Offensive Holding", 2019)
    hold_2020 = season_calls(facts.call_season, "Offensive Holding", 2020)
    hold_2023 = season_calls(facts.call_season, "Offensive Holding", 2023)
    hold_2024 = season_calls(facts.call_season, "Offensive Holding", 2024)
    false_2005 = season_calls(facts.call_season, "False Start", 2005)
    false_2006 = season_calls(facts.call_season, "False Start", 2006)
    dhold_2013 = season_calls(facts.call_season, "Defensive Holding", 2013)
    dhold_2014 = season_calls(facts.call_season, "Defensive Holding", 2014)
    dhold_2019 = season_calls(facts.call_season, "Defensive Holding", 2019)
    dhold_2020 = season_calls(facts.call_season, "Defensive Holding", 2020)
    hands_2013 = season_calls(facts.call_season, "Illegal Use of Hands", 2013)
    hands_2014 = season_calls(facts.call_season, "Illegal Use of Hands", 2014)
    helmet_2018 = season_calls(facts.call_season, "Lowering the Head to Initiate Contact", 2018)
    helmet_2019 = season_calls(facts.call_season, "Lowering the Head to Initiate Contact", 2019)
    hold_2013 = season_calls(facts.call_season, "Offensive Holding", 2013)
    hold_2014 = season_calls(facts.call_season, "Offensive Holding", 2014)
    hold_2015 = season_calls(facts.call_season, "Offensive Holding", 2015)
    hold_2018 = season_calls(facts.call_season, "Offensive Holding", 2018)
    false_2013 = season_calls(facts.call_season, "False Start", 2013)
    false_2014 = season_calls(facts.call_season, "False Start", 2014)
    opi_2013 = season_calls(facts.call_season, "Offensive Pass Interference", 2013)
    opi_2014 = season_calls(facts.call_season, "Offensive Pass Interference", 2014)
    dpi_2014 = season_calls(facts.call_season, "Defensive Pass Interference", 2014)
    dpi_2016 = season_calls(facts.call_season, "Defensive Pass Interference", 2016)
    dpi_2019 = season_calls(facts.call_season, "Defensive Pass Interference", 2019)
    ur_2013 = season_calls(facts.call_season, "Unnecessary Roughness", 2013)
    ur_2014 = season_calls(facts.call_season, "Unnecessary Roughness", 2014)
    rough_2016 = season_calls(facts.call_season, "Roughing the Passer", 2016)
    ic_path = ", ".join(str(season_calls(facts.call_season, "Illegal Contact", year)) for year in range(2014, 2020))
    hands_path = ", ".join(
        str(season_calls(facts.call_season, "Illegal Use of Hands", year)) for year in range(2014, 2020)
    )
    form_2023 = season_calls(facts.call_season, "Illegal Formation", 2023)
    form_2024 = season_calls(facts.call_season, "Illegal Formation", 2024)
    form_2025 = season_calls(facts.call_season, "Illegal Formation", 2025)
    hip_2024 = season_calls(facts.call_season, "Hip Drop Tackle", 2024)
    hip_2025 = season_calls(facts.call_season, "Hip Drop Tackle", 2025)
    dpi_2003 = season_calls(facts.call_season, "Defensive Pass Interference", 2003)
    dpi_2004 = season_calls(facts.call_season, "Defensive Pass Interference", 2004)
    dpi_2020 = season_calls(facts.call_season, "Defensive Pass Interference", 2020)
    taunt_2025 = season_calls(facts.call_season, "Taunting", 2025)
    false_early = era_call_rate(facts.call_season, facts.season_rates, "False Start", 1999, 2003) / 256
    hold_early = era_call_rate(facts.call_season, facts.season_rates, "Offensive Holding", 1999, 2003) / 256
    dpi_early = era_call_rate(facts.call_season, facts.season_rates, "Defensive Pass Interference", 1999, 2003) / 256
    offside_early = era_call_rate(facts.call_season, facts.season_rates, "Defensive Offside", 1999, 2003) / 256
    early_window = facts.season_rates.loc[facts.season_rates["season"].between(1999, 2003)]
    early_games = float(early_window["games"].sum())
    early_yards = float((early_window["yards"] * early_window["games"]).sum() / early_games)
    _games, _penalties = frames()
    early_penalties = _penalties.loc[
        _penalties["season_type"].eq("REG") & _penalties["season"].between(1999, 2003)
    ]
    def_early = len(early_penalties.loc[early_penalties["side"].eq("Defense")]) / early_games
    flags_1999 = float(facts.season_rates.loc[facts.season_rates["season"].eq(1999), "flags"].iloc[0])
    flags_2001 = float(facts.season_rates.loc[facts.season_rates["season"].eq(2001), "flags"].iloc[0])
    flags_2002 = float(facts.season_rates.loc[facts.season_rates["season"].eq(2002), "flags"].iloc[0])
    flags_2003 = float(facts.season_rates.loc[facts.season_rates["season"].eq(2003), "flags"].iloc[0])
    flags_2004 = float(facts.season_rates.loc[facts.season_rates["season"].eq(2004), "flags"].iloc[0])
    yards_2005 = float(facts.season_rates.loc[facts.season_rates["season"].eq(2005), "yards"].iloc[0])
    yards_2015 = float(facts.season_rates.loc[facts.season_rates["season"].eq(2015), "yards"].iloc[0])
    opi_2002 = season_calls(facts.call_season, "Offensive Pass Interference", 2002)
    dhold_2003 = season_calls(facts.call_season, "Defensive Holding", 2003)
    dhold_2004 = season_calls(facts.call_season, "Defensive Holding", 2004)
    dhold_2005 = season_calls(facts.call_season, "Defensive Holding", 2005)
    flags_2006_08 = era_flags(facts.season_rates, 2006, 2008)
    flags_2006_13 = era_flags(facts.season_rates, 2006, 2013)
    unsport_2026 = season_calls(facts.call_season, "Unsportsmanlike Conduct", pace.current_season)
    chapter(
        "early",
        "The early years",
        "1999 to 2003",
        f"These seasons are where we begin, the reference point before the later points of emphasis and rule changes. "
        f"Across these years a game averaged {facts.early_flags:.1f} flags and {early_yards:.0f} penalty yards. "
        f"On offense, false starts and offensive holding were the most common calls, "
        f"about {false_early:.1f} and {hold_early:.1f} a game, accounting for about {false_early + hold_early:.0f} between them. "
        f"The defense drew about {def_early:.1f} penalties a game. "
        f"Pass interference and offside were the most common on that side, about {dpi_early:.1f} and {offside_early:.1f}, near 1 each. "
        f"2001 was the quiet year in this stretch, {flags_2001:.1f} flags a game, "
        "and the rate does not fall that far again until 2008. "
        f"In 2002 the Texans joined and the league went to the 32 clubs it still has. "
        f"That season averaged {flags_2002:.1f} flags a game, next to {flags_1999:.1f} in 1999. "
        "The extra club did not lift the rate. "
        f"Offensive pass interference bottomed in 2002, at {opi_2002} calls, the fewest of any finished season.",
    )
    chart_heading("Most and least penalized, 2002")
    st.plotly_chart(
        early_years_figure(_games, _penalties),
        width="stretch",
        config={"displayModeBar": False, "scrollZoom": False, "doubleClick": False},
    )
    chapter(
        "contact",
        "Illegal contact",
        "2004 and 2005",
        "In 1978 the league drew a line five yards past the line of scrimmage. "
        "A defender could still jam a receiver inside that window. "
        "Contact the defender started beyond it, before the throw, became illegal contact. "
        "Through the 1970s, defensive backs had been allowed to grab receivers all the way down the field, "
        "and scoring had dropped. This file opens in 1999, with that rule already on the books. "
        "The league sent officials back to it after the 2003 AFC Championship. "
        "The Patriots' secondary grabbed Colts receivers down the field, and the flags did not come. "
        "Colts general manager Bill Polian took that game to the league. "
        "Passing yards around the league had fallen too. In 2004 the crews were told to throw illegal contact. "
        f"The call went from {ic_2003} in 2003 to {ic_2004} in 2004, the most of any season in this file, "
        f"and stayed at {ic_2005} in 2005. "
        f"Defensive pass interference fell from {dpi_2003} to {dpi_2004} in that first year. "
        f"Defensive holding, the other call they pointed at, went from {dhold_2003} to {dhold_2004}, then {dhold_2005} in 2005. "
        f"False starts reached {false_2005} in 2005, also the most of any season. "
        f"The game total moved from {flags_2003:.1f} in 2003 to {flags_2004:.1f}, then {facts.season_2005:.1f}. "
        f"That 2005 rate is still the high mark for a finished season. "
        f"The yards, {yards_2005:.0f} a game, sit one behind {yards_2015:.0f} in 2015.",
    )
    chart_heading("The rise from 2003 to 2005")
    st.plotly_chart(
        contact_years_figure(_games, _penalties),
        width="stretch",
        config={"displayModeBar": False, "scrollZoom": False, "doubleClick": False},
    )
    chapter(
        "quiet",
        "The quiet middle",
        "2006 to 2013",
        f"About {facts.season_2005 - flags_2006_08:.0f} fewer flags a game, and then a long quiet middle. "
        f"2005 was {facts.season_2005:.1f}. From 2006 through 2008 the average was {flags_2006_08:.1f}. "
        f"Offensive holding fell from {hold_2005} calls in 2005 to {hold_2006} in 2006, about 0.9 flags a game across 2006-2008. "
        f"False starts fell from {false_2005} to {false_2006}, about 0.6 flags a game. "
        f"Defensive holding accounts for about 0.3. Illegal contact, the emphasized call, accounts for about 0.1, "
        f"from {ic_2005} in 2005 to {ic_2006} in 2006. "
        "The contact rule stayed above its 2003 count for a few seasons and was back near that old rate by the early 2010s. "
        f"From 2006 through 2013 the whole game averaged {flags_2006_13:.1f} flags. "
        "The point of emphasis moved the total for two seasons. It did not set the next decade.",
    )
    chart_heading("Penalty decrease from 2005-2006")
    st.plotly_chart(
        drop_years_figure(_games, _penalties),
        width="stretch",
        config={"displayModeBar": False, "scrollZoom": False, "doubleClick": False},
    )
    chart_heading("Player Safety Initiatives")
    st.plotly_chart(
        safety_years_figure(_penalties),
        width="stretch",
        config={"displayModeBar": False, "scrollZoom": False, "doubleClick": False},
    )
    flags_2013 = float(facts.season_rates.loc[facts.season_rates["season"].eq(2013), "flags"].iloc[0])
    flags_2014 = float(facts.season_rates.loc[facts.season_rates["season"].eq(2014), "flags"].iloc[0])
    flags_2015 = float(facts.season_rates.loc[facts.season_rates["season"].eq(2015), "flags"].iloc[0])
    yards_2013 = float(facts.season_rates.loc[facts.season_rates["season"].eq(2013), "yards"].iloc[0])
    yards_2014 = float(facts.season_rates.loc[facts.season_rates["season"].eq(2014), "yards"].iloc[0])
    yards_2015 = float(facts.season_rates.loc[facts.season_rates["season"].eq(2015), "yards"].iloc[0])
    settled = facts.season_rates.loc[facts.season_rates["season"].between(2016, 2019)]
    chapter(
        "plateau",
        "The long plateau",
        "2014 to 2019",
        "In 2014 Dean Blandino, then vice president of officiating, pointed crews at downfield contact. "
        "The push followed a season in which the Seahawks' secondary had spent the year grabbing receivers. "
        "A defender could still jam a receiver inside five yards. Past that line, contact that broke up the route "
        "was illegal contact, and Blandino asked for those flags when the contact was clearly beyond five yards. "
        "Grabbing an eligible receiver's jersey counted as defensive holding even when the official did not think "
        "the grab slowed the route. Pushing off at the top of a route was offensive pass interference. "
        "Hands to the face got a tighter look at the same time.\n\n"
        f"The game went from {flags_2013:.1f} flags and {yards_2013:.0f} yards to {flags_2014:.1f} flags and {yards_2014:.0f} yards. "
        f"Illegal use of hands went from {hands_2013} to {hands_2014}, the most of any season in this file. "
        f"Illegal contact went from {ic_2013} to {ic_2014}. "
        f"Defensive holding went from {dhold_2013} to {dhold_2014}. "
        f"Offensive pass interference went from {opi_2013} to {opi_2014}. "
        f"Offensive holding rose from {hold_2013} to {hold_2014}, and false starts from {false_2013} to {false_2014}. "
        f"Unnecessary roughness fell from {ur_2013} to {ur_2014} in that same season. "
        "The total picked up a full flag.\n\n"
        f"Illegal contact and illegal hands came back down. Contact ran {ic_path}. Hands ran {hands_path}. "
        f"Both had already fallen by 2015, and 2015 was the loud year of the stretch: {flags_2015:.1f} flags and {yards_2015:.0f} yards, "
        "the most penalty yards of any finished season in this file. "
        f"Offensive holding supplied that jump, from {hold_2014} to {hold_2015}.\n\n"
        f"Defensive holding stayed up and finished at {dhold_2019} in 2019, the most of any season here. "
        f"Offensive holding kept climbing and closed at {hold_2019}, the most of any finished season in the file. "
        f"Pass interference dropped to {dpi_2014} in the emphasis year, then reached {dpi_2016} in 2016 and {dpi_2019} in 2019.\n\n"
        "In 2016 Blandino pointed crews at low hits on a quarterback in the passing posture, "
        "at runners who slide feet first, and at crown-of-the-helmet hits outside the tackle box. "
        f"Roughing that year was {rough_2016}, the low count of this stretch. "
        f"The helmet foul added in 2018, lowering the head to initiate contact, drew {helmet_2018} flags and then {helmet_2019}. "
        "In 2019 the league pointed at offensive holding again, especially on the backside of a run, "
        f"and holding moved from {hold_2018} to {hold_2019}. "
        f"From 2016 through 2019 the game sat between {float(settled['flags'].min()):.1f} and {float(settled['flags'].max()):.1f} flags, "
        f"with about {float(settled['yards'].mean()):.0f} yards.",
    )
    chart_heading("The jump to the unlucky 13s")
    st.plotly_chart(
        plateau_level_figure(_games, _penalties),
        width="stretch",
        config={"displayModeBar": False, "scrollZoom": False, "doubleClick": False},
    )
    chart_heading("What held the new level")
    st.plotly_chart(
        plateau_calls_figure(_penalties),
        width="stretch",
        config={"displayModeBar": False, "scrollZoom": False, "doubleClick": False},
    )
    chapter(
        "empty",
        "Covid year",
        "2020",
        f"The total fell to {facts.season_2020:.1f}, and the drop is significant. "
        f"The five seasons before sat between {facts.plateau_low:.1f} and {facts.plateau_high:.1f}. "
        f"2020 through 2023 sat between {facts.low_low:.1f} and {facts.low_high:.1f}. "
        f"A comparison of those season averages has p {format_p(facts.covid_p)}. "
        f"Offensive holding fell from {hold_2019} to {hold_2020}. Defensive holding fell from {dhold_2019} to {dhold_2020}. "
        f"Defensive pass interference rose to {dpi_2020}, the most of any season. "
        "In 2019 the league let booths review pass interference and dropped the experiment after one season. "
        "A review is not a new flag, and the interference count kept climbing into the quiet year.",
    )
    chart_heading("The Covid Collapse")
    st.plotly_chart(covid_diverging_figure(_penalties), width="stretch", config={"displayModeBar": False, "scrollZoom": False, "doubleClick": False})
    chapter(
        "formation",
        "New Rules and new rulers",
        "2021 to 2025",
        f"Formation stuck. Holding came back. {pace.current_season} is at {pace.current_flags:.1f} through {pace.current_games} games. "
        f"Illegal formation went from {form_2023} calls in 2023 to {form_2024}, then {form_2025}. "
        "Officials were throwing it on tackles lined up off the ball, the point of emphasis that showed up in the 2024 opener. "
        f"Offensive holding climbed from {hold_2023} to {hold_2024}. "
        f"2024 and 2025 together averaged {facts.rebound_flags:.1f} flags, under the old plateau and above the quiet years. "
        f"Hip-drop tackle, banned in 2024, is {hip_2024} call in 2024 and {hip_2025} in 2025. "
        f"Taunting's busiest season in this file is {facts.taunt_peak_season}, with {facts.taunt_peak_calls} calls. "
        f"2025 had {taunt_2025}. The extra flags in {pace.current_season} so far are offensive holding and false starts. "
        "The 2026 points-of-emphasis pages attached here are the high school code, from the NFHS. "
        "They tell high school crews to throw flagrant fouls, unsportsmanlike conduct, illegally helping the runner, and sideline fouls, "
        "and they say the tush push is legal in the NFL. This file has no helping-the-runner penalty. "
        f"Unsportsmanlike conduct in {pace.current_season} is {unsport_2026} calls through the games so far. "
        "That memo does not explain the NFL pace.",
    )
    chart_heading("The new can't outpace the old")
    st.plotly_chart(added_2024_figure(_penalties), width="stretch", config={"displayModeBar": False, "scrollZoom": False, "doubleClick": False})
    chapter(
        "epidemic",
        "So is there actually a penalty problem?",
        "2026 through 6 weeks",
        "So a new wave of leaders has entered the building and the penalties are began to rise again. "
        "While the season is still young and only 6 weeks deep so far, the 2026 year is looking like it might "
        "break the record for most penalties ever. It currently sits the highest average per game just under "
        "15 penalties a game. The biggest increases year over year have been offensive holding up 14%, "
        "delay of game up 38%, false starts up 10%. While these percentages might not seem too impactful, "
        "holding and false starts are the most penalized plays almost every year, so seeing an increase "
        "across the board is not a promising site if you are a football purist who likes to just let them play. "
        "Maybe we're all overacting as the starts to seasons are always the highest penalized periods over "
        "the year, but then again, maybe we aren't and there truly is a flag problem. Based off of the math, "
        "we are seeing a flag on just about 12% of all plays meaning 1 out of every 8 plays will see a flag. "
        "If your team is lucky enough to get 2 first downs, it's unlikely they got that far without seeing "
        "at least one flag on them or the defense to aid in those first downs.",
    )
    chart_heading("The highest opening since 2005")
    st.plotly_chart(
        opening_strip_figure(_games, _penalties),
        width="stretch",
        config={"displayModeBar": False, "scrollZoom": False, "doubleClick": False},
    )

    st.markdown(
        '<hr class="part-break">'
        '<p class="part-title">A deeper dive across the years</p>',
        unsafe_allow_html=True,
    )
    story_paragraph(
        "dive",
        "The seasons above follow one change at a time. "
        "This part is the wide view. It looks at the calls the league pointed at, "
        "the two that fill a game, the regular season against the playoffs, "
        "and the split between offense and defense.",
        "chapter",
    )
    holding_2019 = season_calls(facts.call_season, "Offensive Holding", 2019)
    holding_2020 = season_calls(facts.call_season, "Offensive Holding", 2020)
    false_2019 = season_calls(facts.call_season, "False Start", 2019)
    false_2020 = season_calls(facts.call_season, "False Start", 2020)
    false_2023 = season_calls(facts.call_season, "False Start", 2023)
    opi_2013 = season_calls(facts.call_season, "Offensive Pass Interference", 2013)
    opi_2014 = season_calls(facts.call_season, "Offensive Pass Interference", 2014)
    taunt_2020 = season_calls(facts.call_season, "Taunting", 2020)
    taunt_2021 = season_calls(facts.call_season, "Taunting", 2021)
    taunt_2022 = season_calls(facts.call_season, "Taunting", 2022)
    dpi_counts = facts.call_season.loc[
        facts.call_season["penalty_type"].eq("Defensive Pass Interference")
        & facts.call_season["season"].lt(pace.current_season)
    ]
    dpi_peak = dpi_counts.loc[dpi_counts["calls"].idxmax()]
    exceptions = ", ".join(str(season) for season in facts.post_higher_seasons)
    games, penalties = frames()
    prominence, offense_rate, defense_rate = side_prominence(games, penalties)

    def _called(side: str, count: int) -> str:
        names = (
            prominence.loc[prominence["side"].eq(side)]
            .sort_values("per_game", ascending=False)
            .head(count)["penalty_type"]
            .tolist()
        )
        pretty = [name.lower() for name in names]
        if len(pretty) < 2:
            return pretty[0] if pretty else ""
        return ", ".join(pretty[:-1]) + f" and {pretty[-1]}"

    part_heading("The Who?", "Offense vs. Defense who commits more fouls")
    by_side = side_season_frame(games, penalties)
    side_games = float(by_side["games"].sum())

    def _yard_pair(side: str) -> tuple[float, float]:
        per_game = float((by_side[f"{side} yards"] * by_side["games"]).sum() / side_games)
        flags = float((by_side[side] * by_side["games"]).sum())
        each = per_game * side_games / flags if flags else 0.0
        return per_game, each

    offense_yards, offense_each = _yard_pair("Offense")
    defense_yards, defense_each = _yard_pair("Defense")
    story_paragraph(
        "sides",
        "Gold is the offense. Blue is the defense. "
        f"Most of the offense's share is {_called('Offense', 2)}. "
        f"Most of the defense's share is {_called('Defense', 3)}. "
        f"The yards are closer than the flags. The offense is charged {offense_yards:.1f} yards a game "
        f"and the defense {defense_yards:.1f}. "
        f"Each offensive foul averages {offense_each:.1f} yards. Each defensive foul averages {defense_each:.1f}.",
        "chapter",
    )
    chart_heading("Offense and defense by season")
    st.plotly_chart(
        side_season_figure(by_side),
        width="stretch",
        config={"displayModeBar": False, "scrollZoom": False, "doubleClick": False},
    )
    chart_heading("Most common calls")
    offense_col, defense_col = st.columns(2)
    with offense_col:
        st.plotly_chart(
            side_prominence_figure(prominence, "Offense"),
            width="stretch",
            config={"displayModeBar": False, "scrollZoom": False, "doubleClick": False},
        )
    with defense_col:
        st.plotly_chart(
            side_prominence_figure(prominence, "Defense"),
            width="stretch",
            config={"displayModeBar": False, "scrollZoom": False, "doubleClick": False},
        )

    history = team_summary(
        games.loc[games["season_type"].eq("REG")],
        penalties.loc[penalties["season_type"].eq("REG")],
    )
    history_league = float(history["penalties"].sum() / history["games"].sum()) if not history.empty else 0.0
    by_flags = history.sort_values(["penalties_per_game", "name"], ascending=[False, True])
    top = by_flags.iloc[0]
    second = by_flags.iloc[1]
    third = by_flags.iloc[2]
    clean = by_flags.iloc[-1]
    st.markdown(
        '<p class="reading">Who has the least discipline</p>',
        unsafe_allow_html=True,
    )
    story_paragraph(
        "discipline",
        f"The {top['name']} are the most penalized club in this file, "
        f"{top['penalties_per_game']:.2f} flags a game across {int(top['games'])} regular-season games. "
        f"The {second['name']} follow at {second['penalties_per_game']:.2f}, "
        f"then the {third['name']} at {third['penalties_per_game']:.2f}. "
        f"The {clean['name']} are the cleanest at {clean['penalties_per_game']:.2f}. "
        f"A typical club sits at {history_league:.2f}. "
        "Each bar is that club's color. The gold line is the league rate.",
        "chapter",
    )
    chart_heading("Average Penalties by team")
    st.plotly_chart(
        ranking_figure(history, "penalties_per_game", history_league, False),
        width="stretch",
        config={"displayModeBar": False, "scrollZoom": False, "doubleClick": False},
    )

    part_heading("The What", "The calls that fill a game")
    story_paragraph(
        "volume",
        "Holding and false starts sit on their own chart because they run in the hundreds. "
        f"Offensive holding was a focused call in 2019, {holding_2019} flags, "
        "the most of any finished season in this file. "
        f"It fell to {holding_2020} in 2020. "
        f"False starts went from {false_2019} to {false_2020} in that empty year "
        f"and were at {false_2023} by 2023. "
        "Pass interference, counting both sides, is the third line. "
        "When those calls move, the flags in the game move with them.",
        "chapter",
    )
    chart_heading("Holding, false starts, and pass interference")
    st.plotly_chart(
        call_path_figure(
            call_season_with_interference(facts.call_season),
            pace.current_season,
            VOLUME_CALLS,
        ),
        width="stretch",
        config={"displayModeBar": False, "scrollZoom": False, "doubleClick": False},
    )
    isolate_hint()

    part_heading("The When", "From opening night to closing night, when do the flags fly")
    full_board = game_board(games, penalties)
    early_weeks, later_weeks = week_window(full_board)
    story_paragraph(
        "when_body",
        f"The year opens louder than it finishes. "
        f"Weeks 1 to 5 average {early_weeks.mean():.1f} flags in the game. "
        f"From week 6 through the end of the regular season, that falls to {later_weeks.mean():.1f}. "
        "The chart below does not stop at week 18. It keeps going through the wild card, "
        "the divisional round, the conference games, and the Super Bowl.",
        "chapter",
    )
    chart_heading("Week 1 through the Super Bowl")
    st.plotly_chart(
        week_figure(week_totals(full_board), []),
        width="stretch",
        config={"displayModeBar": False, "scrollZoom": False, "doubleClick": False},
    )
    story_paragraph(
        "playoffs_line",
        f"The regular season runs about {facts.reg_flags - facts.post_flags:.0f} flags hotter than the playoffs, "
        f"in {facts.reg_higher} of {facts.seasons_compared} seasons.",
        "reading",
    )
    story_paragraph(
        "playoffs",
        f"Regular-season games averaged {facts.reg_flags:.1f} flags. "
        f"Playoff games averaged {facts.post_flags:.1f}. The seasons where the playoffs were higher: {exceptions}. "
        f"The Super Bowl averages {facts.super_bowl:.1f}, about {facts.super_bowl - facts.other_playoffs:.1f} flags "
        f"above the other playoff rounds ({facts.other_playoffs:.1f} across wild card, divisional, and conference). "
        f"That is {facts.super_bowl_games} games, and the gap does not clear the usual line "
        f"(p {format_p(facts.super_bowl_p)}). A single flag on that stage can feel enormous. "
        "This file counts the flag. It does not record whether the flag changed the winner. "
        "Gold is the regular season. Blue is the playoffs. The dotted line is the season still being played.",
        "chapter",
    )
    chart_heading("Regular season and the playoffs")
    st.plotly_chart(
        game_pace_figure(pace.by_scope, pace.current_season),
        width="stretch",
        config={"displayModeBar": False, "scrollZoom": False, "doubleClick": False},
    )
    chart_heading("Wild card through the Super Bowl")
    st.plotly_chart(
        playoff_round_figure(games, penalties),
        width="stretch",
        config={"displayModeBar": False, "scrollZoom": False, "doubleClick": False},
    )

    part_heading("The Where", "Is home field advantage real?")
    sites = site_season_frame(games, penalties)
    finished_sites = sites.loc[sites["season"].lt(pace.current_season)]
    road_seasons = int((finished_sites["Away"] > finished_sites["Home"]).sum())
    story_paragraph(
        "where",
        "Gold is home. Blue is the road. "
        f"The road club drew more flags in {road_seasons} of {len(finished_sites)} finished seasons.",
        "chapter",
    )
    chart_heading("Home and away by season")
    st.plotly_chart(
        site_season_figure(sites),
        width="stretch",
        config={"displayModeBar": False, "scrollZoom": False, "doubleClick": False},
    )

    part_heading("The Why", "The calls the league pointed at")
    story_paragraph(
        "pointed",
        "Illegal contact is the call they came back to, in 2004, 2014, and 2022. "
        "Defensive holding and illegal use of hands were in the 2014 look. "
        "Illegal formation was 2024. "
        f"Offensive pass interference, the push-off, went from {opi_2013} in 2013 to {opi_2014} in 2014 "
        "and stayed in that higher band. "
        f"Taunting went from {taunt_2020} in 2020 to {taunt_2021} in 2021, when two taunts meant an ejection, "
        f"and fell to {taunt_2022} the next year. "
        f"Defensive pass interference climbed from about {facts.dpi_early_calls:.0f} calls in a full season "
        f"to about {facts.dpi_late_calls:.0f}. "
        f"The busiest year was {int(dpi_peak['season'])}, with {int(dpi_peak['calls'])} calls.",
        "chapter",
    )
    chart_heading("NFL Points of Emphasis")
    st.plotly_chart(
        call_path_figure(facts.call_season, pace.current_season, EMPHASIS_CALLS, emphasis=True),
        width="stretch",
        config={"displayModeBar": False, "scrollZoom": False, "doubleClick": False},
    )
    isolate_hint(emphasis=True)


def team_names(codes: tuple[str, ...]) -> str:
    ordered = sorted(codes, key=lambda code: TEAM_NAMES.get(code, code))
    return ", ".join(TEAM_NAMES.get(code, code) for code in ordered)


def club_name(code: str) -> str:
    return TEAM_NAMES.get(code, code)


def game_reading(board: pandas.DataFrame) -> str:
    if board.empty:
        return "No games in this slice."
    ordered = board.sort_values(["flags", "season", "week"], ascending=[False, True, True])
    top = ordered.iloc[0]
    low = ordered.iloc[-1]
    return (
        f"{len(board):,} games. The mean is {board['flags'].mean():.1f} flags, "
        f"the median is {board['flags'].median():.0f}, "
        f"and the games run from {int(low['flags'])} to {int(top['flags'])}. "
        f"Week {int(top['week'])} of {int(top['season'])} is the high one: "
        f"{club_name(top['away_team'])} {int(top['away_flags'])}, "
        f"{club_name(top['home_team'])} {int(top['home_flags'])}."
    )


def arc_reading(
    board: pandas.DataFrame,
    games: pandas.DataFrame,
    penalties: pandas.DataFrame,
    team: str | None,
) -> str:
    if board.empty:
        return "No games in this slice."
    early, later = week_window(board)
    if early.empty and later.empty:
        return "The week split uses regular-season weeks. This slice has none."
    whose = f"In {club_name(team)} games, " if team else ""
    if later.empty:
        text = (
            f"{whose}weeks 1 to 5 are the whole regular-season slice here: "
            f"{early.mean():.1f} flags in the game across {len(early)} games."
        )
    elif early.empty:
        text = (
            f"{whose}weeks 6 to 18 averaged {later.mean():.1f} flags in the game across {len(later)} games."
        )
    else:
        text = (
            f"{whose}weeks 1 to 5 averaged {early.mean():.1f} flags in the game across {len(early)} games. "
            f"Weeks 6 to 18 averaged {later.mean():.1f} across {len(later)} games."
        )
    if text[:1].islower():
        text = text[0].upper() + text[1:]
    if team:
        charged_early, charged_later = club_opening_flags(
            games.loc[games["team"].eq(team)],
            penalties.loc[penalties["team"].eq(team)],
        )
        game_early, game_later = opening_games(board, team)
        name = club_name(team)
        if charged_early.empty:
            return text
        if charged_later.empty or game_later.empty:
            text += (
                f" The {name} have {len(charged_early)} games here, still inside their first five. "
                f"They have been charged {charged_early.mean():.1f} flags a game, "
                f"and those games have averaged {game_early.mean():.1f}."
            )
        else:
            text += (
                f" {name} games 1 to 5 averaged {charged_early.mean():.1f} flags charged to them "
                f"and {game_early.mean():.1f} in the game. "
                f"Later games averaged {charged_later.mean():.1f} and {game_later.mean():.1f}."
            )
        return text
    charged_early, charged_later = club_opening_flags(games, penalties)
    if not charged_early.empty and not charged_later.empty:
        text += (
            f" A club's own flags averaged {charged_early.mean():.1f} in its first five games "
            f"and {charged_later.mean():.1f} after that."
        )
    return text


def quarter_reading(frame: pandas.DataFrame, team_only: bool) -> str:
    if frame.empty or frame["per_game"].isna().all():
        return "No flags in this slice."
    regulation = frame.loc[frame["quarter"].ne("Overtime")].dropna(subset=["per_game"])
    if regulation.empty:
        return "No regulation flags in this slice."
    lead = regulation.sort_values("per_game", ascending=False).iloc[0]
    whose = "Flags charged to the selected clubs. " if team_only else ""
    text = (
        f"{whose}The {lead['quarter']} quarter has the most flags, {lead['per_game']:.1f} per game "
        f"({int(lead['flags']):,} flags, {int(lead['games']):,} games)."
    )
    overtime = frame.loc[frame["quarter"].eq("Overtime")]
    if not overtime.empty and int(overtime.iloc[0]["games"]) > 0 and overtime.iloc[0]["per_game"] is not None:
        row = overtime.iloc[0]
        text += (
            f" Overtime adds {row['per_game']:.1f} per game that reaches it "
            f"({int(row['flags']):,} flags, {int(row['games']):,} games)."
        )
    return text


def game_table(board: pandas.DataFrame, order: str) -> pandas.DataFrame:
    frame = board.copy()
    frame["Home"] = frame["home_team"].map(club_name)
    frame["Away"] = frame["away_team"].map(club_name)
    frame["Date"] = frame["game_date"] if "game_date" in frame.columns else ""
    frame["Month"] = frame["month"].fillna("") if "month" in frame.columns else ""
    if order == "Fewest flags":
        frame = frame.sort_values(["flags", "season", "week"], ascending=[True, True, True])
    elif order == "By week":
        frame = frame.sort_values(["season", "week", "flags"], ascending=[True, True, False])
    else:
        frame = frame.sort_values(["flags", "season", "week"], ascending=[False, True, True])
    view = pandas.DataFrame(
        {
            "Season": frame["season"].astype(int),
            "Week": frame["week"].astype(int),
            "Date": frame["Date"].astype(str),
            "Month": frame["Month"],
            "Away": frame["Away"],
            "Away flags": frame["away_flags"].astype(int),
            "Home": frame["Home"],
            "Home flags": frame["home_flags"].astype(int),
            "Flags": frame["flags"].astype(int),
            "Away yards": frame["away_yards"].round(0).astype(int),
            "Home yards": frame["home_yards"].round(0).astype(int),
            "Yards": frame["yards"].round(0).astype(int),
        }
    )
    return view


def explore_figure(board: pandas.DataFrame, label: str) -> go.Figure:
    grouped = (
        board.groupby("season", as_index=False)
        .agg(games=("game_id", "nunique"), flags=("flags", "mean"), yards=("yards", "mean"))
        .sort_values("season")
    )
    figure = go.Figure(
        go.Scatter(
            x=grouped["season"],
            y=grouped["flags"],
            mode="lines+markers",
            name=label,
            customdata=grouped.loc[:, ["games", "yards"]].to_numpy(),
            line=smooth_line("#e6b325"),
            marker=dict(size=7, color="#e6b325"),
            hovertemplate="%{x}<br>%{y:.2f} in the game<br>%{customdata[0]} games<br>%{customdata[1]:.0f} yards<extra></extra>",
        )
    )
    figure.update_layout(
        **chart_layout(420),
        xaxis=season_axis(grouped["season"]),
        yaxis=dict(
            gridcolor="#3a4150",
            zeroline=False,
            tickfont=dict(color="#d9d3c5"),
            title=dict(text=label, font=dict(color="#d9d3c5", size=13)),
        ),
        showlegend=False,
    )
    lift_markers(figure)
    return figure


def play_controls(games: pandas.DataFrame, penalties: pandas.DataFrame) -> tuple[Filters, str, bool, int]:
    seasons = (int(games["season"].min()), int(games["season"].max()))
    team_options = sorted(games["team"].unique())
    type_options = sorted(penalties["penalty_type"].dropna().unique())
    st.markdown('<p class="section-label">Try a slice</p>', unsafe_allow_html=True)
    story_paragraph(
        "slice_intro",
        "The story above uses every regular season. "
        "These controls redraw the chart under them, the game table, and the team totals at the bottom.",
        "chapter",
    )
    season_min, season_max = st.slider("Seasons", seasons[0], seasons[1], (seasons[0], seasons[1]))
    week_min, week_max = st.slider("Weeks", 1, 22, (1, 22))
    scope_col, side_col, site_col = st.columns(3)
    with scope_col:
        game_scope = st.radio("Games", ["Regular season", "Playoffs", "Whole season"], index=0)
    with side_col:
        side = st.radio("Flagged team", ["Both sides", "Offense", "Defense"], index=0)
    with site_col:
        site = st.radio("Where", ["Home and away", "Home", "Away"], index=0)
    picked_teams = st.multiselect(
        "Teams",
        team_options,
        format_func=lambda code: f"{code}  {TEAM_NAMES.get(code, code)}",
        placeholder="All teams",
    )
    picked_types = st.multiselect("Penalty types", type_options, placeholder="All types")
    season_types = {
        "Regular season": ("REG",),
        "Playoffs": ("POST",),
        "Whole season": ("REG", "POST"),
    }[game_scope]
    sides = {"Both sides": (), "Offense": ("Offense",), "Defense": ("Defense",)}[side]
    sites = {"Home and away": (), "Home": ("Home",), "Away": ("Away",)}[site]
    filters = Filters(
        season_min=int(season_min),
        season_max=int(season_max),
        week_min=int(week_min),
        week_max=int(week_max),
        season_types=season_types,
        teams=tuple(picked_teams),
        sides=sides,
        sites=sites,
        penalty_types=tuple(picked_types),
    )
    return filters, "penalties_per_game", False, 32


def game_section(games: pandas.DataFrame, penalties: pandas.DataFrame, pace) -> tuple[Filters, str, bool, int]:
    filters, metric, low_first, on_chart = play_controls(games, penalties)
    board_filters = replace(filters, teams=(), sites=())
    board_games, board_penalties = subset(games, penalties, board_filters)
    board = game_board(board_games, board_penalties)
    meetings = False
    if len(filters.teams) >= 2:
        meetings = st.checkbox("Only games between the picked teams")
    shown = games_involving(board, filters.teams, meetings)
    if filters.penalty_types:
        chart_label = "Selected calls in the game"
    else:
        chart_label = "Flags in the game"
    if shown.empty:
        st.caption("No games in this slice. Widen the seasons or the weeks.")
    else:
        st.plotly_chart(explore_figure(shown, chart_label), width="stretch", config={"displayModeBar": False, "scrollZoom": False, "doubleClick": False})

    st.markdown('<p class="section-label">Games</p>', unsafe_allow_html=True)
    story_paragraph("games_line", game_reading(shown), "reading")
    if shown.empty:
        return filters, metric, low_first, on_chart
    order = st.radio("Order", ["Most flags", "Fewest flags", "By week"], horizontal=True, index=0)
    table = game_table(shown, order)
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        height=420,
        column_config={
            "Date": st.column_config.TextColumn("Date", width="medium"),
            "Month": st.column_config.TextColumn("Month", width="small"),
            "Away flags": st.column_config.NumberColumn("Away flags", format="%d"),
            "Home flags": st.column_config.NumberColumn("Home flags", format="%d"),
            "Flags": st.column_config.NumberColumn("Flags", format="%d"),
            "Away yards": st.column_config.NumberColumn("Away yards", format="%d"),
            "Home yards": st.column_config.NumberColumn("Home yards", format="%d"),
            "Yards": st.column_config.NumberColumn("Yards", format="%d"),
        },
    )
    st.download_button(
        "Download these games",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name="games_flags.csv",
        mime="text/csv",
        key="download_games",
    )
    st.caption(
        "One row is one game. Away flags and home flags are counted separately, then added. "
        "Seasons, weeks, Games, teams, side, and penalty type set this table. "
        "Home and away stay on the team lookup, because this row already has both clubs. "
        "Click a column header to sort it yourself."
    )
    with st.expander(
        "What you can chart from this table",
        expanded=st.session_state.get("story_editing") == "chart_note",
    ):
        story_paragraph(
            "chart_note",
            "Each row can support a chart of game totals, either club's flags, yards, week, month, or season.\n\n"
            "Penalty type, offense or defense, and the quarter of the game live on each flag, one level down. "
            "A row here cannot be split into holding versus offside. That split needs the flag list.",
            "chapter",
        )

    return filters, metric, low_first, on_chart


def scope_text(filters: Filters) -> str:
    if filters.season_min == filters.season_max:
        season = str(filters.season_min)
    else:
        season = f"{filters.season_min} to {filters.season_max}"
    if filters.season_types == ("REG",):
        games = "regular season"
    elif filters.season_types == ("POST",):
        games = "playoffs"
    else:
        games = "whole season"
    parts = [f"{season} {games}"]
    if filters.week_min > 1 or filters.week_max < 22:
        parts.append(f"weeks {filters.week_min} to {filters.week_max}")
    if filters.sides == ("Offense",):
        parts.append("offensive flags only")
    elif filters.sides == ("Defense",):
        parts.append("defensive flags only")
    if filters.sites == ("Home",):
        parts.append("home games")
    elif filters.sites == ("Away",):
        parts.append("away games")
    if len(filters.penalty_types) == 1:
        parts.append(filters.penalty_types[0])
    elif 1 < len(filters.penalty_types) <= 3:
        parts.append(", ".join(filters.penalty_types))
    elif len(filters.penalty_types) > 3:
        parts.append(f"{len(filters.penalty_types)} penalty types")
    return ", ".join(parts) + "."


def reading(summary: pandas.DataFrame, league: float | None, metric: str, scope: str) -> str:
    if summary.empty or league is None:
        return "No games match this filter. Widen the season or week range."
    label = METRIC_LABELS[metric].lower()
    leader = summary.loc[summary[metric].idxmax()]
    trailer = summary.loc[summary[metric].idxmin()]
    if len(summary) == 1:
        gap = float(leader[metric] - league)
        if abs(gap) < 0.005:
            comparison = "even with the league rate"
        elif gap > 0:
            comparison = f"{gap:.2f} over the league rate of {league:.2f}"
        else:
            comparison = f"{abs(gap):.2f} under the league rate of {league:.2f}"
        return (
            f"{scope} The {leader['name']} are at {leader[metric]:.2f} {label} "
            f"across {int(leader['games'])} games, {comparison}."
        )
    return (
        f"{scope} The {leader['name']} lead at {leader[metric]:.2f} {label} "
        f"across {int(leader['games'])} games. The {trailer['name']} are at "
        f"{trailer[metric]:.2f}. League rate: {league:.2f}."
    )


def display_table(summary: pandas.DataFrame, metric: str, low_first: bool) -> pandas.DataFrame:
    frame = summary.sort_values([metric, "team"], ascending=[low_first, True]).copy()
    frame["penalties_per_game"] = frame["penalties_per_game"].round(2)
    frame["yards_per_game"] = frame["yards_per_game"].round(2)
    frame["yards"] = frame["yards"].round(0).astype(int)
    return frame.loc[
        :,
        [
            "team",
            "name",
            "games",
            "penalties",
            "penalties_per_game",
            "yards",
            "yards_per_game",
            "most_common",
        ],
    ]


def penalties_page() -> None:
    st.html('<p class="back"><a href="/">Home</a></p>')
    try:
        from cache_nfl_data import refresh_due, refresh_is_due
    except ImportError:
        refresh_is_due = None

    if refresh_is_due is not None and refresh_is_due():
        with st.spinner("Adding this week's games..."):
            try:
                refresh_due()
            except OSError as exc:
                st.warning(f"This week's update didn't finish ({exc}). Showing the last saved week.")
        st.cache_data.clear()
    games, penalties = frames()
    seasons = (int(games["season"].min()), int(games["season"].max()))

    pace = game_pace(games, penalties)

    st.markdown(
        f"""
        <header class="mast">
          <p class="kicker">NFL, {seasons[0]} to {seasons[1]}</p>
          <h1>Does the NFL have a penalty problem?</h1>
        </header>
        """,
        unsafe_allow_html=True,
    )
    story_paragraph("opening", OPENING_DEFAULT, "deck")
    analysis_section(pace)


# Subjects on the front page. A project needs a title, a one-line description,
# a url_path, and the function that draws its full page.
CATALOG = [
    {
        "name": "Baseball",
        "projects": [
            {
                "title": "Whose contact was better?",
                "dek": "Exit velocity and launch angle, set against the result and against Savant.",
                "url_path": "expected-contact",
                "render": contact_page,
            }
        ],
    },
    {
        "name": "Basketball",
        "projects": [
            {
                "title": "Do back-to-backs cost efficiency?",
                "dek": "Win rate and true shooting on the second night, with a hot and cold switch for the clubs and players at each end, since 2021-22.",
                "url_path": "nba-logs",
                "render": nba_page,
            }
        ],
    },
    {
        "name": "Football",
        "projects": [
            {
                "title": "Does the NFL have a penalty problem?",
                "dek": "Flags and penalty yards for every regular-season game since 1999.",
                "url_path": "nfl-penalties",
                "render": penalties_page,
            }
        ],
    },
    {"name": "Energy", "projects": []},
]


def index_html() -> str:
    subjects = []
    for section in CATALOG:
        projects = section["projects"]
        if projects:
            body = "".join(
                (
                    f'<a class="project" href="/{html.escape(project["url_path"], quote=True)}">'
                    f'<h3 class="project-title">{html.escape(project["title"])}</h3>'
                    f'<span class="project-dek">{html.escape(project["dek"])}</span>'
                    "</a>"
                )
                for project in projects
            )
        else:
            body = '<p class="empty-subject">Nothing in this subject yet.</p>'
        subjects.append(
            "<details class=\"subject\">"
            f"<summary><h2 class=\"subject-name\">{html.escape(section['name'])}</h2>"
            f"<span class=\"subject-count\">{len(projects)}</span></summary>"
            f"{body}</details>"
        )
    return (
        '<div class="index">'
        '<p class="back"><a href="https://trevorfaust.github.io/" target="_top">Home</a></p>'
        '<header class="mast">'
        '<p class="kicker">Trevor Faust</p>'
        "<h1>Analytics</h1>"
        '<p class="deck">Baseball, basketball, football, and energy.<br>'
        "Open a subject to see the projects in it.</p>"
        "</header>"
        f'<div class="subjects">{"".join(subjects)}</div>'
        '<p class="colophon"><a href="https://trevorfaust.github.io/" target="_top">trevorfaust.github.io</a></p>'
        "</div>"
    )


def index_page() -> None:
    st.html(index_html())


def boot() -> None:
    st.set_page_config(
        page_title="Analytics",
        page_icon="favicon.png",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(PAGE_STYLE, unsafe_allow_html=True)
    pages = [st.Page(index_page, title="Analytics", default=True)]
    for section in CATALOG:
        for project in section["projects"]:
            pages.append(
                st.Page(project["render"], title=project["title"], url_path=project["url_path"])
            )
    st.navigation(pages, position="hidden").run()


if __name__ == "__main__":
    boot()
