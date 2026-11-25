"""Basketball page: a story about back-to-backs, the clubs, and a few players.

The counts come from the team and player game logs in data/nba.
"""

from __future__ import annotations

import calendar
import html
import json

import pandas
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

import nba

GOLD = "#e6b325"
CREAM = "#f3efe4"
INK = "#d9d3c5"
RED = "#d6453d"
BLUE = "#8eb8c4"
GREEN = "#9aaf86"
GRID = "#2c3529"
REST_COLORS = {
    "Back-to-back": RED,
    "1 day off": GOLD,
    "2 days off": BLUE,
    "3 or more days off": GREEN,
}
ENDS = 8
CHART = {"displayModeBar": False, "scrollZoom": False, "doubleClick": False}


def chart_layout(height: int) -> dict:
    return dict(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Barlow, sans-serif", color=CREAM, size=14),
        margin=dict(l=8, r=28, t=28, b=8),
        hoverlabel=dict(bgcolor="#1b2219", bordercolor="#3a4436", font_size=13, font_family="Barlow, sans-serif"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(color=CREAM)),
        barmode="group",
    )


def axis() -> dict:
    return dict(gridcolor=GRID, zeroline=False, tickfont=dict(color=INK), title=None)


def rate(value: float) -> str:
    return f"{value * 100:.1f}%"


def signed_points(value: float) -> str:
    return f"{value:+.1f}"


def count(value: int) -> str:
    return f"{value:,}"


def points(value: float) -> str:
    return f"{value:.1f}"


def claim(text: str, *, key: str) -> None:
    from app import story_paragraph

    story_paragraph(key, text, "reading")


def walk(text: str, *, key: str) -> None:
    from app import story_paragraph

    story_paragraph(key, text, "chapter")


def face_styles() -> None:
    st.html(
        """
        <style>
          [class*="st-key-face-clubs-warm"],
          [class*="st-key-face-players-warm"] { display: none; }
        </style>
        """
    )


def face_bridge() -> None:
    """The hot and cold switch lives in a frame. This keeps the matching paragraph in view."""
    st.html(
        """
        <script>
          var nbaFaces = {"nba-clubs-hot": "clubs", "nba-players-hot": "players"};
          function applyNbaFace(id, hot) {
            const name = nbaFaces[id];
            if (!name) return;
            const cold = document.querySelector('[class*="st-key-face-' + name + '-cold"]');
            const warm = document.querySelector('[class*="st-key-face-' + name + '-warm"]');
            if (cold) cold.style.display = hot ? "none" : "flex";
            if (warm) warm.style.display = hot ? "flex" : "none";
          }
          function bootNbaFaces() {
            Object.keys(nbaFaces).forEach((id) => {
              let hot = false;
              try { hot = sessionStorage.getItem(id) === "1"; } catch (error) {}
              applyNbaFace(id, hot);
            });
          }
          if (!window.__nbaFaceBridge) {
            window.__nbaFaceBridge = true;
            window.addEventListener("message", (event) => {
              const data = event.data || {};
              if (data.type !== "nba-hot") return;
              applyNbaFace(data.id, !!data.hot);
              document.querySelectorAll("iframe").forEach((frame) => {
                try { frame.contentWindow.postMessage(data, "*"); } catch (error) {}
              });
            });
          }
          bootNbaFaces();
          setTimeout(bootNbaFaces, 250);
        </script>
        """,
        unsafe_allow_javascript=True,
    )


def section(label: str) -> None:
    st.markdown(f'<p class="section-label">{html.escape(label)}</p>', unsafe_allow_html=True)


def chart_heading(title: str) -> None:
    st.markdown(f'<p class="chart-title">{html.escape(title)}</p>', unsafe_allow_html=True)


def row_for(table: pandas.DataFrame, label: str) -> pandas.Series:
    return table.loc[table["label"].eq(label)].iloc[0]


@st.cache_data(show_spinner="Loading NBA game logs...")
def cached_schedule(_mtime: float) -> pandas.DataFrame:
    return nba.attach_schedule(nba.load_team_logs())


@st.cache_data(show_spinner="Loading player game logs...")
def cached_players(_mtime: float) -> pandas.DataFrame:
    if not nba.player_logs_ready():
        return pandas.DataFrame()
    return nba.attach_players(
        nba.load_player_logs(),
        nba.load_positions(),
        nba.load_player_advanced(),
        nba.attach_schedule(nba.load_team_logs()),
    )


def nights_figure(table: pandas.DataFrame) -> go.Figure:
    colors = [REST_COLORS.get(label, GOLD) for label in table["label"]]
    figure = go.Figure(
        go.Bar(
            x=table["label"],
            y=table["ts"],
            marker_color=colors,
            text=[rate(value) for value in table["ts"]],
            textposition="outside",
            textfont=dict(color=CREAM, family="Barlow, sans-serif", size=13),
            cliponaxis=False,
            customdata=table.loc[:, ["games", "pts_per_game", "margin", "win_pct"]].to_numpy(),
            hovertemplate=(
                "<b>%{x}</b><br>True shooting: %{y:.1%}<br>Games: %{customdata[0]:,}<br>"
                "Points: %{customdata[1]:.1f}<br>Margin: %{customdata[2]:+.1f}<br>"
                "Wins: %{customdata[3]:.1%}<extra></extra>"
            ),
        )
    )
    low = float(table["ts"].min()) - 0.012
    high = float(table["ts"].max()) + 0.016
    figure.update_layout(**chart_layout(420), xaxis=dict(tickfont=dict(color=INK), title=None), yaxis=axis())
    figure.update_yaxes(tickformat=".1%", range=[low, high])
    return figure


def win_figure(table: pandas.DataFrame) -> go.Figure:
    colors = [REST_COLORS.get(label, GOLD) for label in table["label"]]
    figure = go.Figure(
        go.Bar(
            x=table["label"],
            y=table["win_pct"],
            marker_color=colors,
            text=[rate(value) for value in table["win_pct"]],
            textposition="outside",
            textfont=dict(color=CREAM, family="Barlow, sans-serif", size=13),
            cliponaxis=False,
            customdata=table.loc[:, ["games", "margin", "ts"]].to_numpy(),
            hovertemplate=(
                "<b>%{x}</b><br>Win rate: %{y:.1%}<br>Games: %{customdata[0]:,}<br>"
                "Margin: %{customdata[1]:+.1f}<br>True shooting: %{customdata[2]:.1%}<extra></extra>"
            ),
        )
    )
    low = float(table["win_pct"].min()) - 0.04
    high = float(table["win_pct"].max()) + 0.05
    figure.update_layout(**chart_layout(420), xaxis=dict(tickfont=dict(color=INK), title=None), yaxis=axis())
    figure.update_yaxes(tickformat=".0%", range=[low, high])
    return figure


def trip_figure(table: pandas.DataFrame) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=table["label"],
            y=table["b2b_win"],
            name="Back-to-back",
            marker_color=RED,
            text=[rate(value) for value in table["b2b_win"]],
            textposition="outside",
            textfont=dict(color=CREAM, family="Barlow, sans-serif", size=12),
            cliponaxis=False,
            customdata=table.loc[:, ["b2b_n", "b2b_margin", "b2b_ts"]].to_numpy(),
            hovertemplate=(
                "<b>%{x}</b><br>Back-to-back win rate: %{y:.1%}<br>Games: %{customdata[0]:,}<br>"
                "Margin: %{customdata[1]:+.1f}<br>True shooting: %{customdata[2]:.1%}<extra></extra>"
            ),
        )
    )
    figure.add_trace(
        go.Bar(
            x=table["label"],
            y=table["one_win"],
            name="One day off",
            marker_color=GOLD,
            text=[rate(value) for value in table["one_win"]],
            textposition="outside",
            textfont=dict(color=CREAM, family="Barlow, sans-serif", size=12),
            cliponaxis=False,
            customdata=table.loc[:, ["one_n", "one_margin", "one_ts"]].to_numpy(),
            hovertemplate=(
                "<b>%{x}</b><br>One day off win rate: %{y:.1%}<br>Games: %{customdata[0]:,}<br>"
                "Margin: %{customdata[1]:+.1f}<br>True shooting: %{customdata[2]:.1%}<extra></extra>"
            ),
        )
    )
    values = pandas.concat([table["b2b_win"], table["one_win"]])
    figure.update_layout(**chart_layout(460), xaxis=dict(tickfont=dict(color=INK), title=None), yaxis=axis())
    figure.update_yaxes(tickformat=".0%", range=[float(values.min()) - 0.07, float(values.max()) + 0.09])
    return figure


def home_edge_figure(table: pandas.DataFrame, baseline: float) -> go.Figure:
    colors = {"Only the visitor": GREEN, "Both clubs": GOLD, "Only the home club": RED}
    figure = go.Figure(
        go.Bar(
            x=table["label"],
            y=table["win_pct"],
            marker_color=[colors.get(label, GOLD) for label in table["label"]],
            text=[rate(value) for value in table["win_pct"]],
            textposition="outside",
            textfont=dict(color=CREAM, family="Barlow, sans-serif", size=13),
            cliponaxis=False,
            customdata=table.loc[:, ["games", "ts", "opp_ts", "margin"]].to_numpy(),
            hovertemplate=(
                "<b>%{x}</b><br>Home win rate: %{y:.1%}<br>Games: %{customdata[0]:,}<br>"
                "Home true shooting: %{customdata[1]:.1%}<br>Visitor true shooting: %{customdata[2]:.1%}<br>"
                "Home margin: %{customdata[3]:+.1f}<extra></extra>"
            ),
        )
    )
    figure.add_hline(y=baseline, line_dash="dash", line_color=GOLD, line_width=2)
    low = min(float(table["win_pct"].min()), baseline) - 0.06
    high = max(float(table["win_pct"].max()), baseline) + 0.07
    figure.update_layout(
        **chart_layout(420),
        xaxis=dict(tickfont=dict(color=INK), title=None),
        yaxis=axis(),
        showlegend=False,
    )
    figure.update_yaxes(tickformat=".0%", range=[low, high])
    return figure


def club_rest_figure(club: pandas.DataFrame, league: pandas.DataFrame) -> go.Figure:
    merged = league.loc[:, ["label", "ts"]].merge(club.loc[:, ["label", "ts", "games", "win_pct"]], on="label", how="left")
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=merged["label"],
            y=merged["ts_y"],
            name="Club",
            marker_color=RED,
            customdata=merged.loc[:, ["games", "win_pct"]].to_numpy(),
            hovertemplate="%{x}<br>Club: %{y:.1%}<br>Games: %{customdata[0]:,.0f}<br>Wins: %{customdata[1]:.1%}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Bar(
            x=merged["label"],
            y=merged["ts_x"],
            name="League",
            marker_color=GOLD,
            hovertemplate="%{x}<br>League: %{y:.1%}<extra></extra>",
        )
    )
    figure.update_layout(**chart_layout(420), xaxis=dict(tickfont=dict(color=INK), title=None), yaxis=axis())
    values = pandas.concat([merged["ts_x"], merged["ts_y"]]).dropna()
    if not values.empty:
        figure.update_yaxes(tickformat=".1%", range=[float(values.min()) - 0.012, float(values.max()) + 0.016])
    return figure


def club_gap_figure(gaps: pandas.DataFrame, league_gap: float, worst: bool = True) -> go.Figure:
    frame = gaps.sort_values(["ts_gap", "team"], ascending=[not worst, True])
    figure = go.Figure(
        go.Bar(
            x=frame["ts_gap"],
            y=frame["name"],
            orientation="h",
            marker_color=[RED if value < league_gap else BLUE for value in frame["ts_gap"]],
            customdata=frame.loc[:, ["b2b_games", "b2b_ts", "rest_ts", "margin_gap"]].to_numpy(),
            hovertemplate=(
                "<b>%{y}</b><br>True shooting gap: %{x:+.1%}<br>"
                "Back-to-back: %{customdata[1]:.1%}<br>Other games: %{customdata[2]:.1%}<br>"
                "Margin gap: %{customdata[3]:+.1f}<br>Back-to-backs: %{customdata[0]}<extra></extra>"
            ),
        )
    )
    figure.add_vline(x=league_gap, line_dash="dash", line_color=GOLD, line_width=2)
    figure.update_layout(
        **chart_layout(max(420, 36 * len(frame) + 70)),
        yaxis=dict(
            categoryorder="array",
            categoryarray=frame["name"].tolist(),
            showgrid=False,
            tickfont=dict(color=CREAM, size=13),
            title=None,
        ),
        xaxis=axis(),
        showlegend=False,
    )
    figure.update_xaxes(tickformat="+.1%")
    return figure


def cast_figure(cases: pandas.DataFrame, worst: bool = True) -> go.Figure:
    ordered = cases.sort_values(["ts_gap", "name"], ascending=[not worst, True])
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            y=ordered["name"],
            x=ordered["b2b_ts"],
            name="Played yesterday",
            orientation="h",
            marker_color=RED,
            customdata=ordered.loc[:, ["b2b_n", "b2b_min", "b2b_pm"]].to_numpy(),
            hovertemplate=(
                "<b>%{y}</b><br>Played yesterday: %{x:.1%}<br>Games: %{customdata[0]:,}<br>"
                "Minutes: %{customdata[1]:.1f}<br>Plus-minus: %{customdata[2]:+.1f}<extra></extra>"
            ),
        )
    )
    figure.add_trace(
        go.Bar(
            y=ordered["name"],
            x=ordered["rest_ts"],
            name="One day off",
            orientation="h",
            marker_color=GOLD,
            customdata=ordered.loc[:, ["rest_n", "rest_min", "rest_pm"]].to_numpy(),
            hovertemplate=(
                "<b>%{y}</b><br>One day off: %{x:.1%}<br>Games: %{customdata[0]:,}<br>"
                "Minutes: %{customdata[1]:.1f}<br>Plus-minus: %{customdata[2]:+.1f}<extra></extra>"
            ),
        )
    )
    values = pandas.concat([ordered["b2b_ts"], ordered["rest_ts"]])
    figure.update_layout(
        **chart_layout(max(420, 64 * len(ordered) + 40)),
        yaxis=dict(
            categoryorder="array",
            categoryarray=ordered["name"].tolist(),
            showgrid=False,
            tickfont=dict(color=CREAM, size=14),
            title=None,
        ),
        xaxis=axis(),
    )
    figure.update_xaxes(tickformat=".0%", range=[float(values.min()) - 0.025, float(values.max()) + 0.04])
    return figure


FLAME = """
<svg viewBox="0 0 64 78" aria-hidden="true">
  <defs>
    <linearGradient id="flame-outer" x1="32" y1="74" x2="32" y2="6" gradientUnits="userSpaceOnUse">
      <stop stop-color="#d6453d"/>
      <stop offset="0.55" stop-color="#f08a24"/>
      <stop offset="1" stop-color="#ffe27a"/>
    </linearGradient>
    <linearGradient id="flame-inner" x1="32" y1="70" x2="32" y2="28" gradientUnits="userSpaceOnUse">
      <stop stop-color="#ffb703"/>
      <stop offset="1" stop-color="#fff4c2"/>
    </linearGradient>
  </defs>
  <path fill="url(#flame-outer)" d="M32 2c1.2 8.4 6.4 13.2 7.2 21.4.4 4.2-1.6 6.6-1.2 10.2 4.8-2.6 9.4-8.2 10.6-15.2 7.6 8.2 11.4 17.6 11.4 27.4 0 15.8-11.6 28.2-28 28.2S4.2 61.6 4.2 45.8C4.2 33.2 12 23.4 18.6 16.2c1.6 7.2 5.4 10.8 8.8 14.2C24.6 22.2 24.2 12.4 32 2z"/>
  <path fill="url(#flame-inner)" d="M32 28c.6 5.2 4 8.2 4.4 13.2.2 2.6-1 4.2-.6 6.6 2.6-1.6 5.2-5 5.8-9.2 3.6 4.4 5.6 9.2 5.6 14.2 0 8.6-6.2 15.2-15.2 15.2S17 61.4 17 52.8c0-6.6 4-12.2 7.6-16.4 1 4.2 3.2 6.4 5.2 8.4-.8-5.2-1.2-11.2 2.2-16.8z"/>
</svg>
"""

ICE = """
<svg viewBox="0 0 72 78" aria-hidden="true">
  <ellipse cx="36" cy="68" rx="24" ry="5.5" fill="#9fd4ee" opacity="0.85"/>
  <path d="M16 30 L36 18 L56 30 L56 50 L36 62 L16 50 Z" fill="#eaf7ff" stroke="#7eb8d4" stroke-width="1.4"/>
  <path d="M16 30 L36 40 L36 62 L16 50 Z" fill="#b7e2f6"/>
  <path d="M56 30 L36 40 L36 62 L56 50 Z" fill="#d4f0fb"/>
  <path d="M16 30 L36 18 L56 30 L36 40 Z" fill="#ffffff"/>
  <path d="M24 26.5 L36 20.2 L32 27.5 Z" fill="#ffffff" opacity="0.9"/>
  <circle cx="52" cy="70" r="2.2" fill="#b7e2f6"/>
</svg>
"""


def _figure_spec(figure: go.Figure) -> str:
    payload = json.loads(figure.to_json())
    layout = payload.get("layout") or {}
    layout.pop("template", None)
    payload["layout"] = layout
    return json.dumps(payload).replace("<", "\\u003c")


def _switch_button(switch_id: str) -> None:
    page = f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8" />
      <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow:wght@700&display=swap" />
      <style>
        html, body {{ margin: 0; background: #17191e; }}
        .switch {{
          display: flex; align-items: flex-end; justify-content: center; gap: 16px;
          width: 100%; margin: 0; padding: 2px 0 0; border: 0; background: transparent; cursor: pointer;
        }}
        .end {{
          display: flex; flex-direction: column; align-items: center; gap: 2px; width: 78px;
          opacity: 0.42; filter: grayscale(0.35); transition: opacity 140ms ease, filter 140ms ease;
        }}
        .end span {{
          font-family: Barlow, sans-serif; font-size: 11px; font-weight: 700;
          letter-spacing: 0.14em; text-transform: uppercase; color: #d9d3c5;
        }}
        .end svg {{ width: 64px; height: 72px; display: block; }}
        .track {{
          position: relative; width: 96px; height: 42px; margin-bottom: 16px; border-radius: 999px;
          background: #1a2833; border: 1px solid #8eb8c4; transition: background 140ms ease, border-color 140ms ease;
        }}
        .knob {{
          position: absolute; top: 4px; left: 4px; width: 32px; height: 32px; border-radius: 50%;
          background: #f3efe4; box-shadow: 0 1px 4px rgba(0, 0, 0, 0.35); transform: translateX(54px);
          transition: transform 140ms ease, background 140ms ease;
        }}
        .switch.is-hot .knob {{ transform: translateX(0); background: #ffe7c2; }}
        .switch.is-hot .track {{ background: #3a2218; border-color: #d6453d; }}
        .switch.is-hot .end.hot, .switch:not(.is-hot) .end.cold {{ opacity: 1; filter: none; }}
        .switch.is-hot .end.hot svg {{ filter: drop-shadow(0 0 8px rgba(240, 120, 40, 0.9)); }}
        .switch:not(.is-hot) .end.cold svg {{ filter: drop-shadow(0 0 8px rgba(150, 210, 235, 0.95)); }}
      </style>
    </head>
    <body>
      <button type="button" class="switch" id="switch" aria-pressed="false">
        <span class="end hot">{FLAME}<span>Hot</span></span>
        <span class="track" aria-hidden="true"><span class="knob"></span></span>
        <span class="end cold">{ICE}<span>Ice cold</span></span>
      </button>
      <script>
        const key = {json.dumps(switch_id)};
        const button = document.getElementById("switch");
        function paint(hot) {{
          button.classList.toggle("is-hot", hot);
          button.setAttribute("aria-pressed", hot ? "true" : "false");
          button.setAttribute("aria-label", hot ? "Showing hot. Switch to ice cold." : "Showing ice cold. Switch to hot.");
        }}
        function publish(hot) {{
          try {{ sessionStorage.setItem(key, hot ? "1" : "0"); }} catch (error) {{}}
          window.parent.postMessage({{type: "nba-hot", id: key, hot: hot}}, "*");
        }}
        button.addEventListener("click", () => {{
          const next = !button.classList.contains("is-hot");
          paint(next);
          publish(next);
        }});
        let hot = false;
        try {{ hot = sessionStorage.getItem(key) === "1"; }} catch (error) {{}}
        paint(hot);
        publish(hot);
      </script>
    </body>
    </html>
    """
    components.html(page, height=128, scrolling=False)


def hot_cold_switch(
    *,
    switch_id: str,
    face: str,
    cold_key: str,
    hot_key: str,
    cold_claim: str,
    hot_claim: str,
    cold_title: str,
    hot_title: str,
    cold_fig: go.Figure,
    hot_fig: go.Figure,
    caption: str,
) -> None:
    """Both charts are already in the page. The switch only shows one of them."""
    _switch_button(switch_id)
    with st.container(key=f"face-{face}-cold"):
        claim(cold_claim, key=cold_key)
    with st.container(key=f"face-{face}-warm"):
        claim(hot_claim, key=hot_key)
    chart_height = int(cold_fig.layout.height or 420)
    deck_height = chart_height + 90
    frame_height = deck_height + 96
    page = f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8" />
      <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow:wght@500;700&family=Fraunces:opsz,wght@9..144,560&display=swap" />
      <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
      <style>
        html, body {{
          margin: 0;
          background: #17191e;
          color: #f3efe4;
          font-family: Barlow, sans-serif;
        }}
        .switch {{
          display: flex;
          align-items: flex-end;
          justify-content: center;
          gap: 16px;
          width: 100%;
          margin: 0 0 8px;
          padding: 2px 0 0;
          border: 0;
          background: transparent;
          color: inherit;
          cursor: pointer;
        }}
        .end {{
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 2px;
          width: 78px;
          opacity: 0.42;
          filter: grayscale(0.35);
          transition: opacity 140ms ease, filter 140ms ease;
        }}
        .end span {{
          font-size: 11px;
          font-weight: 700;
          letter-spacing: 0.14em;
          text-transform: uppercase;
          color: #d9d3c5;
        }}
        .end svg {{ width: 64px; height: 72px; display: block; }}
        .track {{
          position: relative;
          width: 96px;
          height: 42px;
          margin-bottom: 16px;
          border-radius: 999px;
          background: #1a2833;
          border: 1px solid #8eb8c4;
          transition: background 140ms ease, border-color 140ms ease;
        }}
        .knob {{
          position: absolute;
          top: 4px;
          left: 4px;
          width: 32px;
          height: 32px;
          border-radius: 50%;
          background: #f3efe4;
          box-shadow: 0 1px 4px rgba(0, 0, 0, 0.35);
          transform: translateX(54px);
          transition: transform 140ms ease, background 140ms ease;
        }}
        .switch.is-hot .knob {{
          transform: translateX(0);
          background: #ffe7c2;
        }}
        .switch.is-hot .track {{
          background: #3a2218;
          border-color: #d6453d;
        }}
        .switch.is-hot .end.hot,
        .switch:not(.is-hot) .end.cold {{
          opacity: 1;
          filter: none;
        }}
        .switch.is-hot .end.hot svg {{ filter: drop-shadow(0 0 8px rgba(240, 120, 40, 0.9)); }}
        .switch:not(.is-hot) .end.cold svg {{ filter: drop-shadow(0 0 8px rgba(150, 210, 235, 0.95)); }}
        .deck {{ position: relative; height: {deck_height}px; }}
        .face {{
          position: absolute;
          inset: 0;
          visibility: hidden;
        }}
        .face.on {{ visibility: visible; }}
        .reading {{
          margin: 8px auto 6px;
          max-width: 68rem;
          color: #f3efe4;
          font-family: Fraunces, serif;
          font-size: clamp(1.45rem, 2.4vw, 2rem);
          font-weight: 560;
          letter-spacing: -0.03em;
          line-height: 1.18;
          text-align: center;
        }}
        .chart-title {{
          margin: 10px auto 8px;
          color: #f3efe4;
          font-family: Fraunces, serif;
          font-size: clamp(1.15rem, 1.8vw, 1.45rem);
          text-align: center;
        }}
        .plot {{
          height: {chart_height}px;
          background: #21242b;
          border: 1px solid #3a4150;
          border-radius: 12px;
          overflow: hidden;
        }}
        .caption {{
          margin: 8px auto 0;
          max-width: 68rem;
          color: #d9d3c5;
          font-size: 0.95rem;
          line-height: 1.45;
          text-align: center;
        }}
      </style>
    </head>
    <body>
      <div class="deck">
        <div class="face on" id="cold-face">
          <p class="chart-title">{html.escape(cold_title)}</p>
          <div class="plot" id="cold-plot"></div>
        </div>
        <div class="face" id="hot-face">
          <p class="chart-title">{html.escape(hot_title)}</p>
          <div class="plot" id="hot-plot"></div>
        </div>
      </div>
      <p class="caption">{html.escape(caption)}</p>
      <script>
        const coldSpec = { _figure_spec(cold_fig) };
        const hotSpec = { _figure_spec(hot_fig) };
        function fit(spec) {{
          spec.layout.margin = Object.assign({{}}, spec.layout.margin, {{l: 188, r: 28, t: 24, b: 12}});
          if (spec.layout.yaxis) spec.layout.yaxis.automargin = true;
          return spec;
        }}
        fit(coldSpec);
        fit(hotSpec);
        const config = {{displayModeBar: false, scrollZoom: false, doubleClick: false, responsive: true}};
        const key = {json.dumps(switch_id)};
        const coldFace = document.getElementById("cold-face");
        const hotFace = document.getElementById("hot-face");
        function apply(hot) {{
          coldFace.classList.toggle("on", !hot);
          hotFace.classList.toggle("on", hot);
          const plotId = hot ? "hot-plot" : "cold-plot";
          requestAnimationFrame(() => {{
            try {{ Plotly.Plots.resize(plotId); }} catch (error) {{}}
          }});
        }}
        window.addEventListener("message", (event) => {{
          const data = event.data || {{}};
          if (data.type !== "nba-hot" || data.id !== key) return;
          apply(!!data.hot);
        }});
        Promise.all([
          Plotly.newPlot("cold-plot", coldSpec.data, coldSpec.layout, config),
          Plotly.newPlot("hot-plot", hotSpec.data, hotSpec.layout, config)
        ]).then(() => {{
          let hot = false;
          try {{ hot = sessionStorage.getItem(key) === "1"; }} catch (error) {{}}
          apply(hot);
        }});
      </script>
    </body>
    </html>
    """
    components.html(page, height=frame_height, scrolling=False)


def rest_table(table: pandas.DataFrame) -> None:
    shown = table.loc[:, ["label", "games", "ts", "pts_per_game", "opp_pts", "margin", "win_pct", "ortg", "pace"]].copy()
    shown["ts"] = shown["ts"] * 100
    shown["win_pct"] = shown["win_pct"] * 100
    shown.columns = [
        "Nights off",
        "Games",
        "True shooting",
        "Points",
        "Opponent",
        "Margin",
        "Win rate",
        "Offensive rating",
        "Possessions",
    ]
    st.dataframe(
        shown,
        hide_index=True,
        width="stretch",
        column_config={
            "Games": st.column_config.NumberColumn(format="%d"),
            "True shooting": st.column_config.NumberColumn(format="%.1f"),
            "Points": st.column_config.NumberColumn(format="%.1f"),
            "Opponent": st.column_config.NumberColumn(format="%.1f"),
            "Margin": st.column_config.NumberColumn(format="%+.1f"),
            "Win rate": st.column_config.NumberColumn(format="%.1f"),
            "Offensive rating": st.column_config.NumberColumn(format="%.1f"),
            "Possessions": st.column_config.NumberColumn(format="%.1f"),
        },
    )


def missing_cache() -> None:
    st.markdown(
        """
        <header class="mast">
          <p class="kicker">NBA</p>
          <h1>Do back-to-backs cost efficiency?</h1>
          <p class="deck">The team game log is not on disk yet.</p>
        </header>
        """,
        unsafe_allow_html=True,
    )
    st.code(r".\.venv\Scripts\python.exe cache_nba_data.py", language="powershell")


def nba_page() -> None:
    st.html('<p class="back"><a href="/">Home</a></p>')
    face_styles()
    if not any((nba.CACHE / "team_gamelogs").glob("*.parquet")):
        missing_cache()
        return

    schedule = cached_schedule(nba.cache_mtime())
    on_disk = nba.load_team_logs()
    held = nba.held_seasons(on_disk)
    if not held.empty:
        schedule = schedule.loc[schedule["SEASON"].isin(nba.mature_seasons(on_disk))].copy()
    schedule = schedule.sort_values(["SEASON", "TEAM_ID", "GAME_DATE", "GAME_ID"])
    schedule["PREV_SITE"] = schedule.groupby(["SEASON", "TEAM_ID"], sort=False)["SITE"].shift(1)
    played = nba.with_rest(schedule)
    if played.empty:
        missing_cache()
        return

    seasons = nba.seasons_in(played)
    rest = nba.summarize_labels(played, "REST_LABEL", nba.REST_ORDER)
    edge = nba.summarize_labels(played, "EDGE", nba.EDGE_ORDER)
    sites = nba.summarize_labels(played, "SITE", ["Home", "Away"])
    cross = nba.cross_summary(played)
    by_season = nba.season_compare(played).assign(ts_gap=lambda frame: frame["b2b_ts"] - frame["rest_ts"])
    gaps = nba.team_gaps(played)
    second = row_for(rest, "Back-to-back")
    one = row_for(rest, "1 day off")
    two = row_for(rest, "2 days off")
    long_rest = row_for(rest, "3 or more days off")
    home = row_for(sites, "Home")
    tired_only = row_for(edge, "You on a back-to-back")
    both_tired = row_for(edge, "Both on a back-to-back")
    neither = row_for(edge, "Neither on a back-to-back")
    home_b2b = cross.loc[cross["label"].eq("Home, back-to-back")].iloc[0]
    road_rest = cross.loc[cross["label"].eq("Road, rested")].iloc[0]
    by_win = gaps.sort_values(["win_gap", "team"])
    low_win = by_win.iloc[0]
    next_win = by_win.iloc[1]
    high_win = by_win.iloc[-1]
    widest = by_season.sort_values("ts_gap").iloc[0]
    narrow = by_season.sort_values("ts_gap").iloc[-1]
    other = nba.pooled(played.loc[~played["IS_B2B"]])
    on_b2b = nba.pooled(played.loc[played["IS_B2B"]])
    league_gap = float(on_b2b["ts"] - other["ts"])
    worst_shot = gaps.iloc[0]

    players = cached_players(nba.cache_mtime())
    if not players.empty and not held.empty:
        players = players.loc[~players["SEASON"].isin(set(held.index))].copy()
    cases = nba.player_cases(players) if not players.empty else pandas.DataFrame()
    position = nba.rest_contrast(players, "POSITION_GROUP", nba.POSITION_ORDER) if not players.empty else pandas.DataFrame()

    def trip_stats(previous: str, site: str, one_day: bool) -> dict:
        mask = played["PREV_SITE"].eq(previous) & played["SITE"].eq(site)
        if one_day:
            mask = mask & played["REST_DAYS"].eq(1)
        else:
            mask = mask & played["IS_B2B"]
        return nba.pooled(played.loc[mask])

    trip_rows = []
    for label, previous, site in (
        ("Stayed home", "Home", "Home"),
        ("Left home", "Home", "Away"),
        ("Came home", "Away", "Home"),
        ("Stayed on the road", "Away", "Away"),
    ):
        second_trip = trip_stats(previous, site, False)
        one_trip = trip_stats(previous, site, True)
        trip_rows.append(
            {
                "label": label,
                "b2b_win": second_trip["win_pct"],
                "one_win": one_trip["win_pct"],
                "b2b_n": second_trip["games"],
                "one_n": one_trip["games"],
                "b2b_ts": second_trip["ts"],
                "one_ts": one_trip["ts"],
                "b2b_margin": second_trip["margin"],
                "one_margin": one_trip["margin"],
            }
        )
    trips = pandas.DataFrame(trip_rows)
    stayed = trips.loc[trips["label"].eq("Stayed home")].iloc[0]
    left = trips.loc[trips["label"].eq("Left home")].iloc[0]
    came = trips.loc[trips["label"].eq("Came home")].iloc[0]
    road_trip = trips.loc[trips["label"].eq("Stayed on the road")].iloc[0]
    fresh_one = nba.pooled(played.loc[played["REST_DAYS"].eq(1) & played["OPP_B2B"].eq(False)])
    records = schedule.groupby(["SEASON", "TEAM_ABBREVIATION"], as_index=False).agg(
        opp_win=("WL", lambda series: float((series == "W").mean()))
    )
    records = records.rename(columns={"TEAM_ABBREVIATION": "OPPONENT"})
    faced = played.merge(records, on=["SEASON", "OPPONENT"], how="left")
    b2b_opp_win = float(faced.loc[faced["IS_B2B"], "opp_win"].mean())
    one_opp_win = float(faced.loc[faced["REST_DAYS"].eq(1), "opp_win"].mean())
    month_rows = []
    for month, group in played.groupby(played["GAME_DATE"].dt.month):
        month_b2b = nba.pooled(group.loc[group["IS_B2B"]])
        month_one = nba.pooled(group.loc[group["REST_DAYS"].eq(1)])
        if not month_b2b or not month_one or month_b2b["games"] < 100:
            continue
        month_rows.append(
            {
                "month": calendar.month_name[int(month)],
                "b2b": month_b2b,
                "one": month_one,
                "ts_gap": month_b2b["ts"] - month_one["ts"],
                "win_gap": month_b2b["win_pct"] - month_one["win_pct"],
            }
        )
    quiet = min(month_rows, key=lambda row: abs(row["ts_gap"]))
    harsh = min(month_rows, key=lambda row: row["win_gap"])

    def second_night_mix(name: str) -> tuple[float, float]:
        club = played.loc[played["TEAM_NAME"].eq(name) & played["IS_B2B"]]
        return float(club["SITE"].eq("Away").mean()), float(club["EDGE"].eq("You on a back-to-back").mean())

    low_road, low_fresh = second_night_mix(low_win["name"])
    shot_road, shot_fresh = second_night_mix(worst_shot["name"])
    league_b2b = played.loc[played["IS_B2B"]]
    league_road = float(league_b2b["SITE"].eq("Away").mean())
    league_fresh = float(league_b2b["EDGE"].eq("You on a back-to-back").mean())

    st.markdown(
        f"""
        <header class="mast">
          <p class="kicker">NBA, {html.escape(seasons[0])} to {html.escape(seasons[-1])}</p>
          <h1>Do back-to-backs cost efficiency?</h1>
        </header>
        """,
        unsafe_allow_html=True,
    )
    from app import story_paragraph

    story_paragraph(
        "nba_deck",
        "A back-to-back is a game on the day after that club's previous game. "
        "True shooting pools every field goal and free throw, points divided by twice the attempts plus 0.44 free throws. "
        "The question is whether the second night takes that shot away, for the club and for the players who had to play it.",
        "deck",
    )

    section("The second night")
    claim(
        f"Teams won {rate(second['win_pct'])} of back-to-backs and {rate(one['win_pct'])} with one day off. "
        f"With three or more days off they won {rate(long_rest['win_pct'])}. "
        f"True shooting moved from {rate(second['ts'])} to {rate(long_rest['ts'])} across that same span, "
        f"on {count(int(second['games']))} back-to-backs and {count(int(long_rest['games']))} long layoffs.",
        key="nba_second_claim",
    )
    walk(
        f"A back-to-back shot {rate(second['ts'])} across {count(int(second['games']))} games. "
        f"One day off shot {rate(one['ts'])} across {count(int(one['games']))}. "
        f"Two days off shot {rate(two['ts'])} across {count(int(two['games']))}. "
        f"Three or more days off shot {rate(long_rest['ts'])} across {count(int(long_rest['games']))}.",
        key="nba_nights",
    )
    chart_heading("True shooting by nights off")
    st.plotly_chart(nights_figure(rest), width="stretch", config=CHART)
    st.caption("The axis starts in the mid-50s so the half-point is visible. Every group is still in the high 50s.")
    walk(
        f"Field goal percentage is made field goals divided by attempts. A layup and a three each count as one make, and free throws stay out of it. "
        f"True shooting counts the points those shots produced: points divided by twice the field-goal attempts plus 0.44 free throws. "
        f"The made three is worth the extra point, and the free throws count. "
        f"On a back-to-back, clubs made {rate(second['fg'])} of their field goals and true shooting was {rate(second['ts'])}. "
        f"With one day off they made {rate(one['fg'])}, and true shooting was {rate(one['ts'])}.",
        key="nba_true_shooting",
    )
    chart_heading("Wins by nights off")
    st.plotly_chart(win_figure(rest), width="stretch", config=CHART)
    st.caption("Same games as the chart above. The axis is tightened. A win rate of 50% is a .500 club.")
    walk(
        f"Points fell from {points(one['pts_per_game'])} to {points(second['pts_per_game'])}. "
        f"The opponent scored {points(second['opp_pts'])} instead of {points(one['opp_pts'])}, and the margin moved from "
        f"{signed_points(one['margin'])} to {signed_points(second['margin'])}. "
        f"Possessions were {points(second['pace'])} a game on the second night and {points(one['pace'])} with one day off. "
        f"Offensive rating moved from {points(one['ortg'])} to {points(second['ortg'])} per 100 possessions.",
        key="nba_points",
    )
    walk(
        f"Two days off shot {rate(two['ts'])}, in line with one day off, and won {rate(two['win_pct'])}. "
        f"Three or more days off shot {rate(long_rest['ts'])} across {count(int(long_rest['games']))} games. "
        f"That bucket holds the All-Star break and the other long layoffs.",
        key="nba_two_days",
    )
    walk(
        f"Two-point field goals fell from {rate(one['fg2'])} to {rate(second['fg2'])}. "
        f"Threes went from {rate(one['fg3'])} to {rate(second['fg3'])}, and three-point attempts stayed at "
        f"{rate(second['three_rate'])} of field goals against {rate(one['three_rate'])}. "
        f"Free throws were {rate(second['ft'])} on a back-to-back and {rate(one['ft'])} with one day off. "
        f"Turnovers were {rate(second['tov_rate'])} of possessions and {rate(one['tov_rate'])}.",
        key="nba_shot_mix",
    )
    walk(
        f"The gap is in each of the {len(by_season)} seasons, from {narrow['ts_gap'] * 100:.1f} points in {narrow['season']} "
        f"to {widest['ts_gap'] * 100:.1f} points in {widest['season']}. "
        f"{quiet['month']} is the quiet month for the shot: back-to-backs shot {rate(quiet['b2b']['ts'])} and one day off shot "
        f"{rate(quiet['one']['ts'])}. {harsh['month']} is the wide month for wins, {rate(harsh['b2b']['win_pct'])} against "
        f"{rate(harsh['one']['win_pct'])} with one day off.",
        key="nba_seasons",
    )
    walk(
        f"Home teams shot {rate(home['ts'])} and won {rate(home['win_pct'])}. "
        f"A home back-to-back shot {rate(home_b2b['ts'])}. A rested team on the road shot {rate(road_rest['ts'])}. "
        f"The building is about a point of true shooting. The second night, set against one day off, is half a point.",
        key="nba_home",
    )
    if not position.empty:
        named = {"Guard": "Guards", "Forward": "Forwards", "Center": "Centers"}
        levels = ". ".join(
            f"{named.get(row.label, row.label)} shot {rate(row.b2b_ts)} on a back-to-back and {rate(row.rest_ts)} with one day off"
            for row in position.itertuples()
        )
        extra = ""
        if abs(float(position["ts_gap"].min())) < 0.01 and abs(float(position["ts_gap"].max())) < 0.01:
            extra = " The step at each spot is under a point."
        guards = position.loc[position["label"].eq("Guard")]
        centers = position.loc[position["label"].eq("Center")]
        if (
            not guards.empty
            and not centers.empty
            and centers.iloc[0]["b2b_ts"] > guards.iloc[0]["b2b_ts"]
            and centers.iloc[0]["rest_ts"] > guards.iloc[0]["rest_ts"]
        ):
            extra += " Centers sit higher on both nights."
        walk(f"{levels}.{extra}", key="nba_position")

    def at(site: str, edge: str) -> dict:
        return nba.pooled(played.loc[played["SITE"].eq(site) & played["EDGE"].eq(edge)])

    home_only_visitor = at("Home", "Opponent on a back-to-back")
    road_only_visitor = at("Away", "You on a back-to-back")
    home_both = at("Home", "Both on a back-to-back")
    road_both = at("Away", "Both on a back-to-back")
    home_only_home = at("Home", "You on a back-to-back")
    road_only_home = at("Away", "Opponent on a back-to-back")
    home_clear = at("Home", "Neither on a back-to-back")
    road_clear = at("Away", "Neither on a back-to-back")
    matchups = pandas.DataFrame(
        [
            {
                "label": "Only the visitor",
                "win_pct": home_only_visitor["win_pct"],
                "games": home_only_visitor["games"],
                "ts": home_only_visitor["ts"],
                "opp_ts": road_only_visitor["ts"],
                "margin": home_only_visitor["margin"],
            },
            {
                "label": "Both clubs",
                "win_pct": home_both["win_pct"],
                "games": home_both["games"],
                "ts": home_both["ts"],
                "opp_ts": road_both["ts"],
                "margin": home_both["margin"],
            },
            {
                "label": "Only the home club",
                "win_pct": home_only_home["win_pct"],
                "games": home_only_home["games"],
                "ts": home_only_home["ts"],
                "opp_ts": road_only_home["ts"],
                "margin": home_only_home["margin"],
            },
        ]
    )

    section("The other club, and the trip")
    claim(
        f"A back-to-back against a rested opponent won {rate(tired_only['win_pct'])} and shot {rate(tired_only['ts'])}, "
        f"on {count(int(tired_only['games']))} games. "
        f"A team with one day off, also against a rested opponent, won {rate(fresh_one['win_pct'])} and shot {rate(fresh_one['ts'])}, "
        f"on {count(int(fresh_one['games']))} games. "
        f"The tired club scored {points(tired_only['pts_per_game'])} and allowed {points(tired_only['opp_pts'])}. "
        f"The one-day club scored {points(fresh_one['pts_per_game'])} and allowed {points(fresh_one['opp_pts'])}.",
        key="nba_rested_claim",
    )
    chart_heading("Home win rate, by who played yesterday")
    st.plotly_chart(home_edge_figure(matchups, home_clear["win_pct"]), width="stretch", config=CHART)
    st.caption(
        f"Each bar is the home club. The gold line is a home club on a night when neither side was on a back-to-back, "
        f"{rate(home_clear['win_pct'])}. The visitor won the rest of the games in each bar."
    )
    walk(
        f"When only the visitor was on a back-to-back, the home club won {rate(home_only_visitor['win_pct'])} "
        f"and shot {rate(home_only_visitor['ts'])}. The visitor won {rate(road_only_visitor['win_pct'])} "
        f"and shot {rate(road_only_visitor['ts'])}, on {count(int(home_only_visitor['games']))} games.",
        key="nba_visitor_only",
    )
    walk(
        f"When both clubs were on a back-to-back, the home club won {rate(home_both['win_pct'])} and shot "
        f"{rate(home_both['ts'])}, and the visitor shot {rate(road_both['ts'])}, on {count(int(home_both['games']))} games. "
        f"When neither club was tired, the home club won {rate(home_clear['win_pct'])} and shot {rate(home_clear['ts'])}.",
        key="nba_both_home",
    )
    walk(
        f"When only the home club was on a back-to-back, that club won {rate(home_only_home['win_pct'])} "
        f"and shot {rate(home_only_home['ts'])}. The rested visitor won {rate(road_only_home['win_pct'])} "
        f"and shot {rate(road_only_home['ts'])}, on {count(int(home_only_home['games']))} games. "
        f"On an ordinary road night the visitor won {rate(road_clear['win_pct'])} and shot {rate(road_clear['ts'])}.",
        key="nba_home_only",
    )
    chart_heading("Wins by the trip into the game")
    st.plotly_chart(trip_figure(trips), width="stretch", config=CHART)
    st.caption(
        "A back-to-back against the same kind of trip with one day off. "
        "Stayed home means both games were at home. Left home means a home game, then a road game the next night."
    )
    walk(
        f"Leaving home the next night won {rate(left['b2b_win'])} of {count(int(left['b2b_n']))} games, "
        f"against {rate(left['one_win'])} for that trip with a day in between. "
        f"Staying home won {rate(stayed['b2b_win'])} of {count(int(stayed['b2b_n']))} second nights, "
        f"against {rate(stayed['one_win'])} for a home game with one day off. "
        f"Coming home the next night, true shooting was {rate(came['b2b_ts'])} against {rate(came['one_ts'])}, "
        f"and the win rate was {rate(came['b2b_win'])} against {rate(came['one_win'])}. "
        f"The second city of a road trip won {rate(road_trip['b2b_win'])}, against {rate(road_trip['one_win'])} with a day off.",
        key="nba_trip",
    )
    walk(
        f"When both clubs were on a back-to-back, true shooting was {rate(both_tired['ts'])} and the games averaged "
        f"{points(both_tired['pts_per_game'])} points, on {count(int(both_tired['games']))} games. "
        f"When neither club was, true shooting was {rate(neither['ts'])} and the games averaged "
        f"{points(neither['pts_per_game'])} points.",
        key="nba_both_score",
    )
    walk(
        f"Second-night opponents won {rate(b2b_opp_win)} of their games that season. "
        f"Opponents of a team with one day off won {rate(one_opp_win)}. "
        f"The box score has the date, the building, and the shots. It has no injury report and no miles between cities.",
        key="nba_opponent_record",
    )

    section("The clubs")
    cold_clubs = gaps.nsmallest(ENDS, "ts_gap")
    hot_clubs = gaps.nlargest(ENDS, "ts_gap")
    cold_club = cold_clubs.loc[cold_clubs["ts_gap"].idxmin()]
    hot_club = hot_clubs.loc[hot_clubs["ts_gap"].idxmax()]
    hot_cold_switch(
        switch_id="nba-clubs-hot",
        face="clubs",
        cold_key="nba_clubs_cold",
        hot_key="nba_clubs_hot",
        cold_claim=(
            f"The {cold_club['name']} shot {rate(cold_club['b2b_ts'])} on {count(int(cold_club['b2b_games']))} back-to-backs "
            f"and {rate(cold_club['rest_ts'])} in their other games. The margin moved from "
            f"{signed_points(cold_club['rest_margin'])} to {signed_points(cold_club['b2b_margin'])}, and the win rate moved from "
            f"{rate(cold_club['rest_win'])} to {rate(cold_club['b2b_win'])}."
        ),
        hot_claim=(
            f"The {hot_club['name']} shot {rate(hot_club['b2b_ts'])} on {count(int(hot_club['b2b_games']))} back-to-backs "
            f"and {rate(hot_club['rest_ts'])} in their other games. The margin moved from "
            f"{signed_points(hot_club['rest_margin'])} to {signed_points(hot_club['b2b_margin'])}, and the win rate moved from "
            f"{rate(hot_club['rest_win'])} to {rate(hot_club['b2b_win'])}."
        ),
        cold_title="Where the drop is widest",
        hot_title="Where the second night shoots better",
        cold_fig=club_gap_figure(cold_clubs, league_gap, worst=True),
        hot_fig=club_gap_figure(hot_clubs, league_gap, worst=False),
        caption=(
            f"Ice cold is the {ENDS} widest drops. Hot is the {ENDS} widest gains. "
            f"Back-to-back true shooting minus the other games. The gold line is the league gap, {league_gap * 100:.1f} points. "
            "Red is a larger drop than the league."
        ),
    )
    walk(
        f"Win rate and true shooting land on different clubs. "
        f"The {low_win['name']} won {rate(low_win['b2b_win'])} of {count(int(low_win['b2b_games']))} back-to-backs "
        f"and {rate(low_win['rest_win'])} of their other games. True shooting was {abs(low_win['ts_gap']) * 100:.1f} points "
        f"{'lower' if low_win['ts_gap'] < 0 else 'higher'}, "
        f"and the margin moved {signed_points(low_win['margin_gap'])}. "
        f"The {next_win['name']} won {rate(next_win['b2b_win'])} of their back-to-backs and {rate(next_win['rest_win'])} otherwise. "
        f"The {high_win['name']} won {rate(high_win['b2b_win'])} on the second night and {rate(high_win['rest_win'])} otherwise, "
        f"with true shooting {abs(high_win['ts_gap']) * 100:.1f} points {'lower' if high_win['ts_gap'] < 0 else 'higher'}.",
        key="nba_win_clubs",
    )
    walk(
        f"The {low_win['name']} played {rate(low_road)} of their {count(int(low_win['b2b_games']))} second nights on the road, "
        f"and {rate(low_fresh)} of them faced a rested opponent. The {worst_shot['name']} played {rate(shot_road)} of their "
        f"{count(int(worst_shot['b2b_games']))} second nights on the road, and {rate(shot_fresh)} of them faced a rested opponent. "
        f"Across the league, {rate(league_road)} of back-to-backs are on the road and {rate(league_fresh)} face a rested opponent.",
        key="nba_slate",
    )

    with st.expander("Look up one club"):
        club_names = gaps.sort_values("name")
        default_club = worst_shot["name"]
        choice = st.selectbox(
            "Club",
            club_names["name"].tolist(),
            index=int(club_names["name"].tolist().index(default_club)) if default_club in set(club_names["name"]) else 0,
        )
        season_choice = st.selectbox("Season for this club", ["All seasons", *list(reversed(seasons))])
        club_games = played.loc[played["TEAM_NAME"].eq(choice)].copy()
        if season_choice != "All seasons":
            club_games = club_games.loc[club_games["SEASON"].eq(season_choice)]
        club_rest = nba.summarize_labels(club_games, "REST_LABEL", nba.REST_ORDER)
        league_for_club = rest if season_choice == "All seasons" else nba.summarize_labels(
            played.loc[played["SEASON"].eq(season_choice)], "REST_LABEL", nba.REST_ORDER
        )
        if club_rest.empty:
            st.caption("No games for that club in this season.")
        else:
            club_second = club_rest.loc[club_rest["label"].eq("Back-to-back")]
            club_one = club_rest.loc[club_rest["label"].eq("1 day off")]
            if not club_second.empty and not club_one.empty:
                claim(
                    f"{choice} won {rate(club_second.iloc[0]['win_pct'])} of {count(int(club_second.iloc[0]['games']))} back-to-backs "
                    f"and {rate(club_one.iloc[0]['win_pct'])} of {count(int(club_one.iloc[0]['games']))} games with one day off. "
                    f"They shot {rate(club_second.iloc[0]['ts'])} and {rate(club_one.iloc[0]['ts'])}.",
                    key="nba_one_club",
                )
            chart_heading(choice)
            st.plotly_chart(club_rest_figure(club_rest, league_for_club), width="stretch", config=CHART)
            st.caption("True shooting for this club against the league in the same window. The axis is tightened.")
            club_home = nba.pooled(club_games.loc[club_games["SITE"].eq("Home")])
            club_away = nba.pooled(club_games.loc[club_games["SITE"].eq("Away")])
            if club_home and club_away and club_home.get("ts") is not None and club_away.get("ts") is not None:
                walk(
                    f"At home, {choice} shot {rate(club_home['ts'])}, won {rate(club_home['win_pct'])}, "
                    f"and finished {signed_points(club_home['margin'])} points over {count(int(club_home['games']))} games. "
                    f"On the road, {rate(club_away['ts'])}, {rate(club_away['win_pct'])}, "
                    f"and {signed_points(club_away['margin'])} points over {count(int(club_away['games']))} games.",
                    key="nba_one_club_site",
                )

    section("The players")
    if cases.empty:
        walk(
            "Player game logs are not on disk yet. Run cache_nba_data.py and the names fill in.",
            key="nba_players_missing",
        )
    else:
        cold_players = cases.nsmallest(ENDS, "ts_gap")
        hot_players = cases.nlargest(ENDS, "ts_gap")
        cold_player = cold_players.loc[cold_players["ts_gap"].idxmin()]
        hot_player = hot_players.loc[hot_players["ts_gap"].idxmax()]
        hot_cold_switch(
            switch_id="nba-players-hot",
            face="players",
            cold_key="nba_players_cold",
            hot_key="nba_players_hot",
            cold_claim=(
                f"{cold_player['name']} shot {rate(cold_player['b2b_ts'])} on {count(int(cold_player['b2b_n']))} nights "
                f"after playing the day before, and {rate(cold_player['rest_ts'])} on {count(int(cold_player['rest_n']))} games "
                f"with a day off. The minutes were {points(cold_player['b2b_min'])} and {points(cold_player['rest_min'])}."
            ),
            hot_claim=(
                f"{hot_player['name']} shot {rate(hot_player['b2b_ts'])} on {count(int(hot_player['b2b_n']))} nights "
                f"after playing the day before, and {rate(hot_player['rest_ts'])} on {count(int(hot_player['rest_n']))} games "
                f"with a day off. The minutes were {points(hot_player['b2b_min'])} and {points(hot_player['rest_min'])}."
            ),
            cold_title="Widest drops on a second night",
            hot_title="Widest gains on a second night",
            cold_fig=cast_figure(cold_players, worst=True),
            hot_fig=cast_figure(hot_players, worst=False),
            caption=(
                f"Ice cold is the {ENDS} widest drops. Hot is the {ENDS} widest gains. "
                "Red is a night after he played. Gold is a day between his games. "
                "A player needs 30 second nights and 50 games with one day off."
            ),
        )

    section("The count")
    rest_table(rest)
    face_bridge()
    st.caption(
        "Regular season, from the team game log. True shooting, win rate, and the free-throw and turnover shares "
        "in the text are points on a 100-point scale. Possessions are field-goal attempts, 0.44 free throws, "
        "minus offensive rebounds, plus turnovers."
    )
