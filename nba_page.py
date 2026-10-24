"""Basketball page: efficiency on back-to-backs, extra rest, and home court.

The split is built from team game logs in data/nba. One row is one team in one game.
"""

from __future__ import annotations

import html

import pandas
import plotly.graph_objects as go
import streamlit as st

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


def claim(text: str) -> None:
    st.markdown(f'<p class="reading">{html.escape(text)}</p>', unsafe_allow_html=True)


def walk(text: str) -> None:
    st.markdown(f'<p class="chapter">{html.escape(text)}</p>', unsafe_allow_html=True)


def section(label: str) -> None:
    st.markdown(f'<p class="section-label">{html.escape(label)}</p>', unsafe_allow_html=True)


def row_for(table: pandas.DataFrame, label: str) -> pandas.Series:
    return table.loc[table["label"].eq(label)].iloc[0]


@st.cache_data(show_spinner="Loading NBA game logs...")
def cached_schedule(_mtime: float) -> pandas.DataFrame:
    return nba.attach_schedule(nba.load_team_logs())


def level_figure(table: pandas.DataFrame, column: str, as_percent: bool, height: int = 420) -> go.Figure:
    colors = [REST_COLORS.get(label, GOLD) for label in table["label"]]
    figure = go.Figure(
        go.Bar(
            x=table["label"],
            y=table[column],
            marker_color=colors,
            text=[rate(value) if as_percent else f"{value:.1f}" for value in table[column]],
            textposition="outside",
            textfont=dict(color=CREAM, family="Barlow, sans-serif", size=13),
            cliponaxis=False,
            customdata=table.loc[:, ["games", "pts_per_game", "margin", "win_pct"]].to_numpy(),
            hovertemplate=(
                "<b>%{x}</b><br>Value: %{y:.3f}<br>Games: %{customdata[0]:,}<br>"
                "Points: %{customdata[1]:.1f}<br>Margin: %{customdata[2]:+.1f}<br>"
                "Wins: %{customdata[3]:.1%}<extra></extra>"
            ),
        )
    )
    figure.update_layout(**chart_layout(height), xaxis=dict(tickfont=dict(color=INK), title=None), yaxis=axis())
    if as_percent:
        low = float(table[column].min()) - 0.012
        high = float(table[column].max()) + 0.016
        figure.update_yaxes(tickformat=".1%", range=[low, high])
    else:
        span = float(table[column].max() - table[column].min())
        pad = max(1.2, span * 0.45)
        figure.update_yaxes(range=[float(table[column].min()) - pad, float(table[column].max()) + pad])
    return figure


def edge_figure(table: pandas.DataFrame) -> go.Figure:
    colors = [RED, BLUE, GREEN, GOLD]
    figure = go.Figure(
        go.Bar(
            x=table["label"],
            y=table["margin"],
            marker_color=colors[: len(table)],
            text=[f"{value:+.1f}" for value in table["margin"]],
            textposition="outside",
            textfont=dict(color=CREAM, family="Barlow, sans-serif", size=13),
            cliponaxis=False,
            customdata=table.loc[:, ["games", "ts", "win_pct", "pts_per_game"]].to_numpy(),
            hovertemplate=(
                "<b>%{x}</b><br>Margin: %{y:+.1f}<br>True shooting: %{customdata[1]:.1%}<br>"
                "Wins: %{customdata[2]:.1%}<br>Points: %{customdata[3]:.1f}<br>"
                "Games: %{customdata[0]:,}<extra></extra>"
            ),
        )
    )
    figure.add_hline(y=0, line_color=INK, line_width=1)
    figure.update_layout(
        **chart_layout(420),
        xaxis=dict(tickfont=dict(color=INK), title=None),
        yaxis=axis(),
    )
    return figure


def season_figure(seasons: pandas.DataFrame) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=seasons["season"],
            y=seasons["b2b_ts"],
            mode="lines+markers",
            name="Back-to-back",
            line=dict(color=RED, width=3, shape="spline", smoothing=0.6),
            marker=dict(size=8, color=RED),
            customdata=seasons.loc[:, ["b2b_games", "b2b_pts", "b2b_win"]].to_numpy(),
            hovertemplate=(
                "%{x}<br>Back-to-back: %{y:.1%}<br>Games: %{customdata[0]:,}<br>"
                "Points: %{customdata[1]:.1f}<br>Wins: %{customdata[2]:.1%}<extra></extra>"
            ),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=seasons["season"],
            y=seasons["rest_ts"],
            mode="lines+markers",
            name="At least one day off",
            line=dict(color=GOLD, width=3, shape="spline", smoothing=0.6),
            marker=dict(size=8, color=GOLD),
            hovertemplate="%{x}<br>Rested: %{y:.1%}<extra></extra>",
        )
    )
    low = float(seasons.loc[:, ["b2b_ts", "rest_ts"]].min().min()) - 0.008
    high = float(seasons.loc[:, ["b2b_ts", "rest_ts"]].max().max()) + 0.008
    figure.update_layout(**chart_layout(420), xaxis=dict(tickfont=dict(color=INK), title=None), yaxis=axis())
    figure.update_yaxes(tickformat=".1%", range=[low, high])
    return figure


def home_figure(seasons: pandas.DataFrame) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=seasons["season"],
            y=seasons["home_margin"],
            mode="lines+markers",
            name="Home margin",
            line=dict(color=GOLD, width=3, shape="spline", smoothing=0.6),
            marker=dict(size=8, color=GOLD),
            customdata=seasons.loc[:, ["home_ts", "away_ts"]].to_numpy(),
            hovertemplate=(
                "%{x}<br>Home margin: %{y:+.1f}<br>Home true shooting: %{customdata[0]:.1%}<br>"
                "Road true shooting: %{customdata[1]:.1%}<extra></extra>"
            ),
        )
    )
    figure.add_hline(y=0, line_color=INK, line_width=1)
    figure.update_layout(**chart_layout(380), xaxis=dict(tickfont=dict(color=INK), title=None), yaxis=axis())
    return figure


def cross_figure(table: pandas.DataFrame, column: str, as_percent: bool) -> go.Figure:
    rested = table.loc[table["b2b"].eq("Rested")]
    second = table.loc[table["b2b"].eq("Back-to-back")]
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=rested["site"],
            y=rested[column],
            name="Rested",
            marker_color=GOLD,
            customdata=rested.loc[:, ["games", "margin", "ts", "win_pct"]].to_numpy(),
            hovertemplate=(
                "<b>%{x}, rested</b><br>%{y:.3f}<br>Games: %{customdata[0]:,}<br>"
                "Margin: %{customdata[1]:+.1f}<br>True shooting: %{customdata[2]:.1%}<br>"
                "Wins: %{customdata[3]:.1%}<extra></extra>"
            ),
        )
    )
    figure.add_trace(
        go.Bar(
            x=second["site"],
            y=second[column],
            name="Back-to-back",
            marker_color=RED,
            customdata=second.loc[:, ["games", "margin", "ts", "win_pct"]].to_numpy(),
            hovertemplate=(
                "<b>%{x}, back-to-back</b><br>%{y:.3f}<br>Games: %{customdata[0]:,}<br>"
                "Margin: %{customdata[1]:+.1f}<br>True shooting: %{customdata[2]:.1%}<br>"
                "Wins: %{customdata[3]:.1%}<extra></extra>"
            ),
        )
    )
    figure.update_layout(**chart_layout(420), xaxis=dict(tickfont=dict(color=INK), title=None), yaxis=axis())
    if as_percent:
        low = float(table[column].min()) - 0.012
        high = float(table[column].max()) + 0.012
        figure.update_yaxes(tickformat=".1%", range=[low, high])
    else:
        figure.add_hline(y=0, line_color=INK, line_width=1)
    return figure


def club_gap_figure(gaps: pandas.DataFrame, league_gap: float) -> go.Figure:
    frame = gaps.sort_values(["ts_gap", "team"], ascending=[False, True])
    figure = go.Figure(
        go.Bar(
            x=frame["ts_gap"],
            y=frame["name"],
            orientation="h",
            marker_color=[RED if value < league_gap else BLUE for value in frame["ts_gap"]],
            customdata=frame.loc[:, ["team", "b2b_games", "margin_gap", "win_gap", "b2b_ts"]].to_numpy(),
            hovertemplate=(
                "<b>%{y}</b><br>True shooting gap: %{x:+.1%}<br>"
                "Margin gap: %{customdata[2]:+.1f}<br>Win gap: %{customdata[3]:+.1%}<br>"
                "Back-to-backs: %{customdata[1]}<extra></extra>"
            ),
        )
    )
    figure.add_vline(x=league_gap, line_dash="dash", line_color=GOLD, line_width=2)
    figure.update_layout(
        **chart_layout(max(520, 26 * len(frame) + 80)),
        yaxis=dict(
            categoryorder="array",
            categoryarray=frame["name"].tolist(),
            showgrid=False,
            tickfont=dict(color=CREAM, size=13),
            title=None,
        ),
        xaxis=axis(),
    )
    figure.update_xaxes(tickformat="+.1%")
    return figure


def club_rest_figure(club: pandas.DataFrame, league: pandas.DataFrame) -> go.Figure:
    merged = league.loc[:, ["label", "ts"]].merge(club.loc[:, ["label", "ts", "games"]], on="label", how="left")
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=merged["label"],
            y=merged["ts_y"],
            name="Club",
            marker_color=RED,
            customdata=merged["games"],
            hovertemplate="%{x}<br>Club: %{y:.1%}<br>Games: %{customdata:,}<extra></extra>",
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
    else:
        figure.update_yaxes(tickformat=".1%")
    return figure


def paired_figure(
    table: pandas.DataFrame,
    left: str,
    right: str,
    left_name: str,
    right_name: str,
    as_percent: bool,
) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(go.Bar(x=table["label"], y=table[left], name=left_name, marker_color=RED))
    figure.add_trace(go.Bar(x=table["label"], y=table[right], name=right_name, marker_color=GOLD))
    figure.update_layout(**chart_layout(420), xaxis=dict(tickfont=dict(color=INK), title=None), yaxis=axis())
    values = pandas.concat([table[left], table[right]])
    if as_percent:
        figure.update_yaxes(tickformat=".1%", range=[float(values.min()) - 0.012, float(values.max()) + 0.014])
    else:
        figure.update_yaxes(range=[max(0.0, float(values.min()) - 2), float(values.max()) + 2])
    return figure


def group_name(label: str) -> str:
    names = {
        "Guard": "Guards",
        "Forward": "Forwards",
        "Center": "Centers",
        "Under 15": "Players under 15 minutes",
        "15 to 24": "Players at 15 to 24 minutes",
        "25 to 32": "Players at 25 to 32 minutes",
        "33 or more": "Players at 33 or more minutes",
        "23 and under": "Players 23 and under",
        "24 to 27": "Players 24 to 27",
        "28 to 31": "Players 28 to 31",
        "32 and over": "Players 32 and over",
        "Under 18% usage": "Players under 18% usage",
        "18 to 24% usage": "Players at 18 to 24% usage",
        "24% usage or more": "Players at 24% usage or more",
    }
    return names.get(label, label)


def move_phrase(gap: float) -> str:
    points = gap * 100
    if abs(points) < 0.15:
        return "stay flat"
    if points < 0:
        return f"fall {abs(points):.1f} points"
    return f"rise {points:.1f} points"


def gap_claim(table: pandas.DataFrame) -> str:
    ordered = table.sort_values("ts_gap")
    widest = ordered.iloc[0]
    narrow = ordered.iloc[-1]
    spread = (narrow["ts_gap"] - widest["ts_gap"]) * 100
    levels = ". ".join(
        f"{group_name(row.label)} shot {rate(row.b2b_ts)} on a back-to-back and {rate(row.rest_ts)} with one day off"
        for row in table.itertuples()
    )
    if spread < 0.4:
        lead = f"The back-to-back move stays inside {spread:.1f} points across these groups."
    else:
        lead = (
            f"{group_name(widest['label'])} {move_phrase(widest['ts_gap'])}, the largest move. "
            f"{group_name(narrow['label'])} {move_phrase(narrow['ts_gap'])}."
        )
    return f"{lead} {levels}."


def minutes_claim(table: pandas.DataFrame) -> str:
    heavy = table.loc[table["label"].eq(nba.ROLE_ORDER[-1])]
    short = table.loc[table["label"].eq(nba.ROLE_ORDER[0])]
    if heavy.empty or short.empty:
        return ""
    top = heavy.iloc[0]
    bottom = short.iloc[0]
    return (
        f"Players who average 33 or more minutes played {top['b2b_min']:.1f} on the second night "
        f"and {top['rest_min']:.1f} with one day off. "
        f"The under-15 group played longer on the back-to-back, {bottom['b2b_min']:.1f} minutes against {bottom['rest_min']:.1f}."
    )


def gap_walk(table: pandas.DataFrame) -> str:
    sentences = []
    for row in table.itertuples():
        sentence = (
            f"{group_name(row.label)}: {row.b2b_min:.1f} minutes on the second night, {row.rest_min:.1f} with one day off "
            f"({count(int(row.b2b_games))} and {count(int(row.rest_games))} appearances)"
        )
        home_ready = row.home_b2b_ts is not None and row.home_rest_ts is not None
        road_ready = row.away_b2b_ts is not None and row.away_rest_ts is not None
        if home_ready and road_ready and not pandas.isna(row.home_b2b_ts) and not pandas.isna(row.away_b2b_ts):
            sentence += (
                f". At home, {rate(row.home_b2b_ts)} on a back-to-back against {rate(row.home_rest_ts)} with one day off. "
                f"On the road, {rate(row.away_b2b_ts)} against {rate(row.away_rest_ts)}"
            )
        sentences.append(sentence + ".")
    return " ".join(sentences)


def player_cut(frame: pandas.DataFrame, column: str, order: list[str], title: str, chart_minutes: bool) -> pandas.DataFrame:
    table = nba.rest_contrast(frame, column, order)
    if table.empty:
        return table
    section(title)
    text = gap_claim(table)
    if chart_minutes:
        extra = minutes_claim(table)
        if extra:
            text = f"{text} {extra}"
    claim(text)
    st.plotly_chart(paired_figure(table, "b2b_ts", "rest_ts", "Back-to-back", "One day off", True), width="stretch", config={"displaylogo": False})
    st.caption("True shooting, team back-to-back against one day off. The axis is tightened.")
    if chart_minutes:
        st.plotly_chart(
            paired_figure(table, "b2b_min", "rest_min", "Back-to-back", "One day off", False),
            width="stretch",
            config={"displaylogo": False},
        )
        st.caption("Minutes he played that night. The season average is only how the groups are built.")
    walk(gap_walk(table))
    return table


def appearance_cut(frame: pandas.DataFrame) -> pandas.DataFrame:
    table = nba.group_levels(frame, "APPEARANCE", nba.APPEARANCE_ORDER)
    if table.empty:
        return table
    by_label = {row.label: row for row in table.itertuples()}
    yesterday = by_label.get("Played yesterday")
    sat = by_label.get("Second night, sat yesterday")
    off = by_label.get("Team had a day off")
    section("Played the night before")
    if yesterday is not None and off is not None and sat is not None:
        share = yesterday.games / (yesterday.games + sat.games)
        claim(
            f"On a team back-to-back, {rate(share)} of the appearances are players who played yesterday. "
            f"They shot {rate(yesterday.ts)} in {yesterday.minutes:.1f} minutes, "
            f"with a plus-minus of {signed_points(yesterday.margin)}. "
            f"Players who sat the night before and played on the second night shot {rate(sat.ts)} in {sat.minutes:.1f} minutes. "
            f"When the team had a day off, the floor shot {rate(off.ts)} in {off.minutes:.1f} minutes, "
            f"with a plus-minus of {signed_points(off.margin)}. "
            f"The players who sat also took a shorter shift, {sat.minutes:.1f} minutes against {yesterday.minutes:.1f}, "
            f"so the shooting gap sits next to a different job."
        )
    figure = go.Figure(
        go.Bar(
            x=table["label"],
            y=table["ts"],
            marker_color=[RED, BLUE, GOLD][: len(table)],
            text=[rate(value) for value in table["ts"]],
            textposition="outside",
            textfont=dict(color=CREAM, family="Barlow, sans-serif", size=13),
            cliponaxis=False,
            customdata=table.loc[:, ["games", "minutes", "players"]].to_numpy(),
            hovertemplate=(
                "<b>%{x}</b><br>True shooting: %{y:.1%}<br>Minutes: %{customdata[1]:.1f}<br>"
                "Appearances: %{customdata[0]:,}<br>Players: %{customdata[2]:,}<extra></extra>"
            ),
        )
    )
    low = float(table["ts"].min()) - 0.015
    high = float(table["ts"].max()) + 0.02
    figure.update_layout(**chart_layout(420), xaxis=dict(tickfont=dict(color=INK), title=None), yaxis=axis())
    figure.update_yaxes(tickformat=".1%", range=[low, high])
    st.plotly_chart(figure, width="stretch", config={"displaylogo": False})
    st.caption("Played yesterday is his own back-to-back. The middle bar is a team second night he did not play the day before.")
    return table


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


def floor_sections(players: pandas.DataFrame) -> None:
    section("Who was on the floor")
    if players.empty:
        walk(
            "Position, minutes, age, and usage need the player game log. "
            "Run cache_nba_data.py and this part fills in."
        )
        return
    played = players.loc[players["REST_LABEL"].notna()].copy()
    covered = int(played["POSITION_GROUP"].notna().sum())
    walk(
        "Each row from here is one player in a game he played. "
        f"The roster names a position for {count(covered)} of {count(len(played))} appearances with a known rest day. "
        "A hyphenated spot follows the first letter, so G-F is a guard and C-F is a center. "
        "Usual minutes are his average that season. A run of fewer than 10 games stays out of that chart. "
        "The season advanced line supplies age and usage."
    )
    player_cut(played, "POSITION_GROUP", nba.POSITION_ORDER, "Position", False)
    player_cut(played, "ROLE", nba.ROLE_ORDER, "Usual minutes", True)
    appearance_cut(played)
    player_cut(played, "AGE_BAND", nba.AGE_ORDER, "Age", False)
    player_cut(played, "USAGE_BAND", nba.USAGE_ORDER, "Usage", False)


def missing_cache() -> None:
    st.markdown(
        """
        <header class="mast">
          <p class="kicker">NBA</p>
          <h1>Do back-to-backs and home court change efficiency?</h1>
          <p class="deck">The team game log is not on disk yet.</p>
        </header>
        """,
        unsafe_allow_html=True,
    )
    st.code(".\\.venv\\Scripts\\python.exe cache_nba_data.py", language="text")


def display_rates(table: pandas.DataFrame) -> pandas.DataFrame:
    shown = table.loc[
        :,
        ["label", "games", "ts", "pts_per_game", "opp_pts", "margin", "win_pct", "ortg", "pace", "three_rate", "tov_rate"],
    ].copy()
    for column in ("ts", "win_pct", "three_rate", "tov_rate"):
        shown[column] = shown[column] * 100
    return shown


def nba_page() -> None:
    st.html('<p class="back"><a href="/">Home</a></p>')
    if not any((nba.CACHE / "team_gamelogs").glob("*.parquet")):
        missing_cache()
        return

    schedule = cached_schedule(nba.cache_mtime())
    on_disk = nba.load_team_logs()
    held = nba.held_seasons(on_disk)
    if not held.empty:
        schedule = schedule.loc[schedule["SEASON"].isin(nba.mature_seasons(on_disk))].copy()
    played = nba.with_rest(schedule)
    if played.empty:
        missing_cache()
        return

    seasons = nba.seasons_in(played)
    rest = nba.summarize_labels(played, "REST_LABEL", nba.REST_ORDER)
    edge = nba.summarize_labels(played, "EDGE", nba.EDGE_ORDER)
    sites = nba.summarize_labels(played, "SITE", ["Home", "Away"])
    cross = nba.cross_summary(played)
    by_season = nba.season_compare(played)
    gaps = nba.team_gaps(played)

    second = row_for(rest, "Back-to-back")
    one = row_for(rest, "1 day off")
    two = row_for(rest, "2 days off")
    long_rest = row_for(rest, "3 or more days off")
    home = row_for(sites, "Home")
    away = row_for(sites, "Away")
    tired_only = row_for(edge, "You on a back-to-back")
    both_tired = row_for(edge, "Both on a back-to-back")
    neither = row_for(edge, "Neither on a back-to-back")
    fresh_only = row_for(edge, "Opponent on a back-to-back")
    road_b2b = cross.loc[cross["label"].eq("Road, back-to-back")].iloc[0]
    home_b2b = cross.loc[cross["label"].eq("Home, back-to-back")].iloc[0]
    home_rest = cross.loc[cross["label"].eq("Home, rested")].iloc[0]
    road_rest = cross.loc[cross["label"].eq("Road, rested")].iloc[0]
    openers = int(schedule["REST_LABEL"].isna().sum())
    league_gap = float(second["ts"] - one["ts"])
    by_season = by_season.assign(ts_gap=by_season["b2b_ts"] - by_season["rest_ts"])
    widest = by_season.sort_values("ts_gap").iloc[0]
    narrow = by_season.sort_values("ts_gap").iloc[-1]
    worst = gaps.iloc[0]
    best = gaps.sort_values("ts_gap").iloc[-1]

    st.markdown(
        f"""
        <header class="mast">
          <p class="kicker">NBA, {html.escape(seasons[0])} to {html.escape(seasons[-1])}</p>
          <h1>Do back-to-backs and home court change efficiency?</h1>
        </header>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="deck">True shooting and the scoreboard, split by nights off, by building, '
        "and by who was on the floor. Regular season only.</p>",
        unsafe_allow_html=True,
    )
    claim(
        f"On a back-to-back, teams shot {rate(second['ts'])} and were outscored by {abs(second['margin']):.1f} points. "
        f"With one day off they shot {rate(one['ts'])} and finished {signed_points(one['margin'])} points. "
        f"Home teams shot {rate(home['ts'])} and won {rate(home['win_pct'])} of their games."
    )
    walk(
        f"A back-to-back is a game on the day after that club's previous game. "
        f"One day off means a day in between. True shooting pools every field goal and free throw in the group, "
        f"points divided by twice the attempts plus 0.44 free throws. "
        f"The first game of each season has no previous game, so those {count(openers)} team-games stay out of the rest charts. "
        f"Home and road come from the matchup. This window is {count(int(played.shape[0]))} team-games "
        f"from {seasons[0]} through {seasons[-1]}."
        + (
            " " + " ".join(
                f"{season} has {count(int(games))} team-games on disk and stays out until the season is longer."
                for season, games in held.items()
            )
            if not held.empty
            else ""
        )
    )

    columns = st.columns(4)
    columns[0].metric(
        "Back-to-back true shooting",
        rate(second["ts"]),
        f"{league_gap * 100:+.1f} TS vs one day off",
    )
    columns[1].metric(
        "Home true shooting",
        rate(home["ts"]),
        f"{(home['ts'] - away['ts']) * 100:+.1f} TS vs road",
    )
    columns[2].metric("Road back-to-back margin", signed_points(road_b2b["margin"]))
    columns[3].metric(
        "You tired, they rested",
        signed_points(tired_only["margin"]),
        f"{count(int(tired_only['games']))} games",
    )

    section("Nights off")
    claim(
        f"The second night is {abs(league_gap) * 100:.1f} points of true shooting under one day off, "
        f"on {count(int(second['games']))} games against {count(int(one['games']))}."
    )
    st.plotly_chart(level_figure(rest, "ts", True), width="stretch", config={"displaylogo": False})
    st.caption("The axis starts in the mid-50s so the half-point is visible. Every group is still in the high 50s.")
    walk(
        f"Points fell from {one['pts_per_game']:.1f} to {second['pts_per_game']:.1f}. "
        f"The opponent scored {second['opp_pts']:.1f} instead of {one['opp_pts']:.1f}. "
        f"The margin moved from {signed_points(one['margin'])} to {signed_points(second['margin'])}. "
        f"Possessions were {second['pace']:.1f} per game on a back-to-back and {one['pace']:.1f} with one day off, "
        f"and offensive rating moved from {one['ortg']:.1f} to {second['ortg']:.1f} per 100 possessions. "
        f"The scoring drop is the shooting, in a game of the same length."
    )
    walk(
        f"Two days off shot {rate(two['ts'])}, in line with one day off, and won {rate(two['win_pct'])}. "
        f"Three or more days off shot {rate(long_rest['ts'])} across {count(int(long_rest['games']))} games. "
        f"That bucket holds the All-Star break and the other long layoffs, so it is a thin rung next to the {count(int(one['games']))} games on one day off."
    )
    walk(
        f"The shot mix stays put. Three-point attempts were {rate(second['three_rate'])} of field goals on a back-to-back "
        f"and {rate(one['three_rate'])} with one day off. "
        f"Turnovers were {rate(second['tov_rate'])} of possessions and {rate(one['tov_rate'])}. "
        f"Free-throw rate was {rate(second['ft_rate'])} and {rate(one['ft_rate'])}. "
        f"Wins were {rate(second['win_pct'])} on the second night and {rate(one['win_pct'])} with one day off."
    )

    section("The other club")
    claim(
        f"{count(int(tired_only['games']))} of the {count(int(second['games']))} back-to-backs faced a rested opponent. "
        f"In those games the tired club shot {rate(tired_only['ts'])}, was outscored by {abs(tired_only['margin']):.1f} points, "
        f"and won {rate(tired_only['win_pct'])}."
    )
    st.plotly_chart(edge_figure(edge), width="stretch", config={"displaylogo": False})
    st.caption("Margin is this club's points minus the opponent's. Zero is an even game.")
    walk(
        f"When both clubs were on a back-to-back, true shooting was {rate(both_tired['ts'])} "
        f"and the margin was {signed_points(both_tired['margin'])}, "
        f"on {count(int(both_tired['games']))} team-games. "
        f"When neither club was, true shooting was {rate(neither['ts'])} and the margin was even, "
        f"on {count(int(neither['games']))} team-games. "
        f"The second night looks ordinary when the opponent is on one too."
    )
    walk(
        f"When the opponent was on a back-to-back and you were not, you shot {rate(fresh_only['ts'])}, "
        f"finished {signed_points(fresh_only['margin'])} points, and won {rate(fresh_only['win_pct'])}. "
        f"That is the mirror of the tired side, {count(int(fresh_only['games']))} games. "
        f"The box score has your shots. It does not say whether the miss came from heavy legs or from a defense that had the night off."
    )

    section("Each season")
    claim(
        f"The back-to-back gap in true shooting runs from {narrow['ts_gap'] * 100:.1f} points in {narrow['season']} "
        f"to {widest['ts_gap'] * 100:.1f} points in {widest['season']}."
    )
    st.plotly_chart(season_figure(by_season), width="stretch", config={"displaylogo": False})
    st.caption("The axis is tightened. Rested means at least one day off, including two days and the long layoffs.")
    season_bits = []
    for record in by_season.itertuples(index=False):
        season_bits.append(
            f"{record.season} was {(record.b2b_ts - record.rest_ts) * 100:.1f} points "
            f"({rate(record.b2b_ts)} against {rate(record.rest_ts)}, wins {rate(record.b2b_win)} against {rate(record.rest_win)})"
        )
    walk("By season, back-to-back true shooting minus the rested rate: " + ". ".join(season_bits) + ".")

    section("Home court")
    claim(
        f"Home teams shot {rate(home['ts'])} to {rate(away['ts'])} on the road, "
        f"outscored the opponent by {home['margin']:.1f} points, and won {rate(home['win_pct'])}."
    )
    st.plotly_chart(home_figure(by_season), width="stretch", config={"displaylogo": False})
    st.caption("Home margin is the home club's points minus the road club's, averaged over home team-games.")
    home_low = by_season.sort_values("home_margin").iloc[0]
    home_high = by_season.sort_values("home_margin").iloc[-1]
    walk(
        f"The home margin stays in a band, from {signed_points(home_low['home_margin'])} in {home_low['season']} "
        f"to {signed_points(home_high['home_margin'])} in {home_high['season']}. "
        f"Road teams shot {rate(away['ts'])}, scored {away['pts_per_game']:.1f}, "
        f"and won {rate(away['win_pct'])}, on {count(int(away['games']))} games. "
        f"Home was {count(int(home['games']))} games. "
        f"The building is a full point of true shooting. The second night, measured against one day off, is half a point."
    )

    section("The road back-to-back")
    claim(
        f"The road back-to-back shot {rate(road_b2b['ts'])}, was outscored by {abs(road_b2b['margin']):.1f} points, "
        f"and won {rate(road_b2b['win_pct'])}, on {count(int(road_b2b['games']))} games."
    )
    st.plotly_chart(cross_figure(cross, "ts", True), width="stretch", config={"displaylogo": False})
    st.caption("Rested here means any game that is not a back-to-back. The axis is tightened.")
    st.plotly_chart(cross_figure(cross, "margin", False), width="stretch", config={"displaylogo": False})
    walk(
        f"A rested home team shot {rate(home_rest['ts'])}, finished {signed_points(home_rest['margin'])} points, "
        f"and won {rate(home_rest['win_pct'])}, on {count(int(home_rest['games']))} games. "
        f"A home back-to-back shot {rate(home_b2b['ts'])}, finished {signed_points(home_b2b['margin'])} points, "
        f"and won {rate(home_b2b['win_pct'])}, on {count(int(home_b2b['games']))} games. "
        f"A rested road team shot {rate(road_rest['ts'])} and finished {signed_points(road_rest['margin'])} points. "
        f"A tired team at home still shot better than a rested team on the road, "
        f"{rate(home_b2b['ts'])} against {rate(road_rest['ts'])}."
    )

    section("Which clubs")
    claim(
        f"For the {worst['name']}, true shooting falls {abs(worst['ts_gap']) * 100:.1f} points on back-to-backs, "
        f"and the margin falls {abs(worst['margin_gap']):.1f} points. "
        f"For the {best['name']}, true shooting is {best['ts_gap'] * 100:.1f} points higher, "
        f"and the margin moves {signed_points(best['margin_gap'])} points."
    )
    st.plotly_chart(club_gap_figure(gaps, league_gap), width="stretch", config={"displaylogo": False})
    st.caption(
        f"Each bar is back-to-back true shooting minus the games with at least one day off. "
        f"The gold line is the league gap, {league_gap * 100:.1f} points. "
        f"Every club has at least {int(gaps['b2b_games'].min())} back-to-backs in the file. "
        f"Red is a larger drop than the league."
    )
    if best["ts_gap"] > 0 and best["margin_gap"] < 0:
        walk(
            f"Shooting and the scoreboard part company for some clubs. "
            f"The {best['name']} make more of their shots on the second night and still lose ground on the margin, "
            f"{signed_points(best['margin_gap'])} points versus their rested games. "
            f"Their win rate moves {best['win_gap'] * 100:+.1f} points."
        )

    players = cached_players(nba.cache_mtime())
    if not players.empty and not held.empty:
        players = players.loc[~players["SEASON"].isin(set(held.index))].copy()
    floor_sections(players)

    section("One club")
    walk("The charts above use every club. The picker keeps that league line and adds one team beside it.")
    names = gaps.sort_values("name")
    choice = st.selectbox(
        "Club",
        names["name"].tolist(),
        index=int(names["name"].tolist().index(worst["name"])) if worst["name"] in set(names["name"]) else 0,
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
                f"{choice} shot {rate(club_second.iloc[0]['ts'])} on {int(club_second.iloc[0]['games'])} back-to-backs "
                f"and {rate(club_one.iloc[0]['ts'])} on {int(club_one.iloc[0]['games'])} games with one day off."
            )
        st.plotly_chart(
            club_rest_figure(club_rest, league_for_club),
            width="stretch",
            config={"displaylogo": False},
        )
        club_home = nba.pooled(club_games.loc[club_games["SITE"].eq("Home")])
        club_away = nba.pooled(club_games.loc[club_games["SITE"].eq("Away")])
        if club_home and club_away:
            walk(
                f"At home, {choice} shot {rate(club_home['ts'])} and finished {signed_points(club_home['margin'])} points "
                f"over {count(club_home['games'])} games. "
                f"On the road, {rate(club_away['ts'])} and {signed_points(club_away['margin'])} points "
                f"over {count(club_away['games'])} games."
            )

    section("The table")
    walk(
        "True shooting, points, and the possession estimate for every rest bucket. "
        "Percent columns are points on a 100-point scale. "
        "Offensive rating is points per 100 possessions, with possessions as field-goal attempts, "
        "0.44 free throws, minus offensive rebounds, plus turnovers."
    )
    rest_table = display_rates(rest)
    st.dataframe(
        rest_table,
        hide_index=True,
        width="stretch",
        column_config={
            "label": "Nights off",
            "games": "Games",
            "ts": st.column_config.NumberColumn("TS%", format="%.1f"),
            "pts_per_game": st.column_config.NumberColumn("PTS", format="%.1f"),
            "opp_pts": st.column_config.NumberColumn("Opp PTS", format="%.1f"),
            "margin": st.column_config.NumberColumn("Margin", format="%+.1f"),
            "win_pct": st.column_config.NumberColumn("Win%", format="%.1f"),
            "ortg": st.column_config.NumberColumn("ORTG", format="%.1f"),
            "pace": st.column_config.NumberColumn("Poss", format="%.1f"),
            "three_rate": st.column_config.NumberColumn("3PA rate", format="%.1f"),
            "tov_rate": st.column_config.NumberColumn("TOV rate", format="%.1f"),
        },
    )
    st.download_button(
        "Download rest table",
        data=rest_table.to_csv(index=False).encode("utf-8"),
        file_name="nba_rest_efficiency.csv",
        mime="text/csv",
        key="nba_download_rest",
    )

    gap_table = gaps.loc[:, ["team", "name", "b2b_games", "b2b_ts", "rest_ts", "ts_gap", "margin_gap", "win_gap"]].copy()
    for column in ("b2b_ts", "rest_ts", "ts_gap", "win_gap"):
        gap_table[column] = gap_table[column] * 100
    with st.expander("Club gaps"):
        st.dataframe(
            gap_table,
            hide_index=True,
            width="stretch",
            column_config={
                "team": "Team",
                "name": "Name",
                "b2b_games": "Back-to-backs",
                "b2b_ts": st.column_config.NumberColumn("B2B TS%", format="%.1f"),
                "rest_ts": st.column_config.NumberColumn("Rested TS%", format="%.1f"),
                "ts_gap": st.column_config.NumberColumn("TS gap", format="%+.1f"),
                "margin_gap": st.column_config.NumberColumn("Margin gap", format="%+.1f"),
                "win_gap": st.column_config.NumberColumn("Win gap", format="%+.1f"),
            },
        )
        st.download_button(
            "Download club gaps",
            data=gap_table.to_csv(index=False).encode("utf-8"),
            file_name="nba_b2b_club_gaps.csv",
            mime="text/csv",
            key="nba_download_gaps",
        )

    with st.expander("Game log"):
        if club_games.empty:
            st.caption("Pick a club with games in this season.")
        else:
            log = club_games.sort_values(["GAME_DATE", "GAME_ID"], ascending=[False, False]).loc[
                :,
                [
                    "SEASON",
                    "GAME_DATE",
                    "OPPONENT",
                    "SITE",
                    "WL",
                    "REST_DAYS",
                    "OPP_REST",
                    "PTS",
                    "OPP_PTS",
                    "PLUS_MINUS",
                    "GAME_TS",
                ],
            ].copy()
            log["GAME_DATE"] = log["GAME_DATE"].dt.date
            log["REST_DAYS"] = log["REST_DAYS"].astype("Int64")
            log["OPP_REST"] = log["OPP_REST"].astype("Int64")
            log["GAME_TS"] = log["GAME_TS"] * 100
            st.dataframe(
                log,
                hide_index=True,
                width="stretch",
                column_config={
                    "SEASON": "Season",
                    "GAME_DATE": "Date",
                    "OPPONENT": "Opponent",
                    "SITE": "Site",
                    "WL": "W/L",
                    "REST_DAYS": "Nights off",
                    "OPP_REST": "Opp nights off",
                    "PTS": "PTS",
                    "OPP_PTS": "Opp",
                    "PLUS_MINUS": st.column_config.NumberColumn("+/-", format="%+.0f"),
                    "GAME_TS": st.column_config.NumberColumn("TS%", format="%.1f"),
                },
            )
            st.download_button(
                "Download this game log",
                data=log.to_csv(index=False).encode("utf-8"),
                file_name="nba_rest_game_log.csv",
                mime="text/csv",
                key="nba_download_log",
            )

    st.caption(
        "Regular season, from the team game log cached through nba_api. "
        "Season-level player rates and shot zones are on disk too, and they are not split by rest, so they stay off this page. "
        "Run cache_nba_data.py --refresh once the current season has games."
    )
