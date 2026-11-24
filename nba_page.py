"""Basketball page: a story about back-to-backs, the clubs, and a few players.

The counts come from the team and player game logs in data/nba.
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
# The cases the player section names, worst drop first.
CAST = [
    "Luke Kennard",
    "Max Christie",
    "Duncan Robinson",
    "James Harden",
    "Jayson Tatum",
    "Harrison Barnes",
]
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


def claim(text: str) -> None:
    st.markdown(f'<p class="reading">{html.escape(text)}</p>', unsafe_allow_html=True)


def walk(text: str) -> None:
    st.markdown(f'<p class="chapter">{html.escape(text)}</p>', unsafe_allow_html=True)


def section(label: str) -> None:
    st.markdown(f'<p class="section-label">{html.escape(label)}</p>', unsafe_allow_html=True)


def chart_heading(title: str) -> None:
    st.markdown(f'<p class="chart-title">{html.escape(title)}</p>', unsafe_allow_html=True)


def row_for(table: pandas.DataFrame, label: str) -> pandas.Series:
    return table.loc[table["label"].eq(label)].iloc[0]


def find_player(cases: pandas.DataFrame, name: str) -> pandas.Series | None:
    if cases.empty:
        return None
    hit = cases.loc[cases["name"].eq(name)]
    if hit.empty:
        return None
    return hit.iloc[0]


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


def position_figure(table: pandas.DataFrame) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(go.Bar(x=table["label"], y=table["b2b_ts"], name="Back-to-back", marker_color=RED))
    figure.add_trace(go.Bar(x=table["label"], y=table["rest_ts"], name="One day off", marker_color=GOLD))
    values = pandas.concat([table["b2b_ts"], table["rest_ts"]])
    figure.update_layout(**chart_layout(420), xaxis=dict(tickfont=dict(color=INK), title=None), yaxis=axis())
    figure.update_yaxes(tickformat=".1%", range=[float(values.min()) - 0.012, float(values.max()) + 0.016])
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


def club_gap_figure(gaps: pandas.DataFrame, league_gap: float) -> go.Figure:
    frame = gaps.sort_values(["ts_gap", "team"], ascending=[False, True])
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


def cast_figure(cases: pandas.DataFrame) -> go.Figure:
    ordered = cases.sort_values(["ts_gap", "name"], ascending=[False, True])
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


def club_by_name(gaps: pandas.DataFrame, name: str) -> pandas.Series:
    return gaps.loc[gaps["name"].eq(name)].iloc[0]


def player_story(cases: pandas.DataFrame, age: pandas.DataFrame, roles: pandas.DataFrame) -> None:
    kennard = find_player(cases, "Luke Kennard")
    christie = find_player(cases, "Max Christie")
    robinson = find_player(cases, "Duncan Robinson")
    harden = find_player(cases, "James Harden")
    tatum = find_player(cases, "Jayson Tatum")
    barnes = find_player(cases, "Harrison Barnes")
    if kennard is not None and robinson is not None:
        walk(
            f"Duncan Robinson, with most of these games in Miami, shot {rate(robinson['b2b_ts'])} on "
            f"{count(int(robinson['b2b_n']))} second nights and {rate(robinson['rest_ts'])} on "
            f"{count(int(robinson['rest_n']))} games with a day off. The night dipped from "
            f"{points(robinson['rest_min'])} minutes to {points(robinson['b2b_min'])}. "
            f"Kennard's minutes stayed level. Two shooters, and the second night took the shot that pays them."
        )
    if christie is not None:
        walk(
            f"Max Christie was {christie['age']:.0f} through the middle of this window, with the Lakers and with Dallas. "
            f"He shot {rate(christie['b2b_ts'])} on {count(int(christie['b2b_n']))} second nights and "
            f"{rate(christie['rest_ts'])} on {count(int(christie['rest_n']))} games with a day off, and the second night "
            f"was shorter, {points(christie['b2b_min'])} minutes against {points(christie['rest_min'])}. "
            f"The Lakers, as a team, shot better on back-to-backs. Christie shot worse on the nights he played yesterday."
        )
    if harden is not None:
        walk(
            f"James Harden played {points(harden['b2b_min'])} minutes on a second night and "
            f"{points(harden['rest_min'])} with a day off. He shot {rate(harden['b2b_ts'])} across "
            f"{count(int(harden['b2b_n']))} second nights and {rate(harden['rest_ts'])} across "
            f"{count(int(harden['rest_n']))} games with a day off. The plus-minus on those nights went from "
            f"{signed_points(harden['rest_pm'])} to {signed_points(harden['b2b_pm'])}. "
            f"He took more of these games with the Clippers than with any other club."
        )
    heavy = roles.loc[roles["label"].eq("33 or more")] if not roles.empty else roles
    young = age.loc[age["label"].eq("23 and under")] if not age.empty else age
    older = age.loc[age["label"].eq("32 and over")] if not age.empty else age
    if not heavy.empty and not young.empty and not older.empty:
        heavy_row = heavy.iloc[0]
        young_row = young.iloc[0]
        older_row = older.iloc[0]
        walk(
            f"Players who averaged 33 or more minutes played {points(heavy_row['b2b_min'])} on a second night "
            f"and {points(heavy_row['rest_min'])} with a day off. "
            f"Players 23 and under shot {rate(young_row['b2b_ts'])} the day after they played and "
            f"{rate(young_row['rest_ts'])} with a day off. "
            f"Players 32 and over shot {rate(older_row['b2b_ts'])} and {rate(older_row['rest_ts'])}."
        )
    if tatum is not None and barnes is not None:
        walk(
            f"Jayson Tatum went the other way, and every one of these games was in Boston. He shot {rate(tatum['b2b_ts'])} on "
            f"{count(int(tatum['b2b_n']))} second nights and {rate(tatum['rest_ts'])} with a day off, "
            f"at {points(tatum['b2b_min'])} minutes either way. "
            f"Harrison Barnes was {barnes['age']:.0f} through the middle of the window, split between Sacramento and San Antonio. "
            f"He shot {rate(barnes['b2b_ts'])} on {count(int(barnes['b2b_n']))} second nights and "
            f"{rate(barnes['rest_ts'])} on {count(int(barnes['rest_n']))} games with a day off, "
            f"{points(barnes['b2b_min'])} minutes and {points(barnes['rest_min'])}."
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
    home_rest = cross.loc[cross["label"].eq("Home, rested")].iloc[0]
    road_b2b = cross.loc[cross["label"].eq("Road, back-to-back")].iloc[0]
    road_rest = cross.loc[cross["label"].eq("Road, rested")].iloc[0]
    by_win = gaps.sort_values(["win_gap", "team"])
    low_win = by_win.iloc[0]
    next_win = by_win.iloc[1]
    high_win = by_win.iloc[-1]
    openers = int(schedule["REST_LABEL"].isna().sum())
    widest = by_season.sort_values("ts_gap").iloc[0]
    narrow = by_season.sort_values("ts_gap").iloc[-1]
    other = nba.pooled(played.loc[~played["IS_B2B"]])
    on_b2b = nba.pooled(played.loc[played["IS_B2B"]])
    league_gap = float(on_b2b["ts"] - other["ts"])
    ends = pandas.concat([gaps.head(5), gaps.tail(5)], ignore_index=True)
    nets = club_by_name(gaps, "Brooklyn Nets")
    clippers = club_by_name(gaps, "LA Clippers")
    pistons = club_by_name(gaps, "Detroit Pistons")
    lakers = club_by_name(gaps, "Los Angeles Lakers")

    players = cached_players(nba.cache_mtime())
    if not players.empty and not held.empty:
        players = players.loc[~players["SEASON"].isin(set(held.index))].copy()
    cases = nba.player_cases(players) if not players.empty else pandas.DataFrame()
    cast = cases.loc[cases["name"].isin(CAST)].copy() if not cases.empty else cases
    age = nba.own_rest_contrast(players, "AGE_BAND", nba.AGE_ORDER) if not players.empty else pandas.DataFrame()
    roles = nba.own_rest_contrast(players, "ROLE", nba.ROLE_ORDER) if not players.empty else pandas.DataFrame()
    position = nba.rest_contrast(players, "POSITION_GROUP", nba.POSITION_ORDER) if not players.empty else pandas.DataFrame()
    kennard = find_player(cases, "Luke Kennard")

    held_line = ""
    if not held.empty:
        held_line = " " + " ".join(
            f"{season} has {count(int(games))} team-games on disk and stays out until the season is longer."
            for season, games in held.items()
        )

    st.markdown(
        f"""
        <header class="mast">
          <p class="kicker">NBA, {html.escape(seasons[0])} to {html.escape(seasons[-1])}</p>
          <h1>Do back-to-backs cost efficiency?</h1>
        </header>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="deck">A back-to-back is a game on the day after that club\'s previous game. '
        "True shooting pools every field goal and free throw, points divided by twice the attempts plus 0.44 free throws. "
        "The question is whether the second night takes that shot away, for the club and for the players who had to play it.</p>",
        unsafe_allow_html=True,
    )
    walk(
        f"The file runs from {seasons[0]} through {seasons[-1]}, {count(int(played.shape[0]))} team-games with a previous game to measure. "
        f"Regular season only. The first game of each season stays out, {count(openers)} team-games. "
        f"Home and road come from the matchup.{held_line}"
    )

    section("The second night")
    claim(
        f"Teams won {rate(second['win_pct'])} of back-to-backs and {rate(one['win_pct'])} with one day off. "
        f"With three or more days off they won {rate(long_rest['win_pct'])}. "
        f"True shooting moved from {rate(second['ts'])} to {rate(long_rest['ts'])} across that same span, "
        f"on {count(int(second['games']))} back-to-backs and {count(int(long_rest['games']))} long layoffs."
    )
    chart_heading("True shooting by nights off")
    st.plotly_chart(nights_figure(rest), width="stretch", config=CHART)
    st.caption("The axis starts in the mid-50s so the half-point is visible. Every group is still in the high 50s.")
    chart_heading("Wins by nights off")
    st.plotly_chart(win_figure(rest), width="stretch", config=CHART)
    st.caption("Same games as the chart above. The axis is tightened. A win rate of 50% is a .500 club.")
    walk(
        f"Points fell from {points(one['pts_per_game'])} to {points(second['pts_per_game'])}. "
        f"The opponent scored {points(second['opp_pts'])} instead of {points(one['opp_pts'])}, and the margin moved from "
        f"{signed_points(one['margin'])} to {signed_points(second['margin'])}. "
        f"Possessions were {points(second['pace'])} a game on the second night and {points(one['pace'])} with one day off. "
        f"Offensive rating moved from {points(one['ortg'])} to {points(second['ortg'])} per 100 possessions."
    )
    walk(
        f"Two days off shot {rate(two['ts'])}, in line with one day off, and won {rate(two['win_pct'])}. "
        f"Three or more days off shot {rate(long_rest['ts'])} across {count(int(long_rest['games']))} games. "
        f"That bucket holds the All-Star break and the other long layoffs."
    )
    walk(
        f"Three-point attempts were {rate(second['three_rate'])} of field goals on a back-to-back and "
        f"{rate(one['three_rate'])} with one day off. Turnovers were {rate(second['tov_rate'])} of possessions on a back-to-back "
        f"and {rate(one['tov_rate'])} with one day off. Free throws were {rate(second['ft_rate'])} of field-goal attempts and "
        f"{rate(one['ft_rate'])}."
    )
    walk(
        f"{count(int(tired_only['games']))} of the {count(int(second['games']))} back-to-backs faced a rested opponent. "
        f"In those games the tired club shot {rate(tired_only['ts'])}, was outscored by {abs(tired_only['margin']):.1f} points, "
        f"and won {rate(tired_only['win_pct'])}. "
        f"When both clubs were on a back-to-back, true shooting was {rate(both_tired['ts'])} and the margin was even, "
        f"on {count(int(both_tired['games']))} games. "
        f"When neither club was, true shooting was {rate(neither['ts'])} and the margin was even, "
        f"on {count(int(neither['games']))} games. "
        f"The second night looks ordinary when the opponent is on one too."
    )
    walk(
        f"The gap is in each of the {len(by_season)} seasons, from {narrow['ts_gap'] * 100:.1f} points in {narrow['season']} "
        f"to {widest['ts_gap'] * 100:.1f} points in {widest['season']}."
    )
    walk(
        f"Home teams shot {rate(home['ts'])} and won {rate(home['win_pct'])}. "
        f"A home back-to-back shot {rate(home_b2b['ts'])}. A rested team on the road shot {rate(road_rest['ts'])}. "
        f"The building is about a point of true shooting. The second night, set against one day off, is half a point."
    )
    walk(
        f"A road back-to-back won {rate(road_b2b['win_pct'])} and was outscored by {abs(road_b2b['margin']):.1f} points, "
        f"on {count(int(road_b2b['games']))} games. A rested home team won {rate(home_rest['win_pct'])} and finished "
        f"{signed_points(home_rest['margin'])} points, on {count(int(home_rest['games']))} games."
    )

    section("The clubs")
    claim(
        f"For the Brooklyn Nets, true shooting falls {abs(nets['ts_gap']) * 100:.1f} points on back-to-backs, "
        f"and the margin falls {abs(nets['margin_gap']):.1f} points, across {count(int(nets['b2b_games']))} second nights."
    )
    chart_heading("Where the drop is widest")
    st.plotly_chart(club_gap_figure(ends, league_gap), width="stretch", config=CHART)
    st.caption(
        f"Back-to-back true shooting minus the other games. The gold line is the league gap, {league_gap * 100:.1f} points. "
        f"Red is a larger drop than the league. These are the five widest drops and the five widest gains."
    )
    walk(
        f"The Nets shot {rate(nets['b2b_ts'])} on those nights and {rate(nets['rest_ts'])} in their other games, "
        f"and they were outscored by {abs(nets['b2b_margin']):.1f} points, against {abs(nets['rest_margin']):.1f} in the other games. "
        f"The win rate falls from {rate(nets['rest_win'])} to {rate(nets['b2b_win'])}."
    )
    walk(
        f"The LA Clippers are the next club down the chart, {count(int(clippers['b2b_games']))} back-to-backs. "
        f"They shot {rate(clippers['b2b_ts'])} on the second night and {rate(clippers['rest_ts'])} otherwise, "
        f"and the margin moves from {signed_points(clippers['rest_margin'])} to {signed_points(clippers['b2b_margin'])}. "
        f"The win rate moves from {rate(clippers['rest_win'])} to {rate(clippers['b2b_win'])}."
    )
    walk(
        f"Win rate and true shooting land on different clubs. "
        f"The {low_win['name']} won {rate(low_win['b2b_win'])} of {count(int(low_win['b2b_games']))} back-to-backs "
        f"and {rate(low_win['rest_win'])} of their other games. True shooting was {abs(low_win['ts_gap']) * 100:.1f} points lower, "
        f"and the margin moved {signed_points(low_win['margin_gap'])}. "
        f"The {next_win['name']} won {rate(next_win['b2b_win'])} of their back-to-backs and {rate(next_win['rest_win'])} otherwise. "
        f"The {high_win['name']} won {rate(high_win['b2b_win'])} on the second night and {rate(high_win['rest_win'])} otherwise, "
        f"with true shooting {abs(high_win['ts_gap']) * 100:.1f} points {'lower' if high_win['ts_gap'] < 0 else 'higher'}."
    )
    walk(
        f"The Detroit Pistons are at the other end. The second night shot {rate(pistons['b2b_ts'])} against "
        f"{rate(pistons['rest_ts'])} in their other games. The margin improved by {pistons['margin_gap']:.1f} points "
        f"and the win rate moved {pistons['win_gap'] * 100:+.1f} points. They were still outscored, by "
        f"{abs(pistons['b2b_margin']):.1f} points on the second night and {abs(pistons['rest_margin']):.1f} on the others."
    )
    walk(
        f"The Los Angeles Lakers make more of their shots on the second night, {rate(lakers['b2b_ts'])} against "
        f"{rate(lakers['rest_ts'])}. The margin moves {signed_points(lakers['margin_gap'])} points and the win rate "
        f"moves {lakers['win_gap'] * 100:+.1f} points."
    )

    section("Position")
    if position.empty:
        walk("Roster positions are not on disk yet. Run cache_nba_data.py and this chart fills in.")
    else:
        named = {"Guard": "Guards", "Forward": "Forwards", "Center": "Centers"}
        levels = ". ".join(
            f"{named.get(row.label, row.label)} shot {rate(row.b2b_ts)} on a back-to-back and {rate(row.rest_ts)} with one day off"
            for row in position.itertuples()
        )
        widest_position = abs(float(position["ts_gap"].min())) * 100
        if widest_position < 1:
            lead = "The second night moves true shooting by less than a point at every position."
        else:
            lead = f"The widest position gap is {widest_position:.1f} points of true shooting."
        claim(f"{lead} {levels}.")
        chart_heading("True shooting by position")
        st.plotly_chart(position_figure(position), width="stretch", config=CHART)
        st.caption(
            "A team back-to-back against one day off. The axis is tightened. "
            "The space inside a pair is the second night. The space between groups is the position."
        )
        centers = position.loc[position["label"].eq("Center")]
        if not centers.empty:
            center = centers.iloc[0]
            walk(
                f"Centers at home shot {rate(center['home_b2b_ts'])} on a back-to-back and {rate(center['home_rest_ts'])} "
                f"with one day off. On the road they shot {rate(center['away_b2b_ts'])} and {rate(center['away_rest_ts'])}. "
                f"The minutes stayed level: centers played {points(center['b2b_min'])} on the second night and "
                f"{points(center['rest_min'])} with one day off."
            )

    section("The players")
    if kennard is None:
        walk("Player game logs are not on disk yet. Run cache_nba_data.py and the names fill in.")
    else:
        claim(
            f"Luke Kennard shot {rate(kennard['b2b_ts'])} on {count(int(kennard['b2b_n']))} nights after he had played "
            f"the day before, and {rate(kennard['rest_ts'])} on {count(int(kennard['rest_n']))} games with a day off. "
            f"The minutes were {points(kennard['b2b_min'])} and {points(kennard['rest_min'])}."
        )
        if not cast.empty:
            chart_heading("Six players, on their own second night")
            st.plotly_chart(cast_figure(cast), width="stretch", config=CHART)
            st.caption(
                "A second night means he played yesterday. One day off means a day between his games. "
                "The axis is tightened. A player needs 30 second nights and 50 games with one day off to qualify. "
                "These six are the cases in the text."
            )
        player_story(cases, age, roles)

    section("One club")
    walk("The charts above use the ends of the league. The picker sets one club beside the league line.")
    club_names = gaps.sort_values("name")
    choice = st.selectbox(
        "Club",
        club_names["name"].tolist(),
        index=int(club_names["name"].tolist().index(nets["name"])) if nets["name"] in set(club_names["name"]) else 0,
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
                f"They shot {rate(club_second.iloc[0]['ts'])} and {rate(club_one.iloc[0]['ts'])}."
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
                f"and {signed_points(club_away['margin'])} points over {count(int(club_away['games']))} games."
            )

    section("The count")
    rest_table(rest)
    st.caption(
        "Regular season, from the team game log. True shooting, win rate, and the free-throw and turnover shares "
        "in the text are points on a 100-point scale. Possessions are field-goal attempts, 0.44 free throws, "
        "minus offensive rebounds, plus turnovers."
    )
