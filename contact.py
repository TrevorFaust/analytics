"""Expected contact from exit velocity and launch angle.

Rough draft of the baseball paper. Train on the earlier cached season, score the
later one, and set the predictions next to Baseball Savant's expected value on
the same balls. Walks and strikeouts stay out.
"""

from __future__ import annotations

import base64
import html
import urllib.request
from collections import deque
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy
import pandas
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from sklearn.ensemble import HistGradientBoostingRegressor

RAW = Path(__file__).resolve().parent / "data" / "baseball" / "statcast_bip.parquet"
HEADS = RAW.parent / "heads"
TEAMS = RAW.parent / "batter_team.parquet"
BIRTHS = RAW.parent / "birth_dates.parquet"
MIN_BALLS = 200
NEXT_BALLS = 100
PARK_PRIOR = 2000
FEATURES = ("launch_speed", "launch_angle")
CLUBS = {
    "ARI": "Arizona",
    "AZ": "Arizona",
    "ATL": "Atlanta",
    "BAL": "Baltimore",
    "BOS": "Boston",
    "CHC": "Chicago Cubs",
    "CWS": "Chicago White Sox",
    "CIN": "Cincinnati",
    "CLE": "Cleveland",
    "COL": "Colorado",
    "DET": "Detroit",
    "HOU": "Houston",
    "KC": "Kansas City",
    "LAA": "Los Angeles Angels",
    "LAD": "Los Angeles Dodgers",
    "MIA": "Miami",
    "MIL": "Milwaukee",
    "MIN": "Minnesota",
    "NYM": "New York Mets",
    "NYY": "New York Yankees",
    "ATH": "Athletics",
    "OAK": "Athletics",
    "PHI": "Philadelphia",
    "PIT": "Pittsburgh",
    "SD": "San Diego",
    "SF": "San Francisco",
    "SEA": "Seattle",
    "STL": "St. Louis",
    "TB": "Tampa Bay",
    "TEX": "Texas",
    "TOR": "Toronto",
    "WSH": "Washington",
}
PARKS = {
    "AZ": "Chase Field",
    "ATL": "Truist Park",
    "BAL": "Camden Yards",
    "BOS": "Fenway Park",
    "CHC": "Wrigley Field",
    "CIN": "Great American Ball Park",
    "CLE": "Progressive Field",
    "COL": "Coors Field",
    "CWS": "Rate Field",
    "DET": "Comerica Park",
    "HOU": "Daikin Park",
    "KC": "Kauffman Stadium",
    "LAA": "Angel Stadium",
    "LAD": "Dodger Stadium",
    "MIA": "loanDepot park",
    "MIL": "American Family Field",
    "MIN": "Target Field",
    "NYM": "Citi Field",
    "NYY": "Yankee Stadium",
    "ATH": "Sutter Health Park",
    "OAK": "Oakland Coliseum",
    "PHI": "Citizens Bank Park",
    "PIT": "PNC Park",
    "SD": "Petco Park",
    "SEA": "T-Mobile Park",
    "SF": "Oracle Park",
    "STL": "Busch Stadium",
    "TB": "Tropicana Field",
    "TEX": "Globe Life Field",
    "TOR": "Rogers Centre",
    "WSH": "Nationals Park",
}
RESULT_LABEL = {
    "single": "Single",
    "double": "Double",
    "triple": "Triple",
    "home_run": "Home run",
    "field_out": "Out",
    "force_out": "Out",
    "grounded_into_double_play": "Double play",
    "double_play": "Double play",
    "sac_fly": "Sacrifice fly",
    "sac_fly_double_play": "Double play",
    "field_error": "Reached on an error",
    "sac_bunt": "Sacrifice bunt",
    "fielders_choice": "Fielder's choice",
    "fielders_choice_out": "Out",
    "triple_play": "Triple play",
    "catcher_interf": "Catcher's interference",
}
BATTED_LABEL = {
    "ground_ball": "ground ball",
    "line_drive": "line drive",
    "fly_ball": "fly ball",
    "popup": "popup",
    "home_run": "home run",
}


@dataclass
class ContactStudy:
    train_label: str
    test_label: str
    train_n: int
    test_n: int
    mae_model: float
    mae_savant: float
    mae_bins: float
    hitter_corr_model: float
    hitter_corr_savant: float
    hot_speed: float
    hot_angle: float
    hot_value: float
    popup_value: float
    hr_speed: float
    hr_angle: float
    hr_weight: float
    ld_angle: float
    ld_woba: float
    gb_angle: float
    gb_woba: float
    pu_angle: float
    pu_woba: float
    surface: pandas.DataFrame
    calibration: pandas.DataFrame
    hitters: pandas.DataFrame
    spray: pandas.DataFrame
    zone: pandas.DataFrame | None
    zone_top: float
    zone_bot: float
    league_woba: float
    league_hits: float
    miss_bands: pandas.DataFrame
    miss_facts: pandas.Series
    hard_prior: float
    hard_now: float


def _batter_names(batters: pandas.Series) -> pandas.Series:
    """Statcast's player_name on this pull is the pitcher. Names come from the Chadwick register."""
    from pybaseball import playerid_reverse_lookup

    ids = pandas.to_numeric(batters, errors="coerce").astype("int64")
    found = playerid_reverse_lookup(ids.drop_duplicates().tolist(), key_type="mlbam")
    found = found.drop_duplicates("key_mlbam").set_index("key_mlbam")
    first = ids.map(found["name_first"])
    last = ids.map(found["name_last"])
    raw = (first.fillna("") + " " + last.fillna("")).str.strip()

    def _title_name(text: str) -> str:
        parts = []
        for part in text.split(" "):
            lower = part.lower()
            if lower.startswith("mc") and len(part) > 2:
                parts.append("Mc" + part[2:].capitalize())
            else:
                parts.append(part.capitalize())
        return " ".join(parts)

    name = raw.map(_title_name)
    missing = name.eq("")
    name.loc[missing] = ids.loc[missing].astype(str)
    return name


def _split(balls: pandas.DataFrame) -> tuple[pandas.DataFrame, pandas.DataFrame, str, str]:
    seasons = sorted(int(year) for year in balls["season"].unique())
    if len(seasons) >= 2:
        test_year = seasons[-1]
        train = balls.loc[balls["season"].lt(test_year)].copy()
        test = balls.loc[balls["season"].eq(test_year)].copy()
        train_years = sorted(int(year) for year in train["season"].unique())
        train_label = str(train_years[0]) if len(train_years) == 1 else f"{train_years[0]} through {train_years[-1]}"
        return train, test, train_label, str(test_year)
    year = seasons[0]
    cut = pandas.Timestamp(year=year, month=7, day=1)
    train = balls.loc[balls["game_date"].lt(cut)].copy()
    test = balls.loc[balls["game_date"].ge(cut)].copy()
    if train.empty or test.empty:
        raise ValueError("Need balls on both sides of the split. Let the Statcast download finish another month.")
    return train, test, f"{year} before July", f"{year} from July"


def _mae(predicted: numpy.ndarray, actual: numpy.ndarray) -> float:
    return float(numpy.mean(numpy.abs(predicted - actual)))


def _miss_report(
    guess: numpy.ndarray, actual: numpy.ndarray, events: pandas.Series
) -> tuple[pandas.DataFrame, pandas.Series]:
    """How large each 2025 miss was, and what a call near 1.3 actually became."""
    miss = numpy.abs(guess - actual)
    bands = pandas.cut(
        miss,
        [0, 0.2, 0.5, 1.0, numpy.inf],
        labels=["Under 0.2", "0.2 to 0.5", "0.5 to 1", "Over 1"],
        right=False,
    )
    frame = pandas.DataFrame({"band": bands, "miss": miss})
    total = float(frame["miss"].sum())
    grouped = frame.groupby("band", observed=False).agg(balls=("miss", "size"), average=("miss", "mean"))
    grouped["share"] = grouped["balls"] / grouped["balls"].sum()
    grouped["error_share"] = grouped["average"] * grouped["balls"] / total
    grouped = grouped.reset_index()
    hard = (guess >= 1.2) & (guess < 1.4)
    caught = (guess >= 1.0) & (actual == 0)
    event = events.astype(str).to_numpy()
    def _rate(name: str) -> float:
        return float((event[hard] == name).mean()) if hard.any() else float("nan")

    facts = pandas.Series(
        {
            "hard_n": int(hard.sum()),
            "hard_guess": float(guess[hard].mean()) if hard.any() else float("nan"),
            "hard_stamp": float(actual[hard].mean()) if hard.any() else float("nan"),
            "hard_hr": _rate("home_run"),
            "hard_out": float((actual[hard] == 0).mean()) if hard.any() else float("nan"),
            "hard_double": _rate("double"),
            "caught_n": int(caught.sum()),
            "caught_guess": float(guess[caught].mean()) if caught.any() else float("nan"),
        }
    )
    return grouped, facts


def _calibration(predicted: numpy.ndarray, actual: numpy.ndarray, source: str) -> pandas.DataFrame:
    """Group calls in steps of 0.1. Hold anything under an out at 0 and anything over a home run at 2."""
    frame = pandas.DataFrame({"predicted": numpy.clip(predicted, 0, 2), "actual": actual})
    edges = numpy.round(numpy.arange(0, 2.0 + 1e-9, 0.1), 10)
    frame["bin"] = pandas.cut(frame["predicted"], edges, include_lowest=True)
    grouped = (
        frame.groupby("bin", observed=True)
        .agg(predicted=("predicted", "mean"), actual=("actual", "mean"), balls=("actual", "size"))
        .reset_index()
    )
    grouped = grouped.loc[grouped["balls"].ge(50)].copy()
    grouped["band"] = grouped["bin"].map(lambda interval: round(float(interval.left), 1)).astype(float)
    grouped = grouped.drop(columns=["bin"]).reset_index(drop=True)
    grouped["source"] = source
    return grouped


def _bin_predictions(train: pandas.DataFrame, test: pandas.DataFrame) -> numpy.ndarray:
    train = train.copy()
    test = test.copy()
    train["speed_bin"] = numpy.floor(train["launch_speed"] / 5.0) * 5.0
    train["angle_bin"] = numpy.floor(train["launch_angle"] / 5.0) * 5.0
    lookup = train.groupby(["speed_bin", "angle_bin"])["woba_value"].mean().rename("bins")
    fallback = float(train["woba_value"].mean())
    test["speed_bin"] = numpy.floor(test["launch_speed"] / 5.0) * 5.0
    test["angle_bin"] = numpy.floor(test["launch_angle"] / 5.0) * 5.0
    merged = test.merge(lookup, on=["speed_bin", "angle_bin"], how="left")
    return merged["bins"].fillna(fallback).to_numpy()


def fit_study(balls: pandas.DataFrame) -> ContactStudy:
    needed = {"launch_speed", "launch_angle", "woba_value", "estimated_woba_using_speedangle", "season", "game_date"}
    missing = needed.difference(balls.columns)
    if missing:
        raise ValueError(f"Cached balls are missing {', '.join(sorted(missing))}.")
    balls = balls.dropna(subset=["launch_speed", "launch_angle", "woba_value", "estimated_woba_using_speedangle"])
    train, test, train_label, test_label = _split(balls)

    model = HistGradientBoostingRegressor(
        learning_rate=0.08,
        max_iter=300,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=0.1,
        random_state=7,
    )
    model.fit(train.loc[:, FEATURES].to_numpy(), train["woba_value"].to_numpy())
    model_pred = model.predict(test.loc[:, FEATURES].to_numpy())
    bin_pred = _bin_predictions(train, test)
    savant = test["estimated_woba_using_speedangle"].to_numpy()
    actual = test["woba_value"].to_numpy()
    miss_bands, miss_facts = _miss_report(model_pred, actual, test["events"])
    barrel = train["launch_speed"].ge(100) & train["launch_angle"].between(20, 35)
    barrel_now = test["launch_speed"].ge(100) & test["launch_angle"].between(20, 35)
    hard_prior = float(train.loc[barrel, "woba_value"].mean()) if bool(barrel.any()) else float("nan")
    hard_now = float(test.loc[barrel_now, "woba_value"].mean()) if bool(barrel_now.any()) else float("nan")

    speeds = numpy.arange(60, 116, 2)
    angles = numpy.arange(-20, 56, 2)
    mesh = pandas.DataFrame(
        [(float(speed), float(angle)) for angle in angles for speed in speeds],
        columns=["launch_speed", "launch_angle"],
    )
    mesh["expected"] = model.predict(mesh.loc[:, FEATURES].to_numpy())
    barrels = mesh.loc[mesh["launch_speed"].between(95, 115) & mesh["launch_angle"].between(15, 35)]
    hot = barrels.loc[barrels["expected"].idxmax()]
    popups = mesh.loc[mesh["launch_angle"].ge(45) & mesh["launch_speed"].between(70, 100)]
    popup_value = float(popups["expected"].mean()) if not popups.empty else float("nan")

    scored = test.loc[:, ["batter", "woba_value", "estimated_woba_using_speedangle"]].copy()
    scored["model"] = model_pred
    events = test["events"].astype(str) if "events" in test.columns else pandas.Series("", index=test.index)
    scored["single"] = events.eq("single").to_numpy()
    scored["double"] = events.eq("double").to_numpy()
    scored["triple"] = events.eq("triple").to_numpy()
    scored["homer"] = events.eq("home_run").to_numpy()
    scored["is_hit"] = scored[["single", "double", "triple", "homer"]].any(axis=1)
    hitters = scored.groupby("batter", as_index=False).agg(
        bip=("woba_value", "size"),
        actual=("woba_value", "mean"),
        model=("model", "mean"),
        savant=("estimated_woba_using_speedangle", "mean"),
        singles=("single", "sum"),
        doubles=("double", "sum"),
        triples=("triple", "sum"),
        homers=("homer", "sum"),
    )
    hitters = hitters.loc[hitters["bip"].ge(MIN_BALLS)].copy()
    hitters["name"] = _batter_names(hitters["batter"])
    hitters["gap"] = hitters["model"] - hitters["savant"]
    hitters["hit_rate"] = (hitters["singles"] + hitters["doubles"] + hitters["triples"] + hitters["homers"]) / hitters["bip"]
    hitters = hitters.sort_values("gap", key=lambda column: column.abs(), ascending=False)
    league_woba = float(actual.mean()) if len(actual) else float("nan")
    league_hits = float(scored["is_hit"].mean()) if len(scored) else float("nan")
    facts = _contact_facts(balls)
    spray = _spray_sample(test)
    zone, zone_top, zone_bot = _zone_grid(test)

    calibration = pandas.concat(
        [
            _calibration(model_pred, actual, "This model"),
            _calibration(savant, actual, "MLB expected"),
        ],
        ignore_index=True,
    )
    shared = set(calibration.loc[calibration["source"].eq("This model"), "band"])
    shared &= set(calibration.loc[calibration["source"].eq("MLB expected"), "band"])
    calibration = calibration.loc[calibration["band"].isin(shared)].reset_index(drop=True)
    return ContactStudy(
        train_label=train_label,
        test_label=test_label,
        train_n=int(len(train)),
        test_n=int(len(test)),
        mae_model=_mae(model_pred, actual),
        mae_savant=_mae(savant, actual),
        mae_bins=_mae(bin_pred, actual),
        hitter_corr_model=float(hitters["actual"].corr(hitters["model"])) if len(hitters) > 2 else float("nan"),
        hitter_corr_savant=float(hitters["actual"].corr(hitters["savant"])) if len(hitters) > 2 else float("nan"),
        hot_speed=float(hot["launch_speed"]),
        hot_angle=float(hot["launch_angle"]),
        hot_value=float(hot["expected"]),
        popup_value=popup_value,
        hr_speed=facts["hr_speed"],
        hr_angle=facts["hr_angle"],
        hr_weight=facts["hr_weight"],
        ld_angle=facts["ld_angle"],
        ld_woba=facts["ld_woba"],
        gb_angle=facts["gb_angle"],
        gb_woba=facts["gb_woba"],
        pu_angle=facts["pu_angle"],
        pu_woba=facts["pu_woba"],
        surface=mesh,
        calibration=calibration,
        hitters=hitters,
        spray=spray,
        zone=zone,
        zone_top=zone_top,
        zone_bot=zone_bot,
        league_woba=league_woba,
        league_hits=league_hits,
        miss_bands=miss_bands,
        miss_facts=miss_facts,
        hard_prior=hard_prior,
        hard_now=hard_now,
    )


def _contact_facts(balls: pandas.DataFrame) -> dict[str, float]:
    def pair(mask: pandas.Series) -> tuple[float, float, float]:
        group = balls.loc[mask]
        if group.empty:
            return float("nan"), float("nan"), float("nan")
        return (
            float(group["launch_speed"].median()),
            float(group["launch_angle"].median()),
            float(group["woba_value"].mean()),
        )

    hr_speed, hr_angle, hr_weight = pair(balls["events"].eq("home_run"))
    _, ld_angle, ld_woba = pair(balls["bb_type"].eq("line_drive"))
    _, gb_angle, gb_woba = pair(balls["bb_type"].eq("ground_ball"))
    _, pu_angle, pu_woba = pair(balls["bb_type"].eq("popup"))
    return {
        "hr_speed": hr_speed,
        "hr_angle": hr_angle,
        "hr_weight": hr_weight,
        "ld_angle": ld_angle,
        "ld_woba": ld_woba,
        "gb_angle": gb_angle,
        "gb_woba": gb_woba,
        "pu_angle": pu_angle,
        "pu_woba": pu_woba,
    }


def _spray_label(kind: str, event: str) -> str:
    result = RESULT_LABEL.get(event, "In play")
    if result == "Home run":
        return "Home run"
    batted = BATTED_LABEL.get(kind, "batted ball")
    return f"{result}, {batted}"


_SPRAY_HITS = {"Single": "Singles", "Double": "Doubles", "Triple": "Triples", "Home run": "Home runs"}
_SPRAY_OUTS = {"Out", "Double play", "Sacrifice fly", "Sacrifice bunt", "Triple play"}


def _spray_outcome(label: str) -> str:
    head = str(label).split(",")[0].strip()
    if head in _SPRAY_HITS:
        return head
    if head in _SPRAY_OUTS:
        return "Out"
    return "Other"


def _spray_sample(test: pandas.DataFrame) -> pandas.DataFrame:
    needed = {"hc_x", "hc_y", "bb_type", "events"}
    empty = pandas.DataFrame(columns=["x", "y", "kind", "label", "park"])
    if not needed.issubset(test.columns):
        return empty
    frame = test.dropna(subset=["hc_x", "hc_y"]).copy()
    if frame.empty:
        return empty
    if len(frame) > 8000:
        frame = frame.sample(8000, random_state=7)
    kind = frame["bb_type"].fillna("other").astype(str)
    kind = kind.where(~frame["events"].eq("home_run"), "home_run")
    events = frame["events"].astype(str)
    labels = [_spray_label(one_kind, one_event) for one_kind, one_event in zip(kind, events)]
    if "home_team" in frame.columns:
        park = frame["home_team"].map(PARKS).fillna("Ballpark")
    else:
        park = pandas.Series("Ballpark", index=frame.index)
    return pandas.DataFrame(
        {
            "x": (frame["hc_x"] - 125.42) * 2.495,
            "y": (198.27 - frame["hc_y"]) * 2.495,
            "kind": kind.to_numpy(),
            "label": labels,
            "park": park.to_numpy(),
        }
    )


def _zone_grid(test: pandas.DataFrame, *, hits_only: bool = True) -> tuple[pandas.DataFrame | None, float, float]:
    if not {"plate_x", "plate_z"}.issubset(test.columns):
        return None, 3.4, 1.6
    frame = test.dropna(subset=["plate_x", "plate_z", "woba_value"]).copy()
    frame = frame.loc[frame["plate_x"].between(-2.2, 2.2) & frame["plate_z"].between(0.5, 4.6)]
    if hits_only and "events" in frame.columns:
        frame = frame.loc[frame["events"].isin(("single", "double", "triple", "home_run"))]
    if frame.empty:
        return None, 3.4, 1.6
    frame["xbin"] = (numpy.round(frame["plate_x"] / 0.2) * 0.2).round(2)
    frame["zbin"] = (numpy.round(frame["plate_z"] / 0.2) * 0.2).round(2)
    grouped = (
        frame.groupby(["xbin", "zbin"], as_index=False)
        .agg(woba=("woba_value", "mean"), balls=("woba_value", "size"))
    )
    grouped = grouped.loc[grouped["balls"].ge(25)]
    top = float(frame["sz_top"].median()) if "sz_top" in frame.columns and frame["sz_top"].notna().any() else 3.4
    bot = float(frame["sz_bot"].median()) if "sz_bot" in frame.columns and frame["sz_bot"].notna().any() else 1.6
    return grouped, top, bot


@st.cache_data(show_spinner=False)
def _zone_boards(path: str, mtime_ns: int) -> tuple[pandas.DataFrame | None, pandas.DataFrame | None, float, float]:
    """Hits-only grid and the full balls-in-play grid for the scored season."""
    del mtime_ns
    columns = ["season", "game_date", "plate_x", "plate_z", "woba_value", "events", "sz_top", "sz_bot"]
    balls = pandas.read_parquet(path, columns=columns)
    _train, test, _train_label, _test_label = _split(balls)
    hits, top, bot = _zone_grid(test, hits_only=True)
    everything, _top, _bot = _zone_grid(test, hits_only=False)
    return hits, everything, top, bot


def chart_layout(height: int) -> dict:
    return dict(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Barlow, sans-serif", color="#f3efe4", size=14),
        margin=dict(l=48, r=16, t=12, b=40),
        hoverlabel=dict(bgcolor="#2a303a", bordercolor="#4c5566", font_size=13, font_family="Barlow, sans-serif"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(color="#f3efe4")),
        dragmode=False,
    )


def surface_figure(surface: pandas.DataFrame) -> go.Figure:
    grid = surface.pivot(index="launch_angle", columns="launch_speed", values="expected").sort_index()
    figure = go.Figure(
        go.Heatmap(
            x=grid.columns.astype(float),
            y=grid.index.astype(float),
            z=grid.to_numpy(),
            zmin=0,
            zmax=2,
            colorscale=[
                [0.0, "#2a303a"],
                [0.25, "#3d6f7a"],
                [0.5, "#e6b325"],
                [0.75, "#d08c60"],
                [1.0, "#d6453d"],
            ],
            colorbar=dict(title="Expected weight", tickfont=dict(color="#d9d3c5"), title_font=dict(color="#d9d3c5")),
            hovertemplate="Exit velocity %{x} mph<br>Launch angle %{y}°<br>Expected weight %{z:.3f}<extra></extra>",
        )
    )
    figure.update_layout(
        annotations=[
            dict(x=86, y=-4, text="Ground balls", showarrow=False, font=dict(color="#f3efe4", size=14), bgcolor="rgba(28,31,38,0.55)"),
            dict(x=78, y=16, text="Line drives", showarrow=False, font=dict(color="#1c1f26", size=14), bgcolor="rgba(230,179,37,0.85)"),
            dict(x=108, y=28, text="Home runs", showarrow=False, font=dict(color="#f3efe4", size=14), bgcolor="rgba(214,69,61,0.85)"),
            dict(x=86, y=50, text="Popups", showarrow=False, font=dict(color="#f3efe4", size=14), bgcolor="rgba(28,31,38,0.55)"),
        ]
    )
    layout = chart_layout(560)
    layout["margin"] = dict(l=56, r=90, t=16, b=48)
    figure.update_layout(**layout)
    figure.update_xaxes(title="Exit velocity, how hard (mph)", tickfont=dict(color="#d9d3c5"), gridcolor="#3a4150")
    figure.update_yaxes(title="Launch angle, how high (degrees)", tickfont=dict(color="#d9d3c5"), gridcolor="#3a4150")
    return figure


def calibration_figure(calibration: pandas.DataFrame) -> go.Figure:
    """The vertical axis is the miss. Zero means the results matched the guess."""
    colors = {"This model": "#e6b325", "MLB expected": "#8eb8c4"}
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=[-0.08, 2.12],
            y=[0, 0],
            mode="lines",
            name="Matched the call",
            line=dict(color="#b7b2a4", width=1, dash="dot"),
            hoverinfo="skip",
        )
    )
    gaps = calibration["actual"] - calibration["predicted"]
    gap_pad = max(float(gaps.max() - gaps.min()) * 0.18, 0.02)
    biggest = float(calibration["balls"].max())
    for source, color in colors.items():
        group = calibration.loc[calibration["source"].eq(source)].sort_values("predicted")
        gap = group["actual"] - group["predicted"]
        dot = 7 + 15 * numpy.sqrt(group["balls"].to_numpy(dtype=float) / biggest)
        figure.add_trace(
            go.Scatter(
                x=group["band"].astype(float) + 0.05,
                y=gap,
                customdata=numpy.column_stack(
                    [group["actual"].to_numpy(), group["balls"].to_numpy(), group["predicted"].to_numpy()]
                ),
                mode="lines+markers",
                name=source,
                line=dict(color=color, width=2),
                marker=dict(size=dot, color=color),
                hovertemplate=(
                    f"{source}<br>Band %{{x:.2f}}"
                    "<br>Average call %{customdata[2]:.3f}"
                    "<br>Results %{customdata[0]:.3f}"
                    "<br>Gap %{y:+.3f}"
                    "<br>%{customdata[1]:,.0f} balls<extra></extra>"
                ),
            )
        )
    layout = chart_layout(420)
    layout["margin"] = dict(l=64, r=24, t=56, b=64)
    figure.update_layout(**layout)
    figure.update_xaxes(
        title="Expected wOBA",
        range=[-0.08, 2.12],
        tickmode="array",
        tickvals=[0, 0.9, 1.25, 1.6, 2.0],
        ticktext=["0 out", "0.9 single", "1.25 double", "1.6 triple", "2 home run"],
        tickfont=dict(color="#d9d3c5"),
        gridcolor="#3a4150",
    )
    figure.update_yaxes(
        title="Result minus expected wOBA",
        range=[float(gaps.min()) - gap_pad, float(gaps.max()) + gap_pad],
        tickformat="+.2f",
        zeroline=False,
        tickfont=dict(color="#d9d3c5"),
        gridcolor="#3a4150",
    )
    return figure


def _knockout_backdrop(image: Image.Image) -> Image.Image:
    """MLB headshots sit on a flat gray studio backdrop. Clear the backdrop, keep the jersey."""
    image = image.convert("RGBA")
    pixels = numpy.array(image)
    height, width = pixels.shape[:2]
    rgb = pixels[:, :, :3].astype(numpy.int16)
    backdrop = numpy.median(rgb[:8].reshape(-1, 3), axis=0)
    distance = numpy.max(numpy.abs(rgb - backdrop), axis=2)
    saturation = rgb.max(axis=2) - rgb.min(axis=2)
    near = (distance <= 32) & (saturation <= 28)
    seen = numpy.zeros((height, width), dtype=bool)
    queue: deque[tuple[int, int]] = deque()
    for x in range(width):
        queue.append((0, x))
        queue.append((height - 1, x))
    for y in range(height):
        queue.append((y, 0))
        queue.append((y, width - 1))
    while queue:
        y, x = queue.popleft()
        if y < 0 or x < 0 or y >= height or x >= width or seen[y, x] or not near[y, x]:
            continue
        seen[y, x] = True
        queue.append((y - 1, x))
        queue.append((y + 1, x))
        queue.append((y, x - 1))
        queue.append((y, x + 1))
    pixels[:, :, 3] = numpy.where(seen, 0, 255)
    rows, cols = numpy.where(pixels[:, :, 3] > 0)
    if len(cols) == 0:
        return Image.fromarray(pixels)
    pad = 1
    top = max(int(rows.min()) - pad, 0)
    bottom = min(int(rows.max()) + pad + 1, height)
    left = max(int(cols.min()) - pad, 0)
    right = min(int(cols.max()) + pad + 1, width)
    cropped = Image.fromarray(pixels[top:bottom, left:right])
    side = max(cropped.size)
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.paste(cropped, ((side - cropped.size[0]) // 2, (side - cropped.size[1]) // 2), cropped)
    return canvas


def _ensure_head(batter: int) -> Path | None:
    HEADS.mkdir(parents=True, exist_ok=True)
    path = HEADS / f"{int(batter)}.png"
    if path.exists() and path.stat().st_size > 0:
        return path
    url = (
        "https://img.mlbstatic.com/mlb-photos/image/upload/"
        "d_people:generic:headshot:silo:current.png/w_160,q_auto:best/"
        f"v1/people/{int(batter)}/headshot/67/current"
    )
    try:
        with urllib.request.urlopen(url, timeout=20) as response:
            raw = response.read()
        _knockout_backdrop(Image.open(BytesIO(raw))).save(path, format="PNG")
    except Exception:
        return None
    return path


def _head_uri(batter: int) -> str | None:
    path = _ensure_head(int(batter))
    if path is None:
        return None
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _attach_clubs(hitters: pandas.DataFrame) -> pandas.DataFrame:
    frame = hitters.copy()
    frame["batter"] = pandas.to_numeric(frame["batter"], errors="coerce").astype("int64")
    if not TEAMS.exists():
        frame["club"] = ""
        return frame
    teams = pandas.read_parquet(TEAMS, columns=["batter", "team"])
    teams["batter"] = pandas.to_numeric(teams["batter"], errors="coerce").astype("int64")
    teams = teams.drop_duplicates("batter")
    frame = frame.merge(teams, on="batter", how="left")
    frame["club"] = frame["team"].map(CLUBS).fillna("")
    return frame


_BAR_HOVER = """
<script>
(() => {
  function barFor(node, hover) {
    const bars = [...node.querySelectorAll(".trace.bars .point")];
    const index = node.dataset.hoverIndex;
    if (index !== undefined && index !== "" && bars[Number(index)]) return bars[Number(index)];
    const box = hover.getBoundingClientRect();
    const mid = (box.top + box.bottom) / 2;
    let best = null;
    let bestDist = Infinity;
    bars.forEach((el) => {
      const edge = el.getBoundingClientRect();
      const dist = Math.abs((edge.top + edge.bottom) / 2 - mid);
      if (dist < bestDist) {
        bestDist = dist;
        best = el;
      }
    });
    return best;
  }
  function nudge(node) {
    if (node.dataset.nudging) return;
    const hover = node.querySelector(".hovertext");
    if (!hover) return;
    const bar = barFor(node, hover);
    if (!bar) return;
    const box = hover.getBoundingClientRect();
    const edge = bar.getBoundingClientRect();
    const shift = edge.right + 12 - box.left;
    if (shift <= 1) return;
    const match = /translate\\(([\\-\\d.]+)[, ]+([\\-\\d.]+)\\)/.exec(hover.getAttribute("transform") || "");
    if (!match) return;
    node.dataset.nudging = "1";
    hover.setAttribute("transform", "translate(" + (parseFloat(match[1]) + shift) + "," + match[2] + ")");
    node.dataset.nudging = "";
  }
  function bind(node) {
    if (!node || node.dataset.barHover === "2" || typeof node.on !== "function") return;
    if (!node.querySelector(".trace.bars")) return;
    node.dataset.barHover = "2";
    node.on("plotly_hover", (event) => {
      const point = event.points && event.points[0];
      if (!point || point.pointNumber == null) return;
      node.dataset.hoverIndex = String(point.pointNumber);
      requestAnimationFrame(() => nudge(node));
    });
    const layer = node.querySelector(".hoverlayer");
    if (!layer) return;
    new MutationObserver(() => nudge(node)).observe(layer, {
      subtree: true,
      childList: true,
      attributes: true,
      attributeFilter: ["transform"],
    });
  }
  function scan() {
    document.querySelectorAll(".js-plotly-plot").forEach(bind);
  }
  scan();
  const root = document.querySelector('[data-testid="stMain"]') || document.body;
  if (!window.__barHover) {
    window.__barHover = true;
    new MutationObserver(scan).observe(root, { childList: true, subtree: true });
  }
})();
</script>
"""


_FACE_LIFT = """
<style>
.js-plotly-plot .layer-above image { pointer-events: none; }
</style>
<script>
(() => {
  function bind(node) {
    if (!node || node.dataset.faceLift || typeof node.on !== "function") return;
    const names = (node.data || []).map((trace) => trace.name);
    if (!names.includes("Hitters")) return;
    node.dataset.faceLift = "1";
    node.on("plotly_hover", (event) => {
      const point = event.points && event.points[0];
      if (!point || !point.data || point.data.name !== "Hitters") return;
      const layer = node.querySelector(".layer-above");
      if (!layer) return;
      const images = [...layer.querySelectorAll("image")];
      if (!images.some((image) => image.dataset.faceIndex)) {
        images.forEach((image, index) => {
          image.dataset.faceIndex = String(index);
        });
      }
      const image = layer.querySelector('image[data-face-index="' + point.pointNumber + '"]');
      if (image) layer.appendChild(image);
    });
  }
  function scan() {
    document.querySelectorAll(".js-plotly-plot").forEach(bind);
  }
  scan();
  let tries = 0;
  const timer = setInterval(() => {
    scan();
    tries += 1;
    if (tries > 24) clearInterval(timer);
  }, 250);
  const root = document.querySelector('[data-testid="stMain"]') || document.body;
  if (!window.__contactFaceLift) {
    window.__contactFaceLift = true;
    new MutationObserver(scan).observe(root, { childList: true, subtree: true });
  }
})();
</script>
"""


def _prose(key: str, default: str, css_class: str) -> None:
    """Same Edit control as the other pages. Saved text lives in story.json."""
    from app import story_paragraph

    story_paragraph(key, default, css_class)


def _keep_one_face(
    pool: pandas.DataFrame,
    x_col: str,
    y_col: str,
    face_x: float,
    face_y: float,
    score: pandas.Series,
) -> pandas.DataFrame:
    """When two faces would cover each other, keep the one farther from the middle of the chart."""
    rank = pool.copy()
    rank["_score"] = score.to_numpy()
    rank = rank.sort_values("_score", ascending=False)
    kept: list[tuple[float, float]] = []
    keep_ids: list[int] = []
    for row in rank.itertuples():
        x = float(getattr(row, x_col))
        y = float(getattr(row, y_col))
        if any(abs(x - other_x) < face_x and abs(y - other_y) < face_y for other_x, other_y in kept):
            continue
        kept.append((x, y))
        keep_ids.append(int(row.batter))
    faces = pool.loc[pool["batter"].isin(keep_ids)].copy()
    order = {batter: index for index, batter in enumerate(keep_ids)}
    faces["_order"] = faces["batter"].astype(int).map(order)
    return faces.sort_values("_order").drop(columns=["_order"])


def _hitter_layout(
    hitters: pandas.DataFrame, club: str | None
) -> tuple[pandas.DataFrame, pandas.DataFrame, tuple[float, float, float, float, float, float] | None]:
    pool = hitters if not club else hitters.loc[hitters["club"].eq(club)].copy()
    if pool.empty:
        return pool, pool.iloc[0:0], None
    xmin = float(pool["model"].min())
    xmax = float(pool["model"].max())
    ymin = float(pool["actual"].min())
    ymax = float(pool["actual"].max())
    xspan = max(xmax - xmin, 0.05)
    yspan = max(ymax - ymin, 0.05)
    face_x = xspan / (8 if club else 16)
    face_y = face_x * (xspan / yspan) * 1.55
    pad_x = max(xspan * 0.08, face_x * 0.65)
    pad_y = max(yspan * 0.10, face_y * 0.65)
    if club:
        faces = pool.sort_values("model")
        dots = pool.iloc[0:0]
    else:
        score = (pool["actual"] - pool["model"]).abs()
        if "gap" in pool.columns:
            score = score + pool["gap"].abs() * 1e-6
        faces = _keep_one_face(pool, "model", "actual", face_x, face_y, score)
        dots = pool.loc[~pool["batter"].isin(set(faces["batter"]))].copy()
    bounds = (xmin - pad_x, xmax + pad_x, ymin - pad_y, ymax + pad_y, face_x, face_y)
    return faces, dots, bounds


def _hitter_hover(frame: pandas.DataFrame) -> tuple[numpy.ndarray, str]:
    club = frame["club"] if "club" in frame.columns else pandas.Series("", index=frame.index)
    custom = numpy.column_stack(
        [
            frame["savant"].to_numpy(),
            frame["bip"].to_numpy(),
            frame["gap"].to_numpy(),
            club.fillna("").to_numpy(),
            frame["hit_rate"].to_numpy(),
            frame["singles"].to_numpy(),
            frame["doubles"].to_numpy(),
            frame["triples"].to_numpy(),
            frame["homers"].to_numpy(),
        ]
    )
    template = (
        "%{text}<br>%{customdata[3]}<br>This model %{x:.3f}<br>Results %{y:.3f}"
        "<br>MLB expected %{customdata[0]:.3f}<br>Gap %{customdata[2]:+.3f}"
        "<br>Hits per ball %{customdata[4]:.3f}"
        "<br>%{customdata[5]:.0f} singles, %{customdata[6]:.0f} doubles, %{customdata[7]:.0f} triples, %{customdata[8]:.0f} HR"
        "<br>%{customdata[1]:.0f} balls<extra></extra>"
    )
    return custom, template


def hitter_figure(hitters: pandas.DataFrame, club: str | None = None) -> go.Figure:
    faces, dots, bounds = _hitter_layout(hitters, club)
    if bounds is None:
        figure = go.Figure()
        figure.update_layout(**chart_layout(420))
        return figure
    x0, x1, y0, y1, face_x, face_y = bounds
    figure = go.Figure()
    line_lo = max(x0, y0)
    line_hi = min(x1, y1)
    figure.add_trace(
        go.Scatter(
            x=[line_lo, line_hi],
            y=[line_lo, line_hi],
            mode="lines",
            name="Results match the model",
            line=dict(color="#b7b2a4", width=1, dash="dot"),
            hoverinfo="skip",
        )
    )
    if not dots.empty:
        custom, template = _hitter_hover(dots)
        figure.add_trace(
            go.Scatter(
                x=dots["model"],
                y=dots["actual"],
                mode="markers",
                name="Other hitters",
                marker=dict(size=7, color="#8d8474", opacity=0.9, line=dict(width=0)),
                text=dots["name"],
                customdata=custom,
                hovertemplate=template,
            )
        )
    images, faced = _stamp_faces(faces, "model", "actual", face_x, face_y)
    if not faced.empty:
        custom, template = _hitter_hover(faced)
        figure.add_trace(
            go.Scatter(
                x=faced["model"],
                y=faced["actual"],
                mode="markers",
                name="Hitters",
                marker=dict(
                    size=42 if club else 28,
                    color="rgba(0,0,0,0)",
                    line=dict(width=0, color="rgba(0,0,0,0)"),
                ),
                showlegend=False,
                text=faced["name"],
                customdata=custom,
                hovertemplate=template,
            )
        )
    layout = chart_layout(620 if club else 680)
    layout["margin"] = dict(l=56, r=24, t=40, b=48)
    layout["images"] = images
    layout["hovermode"] = "closest"
    figure.update_layout(**layout)
    figure.update_xaxes(
        title="Model's contact value, per ball",
        range=[x0, x1],
        tickfont=dict(color="#d9d3c5"),
        gridcolor="#3a4150",
        autorange=False,
    )
    figure.update_yaxes(
        title="What actually happened, per ball",
        range=[y0, y1],
        tickfont=dict(color="#d9d3c5"),
        gridcolor="#3a4150",
        autorange=False,
    )
    return figure


def _stamp_faces(
    faces: pandas.DataFrame, x_col: str, y_col: str, face_x: float, face_y: float
) -> tuple[list[dict], pandas.DataFrame]:
    images = []
    keep: list = []
    for row in faces.itertuples():
        source = _head_uri(int(row.batter))
        if source is None:
            continue
        keep.append(row.Index)
        images.append(
            dict(
                source=source,
                xref="x",
                yref="y",
                x=float(getattr(row, x_col)),
                y=float(getattr(row, y_col)),
                sizex=face_x,
                sizey=face_y,
                xanchor="center",
                yanchor="middle",
                sizing="contain",
                layer="above",
            )
        )
    shown = faces.loc[keep] if keep else faces.iloc[0:0]
    return images, shown


def _centered_layout(
    frame: pandas.DataFrame, club: str | None, x_col: str, y_col: str
) -> tuple[pandas.DataFrame, pandas.DataFrame, tuple[float, float, float, float, float, float] | None]:
    pool = frame if not club else frame.loc[frame["club"].eq(club)].copy()
    if pool.empty:
        return pool, pool.iloc[0:0], None
    x_reach = max(abs(float(pool[x_col].min())), abs(float(pool[x_col].max())), 0.02)
    y_reach = max(abs(float(pool[y_col].min())), abs(float(pool[y_col].max())), 0.02)
    xspan = x_reach * 2
    yspan = max(y_reach * 2, 0.04)
    face_x = xspan / (8 if club else 16)
    face_y = face_x * (xspan / yspan) * 1.55
    pad_x = max(xspan * 0.08, face_x * 0.7)
    pad_y = max(yspan * 0.10, face_y * 0.7)
    if club:
        faces = pool.sort_values(x_col)
        dots = pool.iloc[0:0]
    else:
        score = pandas.Series(numpy.hypot(pool[x_col].to_numpy(), pool[y_col].to_numpy()), index=pool.index)
        faces = _keep_one_face(pool, x_col, y_col, face_x, face_y, score)
        dots = pool.loc[~pool["batter"].isin(set(faces["batter"]))].copy()
    bounds = (-x_reach - pad_x, x_reach + pad_x, -y_reach - pad_y, y_reach + pad_y, face_x, face_y)
    return faces, dots, bounds


def _cross_figure(
    frame: pandas.DataFrame,
    club: str | None,
    x_col: str,
    y_col: str,
    x_title: str,
    y_title: str,
    corners: tuple[str, str, str, str],
    hover_fn,
) -> go.Figure:
    """Faces on a cross through zero. Right and up are the high sides."""
    faces, dots, bounds = _centered_layout(frame, club, x_col, y_col)
    figure = go.Figure()
    if bounds is None:
        figure.update_layout(**chart_layout(420))
        return figure
    x0, x1, y0, y1, face_x, face_y = bounds
    figure.add_hline(y=0, line_width=1, line_color="#6a6458")
    figure.add_vline(x=0, line_width=1, line_color="#6a6458")
    spots = (
        (0.99, 0.98, "right", "top", corners[0]),
        (0.01, 0.98, "left", "top", corners[1]),
        (0.99, 0.02, "right", "bottom", corners[2]),
        (0.01, 0.02, "left", "bottom", corners[3]),
    )
    for x, y, xanchor, yanchor, text in spots:
        figure.add_annotation(
            xref="paper",
            yref="paper",
            x=x,
            y=y,
            xanchor=xanchor,
            yanchor=yanchor,
            text=text,
            showarrow=False,
            font=dict(size=12, color="#b7b2a4"),
        )
    if not dots.empty:
        custom, template = hover_fn(dots)
        figure.add_trace(
            go.Scatter(
                x=dots[x_col],
                y=dots[y_col],
                mode="markers",
                name="Other hitters",
                marker=dict(size=7, color="#8d8474", opacity=0.9, line=dict(width=0)),
                text=dots["name"],
                customdata=custom,
                hovertemplate=template,
            )
        )
    images, faced = _stamp_faces(faces, x_col, y_col, face_x, face_y)
    if not faced.empty:
        custom, template = hover_fn(faced)
        figure.add_trace(
            go.Scatter(
                x=faced[x_col],
                y=faced[y_col],
                mode="markers",
                name="Hitters",
                marker=dict(
                    size=42 if club else 28,
                    color="rgba(0,0,0,0)",
                    line=dict(width=0, color="rgba(0,0,0,0)"),
                ),
                showlegend=False,
                text=faced["name"],
                customdata=custom,
                hovertemplate=template,
            )
        )
    layout = chart_layout(680)
    layout["margin"] = dict(l=64, r=24, t=28, b=52)
    layout["images"] = images
    layout["hovermode"] = "closest"
    figure.update_layout(**layout)
    axis = dict(tickfont=dict(color="#d9d3c5"), gridcolor="#3a4150", zeroline=False, autorange=False)
    figure.update_xaxes(title=x_title, range=[x0, x1], **axis)
    figure.update_yaxes(title=y_title, range=[y0, y1], **axis)
    return figure


def _quadrant_hover(frame: pandas.DataFrame) -> tuple[numpy.ndarray, str]:
    club = frame["club"] if "club" in frame.columns else pandas.Series("", index=frame.index)
    custom = numpy.column_stack(
        [
            club.fillna("").to_numpy(),
            frame["bip"].to_numpy(),
            frame["actual"].to_numpy(),
            frame["model"].to_numpy(),
            frame["savant"].to_numpy(),
            frame["hit_rate"].to_numpy(),
            frame["singles"].to_numpy(),
            frame["doubles"].to_numpy(),
            frame["triples"].to_numpy(),
            frame["homers"].to_numpy(),
        ]
    )
    template = (
        "%{text}<br>%{customdata[0]}"
        "<br>Results minus model %{x:+.3f}"
        "<br>Hits per ball vs league %{y:+.3f}"
        "<br>Hits per ball %{customdata[5]:.3f}"
        "<br>Results %{customdata[2]:.3f}<br>This model %{customdata[3]:.3f}<br>MLB expected %{customdata[4]:.3f}"
        "<br>%{customdata[6]:.0f} singles, %{customdata[7]:.0f} doubles, %{customdata[8]:.0f} triples, %{customdata[9]:.0f} HR"
        "<br>%{customdata[1]:.0f} balls<extra></extra>"
    )
    return custom, template


def contact_cross(hitters: pandas.DataFrame, club: str | None, league_hits: float) -> go.Figure:
    frame = hitters.copy()
    frame["beat"] = frame["actual"] - frame["model"]
    frame["hit_edge"] = frame["hit_rate"] - float(league_hits)
    return _cross_figure(
        frame,
        club,
        "beat",
        "hit_edge",
        "Results minus the model",
        "Hits per ball, minus the league",
        (
            "Beat the launch, more hits",
            "Short of the launch, more hits",
            "Beat the launch, fewer hits",
            "Short of the launch, fewer hits",
        ),
        _quadrant_hover,
    )


def _corner_counts(
    horizontal: pandas.Series,
    vertical: pandas.Series,
    high_row: str,
    low_row: str,
    high_col: str,
    low_col: str,
) -> pandas.DataFrame:
    right = horizontal.ge(0)
    up = vertical.ge(0)
    return pandas.DataFrame(
        {
            "Where they sit": [high_row, low_row],
            high_col: [int((right & up).sum()), int((~right & up).sum())],
            low_col: [int((right & ~up).sum()), int((~right & ~up).sum())],
        }
    )


def contact_corners(hitters: pandas.DataFrame, club: str | None, league_hits: float) -> pandas.DataFrame:
    frame = hitters if not club else hitters.loc[hitters["club"].eq(club)]
    return _corner_counts(
        frame["actual"] - frame["model"],
        frame["hit_rate"] - float(league_hits),
        "Above the model",
        "Below the model",
        "More hits than the league",
        "Fewer hits than the league",
    )


def _dirt_shape() -> dict:
    """Skinned infield: a 95-foot arc centered on the mound, closed back to home plate."""
    mound = 60.5
    radius = 95.0
    phi = numpy.linspace(-numpy.pi, numpy.pi, 361)
    xs = radius * numpy.sin(phi)
    ys = mound + radius * numpy.cos(phi)
    fair = ys >= numpy.abs(xs)
    order = numpy.argsort(xs[fair])
    arc_x = xs[fair][order]
    arc_y = ys[fair][order]
    steps = [f"M 0,0 L {arc_x[0]:.2f},{arc_y[0]:.2f}"]
    steps.extend(f"L {x:.2f},{y:.2f}" for x, y in zip(arc_x[1:], arc_y[1:]))
    steps.append("Z")
    return dict(type="path", path=" ".join(steps), fillcolor="#7a5a3c", line=dict(width=0), layer="below")


def _line_pair(x, y, name: str | None, color: str, width: float) -> list[go.Scatter]:
    halo = go.Scatter(
        x=x,
        y=y,
        mode="lines",
        line=dict(color="#10140f", width=width + 3.5),
        hoverinfo="skip",
        showlegend=False,
    )
    mark = go.Scatter(
        x=x,
        y=y,
        mode="lines",
        name=name,
        line=dict(color=color, width=width),
        hoverinfo="skip",
        showlegend=name is not None,
    )
    return [halo, mark]


def spray_figure(spray: pandas.DataFrame) -> go.Figure:
    colors = {
        "ground_ball": "#8eb8c4",
        "line_drive": "#e6b325",
        "fly_ball": "#d08c60",
        "popup": "#d6453d",
        "home_run": "#f3efe4",
    }
    names = {
        "ground_ball": "Ground ball",
        "line_drive": "Line drive",
        "fly_ball": "Fly ball",
        "popup": "Popup",
        "home_run": "Home run",
    }
    base = 90 / numpy.sqrt(2)
    second = 90 * numpy.sqrt(2)
    figure = go.Figure()
    order = ("popup", "ground_ball", "fly_ball", "line_drive", "home_run")
    for kind in order:
        group = spray.loc[spray["kind"].eq(kind)] if "kind" in spray.columns else spray.iloc[0:0]
        has_rows = not group.empty
        if has_rows and "label" in group.columns and "park" in group.columns:
            carry = numpy.hypot(group["x"].to_numpy(), group["y"].to_numpy())
            custom = numpy.column_stack(
                [group["label"].to_numpy(), group["park"].to_numpy(), numpy.rint(carry).astype(int)]
            )
            hover = "%{customdata[0]}<br>%{customdata[1]}<br>%{customdata[2]} ft from home<extra></extra>"
            xs = group["x"]
            ys = group["y"]
        else:
            custom = None
            hover = "skip"
            xs = []
            ys = []
        figure.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                name=names[kind],
                showlegend=has_rows,
                marker=dict(size=6 if kind != "home_run" else 8, color=colors[kind], opacity=0.72, line=dict(width=0)),
                customdata=custom,
                hovertemplate=hover,
            )
        )
    arc = numpy.linspace(-numpy.pi / 4, numpy.pi / 4, 90)
    dirt = numpy.linspace(-numpy.pi * 0.72, numpy.pi * 0.72, 80)
    dirt_x = 95 * numpy.sin(dirt)
    dirt_y = 60.5 + 95 * numpy.cos(dirt)
    dirt_fair = dirt_y >= numpy.abs(dirt_x)
    for trace in _line_pair([0, -520, None, 0, 520], [0, 520, None, 0, 520], None, "#f3efe4", 1.6):
        figure.add_trace(trace)
    for trace in _line_pair(dirt_x[dirt_fair], dirt_y[dirt_fair], None, "#f0d7b0", 2.2):
        figure.add_trace(trace)
    # 2025 Seamheads marks, averaged across the 30 parks: left line 332, center 403, right line 328.
    # cos(2 * angle) holds the fence deep through the alleys and brings it in at the poles.
    left_pole, center_fence, right_pole = 332.0, 403.0, 328.0
    weight = numpy.cos(2.0 * arc)
    pole = numpy.where(arc < 0, left_pole, right_pole)
    radius = pole + (center_fence - pole) * weight
    fence_x = radius * numpy.sin(arc)
    fence_y = radius * numpy.cos(arc)
    for trace in _line_pair(fence_x, fence_y, "Average fence", "#e6b325", 2.6):
        figure.add_trace(trace)
    figure.data[-1].update(
        hovertext=[f"Average fence<br>{foot:.0f} ft from home" for foot in radius],
        hoverinfo="text",
    )
    for trace in _line_pair([0, base, 0, -base, 0], [0, base, second, base, 0], None, "#f3efe4", 1.8):
        figure.add_trace(trace)
    figure.add_trace(
        go.Scatter(
            x=[base, 0, -base, 0],
            y=[base, second, base, 0],
            mode="markers",
            name="Bases",
            showlegend=False,
            marker=dict(symbol="square", size=11, color="#f3efe4", line=dict(color="#10140f", width=1)),
            hovertext=["First base", "Second base, 127 ft from home", "Third base", "Home plate"],
            hoverinfo="text",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[0],
            y=[60.5],
            mode="markers",
            name="Mound",
            showlegend=False,
            marker=dict(symbol="circle", size=14, color="#8d6b4a", line=dict(color="#f0d7b0", width=1)),
            hovertext=["Pitcher's mound, 60 ft 6 in"],
            hoverinfo="text",
        )
    )
    layout = chart_layout(720)
    layout["margin"] = dict(l=48, r=24, t=56, b=36)
    layout["uirevision"] = "spray-feet"
    layout["legend"]["itemclick"] = False
    layout["legend"]["itemdoubleclick"] = False
    layout["shapes"] = [
        _dirt_shape(),
        dict(
            type="circle",
            x0=-9,
            x1=9,
            y0=51.5,
            y1=69.5,
            fillcolor="#8d6b4a",
            line=dict(color="#d9c4a4", width=1),
            layer="below",
        ),
    ]
    figure.update_layout(**layout)
    figure.update_xaxes(
        title="Feet from center",
        range=[-360, 360],
        tickmode="array",
        tickvals=[-300, -200, -100, 0, 100, 200, 300],
        ticktext=["300 ft", "200 ft", "100 ft", "Center", "100 ft", "200 ft", "300 ft"],
        tickfont=dict(color="#d9d3c5"),
        showgrid=False,
        zeroline=False,
        constrain="domain",
        constraintoward="center",
    )
    figure.update_yaxes(
        title="Feet from home plate",
        range=[-55, 510],
        tickmode="array",
        tickvals=[0, 100, 200, 300, 400, 500],
        ticktext=["Home", "100 ft", "200 ft", "300 ft", "400 ft", "500 ft"],
        tickfont=dict(color="#d9d3c5"),
        showgrid=False,
        scaleanchor="x",
        scaleratio=1,
        constrain="domain",
        constraintoward="center",
        zeroline=False,
    )
    figure.add_annotation(
        x=-300,
        y=-48,
        text="Left field",
        showarrow=False,
        xanchor="center",
        font=dict(color="#d9d3c5", size=13),
    )
    figure.add_annotation(
        x=300,
        y=-48,
        text="Right field",
        showarrow=False,
        xanchor="center",
        font=dict(color="#d9d3c5", size=13),
    )
    return figure


def zone_figure(zone: pandas.DataFrame, top: float, bot: float, *, hits_only: bool) -> go.Figure:
    grid = zone.pivot(index="zbin", columns="xbin", values="woba").sort_index()
    counts = zone.pivot(index="zbin", columns="xbin", values="balls").reindex(index=grid.index, columns=grid.columns)
    inches = numpy.abs(grid.columns.astype(float).to_numpy()) * 12
    inch_grid = numpy.broadcast_to(inches, grid.shape)
    custom = numpy.dstack((inch_grid, counts.to_numpy()))
    if hits_only:
        zmin, zmax = 0.9, 1.3
        bar_title = "Avg. hit"
        counted = "hits"
        # Dark at a single, gold through the extra-base hits, red at a home run.
        colorscale = [
            [0.0, "#2a303a"],
            [0.35, "#3d6f7a"],
            [0.6, "#e6b325"],
            [1.0, "#d6453d"],
        ]
    else:
        zmin, zmax = 0.15, 0.5
        bar_title = "Avg. result"
        counted = "balls"
        # A 0.5 is still mostly outs. Use the dark-to-teal step the hits chart
        # uses from 0.9 to just past 1.0. Gold and red stay on the hits chart.
        colorscale = [
            [0.0, "#2a303a"],
            [1.0, "#3d6f7a"],
        ]
    figure = go.Figure(
        go.Heatmap(
            x=grid.columns.astype(float),
            y=grid.index.astype(float),
            z=grid.to_numpy(),
            customdata=custom,
            zmin=zmin,
            zmax=zmax,
            colorscale=colorscale,
            colorbar=dict(title=bar_title, tickfont=dict(color="#d9d3c5"), title_font=dict(color="#d9d3c5")),
            hovertemplate=(
                "%{customdata[0]:.0f} in. from the middle<br>%{y:.1f} ft high<br>"
                "%{customdata[1]:.0f} " + counted + "<br>Average weight %{z:.2f}<extra></extra>"
            ),
        )
    )
    figure.add_shape(
        type="rect",
        x0=-0.83,
        x1=0.83,
        y0=bot,
        y1=top,
        line=dict(color="#f3efe4", width=2),
        fillcolor="rgba(0,0,0,0)",
    )
    layout = chart_layout(640)
    layout["margin"] = dict(l=56, r=90, t=16, b=48)
    figure.update_layout(**layout)
    figure.update_xaxes(
        title="Inches from the middle of the plate",
        range=[-2.0, 2.0],
        tickmode="array",
        tickvals=[-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2],
        ticktext=["24", "18", "12", "6", "0", "6", "12", "18", "24"],
        tickfont=dict(color="#d9d3c5"),
        gridcolor="#3a4150",
        zeroline=False,
        constrain="domain",
        constraintoward="center",
    )
    figure.update_yaxes(
        title="Height as the pitch crosses the plate (feet)",
        range=[0.8, 4.4],
        tickfont=dict(color="#d9d3c5"),
        gridcolor="#3a4150",
        scaleanchor="x",
        scaleratio=1,
        constrain="domain",
        constraintoward="center",
    )
    return figure


@st.fragment
def _zone_board(hits: pandas.DataFrame, everything: pandas.DataFrame, top: float, bot: float) -> None:
    """Hits or every ball in play. The switch reruns this piece only."""
    mode = st.segmented_control(
        "Zone balls",
        ["Hits", "All contact"],
        default="Hits",
        key="contact_zone_mode",
        label_visibility="collapsed",
        required=True,
        width="content",
    )
    hits_only = mode != "All contact"
    st.plotly_chart(
        zone_figure(hits if hits_only else everything, top, bot, hits_only=hits_only),
        width="stretch",
        config={"displayModeBar": False, "scrollZoom": False, "doubleClick": False},
    )
    if hits_only:
        st.caption(
            "Each square needs at least 25 hits from the scored season. Outs are not in the average. "
            "The outline is the median top and bottom of the zone."
        )
    else:
        st.caption(
            "Each square needs at least 25 balls in play from the scored season. Outs are in the average. "
            "The outline is the median top and bottom of the zone."
        )


def _gap_sentence(hitters: pandas.DataFrame) -> str:
    top = hitters.head(3)
    if top.empty:
        return f"No hitter in the holdout has {MIN_BALLS} balls in play yet."
    bits = [f"{row['name']} ({row['gap']:+.3f})" for _, row in top.iterrows()]
    if len(bits) == 1:
        listed = bits[0]
    elif len(bits) == 2:
        listed = f"{bits[0]} and {bits[1]}"
    else:
        listed = f"{bits[0]}, {bits[1]}, and {bits[2]}"
    return (
        f"The widest gaps, among hitters with at least {MIN_BALLS} balls, are {listed}. "
        "A positive number means this model values the contact higher than MLB's expected weight does."
    )


@dataclass
class ForecastReport:
    seasons: list[int]
    steps: pandas.DataFrame
    board: pandas.DataFrame
    balls_mae: float
    league_woba: float = float("nan")
    league_hits: float = float("nan")


def _season_table(frame: pandas.DataFrame) -> pandas.DataFrame:
    def _mode(series: pandas.Series) -> str:
        return str(series.value_counts().index[0])

    work = frame.copy()
    events = work["events"].astype(str) if "events" in work.columns else pandas.Series("", index=work.index)
    work["single"] = events.eq("single").to_numpy()
    work["double"] = events.eq("double").to_numpy()
    work["triple"] = events.eq("triple").to_numpy()
    work["homer"] = events.eq("home_run").to_numpy()
    table = work.groupby("batter", as_index=True).agg(
        bip=("woba_value", "size"),
        actual=("woba_value", "mean"),
        contact=("estimated_woba_using_speedangle", "mean"),
        team=("home_team", _mode),
        singles=("single", "sum"),
        doubles=("double", "sum"),
        triples=("triple", "sum"),
        homers=("homer", "sum"),
    )
    table["hit_rate"] = (table["singles"] + table["doubles"] + table["triples"] + table["homers"]) / table["bip"]
    return table


def _park_effect(frame: pandas.DataFrame) -> pandas.Series:
    grouped = frame.groupby("home_team").agg(
        n=("woba_value", "size"),
        actual=("woba_value", "mean"),
        contact=("estimated_woba_using_speedangle", "mean"),
    )
    gap = grouped["actual"] - grouped["contact"]
    return gap * grouped["n"] / (grouped["n"] + PARK_PRIOR)


def _opening_age(birth: pandas.Timestamp, year: int) -> int | None:
    if pandas.isna(birth):
        return None
    opening = (3, 28)
    born = (int(birth.month), int(birth.day))
    return int(year - birth.year - (opening < born))


def build_forecast(balls: pandas.DataFrame) -> ForecastReport:
    """Carry contact forward one season. Weights are fixed before seeing the result."""
    needed = {"season", "batter", "home_team", "woba_value", "estimated_woba_using_speedangle"}
    empty_steps = pandas.DataFrame(columns=["prior", "nxt", "pairs", "mae_contact", "mae_results", "mae_park"])
    empty_board = pandas.DataFrame(
        columns=[
            "batter",
            "name",
            "club",
            "age",
            "bip",
            "singles",
            "doubles",
            "triples",
            "homers",
            "hit_rate",
            "contact",
            "actual",
            "park",
            "guess",
        ]
    )
    if not needed.issubset(balls.columns):
        return ForecastReport([], empty_steps, empty_board, float("nan"))
    balls = balls.dropna(subset=["woba_value", "estimated_woba_using_speedangle", "batter", "home_team"]).copy()
    balls["batter"] = pandas.to_numeric(balls["batter"], errors="coerce").astype("int64")
    seasons = sorted(int(year) for year in balls["season"].unique())
    tables = {year: _season_table(balls.loc[balls["season"].eq(year)]) for year in seasons}
    effects = {year: _park_effect(balls.loc[balls["season"].eq(year)]) for year in seasons}
    step_rows = []
    balls_mae = float("nan")
    for prior, nxt in zip(seasons, seasons[1:]):
        league = float(balls.loc[balls["season"].eq(prior), "woba_value"].mean())
        left = tables[prior].add_suffix("_0")
        right = tables[nxt].loc[:, ["bip", "actual"]].add_suffix("_1")
        both = left.join(right, how="inner")
        both = both.loc[both["bip_0"].ge(MIN_BALLS) & both["bip_1"].ge(NEXT_BALLS)]
        if both.empty:
            continue
        trust = both["bip_0"] / (both["bip_0"] + MIN_BALLS)
        from_contact = league + trust * (both["contact_0"] - league)
        from_results = league + trust * (both["actual_0"] - league)
        park_bump = 0.5 * both["team_0"].map(effects[prior]).fillna(0.0)
        step_rows.append(
            {
                "prior": prior,
                "nxt": nxt,
                "pairs": int(len(both)),
                "mae_contact": float((from_contact - both["actual_1"]).abs().mean()),
                "mae_results": float((from_results - both["actual_1"]).abs().mean()),
                "mae_park": float((from_contact + park_bump - both["actual_1"]).abs().mean()),
            }
        )
    steps = pandas.DataFrame(step_rows) if step_rows else empty_steps
    if len(seasons) >= 2:
        prior, nxt = seasons[-2], seasons[-1]
        left_balls = tables[prior].loc[tables[prior]["bip"].ge(MIN_BALLS), ["bip"]].add_suffix("_0")
        right_balls = tables[nxt].loc[:, ["bip"]].add_suffix("_1")
        paired_balls = left_balls.join(right_balls, how="inner")
        if not paired_balls.empty:
            balls_mae = float((paired_balls["bip_0"] - paired_balls["bip_1"]).abs().mean())
    if not seasons:
        return ForecastReport(seasons, steps, empty_board, balls_mae)

    last = seasons[-1]
    league = float(balls.loc[balls["season"].eq(last), "woba_value"].mean())
    board = tables[last].loc[tables[last]["bip"].ge(MIN_BALLS)].copy()
    board = board.rename(columns={"team": "park_team"}).reset_index()
    if TEAMS.exists():
        clubs = pandas.read_parquet(TEAMS, columns=["batter", "team"])
        clubs["batter"] = pandas.to_numeric(clubs["batter"], errors="coerce").astype("int64")
        clubs = clubs.drop_duplicates("batter")
        board = board.merge(clubs, on="batter", how="left")
        board["team"] = board["team"].fillna(board["park_team"])
    else:
        board["team"] = board["park_team"]
    trust = board["bip"] / (board["bip"] + MIN_BALLS)
    board["guess"] = league + trust * (board["contact"] - league)
    board["park"] = board["team"].map(effects[last]).fillna(0.0)
    board["club"] = board["team"].map(CLUBS).fillna("")
    if BIRTHS.exists():
        births = pandas.read_parquet(BIRTHS, columns=["batter", "birth"])
        births["batter"] = pandas.to_numeric(births["batter"], errors="coerce").astype("int64")
        births["birth"] = pandas.to_datetime(births["birth"], errors="coerce")
        births = births.drop_duplicates("batter")
        board = board.merge(births, on="batter", how="left")
        board["age"] = board["birth"].map(lambda birth: _opening_age(birth, last + 1)).astype("Int64")
    else:
        board["age"] = pandas.Series(pandas.NA, index=board.index, dtype="Int64")
    board["name"] = _batter_names(board["batter"])
    last_events = balls.loc[balls["season"].eq(last), "events"].astype(str) if "events" in balls.columns else pandas.Series(dtype=str)
    league_hits = float(last_events.isin(["single", "double", "triple", "home_run"]).mean()) if len(last_events) else float("nan")
    keep = [
        "batter",
        "name",
        "club",
        "age",
        "bip",
        "singles",
        "doubles",
        "triples",
        "homers",
        "hit_rate",
        "contact",
        "actual",
        "park",
        "guess",
    ]
    board = board.loc[:, keep]
    board = board.sort_values("guess", ascending=False).reset_index(drop=True)
    return ForecastReport(seasons, steps, board, balls_mae, league, league_hits)


def _forecast_hover(frame: pandas.DataFrame) -> tuple[numpy.ndarray, str]:
    ages = ["" if pandas.isna(age) else str(int(age)) for age in frame["age"]]
    custom = numpy.column_stack(
        [
            frame["club"].fillna("").astype(str).to_numpy(),
            numpy.array(ages, dtype=object),
            frame["bip"].map(lambda balls: f"{int(balls)}").to_numpy(),
            frame["contact"].map(lambda value: f"{value:.3f}").to_numpy(),
            frame["actual"].map(lambda value: f"{value:.3f}").to_numpy(),
            frame["guess"].map(lambda value: f"{value:.3f}").to_numpy(),
            frame["hit_rate"].map(lambda value: f"{value:.3f}").to_numpy(),
            frame["park"].map(lambda value: f"{value:+.3f}").to_numpy(),
            frame["singles"].map(lambda value: f"{int(value)}").to_numpy(),
            frame["doubles"].map(lambda value: f"{int(value)}").to_numpy(),
            frame["triples"].map(lambda value: f"{int(value)}").to_numpy(),
            frame["homers"].map(lambda value: f"{int(value)}").to_numpy(),
        ]
    )
    template = (
        "%{text}<br>%{customdata[0]}<br>Age %{customdata[1]}"
        "<br>Guess minus last year %{x:+.3f}"
        "<br>Hits per ball vs league %{y:+.3f}"
        "<br>Guess %{customdata[5]}<br>Last year %{customdata[4]}<br>Contact %{customdata[3]}"
        "<br>Hits per ball %{customdata[6]}<br>Park %{customdata[7]}"
        "<br>%{customdata[8]} singles, %{customdata[9]} doubles, %{customdata[10]} triples, %{customdata[11]} HR"
        "<br>%{customdata[2]} balls<extra></extra>"
    )
    return custom, template


def forecast_figure(board: pandas.DataFrame, year: int, club: str | None, league_hits: float) -> go.Figure:
    """Guess versus last year's results, crossed with hits per ball versus the league."""
    frame = board.copy()
    frame["rise"] = frame["guess"] - frame["actual"]
    frame["hit_edge"] = frame["hit_rate"] - float(league_hits)
    return _cross_figure(
        frame,
        club,
        "rise",
        "hit_edge",
        f"Guess for {year}, minus {year - 1} results",
        "Hits per ball, minus the league",
        (
            "Small rise, high hit rate",
            "Comes off last year",
            "Rises, results trailed contact",
            "Falls, low hit rate",
        ),
        _forecast_hover,
    )


def forecast_corners(board: pandas.DataFrame, club: str | None, league_hits: float) -> pandas.DataFrame:
    frame = board if not club else board.loc[board["club"].eq(club)]
    return _corner_counts(
        frame["guess"] - frame["actual"],
        frame["hit_rate"] - float(league_hits),
        "Guess above last year",
        "Guess below last year",
        "More hits than the league",
        "Fewer hits than the league",
    )


_SPRAY_OPTIONS = ("Outs", "Singles", "Doubles", "Triples", "Home runs")
_SPRAY_OUTCOME = {"Outs": "Out", "Singles": "Single", "Doubles": "Double", "Triples": "Triple", "Home runs": "Home run"}
_SPRAY_KINDS = (
    ("popup", "Popup"),
    ("ground_ball", "Ground ball"),
    ("fly_ball", "Fly ball"),
    ("line_drive", "Line drive"),
    ("home_run", "Home run"),
)

# 2025 foul lines. Wall height is in feet. None means the listing did not publish one.
_FOUL_LINES = (
    ("American Family Field", 344, 8, 345, 6),
    ("Angel Stadium", 330, 5, 330, 5),
    ("Busch Stadium", 336, 8, 335, 8),
    ("Camden Yards", 333, 7, 318, 21),
    ("Chase Field", 328, 8, 335, 8),
    ("Citi Field", 331, 8, 330, 8),
    ("Citizens Bank Park", 329, 11, 330, 13),
    ("Comerica Park", 345, 6, 330, 8),
    ("Coors Field", 347, 13, 350, 16.5),
    ("Daikin Park", 315, 21, 326, 7),
    ("Dodger Stadium", 330, 4, 330, 4),
    ("Fenway Park", 310, 37, 302, 3),
    ("Globe Life Field", 329, 8, 326, 8),
    ("Great American Ball Park", 328, 12, 325, 8),
    ("Kauffman Stadium", 330, 8, 330, 8),
    ("Nationals Park", 336, 8, 335, 12),
    ("Oracle Park", 339, 8, 309, 25),
    ("Petco Park", 336, 4, 322, 10),
    ("PNC Park", 325, 6, 320, 21),
    ("Progressive Field", 325, 19, 325, 14),
    ("Rate Field", 330, 8, 335, 8),
    ("Rogers Centre", 328, 10, 328, 10),
    ("Steinbrenner Field", 318, None, 314, None),
    ("Sutter Health Park", 330, None, 325, None),
    ("T-Mobile Park", 331, 8, 326, 8),
    ("Target Field", 339, 8, 328, 23),
    ("Truist Park", 335, 6, 325, 16),
    ("Wrigley Field", 355, 16, 353, 16),
    ("Yankee Stadium", 318, 8, 314, 8),
    ("loanDepot park", 344, 7, 335, 7),
)

# Primary club color, bright enough to read on the dark field. Steinbrenner was the Rays' 2025 home.
_MLB_COLORS = {
    "ARI": "#A71930",
    "AZ": "#A71930",
    "ATL": "#CE1141",
    "BAL": "#DF4601",
    "BOS": "#BD3039",
    "CHC": "#0E3386",
    "CWS": "#C4CED4",
    "CIN": "#C6011F",
    "CLE": "#E31937",
    "COL": "#8A4FBF",
    "DET": "#FA4616",
    "HOU": "#EB6E1F",
    "KC": "#004687",
    "LAA": "#BA0021",
    "LAD": "#005A9C",
    "MIA": "#00A3E0",
    "MIL": "#FFC52F",
    "MIN": "#D31145",
    "NYM": "#FF5910",
    "NYY": "#1C4E9A",
    "ATH": "#EFB21E",
    "OAK": "#EFB21E",
    "PHI": "#E81828",
    "PIT": "#FDB827",
    "SD": "#FFC425",
    "SF": "#FD5A1E",
    "SEA": "#0C8A8A",
    "STL": "#C41E3A",
    "TB": "#8FBCE6",
    "TEX": "#C0111F",
    "TOR": "#134A8E",
    "WSH": "#AB0003",
}
_PARK_CLUB = {park: abbr for abbr, park in PARKS.items()}
_PARK_CLUB["Steinbrenner Field"] = "TB"


def fence_lines_figure(side: str) -> go.Figure:
    """The 12 shortest foul lines on one side. Each bar uses that club's color."""
    distance_at = 1 if side == "left" else 3
    wall_at = 2 if side == "left" else 4
    rows = sorted(_FOUL_LINES, key=lambda row: (row[distance_at], row[0]))[:12]
    names = [row[0] for row in rows]
    feet = [row[distance_at] for row in rows]
    walls = ["not listed" if row[wall_at] is None else f"{row[wall_at]:g} ft" for row in rows]
    colors = [_MLB_COLORS.get(_PARK_CLUB.get(name, ""), "#f3efe4") for name in names]
    average = float(numpy.mean([row[distance_at] for row in _FOUL_LINES]))
    figure = go.Figure(
        go.Bar(
            y=names,
            x=feet,
            orientation="h",
            marker=dict(color=colors),
            text=[f"{foot:g}" for foot in feet],
            textposition="outside",
            textfont=dict(color="#f3efe4", size=13),
            customdata=walls,
            hovertemplate="%{y}<br>%{x:.0f} ft from home<br>Wall %{customdata}<extra></extra>",
            cliponaxis=False,
        )
    )
    figure.add_vline(x=average, line_dash="dot", line_color="#f3efe4", line_width=1.2)
    figure.add_annotation(
        x=average,
        y=1,
        yref="paper",
        text=f"Average {average:.0f} ft",
        showarrow=False,
        xanchor="right",
        yanchor="bottom",
        xshift=-6,
        font=dict(color="#f3efe4", size=12),
    )
    # Start just under the shortest porch. Stop at 340 so the average line is not
    # stranded in a long empty stretch.
    shortest = min(feet)
    low = int(numpy.floor((shortest - 8) / 5.0) * 5)
    high = 340
    layout = chart_layout(460)
    layout["margin"] = dict(l=168, r=108, t=36, b=48)
    layout["hovermode"] = "closest"
    layout["hoverlabel"] = dict(
        bgcolor="#2a303a",
        bordercolor="#4c5566",
        font_size=13,
        font_family="Barlow, sans-serif",
        align="left",
    )
    figure.update_layout(**layout)
    figure.update_xaxes(
        title="Feet from home plate",
        range=[low, high],
        dtick=10,
        tickfont=dict(color="#d9d3c5"),
        gridcolor="#3a4150",
    )
    figure.update_yaxes(
        autorange="reversed",
        tickfont=dict(color="#d9d3c5"),
        gridcolor="#3a4150",
    )
    return figure


def _spray_choose_kinds() -> None:
    for kind, _label in _SPRAY_KINDS:
        st.session_state[f"contact_spray_kind_{kind}"] = True


def _spray_kind_label() -> str:
    chosen = [label for kind, label in _SPRAY_KINDS if st.session_state.get(f"contact_spray_kind_{kind}", True)]
    if len(chosen) == len(_SPRAY_KINDS):
        return "All"
    if not chosen:
        return "None"
    if len(chosen) == 1:
        return chosen[0]
    return f"{len(chosen)} types"


def _spray_choose_all() -> None:
    for name in _SPRAY_OPTIONS:
        st.session_state[f"contact_spray_{name}"] = True


def _spray_menu_label() -> str:
    chosen = [name for name in _SPRAY_OPTIONS if st.session_state.get(f"contact_spray_{name}", True)]
    if len(chosen) == len(_SPRAY_OPTIONS):
        return "All"
    if not chosen:
        return "None"
    return ", ".join(chosen)


def _spray_park_rows(spray: pandas.DataFrame) -> list[tuple[str, str, str]]:
    """Park abbreviation, park name, and the checkbox label: park, then club."""
    present = set(spray["park"].astype(str)) if "park" in spray.columns else set()
    rows = []
    for abbr, park in PARKS.items():
        if park in present:
            rows.append((abbr, park, f"{park}, {CLUBS.get(abbr, abbr)}"))
    rows.sort(key=lambda row: row[1].casefold())
    return rows


def _spray_choose_parks(abbrs: tuple[str, ...]) -> None:
    for abbr in abbrs:
        st.session_state[f"contact_spray_park_{abbr}"] = True


def _spray_clear_parks(abbrs: tuple[str, ...]) -> None:
    for abbr in abbrs:
        st.session_state[f"contact_spray_park_{abbr}"] = False


def _spray_park_label(rows: list[tuple[str, str, str]]) -> str:
    chosen = [label for abbr, _park, label in rows if st.session_state.get(f"contact_spray_park_{abbr}", True)]
    if len(chosen) == len(rows):
        return "All parks"
    if not chosen:
        return "No parks"
    if len(chosen) == 1:
        return chosen[0]
    return f"{len(chosen)} parks"


@st.fragment
def _spray_board(spray: pandas.DataFrame) -> None:
    """Filters and the field. A checkbox reruns this piece only, so the rest of the page stays put."""
    spray = spray.copy()
    spray["outcome"] = spray["label"].map(_spray_outcome) if "label" in spray.columns else "Other"
    for spray_name in _SPRAY_OPTIONS:
        st.session_state.setdefault(f"contact_spray_{spray_name}", True)
    for kind, _label in _SPRAY_KINDS:
        st.session_state.setdefault(f"contact_spray_kind_{kind}", True)
    park_rows = _spray_park_rows(spray)
    abbrs = tuple(abbr for abbr, _park, _label in park_rows)
    for abbr in abbrs:
        st.session_state.setdefault(f"contact_spray_park_{abbr}", True)
    result_col, kind_col, park_col = st.columns([1, 1.15, 1.45])
    with result_col:
        st.caption("Results")
        with st.popover(_spray_menu_label(), width="stretch"):
            st.button("All", on_click=_spray_choose_all, key="contact_spray_all")
            for spray_name in _SPRAY_OPTIONS:
                st.checkbox(spray_name, key=f"contact_spray_{spray_name}")
    with kind_col:
        st.caption("Hit type")
        with st.popover(_spray_kind_label(), width="stretch"):
            st.button("All", on_click=_spray_choose_kinds, key="contact_spray_kind_all")
            for kind, label in _SPRAY_KINDS:
                st.checkbox(label, key=f"contact_spray_kind_{kind}")
    with park_col:
        st.caption("Parks")
        with st.popover(_spray_park_label(park_rows), width="stretch"):
            pick, clear = st.columns(2)
            pick.button("All", on_click=_spray_choose_parks, args=(abbrs,), key="contact_spray_park_all")
            clear.button("None", on_click=_spray_clear_parks, args=(abbrs,), key="contact_spray_park_none")
            for abbr, _park, label in park_rows:
                st.checkbox(label, key=f"contact_spray_park_{abbr}")
    spray_choices = [name for name in _SPRAY_OPTIONS if st.session_state.get(f"contact_spray_{name}", True)]
    kind_choices = {kind for kind, _label in _SPRAY_KINDS if st.session_state.get(f"contact_spray_kind_{kind}", True)}
    park_choices = {park for abbr, park, _label in park_rows if st.session_state.get(f"contact_spray_park_{abbr}", True)}
    spray_shown = spray
    if len(spray_choices) != len(_SPRAY_OPTIONS):
        spray_shown = spray_shown.loc[spray_shown["outcome"].isin({_SPRAY_OUTCOME[name] for name in spray_choices})]
    if len(kind_choices) != len(_SPRAY_KINDS):
        spray_shown = spray_shown.loc[spray_shown["kind"].isin(kind_choices)]
    if len(park_choices) != len(park_rows):
        spray_shown = spray_shown.loc[spray_shown["park"].isin(park_choices)]
    st.plotly_chart(
        spray_figure(spray_shown),
        width="stretch",
        key="spray_field",
        config={"displayModeBar": False, "scrollZoom": False, "doubleClick": False},
    )
    outcomes = spray["label"].map(_spray_outcome) if "label" in spray.columns else pandas.Series(dtype=str)
    hr_marks = int(outcomes.eq("Home run").sum())
    triple_marks = int(outcomes.eq("Triple").sum())
    double_marks = int(outcomes.eq("Double").sum())
    single_marks = int(outcomes.eq("Single").sum())
    out_marks = int(outcomes.eq("Out").sum())
    st.caption(
        f"The marks are {len(spray):,} of the scored season's balls in play, picked at random so the field stays readable. "
        f"This graph has {hr_marks:,} home runs and {triple_marks:,} triples, {double_marks:,} doubles, {single_marks:,} singles, and {out_marks:,} outs."
    )


def _forecast_prose(report: ForecastReport) -> str:
    if report.steps.empty:
        return "The season guess needs two seasons in the file. This one has a single year, so there is no finished step to show."
    bits = []
    for row in report.steps.itertuples(index=False):
        bits.append(
            f"{int(row.prior)} contact, pulled toward the league, missed {int(row.nxt)} by {row.mae_contact:.3f} "
            f"across {int(row.pairs)} hitters"
        )
    if len(bits) == 1:
        chain = bits[0]
    else:
        chain = "; ".join(bits[:-1]) + "; and " + bits[-1]
    last = report.steps.iloc[-1]
    coming = int(last.nxt) + 1
    seasons = f"{report.seasons[0]} and {report.seasons[-1]}" if len(report.seasons) == 2 else ", ".join(str(year) for year in report.seasons)
    return (
        f"The file holds {seasons}. Each finished step takes a hitter's contact, the expected weight from exit velocity and launch angle, "
        f"and pulls it toward that season's league average. A hitter with {MIN_BALLS} balls sits halfway to the league. "
        f"Fewer balls pull harder. {chain[0].upper() + chain[1:]}. "
        f"Carrying the results forward instead missed {int(last.nxt)} by {last.mae_results:.3f}. "
        f"Adding half of his home park's gap missed by {last.mae_park:.4f}, against {last.mae_contact:.4f} for contact alone, so the {coming} guess leaves the park in its own column. "
        f"A full season of balls still sits partway toward the league, which is why a great contact year comes out closer to average on the {coming} list. "
        f"Age is opening day {coming}, and this draft leaves age out of the formula. "
        f"Last year's ball count missed the next season by about {report.balls_mae:.0f} balls among hitters who came back. "
        f"Shrinking that count did not tighten the miss, so the guess is a rate. "
        f"The {coming} column uses the same rule on {int(last.nxt)}. "
        f"{_forecast_scale(report, coming)}"
        "The expected weight stored on each ball is what this draft carries forward."
    )


def _forecast_scale(report: ForecastReport, coming: int) -> str:
    earlier = ""
    if report.seasons and min(report.seasons) > 2020:
        earlier = "The file still starts in 2024, so the steps from 2020 through 2023 are not in this count yet. "
    if pandas.isna(report.league_woba) or pandas.isna(report.league_hits):
        return earlier
    return (
        f"The league weight on the latest balls is {report.league_woba:.3f}. "
        f"A {coming} number above that is a better than ordinary contact year. Under it is a soft one. "
        f"Hits per ball is the share of balls in play that became hits. The league share is {report.league_hits:.3f}. "
        f"A card batting average also counts strikeouts. This file starts after the ball is hit, so the two rates use different denominators. "
        f"Singles, doubles, triples, and home runs are last season's counts. "
        f"A {coming} number under last year's results can still sit well above {report.league_woba:.3f}. "
        f"The guess pulls contact toward the league, so a peak season comes out lower. "
        f"The larger rises are hitters whose results trailed the contact. Those seasons also had fewer hits, so the faces sit low and to the right. "
        f"The upper right stays near the cross. A hitter who already turned balls into hits has little room left for the guess to climb. "
        f"{earlier}"
    )


def _guess_gap_copy(study: ContactStudy) -> str:
    """Why the gold line sits under the match, in the words that follow the chart."""
    model = study.calibration.loc[study.calibration["source"].eq("This model")]
    savant = study.calibration.loc[study.calibration["source"].eq("MLB expected")]

    top = model.loc[model["predicted"].idxmax()]
    savant_top = savant.loc[savant["predicted"].idxmax()]
    pile = model.loc[model["band"].eq(0.0)] if "band" in model.columns else model.loc[model["predicted"].le(0.1)]
    return (
        "The gold line is this model. The blue line is MLB expected, and that is a prediction too. "
        "Statcast builds it from how hard the ball left the bat and the angle it left. "
        "The height of either line is the play minus that prediction. "
        "MLB already knows how each of these balls turned out. They publish the prediction because it scores the swing on its own. "
        "A fielder can catch a ball that contact usually turns into a hit, and the stamp on that catch is 0.\n\n"
        "A bigger dot is more balls. Both lines use the same bands, a tenth at a time from 0 to 2, so each line has the same dots. "
        "A call under 0 is drawn at 0. A call over 2 is drawn at 2. A home run stamp is 2, and the chart stops there. "
        f"The crowded dots near an out are the large group: {int(pile['balls'].sum()):,} balls called between 0 and 0.1. "
        f"The last dot is every ball called from 1.9 to 2. This model has {int(top['balls']):,} there, "
        f"called {float(top['predicted']):.3f} against results of {float(top['actual']):.3f}. "
        f"MLB expected has {int(savant_top['balls']):,} in that same band, "
        f"called {float(savant_top['predicted']):.3f} against results of {float(savant_top['actual']):.3f}. "
        f"That dot is not the season's home runs. The {study.test_label} file has 5,645 of those. "
        "Most were called lower, because similar contact still became an out often enough to pull the average down. "
        "Seventy balls had been drawn past 2, where the added-up trees stepped over a home run, and they now sit in that last dot.\n\n"
        "Calls under 0.2 stay on the line. Of those 52,252 balls, 84% were outs. "
        "The call on them averaged 0.083 and the stamps averaged 0.076. A fielder changes very few of them.\n\n"
        "The lines open once the call passes a double. A ball called between 1.2 and 1.8 can still be a home run, an out, or a double. "
        f"In {study.test_label} those 3,772 balls became a home run 55% of the time, an out 24% of the time, and a double 16% of the time. "
        "The call averaged 1.515 and the plays came back at 1.355. "
        f"The trees learned {study.train_label}, when a ball at 100 mph and 20 to 35 degrees averaged {study.hard_prior:.3f}. "
        f"In {study.test_label} that same swing averaged {study.hard_now:.3f}, so the gold line sags through the doubles and the triples.\n\n"
        "The park moves the size of that sag. Across those 3,772 balls the plays came back 0.160 light. "
        "At Globe Life Field, 46% of them were outs. At Petco Park, 42% were outs. "
        "At Dodger Stadium, 10% were outs and 77% left the yard. "
        "The file records which park the ball was hit in. It does not carry a separate grade for the fielders. "
        "A catch is where the park and the defense both show up.\n\n"
        "The line comes back at a home run because a call that high clears the fence almost every time. "
        f"Of the {int(top['balls']):,} balls in that last band, 1,071 were home runs and 29 were outs. "
        "Those 29 are the balls a fielder brought back or that died at the wall. "
        f"They pull the dot under the line: called {float(top['predicted']):.3f} against results of {float(top['actual']):.3f}."
    )


def contact_page() -> None:
    st.html('<p class="back"><a href="/">Home</a></p>')
    if not RAW.exists():
        st.markdown(
            """
            <header class="mast">
              <p class="kicker">Baseball</p>
              <h1>Whose contact was better?</h1>
            </header>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="deck">The page is waiting on the Statcast file. From this folder, run '
            "<code>.\\.venv\\Scripts\\python.exe cache_statcast.py</code> "
            "and reload once <code>data/baseball/statcast_bip.parquet</code> is written.</p>",
            unsafe_allow_html=True,
        )
        return

    study = _cached_study(str(RAW), RAW.stat().st_mtime_ns)
    st.markdown(
        f"""
        <header class="mast">
          <p class="kicker">Baseball, {html.escape(study.train_label)} then {html.escape(study.test_label)}</p>
          <h1>Whose contact was better?</h1>
        </header>
        """,
        unsafe_allow_html=True,
    )
    _prose(
        "contact_deck",
        "wOBA, weighted on-base average, gives each result a weight for the runs it tends to create. "
        "An out is 0. A single is about 0.9. A double is about 1.25. A home run is 2.0. "
        "Over a full season, MLB scales those weights so a league-average wOBA sits near .320, "
        "in the same range as on-base percentage. A .400 season is a star season. A .280 season is a bat the lineup has to hide.",
        "deck",
    )
    _prose(
        "contact_fit",
        f"Statcast already records each of the {study.train_n:,} balls from {study.train_label}. "
        "Two of the recordings are measurements: how hard the ball left the bat, and the angle it left. "
        "The third is the play the fielders made of it, an out, a single, a double, a triple, or a home run. "
        "MLB then stamps that play with a fixed wOBA weight, and that stamp is stored on the ball. "
        "In this file an out is 0, a single is 0.9, a double is 1.25, a triple is 1.6, and a home run is 2. "
        "The same play gets the same weight every time. A single is 0.9 whether it was a soft chopper or a laser. "
        "That weight is the result. It is on the ball before the model runs. "
        "The decision trees come after the stamp. They do not choose the weight. "
        "A tree asks whether the ball was harder than some speed, and whether the angle was higher than some degree, "
        "and it keeps splitting until the balls in a group look alike. "
        "The guess for a new ball is the average stamped weight of the group it lands in. "
        "Walks and strikeouts are not in this file, so the lesson is the value of the contact itself. "
        f"{study.test_label} is already in the file, a full regular season. Those balls stay out of the lesson. "
        f"I then ask the model what each {study.test_label} swing should have been worth, and I compare that guess with the weight already stamped on the ball. "
        f"The average gap between those two numbers is the miss under the counts. That miss is the accuracy check. "
        f"Teaching on {study.test_label} as well would leave no finished season to measure the miss against. "
        f"The file stops at the {study.test_label} regular season. The season after that is the guess at the bottom of the page.",
        "chapter",
    )
    st.markdown('<p class="section-label">How hard, and how high</p>', unsafe_allow_html=True)
    st.plotly_chart(surface_figure(study.surface), width="stretch", config={"displayModeBar": False, "scrollZoom": False, "doubleClick": False})
    st.caption(
        f"Fit on {study.train_label}. Each cell is the model's expected wOBA weight for that exit velocity and launch angle. "
        "This is the contact after the swing. The barrel is the red patch around 100 mph and 25 to 35 degrees, "
        "the band where the home runs sit. It is not a drawing of the strike zone."
    )
    _prose(
        "contact_heat_after",
        "Left to right is how hard the ball came off the bat, and bottom to top is the angle. "
        "Each cell is the wOBA the trees guess for that pair. Dark is weak contact. "
        "The red patch around 110 mph and 30 degrees is where the guess reaches a home run.",
        "chapter",
    )
    hits_zone, all_zone, zone_top, zone_bot = _zone_boards(str(RAW), RAW.stat().st_mtime_ns)
    if hits_zone is not None and not hits_zone.empty and all_zone is not None and not all_zone.empty:
        st.markdown('<p class="section-label">Where the pitch was</p>', unsafe_allow_html=True)
        _prose(
            "contact_zone",
            "This is the strike zone, and only the hits. I dropped the outs. An out weighs 0, and mixing those in dragged every square down. "
            "The box is a typical zone. Across the bottom, the numbers are inches from the middle of the plate, "
            "counted outward on both sides. Inside and outside flip with the hitter, so a minus sign would point the wrong way for half the swings. "
            "Up the chart is the height of the pitch, in feet, as it crosses the plate. "
            "Color is the average weight of the hits from that square. A single is 0.9, a double is 1.25, and a home run is 2. "
            "A square of singles sits near 0.9. Gold and red squares had more doubles and home runs.",
            "chapter",
        )
        _zone_board(hits_zone, all_zone, zone_top, zone_bot)
    st.markdown('<p class="section-label">Does the guess land on the result?</p>', unsafe_allow_html=True)
    _prose(
        "contact_zone_after",
        "The middle of the zone is where those hits carry more weight. "
        "Low and away, a square sits closer to a single.",
        "chapter",
    )
    st.plotly_chart(
        calibration_figure(study.calibration),
        width="stretch",
        config={"displayModeBar": False, "scrollZoom": False, "doubleClick": False},
    )
    st.caption(
        "Expected wOBA is the call, from an out at 0 to a home run at 2. "
        "The height is the result minus that call. Zero, the dotted line, is a match. "
        "A bigger dot is more balls. Both lines use the same bands, so they have the same dots. "
        "A call under 0 or over 2 is held on the scale."
    )
    _prose("contact_check", _guess_gap_copy(study), "chapter")

    st.markdown('<p class="section-label">Where the ball came down</p>', unsafe_allow_html=True)
    _prose(
        "contact_spray_1",
        "Now that we have explored how the balls are hit and rated based off of the exit velocity and angle, we need to see how much a certain other factor effects these balls actual rating. Of course that other factor is the stadium of play. From those balls closest to the 2.0 weight, we judged the average as leaving the bat at about 107 mph and a 27 degree launch angle traveling about 415 feet. Like we said, these balls were absolutely nuked and obviously are a homerun in every park. Except they aren't......these balls fall short of clearing every fence in 4 parks.  That's Fenway, Comercia, Oracle Park, and Coors Field. Obviously I'm looking at only the deepest parts of these fields and if the batter hits that same ball to right or left, it is likely a homerun.",
        "chapter",
    )
    _spray_board(study.spray.copy())
    _prose(
        "contact_spray_2",
        "Some marks sit past the foul lines. A ground ball can pass first or third in fair ground and roll foul, and the dot is where the fielder picked it up. In the strip within 3 degrees of the line, the median ball is 174 feet from home, 27% of them were doubles, and 55% were outs. The home runs and triples drawn there were fair balls. In 2025, 265 home runs sit past the line, and 240 of them are within 3 degrees of it, about 322 feet from home. There are 134 triples past the line, 109 of them within 3 degrees, and those came down about 262 feet out. The line on the chart is a flat 45 degrees, so a ball that hooks around the pole lands a couple of degrees outside it. Farther into foul ground the marks are popups. Past 55 degrees the median spot is 55 feet from home, three quarters of them are popups, and 90% are outs.",
        "chapter",
    )
    _prose(
        "contact_spray_3",
        "The gold line is the average fence. It runs 332 feet down the left field line, 403 feet to center, and 328 feet down the right field line. On balls that traveled 330 to 370 feet and stayed fair, left field was a home run 84% of the time and an out 3% of the time. Right field was 84% and 3%. Center field was an out 46% of the time and a home run 32% of the time. Once the ball carried 370 to 410 feet, center field was a home run 92% of the time, with 7% doubles, and left and right were home runs 99% of the time.",
        "chapter",
    )
    porch_left, porch_right = st.columns(2)
    with porch_left:
        st.markdown(
            '<p class="chart-title" style="text-align:center;color:#f3efe4;font-family:Fraunces,serif;font-size:1.45rem;font-weight:560;letter-spacing:-0.02em;margin:0.35rem auto 0.2rem;">Shortest left fields</p>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            fence_lines_figure("left"),
            width="stretch",
            config={"displayModeBar": False, "scrollZoom": False, "doubleClick": False},
        )
    with porch_right:
        st.markdown(
            '<p class="chart-title" style="text-align:center;color:#f3efe4;font-family:Fraunces,serif;font-size:1.45rem;font-weight:560;letter-spacing:-0.02em;margin:0.35rem auto 0.2rem;">Shortest right fields</p>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            fence_lines_figure("right"),
            width="stretch",
            config={"displayModeBar": False, "scrollZoom": False, "doubleClick": False},
        )
    st.caption(
        "The 12 shortest foul lines in 2025. Each bar is that club's color. "
        "The dotted line is the average foul line on that side, across all 30 parks. "
        "Hover a bar for the wall height."
    )
    st.html(_BAR_HOVER, unsafe_allow_javascript=True)

    st.markdown(
        '<hr class="part-break">'
        '<p class="part-title">The bats the box score buried</p>',
        unsafe_allow_html=True,
    )
    _prose("contact_gaps", _gap_sentence(study.hitters), "reading")
    _prose(
        "contact_hitters",
        "The face is the hitter, with the gray studio backdrop taken off. "
        "A face above the line beat the contact. A face under the line did less with it than the launch said. "
        "All keeps a point for every hitter with 200 balls. Where two faces would cover each other, one face stays and the other hitter remains a dot. "
        "Pick a team and every qualifier on that club gets a face. Hover one of those faces and that hitter draws in front.",
        "chapter",
    )
    _prose(
        "contact_scale",
        f"Results, this model, and MLB expected are the weight of each ball, averaged over the balls he put in play. "
        f"An out is 0. A single is about 0.9. A double is about 1.25. A home run is 2. "
        f"The league on the scored balls is {study.league_woba:.3f}. Above that is a good contact year. Under it is a soft one. "
        f"Hits per ball divides hits by balls in play. The league share is {study.league_hits:.3f}. "
        f"A card batting average also counts strikeouts. This file starts after the ball is hit, so the two rates use different denominators. "
        f"Singles, doubles, triples, and home runs are the counts for {study.test_label}. "
        f"Exit velocity and launch angle move the model. Outs, singles, and extra-base hits move the results. "
        "The gap is this model minus MLB expected.",
        "chapter",
    )
    st.markdown('<p class="section-label">Hitters, per ball in play</p>', unsafe_allow_html=True)
    roster = _attach_clubs(study.hitters)
    club_names = sorted({name for name in roster["club"].tolist() if name})
    picker, _rest = st.columns([1, 2])
    with picker:
        choice = st.selectbox("Team", ["All", *club_names], key="contact_club")
    club = None if choice == "All" else choice
    faces, dots, _bounds = _hitter_layout(roster, club)
    if faces.empty and dots.empty:
        st.markdown(
            f'<p class="chapter">No hitter on {html.escape(choice)} reached {MIN_BALLS} balls in the scored season.</p>',
            unsafe_allow_html=True,
        )
    else:
        st.plotly_chart(
            hitter_figure(roster, club),
            width="stretch",
            config={"displayModeBar": False, "scrollZoom": False, "doubleClick": False},
        )
        st.html(_FACE_LIFT, unsafe_allow_javascript=True)
    if club is None:
        per_team = roster.loc[roster["club"].astype(str).ne("")].groupby("club").size()
        st.caption(
            f"All includes every hitter with at least {MIN_BALLS} balls in play. "
            f"A walk or a strikeout never counts, so clubs on this list run from {int(per_team.min())} to {int(per_team.max())} hitters. "
            f"{len(faces)} faces are drawn. The other {len(dots)} stay as dots where a second face would have covered the first. "
            "Choose a team to see that club's full list. "
            "A hitter who changed clubs is shown with the team they took the most balls with."
        )
    else:
        st.caption(
            f"{choice} shows every hitter on that club with at least {MIN_BALLS} balls in the scored season. "
            "Faces can overlap. The one under the cursor draws in front. "
            "A hitter who changed clubs is listed with the team they took the most balls with."
        )

    st.markdown('<p class="section-label">Above or below the launch</p>', unsafe_allow_html=True)
    _prose(
        "contact_quadrant",
        f"The cross is drawn through zero. "
        f"Right of center, his results beat this model. Left of center, the launch was worth more than the results. "
        f"Above center, more of his balls became hits than the league share of {study.league_hits:.3f}. Below center, fewer did. "
        f"A home run and a single each count once in the hit rate, and the results count the home run as more weight, so a hitter can land in any corner. "
        f"The count under the chart is how many qualifiers sit in each one.",
        "chapter",
    )
    st.plotly_chart(
        contact_cross(roster, club, study.league_hits),
        width="stretch",
        config={"displayModeBar": False, "scrollZoom": False, "doubleClick": False},
    )
    st.html(_FACE_LIFT, unsafe_allow_javascript=True)
    st.dataframe(contact_corners(roster, club, study.league_hits), hide_index=True, width="stretch")
    st.caption(
        "All draws one face where two would cover each other. The rest stay dots. "
        "A team shows every qualifier, and the face under the cursor draws in front."
    )

    table_source = roster if club is None else roster.loc[roster["club"].eq(club)]
    if club is not None:
        table_source = table_source.sort_values("gap", key=lambda column: column.abs(), ascending=False)
    show = (table_source.head(12) if club is None else table_source).loc[
        :, ["name", "club", "bip", "hit_rate", "singles", "doubles", "triples", "homers", "actual", "model", "savant", "gap"]
    ].copy()
    show.columns = ["Hitter", "Team", "Balls", "Hits per ball", "1B", "2B", "3B", "HR", "Results", "This model", "MLB expected", "Gap"]
    for column in ("Hits per ball", "Results", "This model", "MLB expected", "Gap"):
        show[column] = show[column].astype(float).round(3)
    for column in ("1B", "2B", "3B", "HR"):
        show[column] = show[column].astype(int)
    st.dataframe(
        show,
        hide_index=True,
        width="stretch",
        column_config={
            "Balls": st.column_config.NumberColumn(format="%d"),
            "Hits per ball": st.column_config.NumberColumn(format="%.3f"),
            "1B": st.column_config.NumberColumn(format="%d"),
            "2B": st.column_config.NumberColumn(format="%d"),
            "3B": st.column_config.NumberColumn(format="%d"),
            "HR": st.column_config.NumberColumn(format="%d"),
            "Results": st.column_config.NumberColumn(format="%.3f"),
            "This model": st.column_config.NumberColumn(format="%.3f"),
            "MLB expected": st.column_config.NumberColumn(format="%.3f"),
            "Gap": st.column_config.NumberColumn(format="%+.3f"),
        },
    )
    st.caption(
        f"Hits per ball above {study.league_hits:.3f} beat the league share of balls in play that became hits. "
        f"Results above {study.league_woba:.3f} beat the league weight. "
        "1B, 2B, 3B, and HR are the season counts."
    )

    forecast = _cached_forecast(str(RAW), RAW.stat().st_mtime_ns)
    st.markdown('<p class="section-label">Next season</p>', unsafe_allow_html=True)
    _prose("contact_forecast", _forecast_prose(forecast), "chapter")
    if not forecast.steps.empty:
        last_step = forecast.steps.iloc[-1]
        coming = int(last_step.nxt) + 1
        score_cols = st.columns(3)
        score_cols[0].metric(f"{int(last_step.nxt)} from contact", f"{last_step.mae_contact:.3f}")
        score_cols[1].metric(f"{int(last_step.nxt)} from results", f"{last_step.mae_results:.3f}")
        score_cols[2].metric("Hitters in that test", f"{int(last_step.pairs)}")
        st.caption(
            f"Miss is the average distance between the guess and what those hitters actually did in {int(last_step.nxt)}, per ball in play. "
            f"The test keeps hitters with at least {MIN_BALLS} balls in {int(last_step.prior)} who had at least {NEXT_BALLS} the next year."
        )
        board = forecast.board if club is None else forecast.board.loc[forecast.board["club"].eq(club)]
        if board.empty:
            st.markdown(
                f'<p class="chapter">No hitter on {html.escape(choice)} reached {MIN_BALLS} balls for a {coming} guess.</p>',
                unsafe_allow_html=True,
            )
        else:
            st.plotly_chart(
                forecast_figure(board, coming, club, forecast.league_hits),
                width="stretch",
                config={"displayModeBar": False, "scrollZoom": False, "doubleClick": False},
            )
            st.html(_FACE_LIFT, unsafe_allow_javascript=True)
            st.dataframe(forecast_corners(board, None, forecast.league_hits), hide_index=True, width="stretch")
            rises = board.assign(rise=board["guess"] - board["actual"]).nlargest(3, "rise")
            rise_bits = [f"{row['name']} ({row['rise']:+.3f})" for _, row in rises.iterrows() if row["rise"] > 0]
            if len(rise_bits) == 1:
                rise_list = rise_bits[0]
            elif len(rise_bits) == 2:
                rise_list = f"{rise_bits[0]} and {rise_bits[1]}"
            else:
                rise_list = ", ".join(rise_bits[:-1]) + f", and {rise_bits[-1]}" if rise_bits else ""
            rise_sentence = f" The largest rises here are {rise_list}." if rise_list else ""
            st.caption(
                f"Right of center, the {coming} guess is higher than his {int(last_step.nxt)} results. "
                f"Above center, last year's hits per ball beat the league share of {forecast.league_hits:.3f}. That height is last year, not the guess. "
                f"A {coming} number under last year can still be a strong year. The league weight is {forecast.league_woba:.3f}."
                f"{rise_sentence} "
                "All keeps one face where two would cover each other. A team shows every face, and the one under the cursor draws in front."
            )
        show_guess = board.assign(change=board["guess"] - board["actual"]).loc[
            :,
            ["name", "club", "age", "bip", "hit_rate", "singles", "doubles", "triples", "homers", "contact", "actual", "park", "guess", "change"],
        ].copy()
        show_guess.columns = [
            "Hitter",
            "Team",
            "Age",
            "Balls",
            "Hits per ball",
            "1B",
            "2B",
            "3B",
            "HR",
            "Contact",
            "Results",
            "Park",
            str(coming),
            "Change",
        ]
        for column in ("Hits per ball", "Contact", "Results", "Park", str(coming), "Change"):
            show_guess[column] = show_guess[column].astype(float).round(3)
        for column in ("1B", "2B", "3B", "HR"):
            show_guess[column] = show_guess[column].astype(int)
        st.dataframe(
            show_guess,
            hide_index=True,
            width="stretch",
            column_config={
                "Age": st.column_config.NumberColumn(format="%d"),
                "Balls": st.column_config.NumberColumn(format="%d"),
                "Hits per ball": st.column_config.NumberColumn(format="%.3f"),
                "1B": st.column_config.NumberColumn(format="%d"),
                "2B": st.column_config.NumberColumn(format="%d"),
                "3B": st.column_config.NumberColumn(format="%d"),
                "HR": st.column_config.NumberColumn(format="%d"),
                "Contact": st.column_config.NumberColumn(format="%.3f"),
                "Results": st.column_config.NumberColumn(format="%.3f"),
                "Park": st.column_config.NumberColumn(format="%+.3f"),
                str(coming): st.column_config.NumberColumn(format="%.3f"),
                "Change": st.column_config.NumberColumn(format="%+.3f"),
            },
        )
        st.caption(
            f"{coming} uses last season's contact, pulled toward the league weight of {forecast.league_woba:.3f}. "
            "More balls keep the guess closer to his own contact. "
            f"Change is that guess minus the {int(last_step.nxt)} results. "
            f"A negative change on a number above {forecast.league_woba:.3f} is a strong year that comes off a higher one. "
            "Park is the extra weight of balls hit in his home park. "
            f"Age is opening day {coming}. "
            "Hits per ball, singles, doubles, triples, and home runs describe last season."
        )

    st.markdown('<p class="section-label">Still open</p>', unsafe_allow_html=True)
    _prose(
        "contact_open",
        f"The file scored here is {study.train_label}, then {study.test_label}. "
        "The full paper wants every Statcast season from 2015 on, trained on the earlier years and scored on a later year. "
        "Walks and strikeouts stay out of this fit. The number is contact value per ball in play. "
        "Park factors are a later paper on this same file: how much of a park's home-run gap remains after exit velocity and launch angle. "
        "Pitcher velocity is a different cut of Statcast. That one keeps every pitch, not just the balls in play.",
        "chapter",
    )


@st.cache_data(show_spinner="Building the season guesses...")
def _cached_forecast(path: str, mtime_ns: int) -> ForecastReport:
    del mtime_ns
    columns = ["season", "batter", "home_team", "events", "woba_value", "estimated_woba_using_speedangle"]
    return build_forecast(pandas.read_parquet(path, columns=columns))  # hit counts on the guess board


@st.cache_data(show_spinner="Fitting the contact model...")
def _cached_study(path: str, mtime_ns: int) -> ContactStudy:
    del mtime_ns
    return fit_study(pandas.read_parquet(path))  # zone chart averages hits only


if __name__ == "__main__":
    done = fit_study(pandas.read_parquet(RAW))
    print(f"train {done.train_label} n={done.train_n:,}")
    print(f"test  {done.test_label} n={done.test_n:,}")
    print(f"mae model {done.mae_model:.4f}  savant {done.mae_savant:.4f}  grid {done.mae_bins:.4f}")
    print(f"hitter corr model {done.hitter_corr_model:.3f}  savant {done.hitter_corr_savant:.3f}")
    print(done.hitters.head(5).to_string(index=False))
