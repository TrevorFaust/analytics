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
MIN_BALLS = 200
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


def _calibration(predicted: numpy.ndarray, actual: numpy.ndarray, source: str) -> pandas.DataFrame:
    frame = pandas.DataFrame({"predicted": predicted, "actual": actual})
    frame["bin"] = pandas.qcut(frame["predicted"], 10, duplicates="drop")
    grouped = (
        frame.groupby("bin", observed=True)
        .agg(predicted=("predicted", "mean"), actual=("actual", "mean"), balls=("actual", "size"))
        .reset_index(drop=True)
    )
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
    hitters = scored.groupby("batter", as_index=False).agg(
        bip=("woba_value", "size"),
        actual=("woba_value", "mean"),
        model=("model", "mean"),
        savant=("estimated_woba_using_speedangle", "mean"),
    )
    hitters = hitters.loc[hitters["bip"].ge(MIN_BALLS)].copy()
    hitters["name"] = _batter_names(hitters["batter"])
    hitters["gap"] = hitters["model"] - hitters["savant"]
    hitters = hitters.sort_values("gap", key=lambda column: column.abs(), ascending=False)
    facts = _contact_facts(balls)
    spray = _spray_sample(test)
    zone, zone_top, zone_bot = _zone_grid(test)

    calibration = pandas.concat(
        [
            _calibration(model_pred, actual, "This model"),
            _calibration(savant, actual, "Savant"),
            _calibration(bin_pred, actual, "5-by-5 grid"),
        ],
        ignore_index=True,
    )
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
            "x": (frame["hc_x"] - 125.42) * 2.15,
            "y": (198.27 - frame["hc_y"]) * 2.15,
            "kind": kind.to_numpy(),
            "label": labels,
            "park": park.to_numpy(),
        }
    )


def _zone_grid(test: pandas.DataFrame) -> tuple[pandas.DataFrame | None, float, float]:
    if not {"plate_x", "plate_z"}.issubset(test.columns):
        return None, 3.4, 1.6
    frame = test.dropna(subset=["plate_x", "plate_z", "woba_value"]).copy()
    frame = frame.loc[frame["plate_x"].between(-2.2, 2.2) & frame["plate_z"].between(0.5, 4.6)]
    if frame.empty:
        return None, 3.4, 1.6
    frame["xbin"] = (numpy.round(frame["plate_x"] / 0.2) * 0.2).round(2)
    frame["zbin"] = (numpy.round(frame["plate_z"] / 0.2) * 0.2).round(2)
    grouped = (
        frame.groupby(["xbin", "zbin"], as_index=False)
        .agg(woba=("woba_value", "mean"), pitches=("woba_value", "size"))
    )
    grouped = grouped.loc[grouped["pitches"].ge(25)]
    top = float(frame["sz_top"].median()) if "sz_top" in frame.columns and frame["sz_top"].notna().any() else 3.4
    bot = float(frame["sz_bot"].median()) if "sz_bot" in frame.columns and frame["sz_bot"].notna().any() else 1.6
    return grouped, top, bot


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
            zmax=1.6,
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
    colors = {"This model": "#e6b325", "Savant": "#8eb8c4", "5-by-5 grid": "#d08c60"}
    figure = go.Figure()
    ceiling = float(calibration[["predicted", "actual"]].max().max())
    floor = float(calibration[["predicted", "actual"]].min().min())
    figure.add_trace(
        go.Scatter(
            x=[floor, ceiling],
            y=[floor, ceiling],
            mode="lines",
            name="Called it",
            line=dict(color="#b7b2a4", width=1, dash="dot"),
            hoverinfo="skip",
        )
    )
    for source, color in colors.items():
        group = calibration.loc[calibration["source"].eq(source)]
        figure.add_trace(
            go.Scatter(
                x=group["predicted"],
                y=group["actual"],
                mode="lines+markers",
                name=source,
                line=dict(color=color, width=2),
                marker=dict(size=7),
                hovertemplate=f"{source}<br>Predicted %{{x:.3f}}<br>Results %{{y:.3f}}<extra></extra>",
            )
        )
    layout = chart_layout(460)
    layout["margin"] = dict(l=56, r=24, t=40, b=48)
    figure.update_layout(**layout)
    figure.update_xaxes(title="Predicted wOBA weight", tickfont=dict(color="#d9d3c5"), gridcolor="#3a4150")
    figure.update_yaxes(title="Actual wOBA weight", tickfont=dict(color="#d9d3c5"), gridcolor="#3a4150")
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


def _face_rows(pool: pandas.DataFrame, highlighted: set[int]) -> pandas.Series:
    residual = (pool["actual"] - pool["model"]).abs()
    far = residual.ge(float(residual.quantile(0.84)))
    low = float(pool["model"].quantile(0.06))
    high = float(pool["model"].quantile(0.94))
    ends = pool["model"].le(low) | pool["model"].ge(high)
    rings = pool["batter"].isin(highlighted)
    return far | ends | rings


def _shown_hitters(hitters: pandas.DataFrame, club: str | None) -> tuple[pandas.DataFrame, pandas.DataFrame, set[int]]:
    highlighted = set(hitters.head(3)["batter"].tolist())
    pool = hitters if not club else hitters.loc[hitters["club"].eq(club)].copy()
    if club or len(pool) <= 24:
        return pool, pool.iloc[0:0], highlighted
    mask = _face_rows(pool, highlighted)
    return pool.loc[mask].copy(), pool.loc[~mask].copy(), highlighted


def _hitter_hover(frame: pandas.DataFrame) -> tuple[numpy.ndarray, str]:
    club = frame["club"] if "club" in frame.columns else pandas.Series("", index=frame.index)
    custom = numpy.column_stack(
        [
            frame["savant"].to_numpy(),
            frame["bip"].to_numpy(),
            frame["gap"].to_numpy(),
            club.fillna("").to_numpy(),
        ]
    )
    template = (
        "%{text}<br>%{customdata[3]}<br>This model %{x:.3f}<br>Results %{y:.3f}"
        "<br>Savant %{customdata[0]:.3f}<br>Gap %{customdata[2]:+.3f}"
        "<br>%{customdata[1]:.0f} balls<extra></extra>"
    )
    return custom, template


def hitter_figure(hitters: pandas.DataFrame, club: str | None = None) -> go.Figure:
    faces, dots, highlighted = _shown_hitters(hitters, club)
    plotted = pandas.concat([faces, dots], ignore_index=True) if not dots.empty else faces
    if plotted.empty:
        figure = go.Figure()
        figure.update_layout(**chart_layout(420))
        return figure
    xmin = float(plotted["model"].min())
    xmax = float(plotted["model"].max())
    ymin = float(plotted["actual"].min())
    ymax = float(plotted["actual"].max())
    xspan = max(xmax - xmin, 0.05)
    yspan = max(ymax - ymin, 0.05)
    pad_x = xspan * 0.08
    pad_y = yspan * 0.10
    x0, x1 = xmin - pad_x, xmax + pad_x
    y0, y1 = ymin - pad_y, ymax + pad_y
    face_x = xspan / (8 if club else 18)
    face_y = face_x * (xspan / yspan) * 1.55
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
                name="Packed middle",
                marker=dict(size=6, color="#8d8474", opacity=0.8, line=dict(width=0)),
                text=dots["name"],
                customdata=custom,
                hovertemplate=template,
            )
        )
    rings = faces.loc[faces["batter"].isin(highlighted)]
    if not rings.empty:
        figure.add_trace(
            go.Scatter(
                x=rings["model"],
                y=rings["actual"],
                mode="markers",
                marker=dict(size=48 if club else 42, color="rgba(0,0,0,0)", line=dict(color="#e6b325", width=3)),
                showlegend=False,
                hoverinfo="skip",
            )
        )
    if not faces.empty:
        custom, template = _hitter_hover(faces)
        figure.add_trace(
            go.Scatter(
                x=faces["model"],
                y=faces["actual"],
                mode="markers",
                marker=dict(size=18, color="rgba(0,0,0,0)", line=dict(width=0)),
                showlegend=False,
                text=faces["name"],
                customdata=custom,
                hovertemplate=template,
            )
        )
    images = []
    for row in faces.itertuples():
        source = _head_uri(row.batter)
        if source is None:
            continue
        images.append(
            dict(
                source=source,
                xref="x",
                yref="y",
                x=float(row.model),
                y=float(row.actual),
                sizex=face_x,
                sizey=face_y,
                xanchor="center",
                yanchor="middle",
                sizing="contain",
                layer="above",
            )
        )
    layout = chart_layout(620 if club else 680)
    layout["margin"] = dict(l=56, r=24, t=40, b=48)
    layout["images"] = images
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
    return dict(
        type="path",
        path=" ".join(steps),
        fillcolor="#7a5a3c",
        line=dict(width=0),
        layer="below",
    )


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
        group = spray.loc[spray["kind"].eq(kind)]
        if group.empty:
            continue
        if "label" in group.columns and "park" in group.columns:
            custom = numpy.column_stack([group["label"].to_numpy(), group["park"].to_numpy()])
            hover = "%{customdata[0]}<br>%{customdata[1]}<extra></extra>"
        else:
            custom = None
            hover = names[kind]
        figure.add_trace(
            go.Scatter(
                x=group["x"],
                y=group["y"],
                mode="markers",
                name=names[kind],
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
    for trace in _line_pair([0, -420, None, 0, 420], [0, 420, None, 0, 420], "Foul lines", "#f3efe4", 1.6):
        figure.add_trace(trace)
    for trace in _line_pair(dirt_x[dirt_fair], dirt_y[dirt_fair], "Edge of the dirt", "#f0d7b0", 2.2):
        figure.add_trace(trace)
    for trace in _line_pair(330 * numpy.sin(arc), 330 * numpy.cos(arc), "330 ft", "#e6b325", 2.6):
        figure.add_trace(trace)
    for trace in _line_pair([0, base, 0, -base, 0], [0, base, second, base, 0], None, "#f3efe4", 1.8):
        figure.add_trace(trace)
    figure.add_trace(
        go.Scatter(
            x=[base, 0, -base, 0],
            y=[base, second, base, 0],
            mode="markers",
            name="Bases",
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
            marker=dict(symbol="circle", size=14, color="#8d6b4a", line=dict(color="#f0d7b0", width=1)),
            hovertext=["Pitcher's mound, 60 ft 6 in"],
            hoverinfo="text",
        )
    )
    layout = chart_layout(720)
    layout["margin"] = dict(l=48, r=24, t=48, b=36)
    layout["plot_bgcolor"] = "#17352a"
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
        title="Left field to right field",
        range=[-250, 250],
        tickfont=dict(color="#d9d3c5"),
        showgrid=False,
        zeroline=False,
        constrain="domain",
        constraintoward="center",
    )
    figure.update_yaxes(
        title="Feet from home plate",
        range=[-15, 420],
        tickfont=dict(color="#d9d3c5"),
        showgrid=False,
        scaleanchor="x",
        scaleratio=1,
        constrain="domain",
        constraintoward="center",
        zeroline=False,
    )
    return figure


def zone_figure(zone: pandas.DataFrame, top: float, bot: float) -> go.Figure:
    grid = zone.pivot(index="zbin", columns="xbin", values="woba").sort_index()
    inches = numpy.abs(grid.columns.astype(float).to_numpy()) * 12
    custom = numpy.broadcast_to(inches, grid.shape).copy()
    figure = go.Figure(
        go.Heatmap(
            x=grid.columns.astype(float),
            y=grid.index.astype(float),
            z=grid.to_numpy(),
            customdata=custom,
            zmin=0.15,
            zmax=0.7,
            colorscale=[
                [0.0, "#2a303a"],
                [0.35, "#3d6f7a"],
                [0.6, "#e6b325"],
                [1.0, "#d6453d"],
            ],
            colorbar=dict(title="Avg. result", tickfont=dict(color="#d9d3c5"), title_font=dict(color="#d9d3c5")),
            hovertemplate="%{customdata:.0f} in. from the middle<br>%{y:.1f} ft high<br>Average result %{z:.2f}<extra></extra>",
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
        "A positive number means this model values the contact higher than Savant does."
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
    st.markdown(
        '<p class="deck">wOBA, weighted on-base average, gives each result a weight for the runs it tends to create. '
        "An out is 0. A single is about 0.9. A double is about 1.25. A home run is 2.0. "
        "Over a full season, MLB scales those weights so a league-average wOBA sits near .320, "
        "in the same range as on-base percentage. A .400 season is a star season. A .280 season is a bat the lineup has to hide.</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p class="chapter">I take every regular-season ball in play from {html.escape(study.train_label)} '
        "and fit a model on two measurements: how hard the ball came off the bat, and the angle it left, up or down. "
        "The model learns the wOBA weight that kind of contact produced. Walks and strikeouts stay out, "
        f"so the number is the offense in the swing itself. I then score {html.escape(study.test_label)}. "
        "Savant publishes an expected weight from those same two measurements. "
        "The miss under the counts is the average distance between a guess and the weight of the ball that actually happened. "
        f"This model misses by {study.mae_model:.3f}. Savant misses by {study.mae_savant:.3f}. "
        f"A plain grid of 5 mph by 5 degree bins from {html.escape(study.train_label)} misses by {study.mae_bins:.3f}. "
        "One swing is usually an out or a hit, and 0.3 is about the size of that gap. Carry the average forward. "
        f"Among hitters with at least {MIN_BALLS} balls, this model's contact value lines up with the "
        f"{html.escape(study.test_label)} results at {study.hitter_corr_model:.2f}, and Savant's at {study.hitter_corr_savant:.2f}.</p>",
        unsafe_allow_html=True,
    )

    columns = st.columns(4)
    columns[0].metric("Balls fit", f"{study.train_n:,}")
    columns[1].metric("Balls scored", f"{study.test_n:,}")
    columns[2].metric("Model miss", f"{study.mae_model:.3f}")
    columns[3].metric("Savant miss", f"{study.mae_savant:.3f}")
    st.caption(
        "Miss is the average gap between the predicted wOBA weight and the weight of what happened on that ball. "
        "Lower is a closer call. The two models are within a few thousandths of each other."
    )

    st.markdown(
        '<p class="reading">Dark is weak contact. Gold is a hit. Red is the home-run band.</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p class="chapter">Read the sheet left to right for how hard, and bottom to top for how high. '
        f"A home run in these seasons left the bat at a median {study.hr_speed:.0f} mph and {study.hr_angle:.0f} degrees, "
        f"and that event weighs {study.hr_weight:.1f}. "
        f"Line drives, around {study.ld_angle:.0f} degrees, averaged {study.ld_woba:.2f}. "
        f"Ground balls, around {study.gb_angle:.0f} degrees, averaged {study.gb_woba:.2f}. "
        f"Popups, up near {study.pu_angle:.0f} degrees, averaged {study.pu_woba:.3f}. "
        "A hitter who keeps living in the red and the gold band is the one whose numbers should hold up next month. "
        "A hitter whose balls sit on the popup rows, or on weak ground balls, is living on results the contact will have trouble repeating.</p>",
        unsafe_allow_html=True,
    )
    st.markdown('<p class="section-label">How hard, and how high</p>', unsafe_allow_html=True)
    st.plotly_chart(surface_figure(study.surface), width="stretch", config={"displayModeBar": False, "scrollZoom": False, "doubleClick": False})
    st.caption(
        f"Fit on {study.train_label}. Each cell is the model's expected wOBA weight for that exit velocity and launch angle. "
        "This is the contact after the swing. The barrel is the red patch around 100 mph and 25 to 35 degrees, "
        "the band where the home runs sit. It is not a drawing of the strike zone."
    )

    st.markdown('<p class="section-label">Where the ball came down</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="chapter">Same idea, drawn on the field. I sampled '
        f"{html.escape(study.test_label)} balls in play so the grass stays readable. "
        "Line drives fill the gaps. Home runs clear the 330-foot arc. Popups drop around the bases. "
        "The sheet above is the quality of the contact. This drawing is where that contact came down: "
        "a ball in the seats, a ball at an infielder, or a ball through the hole. "
        "Hover a mark for the result and the ballpark. "
        "A fly ball that dies on the track in Detroit can leave the yard in Cincinnati.</p>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(spray_figure(study.spray), width="stretch", config={"displayModeBar": False, "scrollZoom": False, "doubleClick": False})
    st.caption(
        "Home plate is at the bottom. The bases are 90 feet apart, and second base is 127 feet from home. "
        "The pale arc is the edge of the dirt, about 155 feet from home, past the bases. "
        "The gold arc is 330 feet. Left field is to the left."
    )

    if study.zone is not None and not study.zone.empty:
        st.markdown('<p class="section-label">Where the pitch was</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="chapter">This is the strike zone, and only the pitches the hitter put in play. '
            "The box is a typical zone. Across the bottom, the numbers are inches from the middle of the plate, "
            "counted outward on both sides. Inside and outside flip with the hitter, so a minus sign would point the wrong way for half the swings. "
            "Up the chart is the height of the pitch, in feet, as it crosses the plate. "
            "Color is the average result of the balls put in play from that square, on the same wOBA weight as the rest of the page. "
            "An out is 0, a single is about 0.9, and a home run is 2.0. "
            "A square averages many balls, so it lands between those: near 0.25 when they became outs, near 0.55 when more of them became hits. "
            "Gold and red squares are where that contact turned into hits. Dark squares mostly became outs.</p>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            zone_figure(study.zone, study.zone_top, study.zone_bot),
            width="stretch",
            config={"displayModeBar": False, "scrollZoom": False, "doubleClick": False},
        )
        st.caption(
            "Each cell keeps a spot with at least 25 balls in play. The outline is the median top and bottom of the zone. "
            "The color bar is the average result in that square."
        )

    st.markdown(
        f'<p class="chapter">On {study.test_n:,} balls from {html.escape(study.test_label)}, '
        "group the guesses into tenths and compare each tenth with the results. "
        "A model that knew the future would sit on the dotted line. "
        f"These three stay close to it, which is why the hitter averages above are usable and a single swing is not.</p>",
        unsafe_allow_html=True,
    )
    st.markdown('<p class="section-label">Does the guess land on the result?</p>', unsafe_allow_html=True)
    st.plotly_chart(calibration_figure(study.calibration), width="stretch", config={"displayModeBar": False, "scrollZoom": False, "doubleClick": False})
    st.caption("Each point is a tenth of the scored balls, grouped by that model's own prediction.")

    st.markdown(
        f'<p class="reading">{html.escape(_gap_sentence(study.hitters))}</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="chapter">The face is the hitter, with the gray studio backdrop taken off. '
        "A face above the line beat the contact. A face under the line did less with it than the launch said. "
        "Gold rings mark the three widest gaps between this model and Savant. "
        "All keeps every hitter with 200 balls. I draw the packed middle as dots, and leave faces on the hitters away from that line. "
        "Pick a team and every qualifier on that club gets a face. Hover a face for the model, Savant, and the result.</p>",
        unsafe_allow_html=True,
    )
    st.markdown('<p class="section-label">Hitters, per ball in play</p>', unsafe_allow_html=True)
    roster = _attach_clubs(study.hitters)
    club_names = sorted({name for name in roster["club"].tolist() if name})
    picker, _rest = st.columns([1, 2])
    with picker:
        choice = st.selectbox("Team", ["All", *club_names], key="contact_club")
    club = None if choice == "All" else choice
    faces, dots, _rings = _shown_hitters(roster, club)
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
    if club is None:
        st.caption(
            f"All includes every hitter with at least {MIN_BALLS} balls. "
            f"{len(faces)} of them are faces: the ones off the packed middle, the ends of the cloud, and the three gold rings. "
            f"The other {len(dots)} stay as dots. Choose a team to see that club's full list as faces. "
            "A hitter who changed clubs is shown with the team they took the most balls with."
        )
    else:
        st.caption(
            f"{choice} shows every hitter on that club with at least {MIN_BALLS} balls in the scored season. "
            "A hitter who changed clubs is listed with the team they took the most balls with."
        )

    table_source = roster if club is None else roster.loc[roster["club"].eq(club)]
    if club is not None:
        table_source = table_source.sort_values("gap", key=lambda column: column.abs(), ascending=False)
    show = (table_source.head(12) if club is None else table_source).loc[:, ["name", "club", "bip", "actual", "model", "savant", "gap"]].copy()
    show.columns = ["Hitter", "Team", "Balls", "Results", "This model", "Savant", "Gap"]
    for column in ("Results", "This model", "Savant", "Gap"):
        show[column] = show[column].astype(float).round(3)
    st.dataframe(
        show,
        hide_index=True,
        width="stretch",
        column_config={
            "Balls": st.column_config.NumberColumn(format="%d"),
            "Results": st.column_config.NumberColumn(format="%.3f"),
            "This model": st.column_config.NumberColumn(format="%.3f"),
            "Savant": st.column_config.NumberColumn(format="%.3f"),
            "Gap": st.column_config.NumberColumn(format="%+.3f"),
        },
    )

    st.markdown('<p class="section-label">Still open</p>', unsafe_allow_html=True)
    st.markdown(
        f'<p class="chapter">The file scored here is {html.escape(study.train_label)}, then {html.escape(study.test_label)}. '
        "The full paper wants every Statcast season from 2015 on, trained on the earlier years and scored on a later year. "
        "Walks and strikeouts stay out of this fit. The number is contact value per ball in play. "
        "Park factors are a later paper on this same file: how much of a park's home-run gap remains after exit velocity and launch angle. "
        "Pitcher velocity is a different cut of Statcast. That one keeps every pitch, not just the balls in play.</p>",
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner="Fitting the contact model...")
def _cached_study(path: str, mtime_ns: int) -> ContactStudy:
    del mtime_ns
    return fit_study(pandas.read_parquet(path))  # parks on the spray, transparent faces, team filter


if __name__ == "__main__":
    done = fit_study(pandas.read_parquet(RAW))
    print(f"train {done.train_label} n={done.train_n:,}")
    print(f"test  {done.test_label} n={done.test_n:,}")
    print(f"mae model {done.mae_model:.4f}  savant {done.mae_savant:.4f}  grid {done.mae_bins:.4f}")
    print(f"hitter corr model {done.hitter_corr_model:.3f}  savant {done.hitter_corr_savant:.3f}")
    print(done.hitters.head(5).to_string(index=False))
