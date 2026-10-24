"""Download the NBA tables behind the team and player logs page.

One call per season for each file, saved under data/nba:

- team_gamelogs: every regular-season team game
- player_gamelogs: every player appearance, for rest, minutes, and shooting
- player_positions: roster position for that team and season
- player_advanced: per-game advanced rates, including age and usage
- team_shot_zones and player_shot_zones: makes, attempts, and
  field-goal percentage by shot zone

Seasons already on disk are left alone. Pass --refresh to pull the
current season again. Pass --seasons 2024-25 to limit the run.
"""

from __future__ import annotations

import sys
import time
from datetime import date
from pathlib import Path

import pandas
from nba_api.stats.endpoints import (
    CommonTeamRoster,
    LeagueDashPlayerShotLocations,
    LeagueDashPlayerStats,
    LeagueDashTeamShotLocations,
    LeagueGameLog,
    PlayerGameLogs,
)
from nba_api.stats.static import teams as static_teams

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data" / "nba"
FIRST_SEASON = 2021
PAUSE_SECONDS = 0.6
TIMEOUT = 60

SHOT_ZONES = [
    "Restricted Area",
    "In The Paint (Non-RA)",
    "Mid-Range",
    "Left Corner 3",
    "Right Corner 3",
    "Above the Break 3",
    "Corner 3",
    "Backcourt",
]

ADVANCED_COLUMNS = [
    "PLAYER_ID",
    "PLAYER_NAME",
    "TEAM_ID",
    "TEAM_ABBREVIATION",
    "AGE",
    "GP",
    "W",
    "L",
    "MIN",
    "OFF_RATING",
    "DEF_RATING",
    "NET_RATING",
    "AST_PCT",
    "AST_TO",
    "OREB_PCT",
    "DREB_PCT",
    "REB_PCT",
    "TM_TOV_PCT",
    "EFG_PCT",
    "TS_PCT",
    "USG_PCT",
    "PACE",
    "PIE",
    "POSS",
]


def current_season_start(today: date | None = None) -> int:
    today = today or date.today()
    if today.month >= 10:
        return today.year
    return today.year - 1


def season_label(start: int) -> str:
    return f"{start}-{str(start + 1)[-2:]}"


def default_seasons(today: date | None = None) -> list[str]:
    last = current_season_start(today)
    return [season_label(year) for year in range(FIRST_SEASON, last + 1)]


def abbreviation_by_id() -> dict[int, str]:
    return {int(row["id"]): row["abbreviation"] for row in static_teams.get_teams()}


def fetch(build, label: str):
    last_error = None
    for attempt in range(3):
        try:
            return build()
        except Exception as exc:
            last_error = exc
            wait = PAUSE_SECONDS * (attempt + 2)
            print(f"{label} failed ({exc}). Retrying in {wait:.0f}s.", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"{label} failed after 3 tries: {last_error}")


def as_string(frame: pandas.DataFrame, columns: list[str]) -> pandas.DataFrame:
    for column in columns:
        if column in frame.columns:
            frame[column] = frame[column].astype("string")
    return frame


def split_matchup(matchup: str) -> tuple[str, str]:
    text = "" if pandas.isna(matchup) else str(matchup)
    if " vs. " in text:
        return "Home", text.split(" vs. ", 1)[1]
    if " @ " in text:
        return "Away", text.split(" @ ", 1)[1]
    return "", ""


def team_gamelog(season: str) -> pandas.DataFrame:
    endpoint = fetch(
        lambda: LeagueGameLog(
            season=season,
            player_or_team_abbreviation="T",
            season_type_all_star="Regular Season",
            timeout=TIMEOUT,
        ),
        f"{season} team game log",
    )
    frame = endpoint.league_game_log.get_data_frame()
    if frame.empty:
        return frame
    sites, opponents = zip(*(split_matchup(value) for value in frame["MATCHUP"]))
    frame.insert(0, "SEASON", season)
    frame["SITE"] = list(sites)
    frame["OPPONENT"] = list(opponents)
    frame["GAME_DATE"] = pandas.to_datetime(frame["GAME_DATE"])
    frame["OPP_PTS"] = frame["PTS"] - frame["PLUS_MINUS"]
    return as_string(frame, ["TEAM_ABBREVIATION", "TEAM_NAME", "GAME_ID", "MATCHUP", "WL", "SITE", "OPPONENT"])


def player_advanced(season: str) -> pandas.DataFrame:
    endpoint = fetch(
        lambda: LeagueDashPlayerStats(
            season=season,
            measure_type_detailed_defense="Advanced",
            per_mode_detailed="PerGame",
            season_type_all_star="Regular Season",
            timeout=TIMEOUT,
        ),
        f"{season} player advanced",
    )
    frame = endpoint.league_dash_player_stats.get_data_frame()
    if frame.empty:
        return frame
    keep = [column for column in ADVANCED_COLUMNS if column in frame.columns]
    frame = frame.loc[:, keep].copy()
    frame.insert(0, "SEASON", season)
    return as_string(frame, ["PLAYER_NAME", "TEAM_ABBREVIATION"])


def flatten_shots(frame: pandas.DataFrame, id_columns: list[str], season: str) -> pandas.DataFrame:
    if isinstance(frame.columns, pandas.MultiIndex):
        frame = frame.copy()
        frame.columns = [
            bottom if top == "" else f"{top}|{bottom}"
            for top, bottom in frame.columns
        ]
    pieces = []
    for zone in SHOT_ZONES:
        made = f"{zone}|FGM"
        if made not in frame.columns:
            continue
        piece = frame.loc[:, [column for column in id_columns if column in frame.columns]].copy()
        piece["ZONE"] = zone
        piece["FGM"] = pandas.to_numeric(frame[made], errors="coerce")
        piece["FGA"] = pandas.to_numeric(frame[f"{zone}|FGA"], errors="coerce")
        piece["FG_PCT"] = pandas.to_numeric(frame[f"{zone}|FG_PCT"], errors="coerce")
        pieces.append(piece)
    if not pieces:
        return pandas.DataFrame()
    out = pandas.concat(pieces, ignore_index=True)
    out.insert(0, "SEASON", season)
    return out


def team_shot_zones(season: str, abbreviations: dict[int, str]) -> pandas.DataFrame:
    endpoint = fetch(
        lambda: LeagueDashTeamShotLocations(
            season=season,
            per_mode_detailed="PerGame",
            season_type_all_star="Regular Season",
            timeout=TIMEOUT,
        ),
        f"{season} team shot zones",
    )
    frame = flatten_shots(endpoint.shot_locations.get_data_frame(), ["TEAM_ID", "TEAM_NAME"], season)
    if frame.empty:
        return frame
    frame["TEAM_ABBREVIATION"] = frame["TEAM_ID"].map(lambda value: abbreviations.get(int(value), ""))
    return as_string(frame, ["TEAM_NAME", "TEAM_ABBREVIATION", "ZONE"])


PLAYER_LOG_COLUMNS = [
    "PLAYER_ID",
    "PLAYER_NAME",
    "TEAM_ID",
    "TEAM_ABBREVIATION",
    "GAME_ID",
    "GAME_DATE",
    "MATCHUP",
    "WL",
    "MIN",
    "FGM",
    "FGA",
    "FG3M",
    "FG3A",
    "FTM",
    "FTA",
    "OREB",
    "TOV",
    "REB",
    "AST",
    "PTS",
    "PLUS_MINUS",
]


def player_gamelog(season: str) -> pandas.DataFrame:
    endpoint = fetch(
        lambda: PlayerGameLogs(
            season_nullable=season,
            season_type_nullable="Regular Season",
            timeout=TIMEOUT,
        ),
        f"{season} player game log",
    )
    frame = endpoint.player_game_logs.get_data_frame()
    if frame.empty:
        return frame
    keep = [column for column in PLAYER_LOG_COLUMNS if column in frame.columns]
    frame = frame.loc[:, keep].copy()
    frame.insert(0, "SEASON", season)
    frame["GAME_DATE"] = pandas.to_datetime(frame["GAME_DATE"])
    sites, opponents = zip(*(split_matchup(value) for value in frame["MATCHUP"]))
    frame["SITE"] = list(sites)
    frame["OPPONENT"] = list(opponents)
    return as_string(frame, ["PLAYER_NAME", "TEAM_ABBREVIATION", "GAME_ID", "MATCHUP", "WL", "SITE", "OPPONENT"])


def player_positions(season: str) -> pandas.DataFrame:
    frames = []
    for club in static_teams.get_teams():
        team_id = int(club["id"])
        endpoint = fetch(
            lambda team_id=team_id: CommonTeamRoster(team_id=team_id, season=season, timeout=TIMEOUT),
            f"{season} roster {club['abbreviation']}",
        )
        frame = endpoint.common_team_roster.get_data_frame()
        if not frame.empty:
            frames.append(frame)
        time.sleep(PAUSE_SECONDS)
    if not frames:
        return pandas.DataFrame()
    roster = pandas.concat(frames, ignore_index=True)
    roster = roster.rename(columns={"TeamID": "TEAM_ID", "PLAYER": "PLAYER_NAME"})
    keep = [column for column in ("TEAM_ID", "PLAYER_ID", "PLAYER_NAME", "POSITION", "AGE") if column in roster.columns]
    roster = roster.loc[:, keep].copy()
    roster.insert(0, "SEASON", season)
    return as_string(roster, ["PLAYER_NAME", "POSITION"])


def player_shot_zones(season: str) -> pandas.DataFrame:
    endpoint = fetch(
        lambda: LeagueDashPlayerShotLocations(
            season=season,
            per_mode_detailed="PerGame",
            season_type_all_star="Regular Season",
            timeout=TIMEOUT,
        ),
        f"{season} player shot zones",
    )
    frame = flatten_shots(
        endpoint.shot_locations.get_data_frame(),
        ["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION", "AGE"],
        season,
    )
    if frame.empty:
        return frame
    return as_string(frame, ["PLAYER_NAME", "TEAM_ABBREVIATION", "ZONE"])


DATASETS = {
    "team_gamelogs": team_gamelog,
    "player_advanced": player_advanced,
    "team_shot_zones": team_shot_zones,
    "player_shot_zones": player_shot_zones,
}


def season_path(dataset: str, season: str) -> Path:
    return CACHE / dataset / f"{season}.parquet"


def save_season(dataset: str, season: str, frame: pandas.DataFrame) -> None:
    path = season_path(dataset, season)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    print(f"{dataset} {season}: {len(frame):,} rows -> {path.relative_to(ROOT)}", flush=True)


def cache_season(season: str, abbreviations: dict[int, str], refresh: bool) -> None:
    jobs = [
        ("team_gamelogs", lambda: team_gamelog(season)),
        ("player_advanced", lambda: player_advanced(season)),
        ("player_gamelogs", lambda: player_gamelog(season)),
        ("player_positions", lambda: player_positions(season)),
        ("team_shot_zones", lambda: team_shot_zones(season, abbreviations)),
        ("player_shot_zones", lambda: player_shot_zones(season)),
    ]
    for name, build in jobs:
        path = season_path(name, season)
        if path.exists() and not refresh:
            print(f"{name} {season} already saved.", flush=True)
            continue
        frame = build()
        if frame.empty:
            print(f"{name} {season}: empty, left unsaved.", flush=True)
        else:
            save_season(name, season, frame)
        time.sleep(PAUSE_SECONDS)


def requested_seasons(argv: list[str], refresh: bool) -> list[str]:
    if "--seasons" in argv:
        start = argv.index("--seasons") + 1
        labels = []
        for token in argv[start:]:
            if token.startswith("--"):
                break
            labels.append(token)
        if not labels:
            raise SystemExit("Pass at least one season after --seasons, like 2024-25.")
        return labels
    labels = default_seasons()
    if refresh:
        return [labels[-1]]
    return labels


def main() -> None:
    refresh = "--refresh" in sys.argv
    labels = requested_seasons(sys.argv, refresh)
    abbreviations = abbreviation_by_id()
    print(f"Seasons: {', '.join(labels)}", flush=True)
    print(f"Cache folder: {CACHE}", flush=True)
    for season in labels:
        print(f"Season {season}", flush=True)
        cache_season(season, abbreviations, refresh)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
