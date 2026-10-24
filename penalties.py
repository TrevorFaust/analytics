"""Build team-game and penalty tables from the cached play-by-play files.

A penalty row is one play with penalty == 1. nflverse keeps a single
penalty_team and penalty_type on that play. Penalties per game divides
those rows by games played, including games with no flags.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas
import pyarrow.dataset as ds

ROOT = Path(__file__).resolve().parent
PBP_PATH = ROOT / "data" / "cache" / "pbp"
TABLES = ROOT / "data" / "tables"

PLAY_COLUMNS = [
    "season",
    "week",
    "season_type",
    "game_id",
    "play_id",
    "home_team",
    "away_team",
    "posteam",
    "defteam",
    "penalty",
    "penalty_team",
    "penalty_type",
    "penalty_yards",
    "qtr",
    "game_date",
]

TEAM_NAMES = {
    "ARI": "Cardinals",
    "ATL": "Falcons",
    "BAL": "Ravens",
    "BUF": "Bills",
    "CAR": "Panthers",
    "CHI": "Bears",
    "CIN": "Bengals",
    "CLE": "Browns",
    "DAL": "Cowboys",
    "DEN": "Broncos",
    "DET": "Lions",
    "GB": "Packers",
    "HOU": "Texans",
    "IND": "Colts",
    "JAX": "Jaguars",
    "KC": "Chiefs",
    "LA": "Rams",
    "LAC": "Chargers",
    "LV": "Raiders",
    "MIA": "Dolphins",
    "MIN": "Vikings",
    "NE": "Patriots",
    "NO": "Saints",
    "NYG": "Giants",
    "NYJ": "Jets",
    "PHI": "Eagles",
    "PIT": "Steelers",
    "SEA": "Seahawks",
    "SF": "49ers",
    "TB": "Buccaneers",
    "TEN": "Titans",
    "WAS": "Commanders",
}


@dataclass(frozen=True)
class Filters:
    season_min: int
    season_max: int
    week_min: int
    week_max: int
    season_types: tuple[str, ...]
    teams: tuple[str, ...] = ()
    sides: tuple[str, ...] = ()
    sites: tuple[str, ...] = ()
    penalty_types: tuple[str, ...] = ()


def load_plays(seasons: list[int] | None = None) -> pandas.DataFrame:
    dataset = ds.dataset(PBP_PATH, format="parquet", partitioning="hive")
    season_filter = ds.field("season").isin(seasons) if seasons else None
    plays = dataset.to_table(columns=PLAY_COLUMNS, filter=season_filter).to_pandas()
    plays = plays.dropna(subset=["game_id", "play_id", "home_team", "away_team"])
    plays = plays.drop_duplicates(subset=["game_id", "play_id"])
    plays["season"] = plays["season"].astype(int)
    plays["week"] = plays["week"].astype(int)
    return plays


def build_games(plays: pandas.DataFrame) -> pandas.DataFrame:
    base = plays.loc[
        :, ["season", "week", "season_type", "game_id", "home_team", "away_team", "game_date"]
    ].drop_duplicates(subset=["game_id"])
    final_qtr = plays.groupby("game_id")["qtr"].max()
    base["final_qtr"] = base["game_id"].map(final_qtr).round().astype("Int64")
    home = base.rename(columns={"home_team": "team", "away_team": "opponent"})
    home["site"] = "Home"
    away = base.rename(columns={"away_team": "team", "home_team": "opponent"})
    away["site"] = "Away"
    games = pandas.concat([home, away], ignore_index=True)
    columns = [
        "season",
        "week",
        "season_type",
        "game_id",
        "game_date",
        "final_qtr",
        "team",
        "opponent",
        "site",
    ]
    return games.loc[:, columns].sort_values(["season", "week", "game_id", "team"])


def build_penalty_plays(plays: pandas.DataFrame) -> pandas.DataFrame:
    flagged = plays.loc[plays["penalty"].eq(1)].copy()
    flagged = flagged.dropna(subset=["penalty_team", "penalty_type"])
    at_home = flagged["penalty_team"].eq(flagged["home_team"])
    flagged["team"] = flagged["penalty_team"]
    flagged["opponent"] = flagged["away_team"].where(at_home, flagged["home_team"])
    flagged["site"] = "Away"
    flagged.loc[at_home, "site"] = "Home"
    flagged["side"] = "Other"
    flagged.loc[flagged["penalty_team"].eq(flagged["posteam"]), "side"] = "Offense"
    flagged.loc[flagged["penalty_team"].eq(flagged["defteam"]), "side"] = "Defense"
    flagged["penalty_yards"] = flagged["penalty_yards"].fillna(0)
    flagged["qtr"] = flagged["qtr"].round().astype("Int64")
    columns = [
        "season",
        "week",
        "season_type",
        "game_id",
        "game_date",
        "qtr",
        "team",
        "opponent",
        "site",
        "side",
        "penalty_type",
        "penalty_yards",
    ]
    return flagged.loc[:, columns].sort_values(["season", "week", "game_id", "team"])


def build_tables() -> None:
    plays = load_plays()
    games = build_games(plays)
    penalties = build_penalty_plays(plays)
    TABLES.mkdir(parents=True, exist_ok=True)
    games.to_parquet(TABLES / "games.parquet", index=False)
    penalties.to_parquet(TABLES / "penalties.parquet", index=False)
    print(f"Games: {len(games):,} rows -> data/tables/games.parquet")
    print(f"Penalties: {len(penalties):,} rows -> data/tables/penalties.parquet")


def replace_season(year: int) -> None:
    """Swap one season in the built tables. Earlier seasons stay as they are."""
    plays = load_plays([year])
    new_games = build_games(plays)
    new_penalties = build_penalty_plays(plays)
    games_path = TABLES / "games.parquet"
    penalties_path = TABLES / "penalties.parquet"
    if games_path.exists() and penalties_path.exists():
        games = pandas.read_parquet(games_path)
        penalties = pandas.read_parquet(penalties_path)
        games = pandas.concat([games.loc[games["season"].ne(year)], new_games], ignore_index=True)
        penalties = pandas.concat(
            [penalties.loc[penalties["season"].ne(year)], new_penalties],
            ignore_index=True,
        )
    else:
        games = new_games
        penalties = new_penalties
    TABLES.mkdir(parents=True, exist_ok=True)
    games.sort_values(["season", "week", "game_id", "team"]).to_parquet(games_path, index=False)
    penalties.sort_values(["season", "week", "game_id", "team"]).to_parquet(penalties_path, index=False)
    print(f"Replaced {year}: {len(new_games):,} team-games, {len(new_penalties):,} flags.")


def ensure_tables() -> None:
    games_path = TABLES / "games.parquet"
    penalties_path = TABLES / "penalties.parquet"
    if not games_path.exists() or not penalties_path.exists():
        build_tables()


def load_frames() -> tuple[pandas.DataFrame, pandas.DataFrame]:
    ensure_tables()
    games = pandas.read_parquet(TABLES / "games.parquet")
    penalties = pandas.read_parquet(TABLES / "penalties.parquet")
    return games, penalties


def subset(
    games: pandas.DataFrame,
    penalties: pandas.DataFrame,
    filters: Filters,
) -> tuple[pandas.DataFrame, pandas.DataFrame]:
    games = games.loc[
        games["season"].between(filters.season_min, filters.season_max)
        & games["week"].between(filters.week_min, filters.week_max)
        & games["season_type"].isin(filters.season_types)
    ]
    penalties = penalties.loc[
        penalties["season"].between(filters.season_min, filters.season_max)
        & penalties["week"].between(filters.week_min, filters.week_max)
        & penalties["season_type"].isin(filters.season_types)
    ]
    if filters.sites:
        games = games.loc[games["site"].isin(filters.sites)]
        penalties = penalties.loc[penalties["site"].isin(filters.sites)]
    if filters.teams:
        games = games.loc[games["team"].isin(filters.teams)]
        penalties = penalties.loc[penalties["team"].isin(filters.teams)]
    if filters.sides:
        penalties = penalties.loc[penalties["side"].isin(filters.sides)]
    if filters.penalty_types:
        penalties = penalties.loc[penalties["penalty_type"].isin(filters.penalty_types)]
    return games.copy(), penalties.copy()


def team_summary(games: pandas.DataFrame, penalties: pandas.DataFrame) -> pandas.DataFrame:
    played = (
        games.drop_duplicates(["game_id", "team"])
        .groupby("team", as_index=False)
        .agg(games=("game_id", "nunique"))
    )
    counted = penalties.groupby("team", as_index=False).agg(
        penalties=("penalty_type", "size"),
        yards=("penalty_yards", "sum"),
    )
    if penalties.empty:
        common = pandas.DataFrame(columns=["team", "most_common"])
    else:
        common = (
            penalties.groupby("team")["penalty_type"]
            .agg(lambda values: values.value_counts().index[0])
            .rename("most_common")
            .reset_index()
        )
    summary = played.merge(counted, on="team", how="left").merge(common, on="team", how="left")
    summary["penalties"] = summary["penalties"].fillna(0).astype(int)
    summary["yards"] = summary["yards"].fillna(0.0)
    summary["most_common"] = summary["most_common"].fillna("")
    summary["penalties_per_game"] = summary["penalties"] / summary["games"]
    summary["yards_per_game"] = summary["yards"] / summary["games"]
    summary["name"] = summary["team"].map(TEAM_NAMES).fillna(summary["team"])
    return summary


def rates_by_season(games: pandas.DataFrame, penalties: pandas.DataFrame) -> pandas.DataFrame:
    played = (
        games.drop_duplicates(["game_id", "team"])
        .groupby("season", as_index=False)
        .size()
        .rename(columns={"size": "games"})
    )
    counted = penalties.groupby("season", as_index=False).agg(
        penalties=("penalty_type", "size"),
        yards=("penalty_yards", "sum"),
    )
    trend = played.merge(counted, on="season", how="left")
    trend["penalties"] = trend["penalties"].fillna(0).astype(int)
    trend["yards"] = trend["yards"].fillna(0.0)
    trend["penalties_per_game"] = trend["penalties"] / trend["games"]
    trend["yards_per_game"] = trend["yards"] / trend["games"]
    return trend.sort_values("season")


def rates_by_team_season(
    games: pandas.DataFrame, penalties: pandas.DataFrame
) -> pandas.DataFrame:
    played = (
        games.drop_duplicates(["season", "game_id", "team"])
        .groupby(["season", "team"], as_index=False)
        .agg(games=("game_id", "nunique"))
    )
    counted = penalties.groupby(["season", "team"], as_index=False).agg(
        penalties=("penalty_type", "size"),
        yards=("penalty_yards", "sum"),
    )
    trend = played.merge(counted, on=["season", "team"], how="left")
    trend["penalties"] = trend["penalties"].fillna(0).astype(int)
    trend["yards"] = trend["yards"].fillna(0.0)
    trend["penalties_per_game"] = trend["penalties"] / trend["games"]
    trend["yards_per_game"] = trend["yards"] / trend["games"]
    trend["name"] = trend["team"].map(TEAM_NAMES).fillna(trend["team"])
    return trend


def type_summary(penalties: pandas.DataFrame, team_games: int) -> pandas.DataFrame:
    if penalties.empty or team_games == 0:
        return pandas.DataFrame(
            columns=["penalty_type", "penalties", "yards", "penalties_per_game", "yards_per_game"]
        )
    summary = penalties.groupby("penalty_type", as_index=False).agg(
        penalties=("penalty_type", "size"),
        yards=("penalty_yards", "sum"),
    )
    summary["penalties_per_game"] = summary["penalties"] / team_games
    summary["yards_per_game"] = summary["yards"] / team_games
    return summary


def team_games_count(games: pandas.DataFrame) -> int:
    if games.empty:
        return 0
    return int(games.drop_duplicates(["game_id", "team"]).shape[0])


def per_game_rate(games: pandas.DataFrame, penalties: pandas.DataFrame, metric: str) -> float | None:
    played = team_games_count(games)
    if played == 0:
        return None
    if metric == "yards_per_game":
        return float(penalties["penalty_yards"].sum() / played)
    return float(len(penalties) / played)


def game_log(games: pandas.DataFrame, penalties: pandas.DataFrame) -> pandas.DataFrame:
    counted = penalties.groupby(["game_id", "team"], as_index=False).agg(
        penalties=("penalty_type", "size"),
        yards=("penalty_yards", "sum"),
    )
    log = games.drop_duplicates(["game_id", "team"]).merge(
        counted, on=["game_id", "team"], how="left"
    )
    log["penalties"] = log["penalties"].fillna(0).astype(int)
    log["yards"] = log["yards"].fillna(0.0)
    log["name"] = log["team"].map(TEAM_NAMES).fillna(log["team"])
    return log.sort_values(["season", "week", "team"])


def _check() -> None:
    build_tables()
    games, penalties = load_frames()
    filters = Filters(2024, 2024, 1, 22, ("REG",))
    games_2024, penalties_2024 = subset(games, penalties, filters)
    summary = team_summary(games_2024, penalties_2024)
    rate = summary["penalties"].sum() / summary["games"].sum()
    print(summary.sort_values("penalties_per_game", ascending=False).head(8).to_string(index=False))
    print(f"2024 regular season: {rate:.2f} penalties per team game")
    if summary["games"].median() != 17:
        raise SystemExit(f"Expected 17 games in 2024, got {summary['games'].value_counts().to_dict()}")
    if not 4 < rate < 9:
        raise SystemExit(f"2024 rate {rate:.2f} is outside the expected range")


if __name__ == "__main__":
    _check()
