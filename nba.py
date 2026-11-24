"""Read the cached NBA team logs and shape rest, back-to-backs, and home court.

cache_nba_data.py writes the parquet files under data/nba.
A back-to-back is a game on the day after that club's previous game.
True shooting pools every shot in the group.
"""

from __future__ import annotations

from pathlib import Path

import pandas

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data" / "nba"

ZONE_ORDER = [
    "Restricted Area",
    "In The Paint (Non-RA)",
    "Mid-Range",
    "Left Corner 3",
    "Right Corner 3",
    "Above the Break 3",
]
ZONE_LABELS = {
    "Restricted Area": "At the rim",
    "In The Paint (Non-RA)": "Paint",
    "Mid-Range": "Mid-range",
    "Left Corner 3": "Left corner",
    "Right Corner 3": "Right corner",
    "Above the Break 3": "Above the break",
}
MINUTES_FLOOR = 10
REST_ORDER = ["Back-to-back", "1 day off", "2 days off", "3 or more days off"]
EDGE_ORDER = [
    "You on a back-to-back",
    "Both on a back-to-back",
    "Opponent on a back-to-back",
    "Neither on a back-to-back",
]
CROSS_ORDER = [
    ("Home", False, "Home, rested"),
    ("Home", True, "Home, back-to-back"),
    ("Away", False, "Road, rested"),
    ("Away", True, "Road, back-to-back"),
]
POSITION_ORDER = ["Guard", "Forward", "Center"]
ROLE_ORDER = ["Under 15", "15 to 24", "25 to 32", "33 or more"]
AGE_ORDER = ["23 and under", "24 to 27", "28 to 31", "32 and over"]
USAGE_ORDER = ["Under 18% usage", "18 to 24% usage", "24% usage or more"]
APPEARANCE_ORDER = [
    "Played yesterday",
    "Second night, sat yesterday",
    "Team had a day off",
]
ROLE_MIN_GAMES = 10


def cache_ready() -> bool:
    needed = ("team_gamelogs", "player_advanced", "team_shot_zones", "player_shot_zones")
    return all(any((CACHE / name).glob("*.parquet")) for name in needed)


def cache_mtime() -> float:
    stamps = [path.stat().st_mtime for path in CACHE.glob("*/*.parquet")]
    if not stamps:
        return 0.0
    return max(stamps)


def _read_dataset(name: str) -> pandas.DataFrame:
    folder = CACHE / name
    files = sorted(folder.glob("*.parquet"))
    if not files:
        return pandas.DataFrame()
    frame = pandas.concat([pandas.read_parquet(path) for path in files], ignore_index=True)
    if "SEASON" in frame.columns:
        frame["SEASON"] = frame["SEASON"].astype("string")
    return frame


def load_team_logs() -> pandas.DataFrame:
    frame = _read_dataset("team_gamelogs")
    if frame.empty:
        return frame
    frame["GAME_DATE"] = pandas.to_datetime(frame["GAME_DATE"])
    for column in (
        "PTS",
        "OPP_PTS",
        "PLUS_MINUS",
        "FGM",
        "FGA",
        "FG3M",
        "FG3A",
        "FTM",
        "FTA",
        "OREB",
        "TOV",
    ):
        if column in frame.columns:
            frame[column] = pandas.to_numeric(frame[column], errors="coerce")
    return frame


def load_player_advanced() -> pandas.DataFrame:
    frame = _read_dataset("player_advanced")
    if frame.empty:
        return frame
    for column in ("GP", "MIN", "TS_PCT", "USG_PCT", "OFF_RATING", "DEF_RATING", "NET_RATING", "PIE", "EFG_PCT"):
        if column in frame.columns:
            frame[column] = pandas.to_numeric(frame[column], errors="coerce")
    return frame


def load_team_shots() -> pandas.DataFrame:
    return _numeric_shots(_read_dataset("team_shot_zones"))


def load_player_shots() -> pandas.DataFrame:
    return _numeric_shots(_read_dataset("player_shot_zones"))


def _numeric_shots(frame: pandas.DataFrame) -> pandas.DataFrame:
    if frame.empty:
        return frame
    for column in ("FGM", "FGA", "FG_PCT"):
        frame[column] = pandas.to_numeric(frame[column], errors="coerce")
    return frame


def mature_seasons(logs: pandas.DataFrame, minimum_games: int = 400) -> list[str]:
    """Seasons with enough team-games to sit in the pooled story.

    A season that has just tipped off stays on disk and out of the charts.
    """
    if logs.empty:
        return []
    counts = logs.groupby("SEASON").size()
    return sorted(season for season, games in counts.items() if int(games) >= minimum_games)


def held_seasons(logs: pandas.DataFrame, minimum_games: int = 400) -> pandas.Series:
    if logs.empty:
        return pandas.Series(dtype="int64")
    counts = logs.groupby("SEASON").size()
    return counts.loc[counts < minimum_games].sort_index()


def seasons_in(logs: pandas.DataFrame) -> list[str]:
    if logs.empty:
        return []
    return sorted(logs["SEASON"].dropna().unique().tolist())


def teams_in(logs: pandas.DataFrame, season: str) -> pandas.DataFrame:
    view = logs.loc[logs["SEASON"].eq(season), ["TEAM_ABBREVIATION", "TEAM_NAME"]]
    view = view.drop_duplicates().sort_values("TEAM_NAME")
    return view


def season_logs(logs: pandas.DataFrame, season: str, team: str = "") -> pandas.DataFrame:
    view = logs.loc[logs["SEASON"].eq(season)].copy()
    if team:
        view = view.loc[view["TEAM_ABBREVIATION"].eq(team)]
    return view.sort_values(["GAME_DATE", "GAME_ID"])


def _true_shooting(points: float, fga: float, fta: float) -> float | None:
    denom = 2 * (fga + 0.44 * fta)
    if denom <= 0:
        return None
    return points / denom


def team_season_summary(logs: pandas.DataFrame) -> pandas.DataFrame:
    if logs.empty:
        return pandas.DataFrame()
    grouped = logs.groupby(["TEAM_ABBREVIATION", "TEAM_NAME"], as_index=False).agg(
        games=("GAME_ID", "nunique"),
        points=("PTS", "sum"),
        opp_points=("OPP_PTS", "sum"),
        margin=("PLUS_MINUS", "sum"),
        fgm=("FGM", "sum"),
        fga=("FGA", "sum"),
        fg3m=("FG3M", "sum"),
        fg3a=("FG3A", "sum"),
        ftm=("FTM", "sum"),
        fta=("FTA", "sum"),
    )
    grouped["pts_per_game"] = grouped["points"] / grouped["games"]
    grouped["opp_per_game"] = grouped["opp_points"] / grouped["games"]
    grouped["margin_per_game"] = grouped["margin"] / grouped["games"]
    grouped["fg_pct"] = grouped["fgm"] / grouped["fga"]
    grouped["fg3_pct"] = grouped["fg3m"] / grouped["fg3a"]
    grouped["ts_pct"] = [
        _true_shooting(points, fga, fta)
        for points, fga, fta in zip(grouped["points"], grouped["fga"], grouped["fta"])
    ]
    return grouped.sort_values(["pts_per_game", "TEAM_ABBREVIATION"], ascending=[False, True])


def league_rates(logs: pandas.DataFrame) -> dict[str, float | None]:
    if logs.empty:
        return {"pts_per_game": None, "margin_per_game": None, "ts_pct": None, "games": 0}
    games = int(logs["GAME_ID"].nunique())
    return {
        "pts_per_game": float(logs["PTS"].sum() / len(logs)),
        "margin_per_game": 0.0,
        "ts_pct": _true_shooting(float(logs["PTS"].sum()), float(logs["FGA"].sum()), float(logs["FTA"].sum())),
        "games": games,
    }


def player_rows(players: pandas.DataFrame, season: str, team: str, min_games: int) -> pandas.DataFrame:
    """One row per player. A traded player's league row is the combined season."""
    if players.empty:
        return players
    view = players.loc[players["SEASON"].eq(season)].copy()
    if team:
        view = view.loc[view["TEAM_ABBREVIATION"].eq(team)]
    else:
        traded = set(view.loc[view["TEAM_ABBREVIATION"].eq("TOT"), "PLAYER_ID"])
        combined = view["PLAYER_ID"].isin(traded) & view["TEAM_ABBREVIATION"].eq("TOT")
        single = ~view["PLAYER_ID"].isin(traded)
        view = view.loc[combined | single]
    view = view.loc[view["GP"].fillna(0) >= min_games]
    view = view.loc[view["MIN"].fillna(0) >= MINUTES_FLOOR]
    return view.sort_values(["TS_PCT", "PLAYER_NAME"], ascending=[False, True])


def shot_profile(shots: pandas.DataFrame, season: str, team: str = "") -> pandas.DataFrame:
    if shots.empty:
        return shots
    view = shots.loc[shots["SEASON"].eq(season) & shots["ZONE"].isin(ZONE_ORDER)].copy()
    if team:
        view = view.loc[view["TEAM_ABBREVIATION"].eq(team)]
    if view.empty:
        return view
    grouped = view.groupby("ZONE", as_index=False).agg(fgm=("FGM", "sum"), fga=("FGA", "sum"))
    grouped["fg_pct"] = grouped["fgm"] / grouped["fga"].where(grouped["fga"] > 0)
    grouped["label"] = grouped["ZONE"].map(ZONE_LABELS)
    order = {zone: index for index, zone in enumerate(ZONE_ORDER)}
    return grouped.sort_values("ZONE", key=lambda series: series.map(order))


def attach_schedule(logs: pandas.DataFrame) -> pandas.DataFrame:
    """Add nights off for the club and for the opponent.

    Nights off are the days between this game and the previous one, minus one.
    Zero is a back-to-back. The first game of a season has no previous game.
    """
    frame = logs.sort_values(["SEASON", "TEAM_ID", "GAME_DATE", "GAME_ID"]).copy()
    previous = frame.groupby(["SEASON", "TEAM_ID"], sort=False)["GAME_DATE"].shift(1)
    frame["REST_DAYS"] = (frame["GAME_DATE"] - previous).dt.days - 1
    frame["IS_B2B"] = frame["REST_DAYS"].le(0) & frame["REST_DAYS"].notna()
    frame["REST_LABEL"] = pandas.Series(pandas.NA, index=frame.index, dtype="object")
    known = frame["REST_DAYS"].notna()
    frame.loc[known & frame["IS_B2B"], "REST_LABEL"] = REST_ORDER[0]
    frame.loc[known & frame["REST_DAYS"].eq(1), "REST_LABEL"] = REST_ORDER[1]
    frame.loc[known & frame["REST_DAYS"].eq(2), "REST_LABEL"] = REST_ORDER[2]
    frame.loc[known & frame["REST_DAYS"].ge(3), "REST_LABEL"] = REST_ORDER[3]
    opponent = frame.loc[:, ["GAME_ID", "TEAM_ABBREVIATION", "REST_DAYS", "IS_B2B"]].rename(
        columns={"TEAM_ABBREVIATION": "OPPONENT", "REST_DAYS": "OPP_REST", "IS_B2B": "OPP_B2B"}
    )
    frame = frame.merge(opponent, on=["GAME_ID", "OPPONENT"], how="left")
    frame["TS_DENOM"] = 2 * (frame["FGA"] + 0.44 * frame["FTA"])
    frame["POSS"] = frame["FGA"] + 0.44 * frame["FTA"] - frame["OREB"] + frame["TOV"]
    frame["EFG_NUM"] = frame["FGM"] + 0.5 * frame["FG3M"]
    frame["GAME_TS"] = frame["PTS"] / frame["TS_DENOM"].where(frame["TS_DENOM"] > 0)
    you = frame["IS_B2B"].fillna(False)
    them = frame["OPP_B2B"].fillna(False)
    both_known = frame["REST_DAYS"].notna() & frame["OPP_REST"].notna()
    frame["EDGE"] = pandas.Series(pandas.NA, index=frame.index, dtype="object")
    frame.loc[both_known & you & ~them, "EDGE"] = EDGE_ORDER[0]
    frame.loc[both_known & you & them, "EDGE"] = EDGE_ORDER[1]
    frame.loc[both_known & ~you & them, "EDGE"] = EDGE_ORDER[2]
    frame.loc[both_known & ~you & ~them, "EDGE"] = EDGE_ORDER[3]
    return frame


def with_rest(frame: pandas.DataFrame) -> pandas.DataFrame:
    return frame.loc[frame["REST_LABEL"].notna()].copy()


def pooled(frame: pandas.DataFrame) -> dict:
    if frame.empty:
        return {}
    games = int(len(frame))
    points = float(frame["PTS"].sum())
    fga = float(frame["FGA"].sum())
    fta = float(frame["FTA"].sum())
    possessions = float(frame["POSS"].sum())
    denom = float(frame["TS_DENOM"].sum())
    margin = float(frame["PLUS_MINUS"].sum())
    return {
        "games": games,
        "pts_per_game": points / games,
        "opp_pts": (points - margin) / games,
        "margin": margin / games,
        "ts": points / denom if denom else None,
        "efg": float(frame["EFG_NUM"].sum()) / fga if fga else None,
        "fg3": float(frame["FG3M"].sum()) / float(frame["FG3A"].sum()) if frame["FG3A"].sum() else None,
        "three_rate": float(frame["FG3A"].sum()) / fga if fga else None,
        "ft_rate": fta / fga if fga else None,
        "ortg": 100 * points / possessions if possessions else None,
        "pace": possessions / games,
        "tov_rate": float(frame["TOV"].sum()) / possessions if possessions else None,
        "win_pct": float((frame["WL"] == "W").mean()),
    }


def summarize_labels(frame: pandas.DataFrame, column: str, order: list[str]) -> pandas.DataFrame:
    rows = []
    for label in order:
        stats = pooled(frame.loc[frame[column].eq(label)])
        if not stats:
            continue
        stats["label"] = label
        rows.append(stats)
    return pandas.DataFrame(rows)


def cross_summary(frame: pandas.DataFrame) -> pandas.DataFrame:
    rows = []
    for site, is_b2b, label in CROSS_ORDER:
        stats = pooled(frame.loc[frame["SITE"].eq(site) & frame["IS_B2B"].eq(is_b2b)])
        if not stats:
            continue
        stats["label"] = label
        stats["site"] = "Home" if site == "Home" else "Road"
        stats["b2b"] = "Back-to-back" if is_b2b else "Rested"
        rows.append(stats)
    return pandas.DataFrame(rows)


def season_compare(frame: pandas.DataFrame) -> pandas.DataFrame:
    rows = []
    for season, group in frame.groupby("SEASON", sort=True):
        second = pooled(group.loc[group["IS_B2B"]])
        rested = pooled(group.loc[~group["IS_B2B"]])
        home = pooled(group.loc[group["SITE"].eq("Home")])
        away = pooled(group.loc[group["SITE"].eq("Away")])
        rows.append(
            {
                "season": season,
                "b2b_ts": second["ts"],
                "rest_ts": rested["ts"],
                "b2b_pts": second["pts_per_game"],
                "rest_pts": rested["pts_per_game"],
                "b2b_win": second["win_pct"],
                "rest_win": rested["win_pct"],
                "home_ts": home["ts"],
                "away_ts": away["ts"],
                "home_margin": home["margin"],
                "away_margin": away["margin"],
                "b2b_games": second["games"],
                "home_games": home["games"],
            }
        )
    return pandas.DataFrame(rows)


def player_logs_ready() -> bool:
    return any((CACHE / "player_gamelogs").glob("*.parquet")) and any((CACHE / "player_positions").glob("*.parquet"))


def load_player_logs() -> pandas.DataFrame:
    frame = _read_dataset("player_gamelogs")
    if frame.empty:
        return frame
    frame["GAME_DATE"] = pandas.to_datetime(frame["GAME_DATE"])
    for column in ("MIN", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "OREB", "TOV", "PTS", "PLUS_MINUS", "REB", "AST"):
        if column in frame.columns:
            frame[column] = pandas.to_numeric(frame[column], errors="coerce")
    frame["GAME_ID"] = frame["GAME_ID"].astype("string")
    frame["PLAYER_ID"] = pandas.to_numeric(frame["PLAYER_ID"], errors="coerce").astype("Int64")
    frame["TEAM_ID"] = pandas.to_numeric(frame["TEAM_ID"], errors="coerce").astype("Int64")
    return frame


def load_positions() -> pandas.DataFrame:
    frame = _read_dataset("player_positions")
    if frame.empty:
        return frame
    frame["PLAYER_ID"] = pandas.to_numeric(frame["PLAYER_ID"], errors="coerce").astype("Int64")
    frame["TEAM_ID"] = pandas.to_numeric(frame["TEAM_ID"], errors="coerce").astype("Int64")
    if "AGE" in frame.columns:
        frame["AGE"] = pandas.to_numeric(frame["AGE"], errors="coerce")
    return frame


def position_group(value) -> str | None:
    text = "" if pandas.isna(value) else str(value).upper()
    if text.startswith("G"):
        return "Guard"
    if text.startswith("F"):
        return "Forward"
    if text.startswith("C"):
        return "Center"
    return None


def role_label(minutes: float) -> str:
    if minutes < 15:
        return ROLE_ORDER[0]
    if minutes < 25:
        return ROLE_ORDER[1]
    if minutes < 33:
        return ROLE_ORDER[2]
    return ROLE_ORDER[3]


def age_label(age: float) -> str | None:
    if pandas.isna(age):
        return None
    if age <= 23:
        return AGE_ORDER[0]
    if age <= 27:
        return AGE_ORDER[1]
    if age <= 31:
        return AGE_ORDER[2]
    return AGE_ORDER[3]


def usage_label(usage: float) -> str | None:
    if pandas.isna(usage):
        return None
    if usage < 0.18:
        return USAGE_ORDER[0]
    if usage < 0.24:
        return USAGE_ORDER[1]
    return USAGE_ORDER[2]


def _season_profile(advanced: pandas.DataFrame) -> pandas.DataFrame:
    if advanced.empty:
        return pandas.DataFrame(columns=["SEASON", "PLAYER_ID", "AGE", "USG_PCT"])
    frame = advanced.copy()
    frame["PLAYER_ID"] = pandas.to_numeric(frame["PLAYER_ID"], errors="coerce").astype("Int64")
    frame["GP"] = pandas.to_numeric(frame["GP"], errors="coerce")
    traded = set(frame.loc[frame["TEAM_ABBREVIATION"].eq("TOT"), "PLAYER_ID"].dropna())
    combined = frame.loc[frame["PLAYER_ID"].isin(traded) & frame["TEAM_ABBREVIATION"].eq("TOT")]
    single = frame.loc[~frame["PLAYER_ID"].isin(traded)]
    profile = pandas.concat([combined, single], ignore_index=True)
    profile = profile.sort_values("GP", ascending=False).drop_duplicates(["SEASON", "PLAYER_ID"])
    return profile.loc[:, ["SEASON", "PLAYER_ID", "AGE", "USG_PCT"]]


def attach_players(logs: pandas.DataFrame, positions: pandas.DataFrame, advanced: pandas.DataFrame, teams: pandas.DataFrame) -> pandas.DataFrame:
    """Tag each player appearance with schedule rest, position, role, age, and usage.

    A team back-to-back is the club's second night. Played yesterday means this
    player was in the box score the day before. Usual minutes are his average
    in games he played that season, and only seasons with at least 10 games
    get a minutes role.
    """
    frame = logs.loc[logs["MIN"].fillna(0).gt(0)].copy()
    frame = frame.sort_values(["SEASON", "PLAYER_ID", "GAME_DATE", "GAME_ID"])
    previous = frame.groupby(["SEASON", "PLAYER_ID"], sort=False)["GAME_DATE"].shift(1)
    frame["PLAYER_REST"] = (frame["GAME_DATE"] - previous).dt.days - 1
    frame["PLAYED_YESTERDAY"] = frame["PLAYER_REST"].eq(0)
    frame["TS_DENOM"] = 2 * (frame["FGA"] + 0.44 * frame["FTA"])
    frame["POSS"] = frame["FGA"] + 0.44 * frame["FTA"] - frame["OREB"] + frame["TOV"]
    frame["EFG_NUM"] = frame["FGM"] + 0.5 * frame["FG3M"]

    club = teams.loc[:, ["SEASON", "GAME_ID", "TEAM_ID", "IS_B2B", "REST_LABEL", "REST_DAYS"]].copy()
    club["GAME_ID"] = club["GAME_ID"].astype("string")
    club["TEAM_ID"] = pandas.to_numeric(club["TEAM_ID"], errors="coerce").astype("Int64")
    club = club.drop_duplicates(["SEASON", "GAME_ID", "TEAM_ID"])
    frame = frame.merge(club, on=["SEASON", "GAME_ID", "TEAM_ID"], how="left")

    roster = positions.drop_duplicates(["SEASON", "TEAM_ID", "PLAYER_ID"])
    frame = frame.merge(
        roster.loc[:, ["SEASON", "TEAM_ID", "PLAYER_ID", "POSITION"]],
        on=["SEASON", "TEAM_ID", "PLAYER_ID"],
        how="left",
    )
    fallback = (
        positions.dropna(subset=["POSITION"])
        .groupby(["SEASON", "PLAYER_ID"])["POSITION"]
        .agg(lambda values: values.mode().iloc[0] if not values.mode().empty else values.iloc[0])
        .rename("POSITION_FALLBACK")
        .reset_index()
    )
    frame = frame.merge(fallback, on=["SEASON", "PLAYER_ID"], how="left")
    frame["POSITION"] = frame["POSITION"].fillna(frame["POSITION_FALLBACK"])
    frame["POSITION_GROUP"] = frame["POSITION"].map(position_group)

    profile = _season_profile(advanced)
    frame = frame.merge(profile, on=["SEASON", "PLAYER_ID"], how="left", suffixes=("", "_PROFILE"))
    frame["AGE_BAND"] = frame["AGE"].map(age_label)
    frame["USAGE_BAND"] = frame["USG_PCT"].map(usage_label)

    season_games = frame.groupby(["SEASON", "PLAYER_ID"])["GAME_ID"].transform("nunique")
    season_minutes = frame.groupby(["SEASON", "PLAYER_ID"])["MIN"].transform("mean")
    frame["USUAL_MINUTES"] = season_minutes.where(season_games >= ROLE_MIN_GAMES)
    frame["ROLE"] = frame["USUAL_MINUTES"].map(lambda value: None if pandas.isna(value) else role_label(float(value)))

    frame["IS_B2B"] = frame["IS_B2B"].eq(True)
    known_team = frame["REST_LABEL"].notna()
    frame["APPEARANCE"] = pandas.Series(pandas.NA, index=frame.index, dtype="object")
    frame.loc[frame["PLAYED_YESTERDAY"], "APPEARANCE"] = APPEARANCE_ORDER[0]
    frame.loc[~frame["PLAYED_YESTERDAY"] & frame["IS_B2B"], "APPEARANCE"] = APPEARANCE_ORDER[1]
    frame.loc[~frame["PLAYED_YESTERDAY"] & known_team & ~frame["IS_B2B"], "APPEARANCE"] = APPEARANCE_ORDER[2]
    return frame


def pooled_player(frame: pandas.DataFrame) -> dict:
    stats = pooled(frame)
    if not stats:
        return stats
    stats["minutes"] = float(frame["MIN"].sum()) / stats["games"]
    stats["players"] = int(frame["PLAYER_ID"].nunique())
    return stats


def rest_contrast(frame: pandas.DataFrame, column: str, order: list[str]) -> pandas.DataFrame:
    """True shooting and minutes on a team back-to-back against one day off, by group."""
    rows = []
    view = frame.loc[frame[column].notna() & frame["REST_LABEL"].notna()]
    for label in order:
        group = view.loc[view[column].eq(label)]
        on_second = group["IS_B2B"].fillna(False)
        one_day = group["REST_LABEL"].eq("1 day off")
        second = pooled_player(group.loc[on_second])
        one = pooled_player(group.loc[one_day])
        home_second = pooled_player(group.loc[on_second & group["SITE"].eq("Home")])
        home_one = pooled_player(group.loc[one_day & group["SITE"].eq("Home")])
        away_second = pooled_player(group.loc[on_second & group["SITE"].eq("Away")])
        away_one = pooled_player(group.loc[one_day & group["SITE"].eq("Away")])
        if not second or not one or second.get("ts") is None or one.get("ts") is None:
            continue
        rows.append(
            {
                "label": label,
                "b2b_ts": second["ts"],
                "rest_ts": one["ts"],
                "ts_gap": second["ts"] - one["ts"],
                "b2b_min": second["minutes"],
                "rest_min": one["minutes"],
                "min_gap": second["minutes"] - one["minutes"],
                "b2b_games": second["games"],
                "rest_games": one["games"],
                "players": max(second["players"], one["players"]),
                "home_b2b_ts": None if not home_second else home_second.get("ts"),
                "home_rest_ts": None if not home_one else home_one.get("ts"),
                "away_b2b_ts": None if not away_second else away_second.get("ts"),
                "away_rest_ts": None if not away_one else away_one.get("ts"),
            }
        )
    return pandas.DataFrame(rows)


def own_rest_contrast(frame: pandas.DataFrame, column: str, order: list[str]) -> pandas.DataFrame:
    """True shooting on a night he played yesterday, against a night with one day since his last game."""
    rows = []
    view = frame.loc[frame[column].notna() & frame["MIN"].fillna(0).gt(0)]
    for label in order:
        group = view.loc[view[column].eq(label)]
        second = pooled_player(group.loc[group["PLAYED_YESTERDAY"]])
        one = pooled_player(group.loc[group["PLAYER_REST"].eq(1)])
        if not second or not one or second.get("ts") is None or one.get("ts") is None:
            continue
        rows.append(
            {
                "label": label,
                "b2b_ts": second["ts"],
                "rest_ts": one["ts"],
                "ts_gap": second["ts"] - one["ts"],
                "b2b_min": second["minutes"],
                "rest_min": one["minutes"],
                "b2b_games": second["games"],
                "rest_games": one["games"],
            }
        )
    return pandas.DataFrame(rows)


def player_cases(frame: pandas.DataFrame, minimum_second: int = 30, minimum_rest: int = 50) -> pandas.DataFrame:
    """Players with enough of their own back-to-backs and enough games on one day off."""
    rows = []
    view = frame.loc[frame["MIN"].fillna(0).gt(0)]
    for player_id, group in view.groupby("PLAYER_ID", sort=False):
        second = group.loc[group["PLAYED_YESTERDAY"]]
        one = group.loc[group["PLAYER_REST"].eq(1)]
        if len(second) < minimum_second or len(one) < minimum_rest:
            continue
        played_second = pooled_player(second)
        rested = pooled_player(one)
        if not played_second or not rested or played_second.get("ts") is None or rested.get("ts") is None:
            continue
        latest = group.sort_values(["GAME_DATE", "GAME_ID"]).iloc[-1]
        clubs = group.groupby("TEAM_ABBREVIATION").size().sort_values(ascending=False)
        rows.append(
            {
                "player_id": int(player_id),
                "name": str(latest["PLAYER_NAME"]),
                "club": str(clubs.index[0]),
                "club_games": int(clubs.iloc[0]),
                "games": int(len(group)),
                "age": float(group["AGE"].median()),
                "b2b_n": played_second["games"],
                "rest_n": rested["games"],
                "b2b_ts": played_second["ts"],
                "rest_ts": rested["ts"],
                "ts_gap": played_second["ts"] - rested["ts"],
                "b2b_min": played_second["minutes"],
                "rest_min": rested["minutes"],
                "b2b_pm": played_second["margin"],
                "rest_pm": rested["margin"],
            }
        )
    cases = pandas.DataFrame(rows)
    if cases.empty:
        return cases
    return cases.sort_values(["ts_gap", "name"], ascending=[True, True])


def group_levels(frame: pandas.DataFrame, column: str, order: list[str]) -> pandas.DataFrame:
    rows = []
    for label in order:
        stats = pooled_player(frame.loc[frame[column].eq(label)])
        if not stats or stats.get("ts") is None:
            continue
        stats["label"] = label
        rows.append(stats)
    return pandas.DataFrame(rows)


def team_gaps(frame: pandas.DataFrame) -> pandas.DataFrame:
    rows = []
    for (abbrev, name), group in frame.groupby(["TEAM_ABBREVIATION", "TEAM_NAME"], sort=False):
        second = pooled(group.loc[group["IS_B2B"]])
        rested = pooled(group.loc[~group["IS_B2B"]])
        home = pooled(group.loc[group["SITE"].eq("Home")])
        away = pooled(group.loc[group["SITE"].eq("Away")])
        if not second or not rested:
            continue
        rows.append(
            {
                "team": abbrev,
                "name": name,
                "b2b_games": second["games"],
                "b2b_ts": second["ts"],
                "rest_ts": rested["ts"],
                "ts_gap": second["ts"] - rested["ts"],
                "margin_gap": second["margin"] - rested["margin"],
                "win_gap": second["win_pct"] - rested["win_pct"],
                "b2b_win": second["win_pct"],
                "rest_win": rested["win_pct"],
                "b2b_margin": second["margin"],
                "rest_margin": rested["margin"],
                "home_ts": home.get("ts"),
                "away_ts": away.get("ts"),
                "home_margin": home.get("margin"),
            }
        )
    gaps = pandas.DataFrame(rows)
    return gaps.sort_values(["ts_gap", "team"], ascending=[True, True])


def player_shot_profile(shots: pandas.DataFrame, season: str, player_id, team: str = "") -> pandas.DataFrame:
    if shots.empty or player_id is None:
        return pandas.DataFrame()
    view = shots.loc[
        shots["SEASON"].eq(season) & shots["PLAYER_ID"].eq(player_id) & shots["ZONE"].isin(ZONE_ORDER)
    ].copy()
    if team:
        view = view.loc[view["TEAM_ABBREVIATION"].eq(team)]
    elif (view["TEAM_ABBREVIATION"] == "TOT").any():
        view = view.loc[view["TEAM_ABBREVIATION"].eq("TOT")]
    if view.empty:
        return view
    grouped = view.groupby("ZONE", as_index=False).agg(fgm=("FGM", "sum"), fga=("FGA", "sum"))
    grouped["fg_pct"] = grouped["fgm"] / grouped["fga"].where(grouped["fga"] > 0)
    grouped["label"] = grouped["ZONE"].map(ZONE_LABELS)
    order = {zone: index for index, zone in enumerate(ZONE_ORDER)}
    return grouped.sort_values("ZONE", key=lambda series: series.map(order))
