"""Did penalty volume jump at three pre-chosen officiating seasons?

The outcome is flags charged to one team in one regular-season game, and
games with zero flags stay in the average. The mean is a yearly drift plus
a level shift at three seasons the NFL marked as points of emphasis:

- 2004, defensive holding and illegal contact
- 2018, roughing the passer
- 2021, taunting

Each shift stacks on the earlier ones. Standard errors are clustered by
season, because every team that year shares the same officiating climate.
A coefficient is a change in flags per team-game. It bundles the emphasis
with everything else that changed in that season.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas
import statsmodels.api as sm

from penalties import load_frames

YEAR_ZERO = 2000
SHIFTS = (
    ("post_2004", 2004, "From 2004"),
    ("post_2018", 2018, "From 2018"),
    ("post_2021", 2021, "From 2021"),
)
TARGETS = (
    ("Defensive Holding", "post_2004", "2004 holding emphasis"),
    ("Illegal Contact", "post_2004", "2004 holding emphasis"),
    ("Roughing the Passer", "post_2018", "2018 roughing emphasis"),
    ("Taunting", "post_2021", "2021 taunting emphasis"),
)
TERM_LABELS = {
    "const": "2000 baseline",
    "year": "Yearly drift",
    "post_2004": "From 2004",
    "post_2018": "From 2018",
    "post_2021": "From 2021",
}


@dataclass(frozen=True)
class Term:
    key: str
    label: str
    estimate: float
    se: float
    ci_low: float
    ci_high: float
    pvalue: float


@dataclass(frozen=True)
class Fit:
    outcome: str
    terms: tuple[Term, ...]
    n_team_games: int
    n_seasons: int
    r_squared: float

    def term(self, key: str) -> Term:
        for item in self.terms:
            if item.key == key:
                return item
        raise KeyError(key)


@dataclass(frozen=True)
class TargetFit:
    penalty_type: str
    focus_key: str
    focus_label: str
    emphasis_season: int
    prior_rate: float
    emphasis_rate: float
    fit: Fit


@dataclass(frozen=True)
class Study:
    volume: Fit
    targets: tuple[TargetFit, ...]
    season: pandas.DataFrame
    type_season: pandas.DataFrame
    season_min: int
    season_max: int


def team_game_frame(games: pandas.DataFrame, penalties: pandas.DataFrame) -> pandas.DataFrame:
    base = games.loc[games["season_type"].eq("REG"), ["season", "game_id", "team"]]
    base = base.drop_duplicates(["game_id", "team"])
    counted = (
        penalties.loc[penalties["season_type"].eq("REG")]
        .groupby(["game_id", "team"], as_index=False)
        .size()
        .rename(columns={"size": "penalties"})
    )
    frame = base.merge(counted, on=["game_id", "team"], how="left")
    frame["penalties"] = frame["penalties"].fillna(0).astype(float)
    return frame


def _design(frame: pandas.DataFrame) -> pandas.DataFrame:
    design = pandas.DataFrame(index=frame.index)
    design["const"] = 1.0
    design["year"] = frame["season"].astype(float) - YEAR_ZERO
    for key, season, _label in SHIFTS:
        design[key] = frame["season"].ge(season).astype(float)
    return design


def _fit(frame: pandas.DataFrame, outcome: pandas.Series, name: str) -> Fit:
    design = _design(frame)
    model = sm.OLS(outcome.astype(float), design).fit(
        cov_type="cluster",
        cov_kwds={"groups": frame["season"]},
    )
    intervals = model.conf_int(alpha=0.05)
    terms = []
    for key in design.columns:
        terms.append(
            Term(
                key=key,
                label=TERM_LABELS[key],
                estimate=float(model.params[key]),
                se=float(model.bse[key]),
                ci_low=float(intervals.loc[key, 0]),
                ci_high=float(intervals.loc[key, 1]),
                pvalue=float(model.pvalues[key]),
            )
        )
    return Fit(
        outcome=name,
        terms=tuple(terms),
        n_team_games=int(len(frame)),
        n_seasons=int(frame["season"].nunique()),
        r_squared=float(model.rsquared),
    )


def _type_counts(frame: pandas.DataFrame, penalties: pandas.DataFrame, penalty_type: str) -> pandas.Series:
    flagged = penalties.loc[
        penalties["season_type"].eq("REG") & penalties["penalty_type"].eq(penalty_type)
    ]
    counted = flagged.groupby(["game_id", "team"]).size()
    keys = pandas.MultiIndex.from_frame(frame.loc[:, ["game_id", "team"]])
    return counted.reindex(keys, fill_value=0).astype(float).to_numpy()


def _season_type_rates(frame: pandas.DataFrame, penalties: pandas.DataFrame) -> pandas.DataFrame:
    rows = []
    games = frame.groupby("season").size()
    for penalty_type, _key, _label in TARGETS:
        counted = (
            penalties.loc[
                penalties["season_type"].eq("REG") & penalties["penalty_type"].eq(penalty_type)
            ]
            .groupby("season")
            .size()
        )
        for season, n_games in games.items():
            rows.append(
                {
                    "season": int(season),
                    "penalty_type": penalty_type,
                    "rate": float(counted.get(season, 0) / n_games),
                }
            )
    return pandas.DataFrame(rows)


def run_study(games: pandas.DataFrame, penalties: pandas.DataFrame) -> Study:
    frame = team_game_frame(games, penalties)
    volume = _fit(frame, frame["penalties"], "All flags")
    design = _design(frame)
    # Fitted season mean uses the same linear predictor as the volume model.
    coefficients = pandas.Series({term.key: term.estimate for term in volume.terms})
    fitted = design.to_numpy() @ coefficients.loc[design.columns].to_numpy()
    season = (
        frame.assign(fitted=fitted)
        .groupby("season", as_index=False)
        .agg(actual=("penalties", "mean"), fitted=("fitted", "mean"))
    )
    targets = []
    type_season = _season_type_rates(frame, penalties)
    shift_year = {key: season for key, season, _label in SHIFTS}
    for penalty_type, focus_key, focus_label in TARGETS:
        outcome = pandas.Series(_type_counts(frame, penalties, penalty_type), index=frame.index)
        if float(outcome.sum()) == 0:
            continue
        emphasis_season = shift_year[focus_key]
        prior_rate, emphasis_rate = _window_rates(type_season, penalty_type, emphasis_season)
        targets.append(
            TargetFit(
                penalty_type=penalty_type,
                focus_key=focus_key,
                focus_label=focus_label,
                emphasis_season=emphasis_season,
                prior_rate=prior_rate,
                emphasis_rate=emphasis_rate,
                fit=_fit(frame, outcome, penalty_type),
            )
        )
    return Study(
        volume=volume,
        targets=tuple(targets),
        season=season,
        type_season=type_season,
        season_min=int(season["season"].min()),
        season_max=int(season["season"].max()),
    )


def _crosses_zero(term: Term) -> bool:
    return term.ci_low <= 0 <= term.ci_high


def _window_rates(
    type_season: pandas.DataFrame, penalty_type: str, season: int
) -> tuple[float, float]:
    rates = type_season.loc[type_season["penalty_type"].eq(penalty_type)].set_index("season")["rate"]
    prior = rates.reindex([season - 3, season - 2, season - 1]).dropna()
    return float(prior.mean()), float(rates.loc[season])


def volume_reading(study: Study) -> str:
    early = study.season.loc[study.season["season"].between(2000, 2003), "actual"]
    late = study.season.loc[study.season["season"].between(2023, 2025), "actual"]
    early_rate = float(early.mean())
    late_rate = float(late.mean())
    drift = study.volume.term("year")
    clearing = [
        str(season)
        for key, season, _label in SHIFTS
        if not _crosses_zero(study.volume.term(key))
    ]
    if clearing:
        shift_text = (
            "The emphasis seasons whose total-flag interval sits away from zero: "
            + ", ".join(clearing)
            + "."
        )
    else:
        shift_text = (
            "Once a yearly drift is included, the interval for each emphasis season "
            "still covers no change in the total."
        )
    drift_per_game = drift.estimate * 2
    drift_low = drift.ci_low * 2
    drift_high = drift.ci_high * 2
    if _crosses_zero(drift):
        drift_text = (
            f"Drift, the fitted change per year, is {drift_per_game:+.2f} flags in the game "
            f"(95% interval {drift_low:.2f} to {drift_high:.2f}, p {drift.pvalue:.2f}). "
            f"That interval includes zero, so the yearly climb is not significant."
        )
    else:
        drift_text = (
            f"Drift, the fitted change per year, is {drift_per_game:+.2f} flags in the game "
            f"(95% interval {drift_low:.2f} to {drift_high:.2f}, p {drift.pvalue:.2f})."
        )
    return (
        f"A game averaged {early_rate * 2:.2f} flags in 2000-2003 and {late_rate * 2:.2f} in 2023-2025, "
        f"counting both clubs. {drift_text} {shift_text}"
    )


def type_reading(study: Study) -> str:
    parts = []
    for target in study.targets:
        term = target.fit.term(target.focus_key)
        jump = (
            f"{target.penalty_type} moved from {target.prior_rate * 2:.2f} over the three seasons "
            f"before {target.emphasis_season} to {target.emphasis_rate * 2:.2f} in {target.emphasis_season}"
        )
        if _crosses_zero(term):
            parts.append(
                f"{jump}. The permanent shift from that season on includes no change "
                f"({term.estimate * 2:+.2f} in the game, {term.ci_low * 2:.2f} to {term.ci_high * 2:.2f})"
            )
        else:
            word = "more" if term.estimate > 0 else "fewer"
            parts.append(
                f"{jump}. From that season on, the model puts the rate "
                f"{abs(term.estimate) * 2:.2f} {word} in the game "
                f"({term.ci_low * 2:.2f} to {term.ci_high * 2:.2f})"
            )
    if not parts:
        return ""
    return " ".join(part + "." for part in parts)


def limit_reading(study: Study) -> str:
    return (
        f"Each shift is the {study.season_min}-{study.season_max} change from that season forward, "
        "bundled with every other rule and emphasis that arrived the same year. "
        f"The intervals are clustered into {study.volume.n_seasons} seasons. "
        f"Treating each of the {study.volume.n_team_games:,} team-games as its own season "
        "would overstate how precise those shifts are. "
        "2026 is the season still being played, and it stays in the fit as the games played so far."
    )


@dataclass(frozen=True)
class GamePace:
    """Flags charged in one game, both teams combined."""

    by_scope: pandas.DataFrame
    current_season: int
    current_flags: float
    current_games: int
    prior_peak_season: int
    prior_peak_flags: float
    recent_flags: float
    regular_flags: float
    playoff_flags: float
    compared_min: int
    compared_max: int


@dataclass(frozen=True)
class MatchupCut:
    heavy_flags: float | None
    heavy_games: int
    light_flags: float | None
    light_games: int
    heavy_teams: tuple[str, ...]
    light_teams: tuple[str, ...]
    seasons: int


def game_rows(games: pandas.DataFrame) -> pandas.DataFrame:
    home = games.loc[
        games["site"].eq("Home"),
        ["season", "week", "season_type", "game_id", "team", "opponent"],
    ]
    return home.rename(columns={"team": "home_team", "opponent": "away_team"}).drop_duplicates("game_id")


MONTH_NAMES = {
    1: "January",
    2: "February",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}
MONTH_ORDER = ("August", "September", "October", "November", "December", "January", "February")
QUARTER_NAMES = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th", 5: "Overtime"}


def _side_counts(penalties: pandas.DataFrame, value: str) -> pandas.DataFrame:
    if penalties.empty:
        return pandas.DataFrame(columns=["game_id", "team", value])
    if value == "flags":
        counted = penalties.groupby(["game_id", "team"]).size().rename(value).reset_index()
    else:
        counted = penalties.groupby(["game_id", "team"])[value].sum().rename(value).reset_index()
    return counted


def game_board(games: pandas.DataFrame, penalties: pandas.DataFrame) -> pandas.DataFrame:
    """One row per game. Flags are the two clubs' counts added."""
    rows = game_rows(games)
    if rows.empty:
        return rows
    dates = games.drop_duplicates("game_id").set_index("game_id")
    rows = rows.copy()
    if "game_date" in dates.columns:
        rows["game_date"] = rows["game_id"].map(dates["game_date"])
    if "final_qtr" in dates.columns:
        rows["final_qtr"] = rows["game_id"].map(dates["final_qtr"])
    flags = _side_counts(penalties, "flags")
    yards = _side_counts(penalties, "penalty_yards")
    home_flags = flags.rename(columns={"team": "home_team", "flags": "home_flags"})
    away_flags = flags.rename(columns={"team": "away_team", "flags": "away_flags"})
    home_yards = yards.rename(columns={"team": "home_team", "penalty_yards": "home_yards"})
    away_yards = yards.rename(columns={"team": "away_team", "penalty_yards": "away_yards"})
    rows = rows.merge(home_flags, on=["game_id", "home_team"], how="left")
    rows = rows.merge(away_flags, on=["game_id", "away_team"], how="left")
    rows = rows.merge(home_yards, on=["game_id", "home_team"], how="left")
    rows = rows.merge(away_yards, on=["game_id", "away_team"], how="left")
    for column in ("home_flags", "away_flags", "home_yards", "away_yards"):
        rows[column] = rows[column].fillna(0)
    rows["home_flags"] = rows["home_flags"].astype(int)
    rows["away_flags"] = rows["away_flags"].astype(int)
    rows["flags"] = rows["home_flags"] + rows["away_flags"]
    rows["yards"] = rows["home_yards"] + rows["away_yards"]
    if "game_date" in rows.columns:
        month_n = pandas.to_datetime(rows["game_date"], errors="coerce").dt.month
        rows["month"] = month_n.map(MONTH_NAMES)
    return rows


def games_involving(board: pandas.DataFrame, teams: tuple[str, ...], meetings_only: bool) -> pandas.DataFrame:
    if board.empty or not teams:
        return board
    if meetings_only and len(teams) >= 2:
        picked = set(teams)
        both = board["home_team"].isin(picked) & board["away_team"].isin(picked)
        return board.loc[both].copy()
    either = board["home_team"].isin(teams) | board["away_team"].isin(teams)
    return board.loc[either].copy()


def attach_flags(games: pandas.DataFrame, penalties: pandas.DataFrame) -> pandas.DataFrame:
    board = game_board(games, penalties)
    columns = ["season", "week", "season_type", "game_id", "home_team", "away_team", "flags"]
    return board.loc[:, [column for column in columns if column in board.columns]]


SLOT_ORDER = [str(week) for week in range(1, 19)] + [
    "Wild card",
    "Divisional",
    "Conference",
    "Super Bowl",
]


def schedule_slot(season: int, week: int, season_type: str) -> str | None:
    week = int(week)
    season = int(season)
    if season_type == "REG":
        if 1 <= week <= 18:
            return str(week)
        return None
    if season >= 2021:
        rounds = {19: "Wild card", 20: "Divisional", 21: "Conference", 22: "Super Bowl"}
    else:
        rounds = {18: "Wild card", 19: "Divisional", 20: "Conference", 21: "Super Bowl"}
    return rounds.get(week)


def _with_slot(board: pandas.DataFrame) -> pandas.DataFrame:
    slotted = board.copy()
    slotted["slot"] = [
        schedule_slot(season, week, season_type)
        for season, week, season_type in zip(
            slotted["season"], slotted["week"], slotted["season_type"], strict=False
        )
    ]
    return slotted.dropna(subset=["slot"])


def _order_slots(frame: pandas.DataFrame) -> pandas.DataFrame:
    frame["slot"] = pandas.Categorical(frame["slot"], categories=SLOT_ORDER, ordered=True)
    return frame.sort_values("slot")


def week_totals(board: pandas.DataFrame) -> pandas.DataFrame:
    if board.empty:
        return pandas.DataFrame(columns=["slot", "games", "flags"])
    slotted = _with_slot(board)
    grouped = slotted.groupby("slot", as_index=False).agg(games=("game_id", "nunique"), flags=("flags", "mean"))
    return _order_slots(grouped)


def team_week_totals(board: pandas.DataFrame, team: str) -> pandas.DataFrame:
    columns = ["season", "week", "season_type", "game_id", "flags"]
    home = board.loc[board["home_team"].eq(team), columns + ["home_flags"]]
    home = home.rename(columns={"home_flags": "team_flags"})
    away = board.loc[board["away_team"].eq(team), columns + ["away_flags"]]
    away = away.rename(columns={"away_flags": "team_flags"})
    played = pandas.concat([home, away], ignore_index=True)
    if played.empty:
        return pandas.DataFrame(columns=["slot", "games", "team_flags", "flags"])
    played = _with_slot(played)
    grouped = played.groupby("slot", as_index=False).agg(
        games=("game_id", "nunique"),
        team_flags=("team_flags", "mean"),
        flags=("flags", "mean"),
    )
    return _order_slots(grouped)


def month_totals(board: pandas.DataFrame) -> pandas.DataFrame:
    if board.empty or "month" not in board.columns:
        return pandas.DataFrame(columns=["month", "games", "flags"])
    grouped = board.dropna(subset=["month"]).groupby("month", as_index=False).agg(
        games=("game_id", "nunique"),
        flags=("flags", "mean"),
    )
    grouped["month"] = pandas.Categorical(grouped["month"], categories=MONTH_ORDER, ordered=True)
    return grouped.sort_values("month")


def week_window(board: pandas.DataFrame, last_early_week: int = 5) -> tuple[pandas.Series, pandas.Series]:
    regular = board.loc[board["season_type"].eq("REG")] if "season_type" in board.columns else board
    early = regular.loc[regular["week"].le(last_early_week), "flags"]
    later = regular.loc[regular["week"].gt(last_early_week) & regular["week"].le(18), "flags"]
    return early, later


def opening_games(board: pandas.DataFrame, team: str | None = None) -> tuple[pandas.Series, pandas.Series]:
    """A club's first five games in a season, then the rest. Game totals, each game once."""
    if board.empty:
        empty = pandas.Series(dtype=float)
        return empty, empty
    if team:
        played = board.loc[board["home_team"].eq(team) | board["away_team"].eq(team)].copy()
        played = played.sort_values(["season", "week", "game_id"])
        played["game_n"] = played.groupby("season").cumcount() + 1
        return played.loc[played["game_n"].le(5), "flags"], played.loc[played["game_n"].gt(5), "flags"]
    home = board.loc[:, ["season", "week", "game_id", "home_team", "flags"]].rename(columns={"home_team": "team"})
    away = board.loc[:, ["season", "week", "game_id", "away_team", "flags"]].rename(columns={"away_team": "team"})
    played = pandas.concat([home, away], ignore_index=True).sort_values(["season", "team", "week", "game_id"])
    played["game_n"] = played.groupby(["season", "team"]).cumcount() + 1
    early_ids = set(played.loc[played["game_n"].le(5), "game_id"])
    later_ids = set(played.loc[played["game_n"].gt(5), "game_id"])
    # A game can be an opening game for one club and a later game for the other.
    # Keep it with the opening set when either club is still in its first five.
    later_ids -= early_ids
    return (
        board.loc[board["game_id"].isin(early_ids), "flags"],
        board.loc[board["game_id"].isin(later_ids), "flags"],
    )


def club_opening_flags(games: pandas.DataFrame, penalties: pandas.DataFrame) -> tuple[pandas.Series, pandas.Series]:
    """Flags charged to a club in its first five games, then later games."""
    played = games.drop_duplicates(["season", "game_id", "team"]).sort_values(["season", "team", "week", "game_id"])
    if played.empty:
        empty = pandas.Series(dtype=float)
        return empty, empty
    counts = penalties.groupby(["game_id", "team"]).size()
    played = played.copy()
    played["team_flags"] = [
        int(counts.get((game_id, team), 0))
        for game_id, team in zip(played["game_id"], played["team"], strict=False)
    ]
    played["game_n"] = played.groupby(["season", "team"]).cumcount() + 1
    return (
        played.loc[played["game_n"].le(5), "team_flags"],
        played.loc[played["game_n"].gt(5), "team_flags"],
    )


def quarter_breakdown(board: pandas.DataFrame, penalties: pandas.DataFrame) -> pandas.DataFrame:
    if board.empty or "qtr" not in penalties.columns:
        return pandas.DataFrame(columns=["quarter", "flags", "per_game", "games"])
    game_ids = set(board["game_id"])
    flagged = penalties.loc[penalties["game_id"].isin(game_ids) & penalties["qtr"].notna()]
    counts = flagged.groupby("qtr").size()
    n_games = int(board["game_id"].nunique())
    ot_games = 0
    if "final_qtr" in board.columns:
        ot_games = int(board.loc[board["final_qtr"].fillna(0).ge(5), "game_id"].nunique())
    rows = []
    for number, name in QUARTER_NAMES.items():
        flags = int(counts.get(number, 0))
        games = ot_games if number == 5 else n_games
        per_game = None if games == 0 else flags / games
        rows.append({"quarter": name, "flags": flags, "per_game": per_game, "games": games})
    return pandas.DataFrame(rows)


def game_pace(games: pandas.DataFrame, penalties: pandas.DataFrame) -> GamePace:
    rows = attach_flags(games, penalties)
    grouped = rows.groupby(["season", "season_type"], as_index=False).agg(
        games=("game_id", "nunique"),
        flags=("flags", "sum"),
    )
    grouped["flags_per_game"] = grouped["flags"] / grouped["games"]
    regular = grouped.loc[grouped["season_type"].eq("REG")].sort_values("season")
    finished = regular.loc[regular["season"].lt(int(regular["season"].max()))]
    current = regular.iloc[-1]
    peak = finished.loc[finished["flags_per_game"].idxmax()]
    recent = finished.loc[finished["season"].ge(int(current["season"]) - 2)]
    playoff = grouped.loc[grouped["season_type"].eq("POST")]
    shared = set(finished["season"]).intersection(playoff["season"])
    regular_shared = finished.loc[finished["season"].isin(shared)]
    playoff_shared = playoff.loc[playoff["season"].isin(shared)]
    return GamePace(
        by_scope=grouped,
        current_season=int(current["season"]),
        current_flags=float(current["flags_per_game"]),
        current_games=int(current["games"]),
        prior_peak_season=int(peak["season"]),
        prior_peak_flags=float(peak["flags_per_game"]),
        recent_flags=float(recent["flags_per_game"].mean()),
        regular_flags=float(regular_shared["flags"].sum() / regular_shared["games"].sum()),
        playoff_flags=float(playoff_shared["flags"].sum() / playoff_shared["games"].sum()),
        compared_min=int(min(shared)),
        compared_max=int(max(shared)),
    )


def pace_reading(pace: GamePace) -> str:
    gap = pace.current_flags - pace.prior_peak_flags
    if gap > 0:
        place = (
            f"That is {gap:.2f} flags above {pace.prior_peak_season}, the highest finished season in this file"
        )
    else:
        place = (
            f"{pace.current_season} is {abs(gap):.2f} flags under the {pace.prior_peak_season} high"
        )
    return (
        f"Both teams' flags count in one game. In the last two finished seasons that average was "
        f"{pace.recent_flags:.1f}. Through {pace.current_games} games, {pace.current_season} is at "
        f"{pace.current_flags:.1f}. {place}. From {pace.compared_min} to {pace.compared_max}, "
        f"regular-season games averaged {pace.regular_flags:.1f} flags and playoff games averaged "
        f"{pace.playoff_flags:.1f}."
    )


def quarter_matchups(games: pandas.DataFrame, penalties: pandas.DataFrame) -> MatchupCut:
    rows = attach_flags(games, penalties)
    played = games.drop_duplicates(["season", "game_id", "team"])
    counted = penalties.groupby(["season", "team"]).size()
    n_games = played.groupby(["season", "team"]).size()
    if n_games.empty:
        return MatchupCut(None, 0, None, 0, (), (), 0)
    rates = counted.reindex(n_games.index).fillna(0) / n_games
    heavy_ids: list[str] = []
    light_ids: list[str] = []
    heavy_names: set[str] = set()
    light_names: set[str] = set()
    seasons_used = 0
    one_season = (not rates.empty) and rates.index.get_level_values("season").nunique() == 1
    for season, season_rates in rates.groupby(level=0):
        season_rates = season_rates.droplevel(0).sort_values(ascending=False)
        if len(season_rates) < 8:
            continue
        seasons_used += 1
        count = max(1, len(season_rates) // 4)
        heavy = set(season_rates.head(count).index)
        light = set(season_rates.tail(count).index)
        overlap = heavy & light
        heavy -= overlap
        light -= overlap
        if one_season:
            heavy_names = set(heavy)
            light_names = set(light)
        season_games = rows.loc[rows["season"].eq(season)]
        both_heavy = season_games["home_team"].isin(heavy) & season_games["away_team"].isin(heavy)
        both_light = season_games["home_team"].isin(light) & season_games["away_team"].isin(light)
        heavy_ids.extend(season_games.loc[both_heavy, "game_id"].tolist())
        light_ids.extend(season_games.loc[both_light, "game_id"].tolist())

    def _mean(ids: list[str]) -> tuple[float | None, int]:
        if not ids:
            return None, 0
        picked = rows.loc[rows["game_id"].isin(ids), "flags"]
        return float(picked.mean()), int(len(picked))

    heavy_flags, heavy_games = _mean(heavy_ids)
    light_flags, light_games = _mean(light_ids)
    return MatchupCut(
        heavy_flags=heavy_flags,
        heavy_games=heavy_games,
        light_flags=light_flags,
        light_games=light_games,
        heavy_teams=tuple(sorted(heavy_names)),
        light_teams=tuple(sorted(light_names)),
        seasons=seasons_used,
    )


def matchup_reading(cut: MatchupCut, scope: str) -> str:
    if cut.heavy_flags is None or cut.light_flags is None:
        return (
            f"{scope} Not enough clubs in this slice to split a top quarter from a bottom quarter."
        )
    gap = cut.heavy_flags - cut.light_flags
    small = ""
    if cut.heavy_games < 20 or cut.light_games < 20:
        small = " Those samples are small."
    return (
        f"{scope} When both teams are in that season's top quarter, the game averages "
        f"{cut.heavy_flags:.1f} flags across {cut.heavy_games} games. "
        f"When both are in the bottom quarter, it averages {cut.light_flags:.1f} flags "
        f"across {cut.light_games} games. The gap is {gap:.1f} flags.{small}"
    )


LABEL_BUNDLES = (
    ("Line calls at the snap", ("Defensive Offside", "Neutral Zone Infraction", "Encroachment")),
    ("Personal foul or unnecessary roughness", ("Personal Foul", "Unnecessary Roughness")),
    ("Too many men", ("Defensive 12 On-field", "Defensive Too Many Men on Field")),
    ("Face mask", ("Face Mask", "Face Mask (5 Yards)")),
)


@dataclass(frozen=True)
class StoryFacts:
    early_flags: float
    season_2004: float
    season_2005: float
    plateau_mean: float
    plateau_low: float
    plateau_high: float
    season_2020: float
    low_mean: float
    low_low: float
    low_high: float
    rebound_flags: float
    season_sd: float
    covid_p: float
    rtp_early: float
    rtp_2019: float
    rtp_2021: float
    rtp_late: float
    ic_early: float
    ic_2004: float
    ic_late: float
    holding_early: float
    holding_late: float
    taunt_peak_season: int
    taunt_peak_calls: int
    taunt_late_calls: float
    dpi_early_calls: float
    dpi_late_calls: float
    reg_flags: float
    post_flags: float
    seasons_compared: int
    reg_higher: int
    post_higher_seasons: tuple[int, ...]
    super_bowl: float
    other_playoffs: float
    super_bowl_games: int
    super_bowl_p: float
    modern_games: int
    movers: pandas.DataFrame
    season_rates: pandas.DataFrame
    call_season: pandas.DataFrame
    loud_games: pandas.DataFrame


def _regular_games(games: pandas.DataFrame) -> pandas.DataFrame:
    return games.loc[games["season_type"].eq("REG")].drop_duplicates("game_id")


def _window_flags(board: pandas.DataFrame, start: int, end: int) -> pandas.Series:
    return board.loc[
        board["season_type"].eq("REG") & board["season"].between(start, end),
        "flags",
    ]


def _calls_per_game(penalties: pandas.DataFrame, games: pandas.DataFrame, names: tuple[str, ...], start: int, end: int) -> float:
    played = games.loc[games["season"].between(start, end)]
    flagged = penalties.loc[
        penalties["season_type"].eq("REG")
        & penalties["season"].between(start, end)
        & penalties["penalty_type"].isin(names)
    ]
    game_count = int(played["game_id"].nunique())
    if game_count == 0:
        return 0.0
    return len(flagged) / game_count


def story_facts(games: pandas.DataFrame, penalties: pandas.DataFrame) -> StoryFacts:
    from scipy import stats

    board = game_board(games, penalties)
    regular = board.loc[board["season_type"].eq("REG")]
    by_season = regular.groupby("season")["flags"].mean()
    finished = by_season.loc[by_season.index < int(by_season.index.max())]
    plateau = by_season.loc[2015:2019]
    low = by_season.loc[2020:2023]
    covid = stats.ttest_ind(plateau.to_numpy(), low.to_numpy(), equal_var=False)

    played = _regular_games(games)
    modern = played.loc[played["season"].between(2023, 2025)].groupby("season").size()
    modern_games = int(round(float(modern.mean()))) if not modern.empty else 272
    reg_pen = penalties.loc[penalties["season_type"].eq("REG")]

    def scaled(names: tuple[str, ...], start: int, end: int) -> float:
        return _calls_per_game(reg_pen, played, names, start, end) * modern_games

    rtp = reg_pen.loc[reg_pen["penalty_type"].eq("Roughing the Passer")].groupby("season").size()
    games_n = played.groupby("season").size()
    rtp_per = rtp / games_n
    taunt = reg_pen.loc[reg_pen["penalty_type"].eq("Taunting")].groupby("season").size()

    post = board.loc[board["season_type"].eq("POST")].copy()
    post["slot"] = [
        schedule_slot(season, week, season_type)
        for season, week, season_type in zip(post["season"], post["week"], post["season_type"], strict=False)
    ]
    super_bowl = post.loc[post["slot"].eq("Super Bowl"), "flags"]
    other = post.loc[post["slot"].ne("Super Bowl") & post["slot"].notna(), "flags"]
    bowl_test = stats.ttest_ind(super_bowl.to_numpy(), other.to_numpy(), equal_var=False)

    compared = regular.groupby("season")["flags"].mean().to_frame("REG")
    compared["POST"] = post.groupby("season")["flags"].mean()
    compared = compared.dropna()
    compared = compared.loc[compared.index < int(by_season.index.max())]
    gap = compared["REG"] - compared["POST"]

    bundled = {name for _label, names in LABEL_BUNDLES for name in names}
    rows = []
    for label, names in LABEL_BUNDLES:
        early = scaled(names, 2000, 2003)
        late = scaled(names, 2023, 2025)
        rows.append({"Call": label, "2000-2003": early, "2023-2025": late, "Change": late - early})
    counts = reg_pen.groupby("penalty_type").size()
    for name in counts.index:
        if name in bundled:
            continue
        early = scaled((name,), 2000, 2003)
        late = scaled((name,), 2023, 2025)
        if max(early, late) < 20:
            continue
        rows.append({"Call": name, "2000-2003": early, "2023-2025": late, "Change": late - early})
    movers = pandas.DataFrame(rows).sort_values("Change", ascending=False)
    season_rates = (
        regular.groupby("season", as_index=False)
        .agg(games=("game_id", "nunique"), flags=("flags", "mean"), yards=("yards", "mean"))
    )
    call_season = (
        reg_pen.groupby(["season", "penalty_type"], as_index=False)
        .agg(calls=("penalty_type", "size"), yards=("penalty_yards", "sum"))
    )
    loud = board.copy()
    loud["flag_rank"] = loud["flags"].rank(method="min", ascending=False).astype(int)
    loud["yard_rank"] = loud["yards"].rank(method="min", ascending=False).astype(int)
    loud["flag_tied"] = loud.groupby("flags")["game_id"].transform("size").gt(1)
    loud["yard_tied"] = loud.groupby("yards")["game_id"].transform("size").gt(1)
    current_season = int(board["season"].max())
    loud_games = loud.loc[
        loud["season"].eq(current_season) & (loud["flag_rank"].le(15) | loud["yard_rank"].le(15)),
        [
            "season",
            "week",
            "season_type",
            "away_team",
            "home_team",
            "flags",
            "yards",
            "flag_rank",
            "yard_rank",
            "flag_tied",
            "yard_tied",
        ],
    ].sort_values(["flag_rank", "yard_rank"])

    return StoryFacts(
        early_flags=float(_window_flags(board, 1999, 2003).mean()),
        season_2004=float(by_season.loc[2004]),
        season_2005=float(by_season.loc[2005]),
        plateau_mean=float(plateau.mean()),
        plateau_low=float(plateau.min()),
        plateau_high=float(plateau.max()),
        season_2020=float(by_season.loc[2020]),
        low_mean=float(low.mean()),
        low_low=float(low.min()),
        low_high=float(low.max()),
        rebound_flags=float(by_season.loc[2024:2025].mean()),
        season_sd=float(finished.std()),
        covid_p=float(covid.pvalue),
        rtp_early=float(rtp_per.loc[2000:2003].mean()),
        rtp_2019=float(rtp_per.loc[2019]),
        rtp_2021=float(rtp_per.loc[2021]),
        rtp_late=float(rtp_per.loc[2023:2025].mean()),
        ic_early=_calls_per_game(reg_pen, played, ("Illegal Contact",), 2000, 2003),
        ic_2004=_calls_per_game(reg_pen, played, ("Illegal Contact",), 2004, 2004),
        ic_late=_calls_per_game(reg_pen, played, ("Illegal Contact",), 2023, 2025),
        holding_early=_calls_per_game(reg_pen, played, ("Defensive Holding",), 2000, 2003),
        holding_late=_calls_per_game(reg_pen, played, ("Defensive Holding",), 2023, 2025),
        taunt_peak_season=int(taunt.idxmax()),
        taunt_peak_calls=int(taunt.max()),
        taunt_late_calls=scaled(("Taunting",), 2023, 2025),
        dpi_early_calls=scaled(("Defensive Pass Interference",), 2000, 2003),
        dpi_late_calls=scaled(("Defensive Pass Interference",), 2023, 2025),
        reg_flags=float(regular.loc[regular["season"].isin(compared.index), "flags"].mean()),
        post_flags=float(post.loc[post["season"].isin(compared.index), "flags"].mean()),
        seasons_compared=int(len(compared)),
        reg_higher=int((gap > 0).sum()),
        post_higher_seasons=tuple(int(season) for season in gap.loc[gap < 0].index),
        super_bowl=float(super_bowl.mean()),
        other_playoffs=float(other.mean()),
        super_bowl_games=int(len(super_bowl)),
        super_bowl_p=float(bowl_test.pvalue),
        modern_games=modern_games,
        movers=movers,
        season_rates=season_rates,
        call_season=call_season,
        loud_games=loud_games,
    )


def _check() -> None:
    games, penalties = load_frames()
    study = run_study(games, penalties)
    if study.volume.n_seasons < 20:
        raise SystemExit(f"Expected 20+ seasons, got {study.volume.n_seasons}")
    baseline = study.volume.term("const")
    first = study.season.loc[study.season["season"].eq(YEAR_ZERO), "fitted"]
    if first.empty or abs(float(first.iloc[0]) - baseline.estimate) > 1e-8:
        raise SystemExit("Fitted 2000 rate does not match the intercept")
    print(volume_reading(study))
    print(type_reading(study))
    print(limit_reading(study))
    for term in study.volume.terms:
        print(
            f"{term.label:16} {term.estimate:8.3f}  ({term.ci_low:.3f}, {term.ci_high:.3f})  p={term.pvalue:.4f}"
        )


if __name__ == "__main__":
    _check()
