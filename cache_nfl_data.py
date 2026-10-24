"""Download nflverse data used by the penalties-per-game project.

Play-by-play uses nfl.cache_pbp(). Seasons that file skips (the 2026
participation file is not published yet) are saved from play-by-play alone.

nfl.import_weekly_data() still reads player_stats_{year}.parquet, which
stops at 2024. The current weekly release is stats_player_week_{year}.parquet
and runs through 2026, so both are saved.

Snap counts and seasonal rosters come from the matching nfl_data_py imports.
Play-by-play, which feeds the penalty tables, starts in 1999.
Snap counts begin in 2012, so those files keep that later window.
"""

import shutil
import sys
from datetime import date, timedelta
from pathlib import Path
from urllib.error import HTTPError

import pandas
import nfl_data_py as nfl

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data" / "cache"
PBP_PATH = CACHE / "pbp"
PBP_YEARS = list(range(1999, 2027))
OTHER_YEARS = list(range(2012, 2027))


def say(message: str) -> None:
    """Log a line. A Windows console under Streamlit can reject flush."""
    stream = sys.stdout
    if stream is None:
        return
    try:
        stream.write(message + "\n")
        stream.flush()
    except OSError:
        return


def guard_stdout() -> None:
    """Keep say() from aborting the page when the console flush is invalid."""
    stream = sys.stdout
    if stream is None or getattr(stream, "_nfl_guard", False):
        return
    write = stream.write
    flush = stream.flush

    def safe_write(text):
        try:
            return write(text)
        except OSError:
            return 0

    def safe_flush():
        try:
            flush()
        except OSError:
            return None

    stream.write = safe_write
    stream.flush = safe_flush
    stream._nfl_guard = True

PBP_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "pbp/play_by_play_{}.parquet"
)
PART_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "pbp_participation/pbp_participation_{}.parquet"
)
WEEKLY_CURRENT_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "stats_player/stats_player_week_{}.parquet"
)


def cached_pbp_seasons() -> set[int]:
    seasons = set()
    for path in PBP_PATH.glob("season=*"):
        seasons.add(int(path.name.split("=", 1)[1]))
    return seasons


def read_parquet_or_skip(url: str, year: int):
    try:
        return pandas.read_parquet(url.format(year))
    except HTTPError as exc:
        say(f"Missing {year}: {url.format(year)} ({exc.code})")
        return None


def downcast_floats(frame: pandas.DataFrame) -> pandas.DataFrame:
    cols = frame.select_dtypes(include=["float64"]).columns
    frame[cols] = frame[cols].astype("float32")
    return frame


def ensure_pbp(years: list[int]) -> None:
    missing = [year for year in years if year not in cached_pbp_seasons()]
    if missing:
        say(f"Caching play-by-play for {missing}...")
        nfl.cache_pbp(missing, downcast=True, alt_path=str(PBP_PATH))

    still_missing = [year for year in years if year not in cached_pbp_seasons()]
    for year in still_missing:
        say(f"Saving {year} play-by-play without participation...")
        raw = read_parquet_or_skip(PBP_URL, year)
        if raw is None:
            continue
        raw["season"] = year
        if year >= 2016:
            part = read_parquet_or_skip(PART_URL, year)
            if part is not None:
                raw = raw.merge(part, how="left", on=["play_id", "old_game_id"])
        raw = downcast_floats(raw)
        raw.to_parquet(PBP_PATH, partition_cols="season")
        say(f"{year} done.")

    say(f"Play-by-play seasons: {sorted(cached_pbp_seasons())}")


def object_columns_to_string(frame: pandas.DataFrame) -> pandas.DataFrame:
    for col in frame.select_dtypes(include="object").columns:
        frame[col] = frame[col].map(lambda value: pandas.NA if pandas.isna(value) else str(value))
        frame[col] = frame[col].astype("string")
    return frame


def cache_legacy_weekly(years: list[int]) -> None:
    path = CACHE / "weekly_stats.parquet"
    if path.exists():
        say(f"Legacy weekly stats already saved: {path.name}")
        return
    say("Caching weekly stats from nfl.import_weekly_data...")
    frames = []
    for year in years:
        try:
            frames.append(nfl.import_weekly_data([year], downcast=True))
            say(f"weekly {year} done.")
        except HTTPError as exc:
            say(f"weekly {year} not in legacy release ({exc.code})")
    if not frames:
        return
    weekly = pandas.concat(frames, ignore_index=True)
    weekly.to_parquet(path, index=False)
    say(f"Legacy weekly stats: {len(weekly):,} rows -> {path.name}")


def cache_current_weekly(years: list[int]) -> None:
    path = CACHE / "weekly_player_stats.parquet"
    if path.exists():
        say(f"Current weekly stats already saved: {path.name}")
        return
    say("Caching current nflverse weekly player stats...")
    frames = []
    for year in years:
        frame = read_parquet_or_skip(WEEKLY_CURRENT_URL, year)
        if frame is None:
            continue
        frames.append(downcast_floats(frame))
        say(f"player week {year} done.")
    if not frames:
        return
    weekly = pandas.concat(frames, ignore_index=True)
    weekly.to_parquet(path, index=False)
    say(f"Current weekly stats: {len(weekly):,} rows -> {path.name}")


def cache_snaps(years: list[int]) -> None:
    path = CACHE / "snap_counts.parquet"
    if path.exists():
        say(f"Snap counts already saved: {path.name}")
        return
    say("Caching snap counts...")
    frames = []
    for year in years:
        try:
            frames.append(nfl.import_snap_counts([year]))
            say(f"snaps {year} done.")
        except HTTPError as exc:
            say(f"snaps {year} missing ({exc.code})")
    snaps = pandas.concat(frames, ignore_index=True)
    snaps.to_parquet(path, index=False)
    say(f"Snap counts: {len(snaps):,} rows -> {path.name}")


def cache_rosters(years: list[int]) -> None:
    say("Caching seasonal rosters...")
    frames = []
    for year in years:
        try:
            frames.append(nfl.import_seasonal_rosters([year]))
            say(f"rosters {year} done.")
        except HTTPError as exc:
            say(f"rosters {year} missing ({exc.code})")
    rosters = object_columns_to_string(pandas.concat(frames, ignore_index=True))
    path = CACHE / "rosters_seasonal.parquet"
    rosters.to_parquet(path, index=False)
    say(f"Rosters: {len(rosters):,} rows -> {path.name}")


def nfl_season(today: date) -> int:
    if today.month < 3:
        return today.year - 1
    return today.year


def due_tuesday(today: date) -> date:
    offset = (today.weekday() - 1) % 7
    return today - timedelta(days=offset)


def in_season_window(today: date) -> bool:
    return today.month >= 9 or today.month <= 2


def replace_pbp_season(year: int) -> bool:
    folder = PBP_PATH / f"season={year}"
    holding = PBP_PATH / f"_hold_{year}"
    if holding.exists():
        shutil.rmtree(holding)
    if folder.exists():
        folder.rename(holding)
    try:
        ensure_pbp([year])
    except Exception:
        if holding.exists() and not folder.exists():
            holding.rename(folder)
        raise
    if year not in cached_pbp_seasons():
        if folder.exists():
            shutil.rmtree(folder)
        if holding.exists():
            holding.rename(folder)
        return False
    if holding.exists():
        shutil.rmtree(holding)
    return True


def refresh_is_due(today: date | None = None) -> bool:
    today = today or date.today()
    if not in_season_window(today):
        return False
    # The hosted app has the built tables and not the raw download.
    # A missing cache means there is nothing to fold a new week into.
    if not PBP_PATH.exists():
        return False
    stamp = CACHE / "last_tuesday_refresh.txt"
    if not stamp.exists():
        return True
    try:
        last = date.fromisoformat(stamp.read_text(encoding="utf-8").strip())
    except ValueError:
        return True
    return last < due_tuesday(today)


def refresh_due(today: date | None = None) -> bool:
    """Re-download the current season once each Tuesday and fold in the new week.

    Older seasons stay on disk. September through February only.
    """
    guard_stdout()
    today = today or date.today()
    if not refresh_is_due(today):
        return False
    year = nfl_season(today)
    say(f"Tuesday refresh for {year}...")
    if not replace_pbp_season(year):
        say(f"No play-by-play file for {year}.")
        return False
    from penalties import replace_season

    replace_season(year)
    stamp = CACHE / "last_tuesday_refresh.txt"
    stamp.write_text(due_tuesday(today).isoformat(), encoding="utf-8")
    say(f"Updated {year} through the latest week in the file.")
    return True


def main() -> None:
    if "--week" in sys.argv:
        CACHE.mkdir(parents=True, exist_ok=True)
        PBP_PATH.mkdir(parents=True, exist_ok=True)
        refresh_due()
        return
    CACHE.mkdir(parents=True, exist_ok=True)
    PBP_PATH.mkdir(parents=True, exist_ok=True)
    say(f"Play-by-play years: {PBP_YEARS[0]}-{PBP_YEARS[-1]}")
    say(f"Other files: {OTHER_YEARS[0]}-{OTHER_YEARS[-1]}")
    say(f"Cache folder: {CACHE}")
    ensure_pbp(PBP_YEARS)
    cache_legacy_weekly(OTHER_YEARS)
    cache_current_weekly(OTHER_YEARS)
    cache_snaps(OTHER_YEARS)
    cache_rosters(OTHER_YEARS)
    say("Done.")


if __name__ == "__main__":
    main()
