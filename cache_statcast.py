"""Download regular-season balls in play and cache the columns this paper uses.

Statcast is pitch-level. A season is large, so this keeps batted balls only:
exit velocity, launch angle, the wOBA weight of the result, Savant's expected
value on that ball, and the spray coordinates for a later model.

    .\\.venv\\Scripts\\python.exe cache_statcast.py

Parts land in data/baseball/parts. A finished or partial run also writes
data/baseball/statcast_bip.parquet, which the baseball page reads.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas
from pybaseball import cache as pybaseball_cache
from pybaseball import statcast

ROOT = Path(__file__).resolve().parent
PARTS = ROOT / "data" / "baseball" / "parts"
OUT = ROOT / "data" / "baseball" / "statcast_bip.parquet"

# Regular season only. Spring training and the postseason stay out.
SEASONS = (
    (date(2024, 3, 28), date(2024, 9, 30)),
    (date(2025, 3, 27), date(2025, 9, 29)),
)

KEEP = (
    "game_date",
    "game_pk",
    "game_type",
    "at_bat_number",
    "pitch_number",
    "player_name",
    "batter",
    "home_team",
    "away_team",
    "events",
    "description",
    "bb_type",
    "launch_speed",
    "launch_angle",
    "hc_x",
    "hc_y",
    "hit_distance_sc",
    "woba_value",
    "woba_denom",
    "estimated_ba_using_speedangle",
    "estimated_woba_using_speedangle",
    "launch_speed_angle",
    "stand",
    "p_throws",
    "inning",
    "if_fielding_alignment",
    "of_fielding_alignment",
)

NUMERIC = (
    "launch_speed",
    "launch_angle",
    "hc_x",
    "hc_y",
    "hit_distance_sc",
    "woba_value",
    "woba_denom",
    "estimated_ba_using_speedangle",
    "estimated_woba_using_speedangle",
    "launch_speed_angle",
    "batter",
    "inning",
)


def months(start: date, end: date):
    cursor = date(start.year, start.month, 1)
    while cursor <= end:
        if cursor.month == 12:
            nxt = date(cursor.year + 1, 1, 1)
        else:
            nxt = date(cursor.year, cursor.month + 1, 1)
        yield max(cursor, start), min(end, nxt - timedelta(days=1))
        cursor = nxt


def balls_in_play(frame: pandas.DataFrame) -> pandas.DataFrame:
    if frame is None or frame.empty:
        return pandas.DataFrame(columns=list(KEEP))
    columns = [name for name in KEEP if name in frame.columns]
    kept = frame.loc[:, columns].copy()
    if "game_type" in kept.columns:
        kept = kept.loc[kept["game_type"].astype(str).eq("R")]
    for name in NUMERIC:
        if name in kept.columns:
            kept[name] = pandas.to_numeric(kept[name], errors="coerce")
    if "description" in kept.columns:
        bunt = kept["description"].astype(str).str.contains("bunt", case=False, na=False)
        kept = kept.loc[~bunt]
    needed = [name for name in ("launch_speed", "launch_angle", "woba_value") if name in kept.columns]
    kept = kept.dropna(subset=needed)
    kept = kept.loc[kept["launch_speed"].between(20, 130) & kept["launch_angle"].between(-90, 90)]
    keys = [name for name in ("game_pk", "at_bat_number", "pitch_number") if name in kept.columns]
    if keys:
        kept = kept.drop_duplicates(keys)
    return kept.reset_index(drop=True)


def part_path(start: date) -> Path:
    return PARTS / f"{start.year}-{start.month:02d}.parquet"


def download() -> Path:
    pybaseball_cache.enable()
    PARTS.mkdir(parents=True, exist_ok=True)
    for start, end in SEASONS:
        for month_start, month_end in months(start, end):
            path = part_path(month_start)
            if path.exists():
                print(f"keep {path.name}", flush=True)
                continue
            print(f"pull {month_start} to {month_end}", flush=True)
            raw = statcast(str(month_start), str(month_end), verbose=False, parallel=True)
            balls = balls_in_play(raw)
            balls.to_parquet(path, index=False)
            print(f"  {len(balls):,} balls -> {path.name}", flush=True)
            combine()
    return OUT


def combine() -> Path:
    parts = sorted(PARTS.glob("*.parquet"))
    if not parts:
        raise SystemExit(f"No month files in {PARTS}")
    frame = pandas.concat([pandas.read_parquet(path) for path in parts], ignore_index=True)
    keys = [name for name in ("game_pk", "at_bat_number", "pitch_number") if name in frame.columns]
    if keys:
        frame = frame.drop_duplicates(keys)
    frame["game_date"] = pandas.to_datetime(frame["game_date"])
    frame["season"] = frame["game_date"].dt.year.astype(int)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUT, index=False)
    seasons = ", ".join(str(year) for year in sorted(frame["season"].unique()))
    print(f"{len(frame):,} balls in play, seasons {seasons}, wrote {OUT}", flush=True)
    return OUT


ZONE_PARTS = ROOT / "data" / "baseball" / "zone_parts"
ZONE_COLUMNS = ("game_pk", "at_bat_number", "pitch_number", "plate_x", "plate_z", "sz_top", "sz_bot")


def download_zone() -> Path:
    """Add pitch location for a strike-zone grid. The balls-in-play file stays the base."""
    pybaseball_cache.enable()
    ZONE_PARTS.mkdir(parents=True, exist_ok=True)
    for start, end in SEASONS:
        for month_start, month_end in months(start, end):
            path = ZONE_PARTS / f"{month_start.year}-{month_start.month:02d}.parquet"
            if path.exists():
                print(f"keep zone {path.name}", flush=True)
                continue
            print(f"pull zone {month_start} to {month_end}", flush=True)
            raw = statcast(str(month_start), str(month_end), verbose=False, parallel=True)
            if raw is None or raw.empty:
                pandas.DataFrame(columns=list(ZONE_COLUMNS)).to_parquet(path, index=False)
                continue
            columns = [name for name in ZONE_COLUMNS if name in raw.columns]
            kept = raw.loc[:, columns].copy()
            for name in columns:
                kept[name] = pandas.to_numeric(kept[name], errors="coerce")
            keys = [name for name in ("game_pk", "at_bat_number", "pitch_number") if name in kept.columns]
            if keys:
                kept = kept.drop_duplicates(keys)
            kept.to_parquet(path, index=False)
            print(f"  {len(kept):,} pitches -> {path.name}", flush=True)
    return merge_zone()


def merge_zone() -> Path:
    parts = sorted(ZONE_PARTS.glob("*.parquet"))
    if not parts or not OUT.exists():
        raise SystemExit("Need zone month files and the balls-in-play parquet.")
    zone = pandas.concat([pandas.read_parquet(path) for path in parts], ignore_index=True)
    keys = ["game_pk", "at_bat_number", "pitch_number"]
    zone = zone.drop_duplicates(keys)
    balls = pandas.read_parquet(OUT)
    extra = [name for name in ("plate_x", "plate_z", "sz_top", "sz_bot") if name in balls.columns]
    if extra:
        balls = balls.drop(columns=extra)
    merged = balls.merge(zone.loc[:, [name for name in ZONE_COLUMNS if name in zone.columns]], on=keys, how="left")
    merged.to_parquet(OUT, index=False)
    covered = float(merged["plate_x"].notna().mean()) if "plate_x" in merged.columns else 0.0
    print(f"pitch location on {covered:.0%} of balls, wrote {OUT}", flush=True)
    return OUT


if __name__ == "__main__":
    import sys

    if "--zone" in sys.argv:
        download_zone()
    else:
        download()
