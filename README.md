# Analytics

The app opens on a subject index: baseball, basketball, football, and energy. Open a subject to see the projects inside it. Baseball holds a draft of the expected-contact model, fit on one season of balls in play and scored on the next. Basketball asks whether a back-to-back costs a team its shot, then follows that question into a few clubs and a few players. Football holds the penalties page, which walks regular-season flags from 1999 through the season still being played.

## Open it

From this folder:

```
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The index is at http://localhost:8501. The penalties page is at http://localhost:8501/nfl-penalties, and linked from Football. The contact draft is at http://localhost:8501/expected-contact, and linked from Baseball. The efficiency page is at http://localhost:8501/nba-logs, and linked from Basketball. On the penalties page, the story uses every regular season. One chart lower on the page, Try a slice, redraws that chart, the game table, and the team ranking. The note under the team chart is saved in `notes.json` when you click Save note.

## Add a project

In `app.py`, add an entry to that subject's list in `CATALOG`. It needs a title, a one-line description, a `url_path`, and a `render` function for the full page. The index lists it on the next reload.

Edit `app.py` to change colors, chart layout, or the counting text. Streamlit reloads the page when you save the file.

## Refresh the data

Opening the page during the season (September through February) checks the most recent Tuesday. If this week's file is not in yet, it downloads only the current season and adds that week's games. Older seasons stay put.

The same check from the command line. The download library pins an old pandas, so install it with dependencies turned off:

```
.\.venv\Scripts\python.exe -m pip install -r requirements-nfl.txt
.\.venv\Scripts\python.exe cache_nfl_data.py --week
```

A full rebuild, every season:

```
.\.venv\Scripts\python.exe cache_nfl_data.py
.\.venv\Scripts\python.exe penalties.py
```

## Baseball

Whose contact was better reads cached regular-season balls in play. The draft file is the 2024 and 2025 seasons. A gradient boosting model learns the wOBA weight of a ball from exit velocity and launch angle, then scores the later season against the results and against Baseball Savant's expected value on the same balls.

pybaseball and scikit-learn are in `requirements.txt`. Rebuild the file with:

```
.\.venv\Scripts\python.exe cache_statcast.py
```

The page is at http://localhost:8501/expected-contact. Month files sit in `data/baseball/parts`. The page reads `data/baseball/statcast_bip.parquet`.

## Basketball

The basketball page asks whether a back-to-back costs efficiency. The shot moves about a point. The win rate moves several points, and that is the chart the page leans on. It then names the clubs at the ends, the three positions, a few players, and a picker for one club. A back-to-back is a game on the day after that club's previous game. For a player, a second night means he himself played the day before. True shooting pools every shot in the group. The first game of a season stays out of the rest charts. Home and road come from the matchup and show up in the text, without a chart of their own.

The cache starts at 2021-22. Regular season only. A season with fewer than 400 team-games stays on disk and out of the pools. Shot-zone files stay season totals, so they are not part of this story.

```
.\.venv\Scripts\python.exe cache_nba_data.py
```

One season, or a fresh pull of the current season:

```
.\.venv\Scripts\python.exe cache_nba_data.py --seasons 2024-25
.\.venv\Scripts\python.exe cache_nba_data.py --refresh
```

The page is at http://localhost:8501/nba-logs. The league charts use every club. The picker at the bottom sets one club beside that line.

## Tables

`penalties.py` writes:

- `data/tables/games.parquet`: one row per team per game
- `data/tables/penalties.parquet`: one row per flagged play

The dashboard reads those files and aggregates them for the current filters.

## The fit

`analysis.py` asks whether the regular-season rate changed at three pre-chosen officiating seasons: 2004 (defensive holding and illegal contact), 2018 (roughing the passer), and 2021 (taunting). The outcome is flags per team-game, including games with zero. The mean is a yearly drift plus a level shift at each of those seasons. Standard errors are clustered by season. The page walks the waves on the chart: the 1999-2003 baseline, the 2004 illegal-contact crackdown, the quiet years from 2006 to 2013, the 2014-2019 holding plateau, 2018 roughing the passer, the 2020 empty-stadium drop, and the 2024 illegal-formation emphasis. Lines on the chart are points of emphasis or rule changes, plus 2020 for the season played without crowds. The era table uses those same stretches. It then shows which calls rose or fell, regular season against the playoffs, and one penalty type as calls per season.

Try a slice filters the team lookup. Those controls do not change the fit.

## Flags in the game

The top of the page is that timeline, in flags per game. Hover a point for the penalty yards in the same season. Each game is the home club's flags plus the away club's flags. The era table walks 1999-03, 2004-05, 2006-13, 2014-19, 2020-23, and 2024-25. The playoff chart draws regular season and playoffs as separate lines. 2026 is the season still being played. One call over time counts that penalty across the regular season, not as a fraction of a game.

The game table lists every game in the Try a slice controls, with each club's flags and yards. Sort it by most flags, fewest flags, or week. Pick a team to keep that club's games. Pick two teams and check "Only games between the picked teams" for the matchup.

Through the year shows week and month. Quarter of the game splits flags into the 1st, 2nd, 3rd, 4th, and overtime.

Heavy clubs against each other ranks clubs inside each season, then keeps games where both teams are in that season's top quarter, or both are in the bottom quarter. The history lines stay on the full file.

## Counting

A row in `penalties.parquet` is a play with `penalty = 1`. The penalized team is `penalty_team`. Penalties per team divides that count by games played, including games with zero flags. Flags in the game add both clubs. Offense means `penalty_team` had the ball. Home and away follow the penalized team.

nflverse already uses the current team codes (LV, LAC, LA) on older seasons.
