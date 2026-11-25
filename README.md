# Netflix viewing history cleaning
`python clean.py` reads `events_raw.csv` + `catalog.csv` and writes:
- `events_train.csv` – events before 2026-11-10 (model learns from these)
- `events_holdout.csv` – 10–16 Nov 2026, the hidden final week
- `events_dropped.csv` – removed rows with reason

Rules: drop missing watch time, titles not in catalog or without runtime, views before release date, watch > 2x runtime (films) / 10x (series). `completed` recomputed as watch >= 90% of runtime (original kept). Blank device/country -> "unknown"; kids-on-mature-title flagged, not removed.
