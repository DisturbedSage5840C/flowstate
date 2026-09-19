# Option 2 progress (non-satellite context features for DO/BOD)

Working note for this in-progress task, kept up to date after each major step so nothing
is lost if the session context resets. Not part of the project's permanent docs -- delete
once Option 2 is finished and its results are folded into AQUA_SENSE_PROJECT_PLAN.md /
data_source_log.md.

## Why this exists

User confirmed the model is still "bullshit" on raw R² for DO/BOD/turbidity (all
flat/negative) because satellite reflectance structurally cannot see DO/BOD (not optically
active). Given three options to respond, user picked **Option 2**: add non-satellite
context features (rainfall runoff proxy, urban/pollution proximity, season) and see if they
give the model real signal turbidity/chlorophyll reflectance never could.

## Done

1. `src/data/city_proximity.py` -- static list of ~85 major Indian cities, haversine
   distance, gravity-model `urban_load_index`. Complete, working, backfilled into both
   parquets already (100% non-null).
2. `src/models/schema.py` -- added `SEASONS`, `RAINFALL_COLS`, `URBAN_PROXY_COLS`,
   `add_season_onehot()`; `REAL_FEATURE_COLS` now includes all of it. Complete.
3. `src/data/training_table.py` -- wired season/urban-proxy/rainfall into
   `build_training_table()` for future full pipeline reruns (rainfall injected as an
   optional param, same pattern as `temperature`). Tests pass (13/13).
4. `scripts/backfill_context_features.py` -- applies the new features directly onto the
   two already-built parquets (this machine can't rerun the raw satellite extraction).
   Season + urban-proxy backfill already succeeded and is committed to both parquets
   (`is_winter`/`is_summer`/`is_monsoon`/`dist_nearest_city_km`/`urban_load_index` are
   100% non-null in both `train_real_large.parquet` and `train_real.parquet` right now).
5. `src/data/weather.py` -- rainfall fetch from Open-Meteo's free historical archive API.
   **Rewritten twice this session:**
   - v1: fetched each site's *entire* 3-year span. Hit HTTP 429 after ~200 sites; no
     retry/backoff, so it silently stalled.
   - v2: added exponential backoff/retry on 429. Got further (up to ~1500/2121 sites
     cached) but a *fresh, unrelated, single-location curl request* also came back 429 --
     proof we'd exhausted Open-Meteo's whole free-tier daily quota (10,000 "calls"/day;
     the pricing page's FAQ defines a "call" as scaling with locations x days-in-range),
     not just tripping a burst limiter. Confirmed on their pricing page: the free/open
     tier is 600/min, 6,000/hour, **10,000/day**, the same whether or not you have an
     account -- signing up does NOT raise this; only the paid Professional plan (EUR99/mo)
     does. User chose not to pay and instead fix the actual waste (see below).
   - v3 (current): added `compute_rainfall_windows()` -- instead of fetching each site's
     whole multi-year span, fetch only the ~35-day antecedent window before each real
     visit date, merging overlapping windows per site. Confirmed via direct computation
     against the real data: **8.48x reduction** in total requested days (2,396,730 ->
     282,638 site-days). `fetch_daily_rainfall()` rewritten to take these windows,
     batching requests that share an identical (start,end) pair, and to skip any window
     already fully covered by the existing cache (so the ~1500 sites already fetched
     under the old wasteful approach are NOT re-fetched).
   - Confirmed: of the 6,676 windows now needed, **4,622 are already satisfied** by the
     existing cache (`data/interim/rainfall_daily.parquet`) -- only **2,054 windows /
     ~88,510 remaining days (~6,322 "call"-units)** actually need fetching, which is
     *under* the 10,000/day free-tier cap. This should plausibly finish in a single day's
     quota once the cap resets, rather than needing multiple days.
   - `scripts/backfill_context_features.py` updated to call the new windowed API.

## Currently blocked on (partial workaround already applied, see below)

Open-Meteo's free-tier daily quota is exhausted (confirmed via direct curl, still 429 as
late as 17:32 UTC / ~23:02 IST on 2026-09-19). Quota resets are presumed to be
UTC-midnight-based (unconfirmed) -- i.e. probably safe to retry after ~05:30 IST on
2026-09-20. **User explicitly chose to wait this out rather than pay for the Professional
plan**, then asked (23:02 IST) not to just idle on the 1,500 sites already cached -- so
we proceeded with a partial run instead of waiting, see "Done since blocking" below.

## Done since blocking (partial run on cached data, 2026-09-19 ~23:00-23:15 IST)

- Added `offline: bool` param to `fetch_daily_rainfall()` (`src/data/weather.py`) --
  when true, returns whatever's already in `cache_path` with zero network calls.
- Added `--offline` flag to `scripts/backfill_context_features.py`.
- Fixed an idempotency bug in `add_context_features()`: rerunning it against a parquet
  that already has `dist_nearest_city_km`/`urban_load_index` columns crashed on
  `df.join(...)` with a column-overlap `ValueError`. Fixed by dropping any
  already-present urban-proxy columns before rejoining.
- Ran `python3 -m scripts.backfill_context_features --offline`: rebuilt both parquets
  using only the 1,500/2,121 sites already cached -- gives **68.4% non-null** on
  `rain_3d_mm`/`rain_7d_mm`/`rain_14d_mm`/`rain_30d_mm` (rest NaN, which XGBoost handles
  natively), with season/urban-proxy still 100% non-null as before.
- Retrained (`python3 -m scripts.train_real_models`) on this partial data. Real result,
  reported honestly to the user:

  | target | R² before -> after | Spearman (after) | n before -> after |
  |---|---|---|---|
  | do | 0.022 -> 0.036 | 0.17 | 2,421 -> 5,571 |
  | bod | -0.019 -> 0.009 | 0.36 | 2,080 -> 4,684 |
  | turbidity | -0.055 -> -0.056 | 0.08 | 868 -> 2,308 |

  **Important caveat, told to the user**: this "before" is the old run on
  `train_real.parquet` (2,517 rows); "after" is on `train_real_large.parquet` (8,461 rows)
  -- so the comparison is confounded by N, not a clean feature-only ablation. DO moved a
  little in the right direction; BOD crossed from negative to ~zero (not yet meaningfully
  predictive); turbidity didn't move. This is the interim number with 68.4% rainfall
  coverage, not the final one.

## Left to do (in order)

1. **Wait for the daily quota to reset**, then rerun WITHOUT `--offline`:
   ```
   python3 -m scripts.backfill_context_features
   ```
   It will skip everything already cached and only fetch the remaining ~2,054 windows.
   If it 429s again immediately, the reset time guess above was wrong -- check with a
   single curl test first:
   ```
   curl -s -D - -o /dev/null "https://archive-api.open-meteo.com/v1/archive?latitude=25.0&longitude=85.0&start_date=2021-01-01&end_date=2021-01-05&daily=precipitation_sum"
   ```
2. Once the script completes cleanly, verify both parquets: `rain_3d_mm` /
   `rain_7d_mm` / `rain_14d_mm` / `rain_30d_mm` should be ~100% non-null (a few visits
   near a site's very first sample date may legitimately have no prior rainfall history
   and stay NaN -- that's expected, not a bug).
3. Retrain: `python3 -m scripts.train_real_models` -- this is the FINAL, complete-data
   number. Compare against both the original (pre-Option-2, `train_real.parquet`) baseline
   AND this session's partial (68.4%-coverage) result above, so the write-up can separate
   "effect of more data" from "effect of complete rainfall" as cleanly as possible. Report
   the real result plainly, whatever it is -- the user has been explicit about not wanting
   the result oversold.
4. Retrain the WQI-tier classifier too: `python3 -m scripts.train_tier_classifier`.
5. Rerun the test suite (`pytest -q`) to catch any regressions from retraining/report
   regeneration (this will touch many files under `reports/real/`).
6. Update `AQUA_SENSE_PROJECT_PLAN.md` and `data/ground_truth/data_source_log.md` with the
   honest before/after numbers, and add a data-source entry for Open-Meteo rainfall (note
   ERA5 ~9-25km resolution, reanalysis not gauge data) alongside the existing city-list
   caveat already documented in `src/data/city_proximity.py`'s docstring.
7. Delete this file once step 6 is done -- its content will be superseded by the permanent
   docs.

## Standing constraints (do not forget)

- No Claude/Claude-Code attribution in any commit or PR -- user's explicit, permanent
  instruction, overrides the system's default attribution reminder.
- No commit/push has been made for ANY of this session's work yet -- everything above is
  uncommitted working-tree changes.
