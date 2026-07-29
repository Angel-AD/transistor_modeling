# `analyze_shape.py` — gm-shape compliance rules

Ranks/filters already-good-fit models by PHYSICAL SHAPE, not just point-wise RMSE. A model
can score well on `region_knee_combined_gm`/`region_knee_ids_rmse` yet still show Ids-Vds
*bumps* in the knee or *peaks/wiggles* in gm2/gm3 — defects that live in the derivatives and
between the measured points, invisible to a point-wise metric. Evaluated on a DENSE (Vgs, Vds)
grid via exact autograd derivatives (same convention as `get_gm_rmse_metrics`: `gm_n =
d^n Ids/dVgs^n` on input col 0, `gds = dIds/dVds` on col 1).

## Workflow
1. **Filter first** on fit metrics (region-restricted `combined_gm`/`ids_rmse`, or `--top_n`
   best-by-metric via `filter_results.py`) — keep only already-good fits. Shape evaluation
   loads and evaluates every model, so pre-filtering keeps it cheap.
2. Evaluate shape on the survivors, split into TWO independent compliance categories (a
   model can pass one without the other — individually smooth gm2/gm3 curves don't guarantee
   the Ids fit itself is well-shaped, and vice versa), written as SIX files total (slim
   `.xlsx` only — no plain CSV kept for these 6; the base `<name>_shape.csv`, with every row's
   metrics regardless of compliance, stays as CSV+slim):
   - **`gmshape`** — GM-only (gm3 window/tail/start/global checks).
     `_gmshape_ok_by_gmshape_slim.xlsx` (sorted by `region_knee_combined_gm`),
     `_gmshape_ok_by_gdshape_slim.xlsx` (SAME rows, sorted by `gds_residual_worst_max` instead
     -- surfaces which of the gm-good ones are ALSO gds-good at the top).
   - **`gdsshape`** — GDS-only (the Ids fit-error/residual check).
     `_gdsshape_ok_by_gmshape_slim.xlsx`, `_gdsshape_ok_by_gdshape_slim.xlsx` (same idea,
     mirrored).
   - **`bothshape`** — the intersection, compliant with BOTH `gmshape` AND `gdsshape`.
     `_bothshape_ok_slim.xlsx` and `_bothshape_removed_slim.xlsx` (the complement -- every row
     that failed gmshape and/or gdsshape), both sorted by `shape_rank` (the combined
     gmshape+gdshape rank). A dedicated pair of files rather than relying on eyeballing the
     top of `_gmshape_ok_by_gdshape` -- the intersection can be empty, its exact membership
     isn't always obvious from a sorted-but-not-filtered list, and (unlike gmshape/gdsshape
     individually) a row missing from `bothshape_ok` could be failing either check or both, so
     an explicit `_removed` file is useful here specifically.

   No `_removed.csv` for `gmshape`/`gdsshape` individually -- the base `_shape.csv` already has
   every row's metrics (why something failed is visible there). `bothshape` is the one
   exception (see above). (An earlier version of this had `_removed.csv` per category plus a
   `_by_bump` sort tied to the now-deprecated `gds_hump` metric, and dropped `bothshape`
   entirely -- simplified down to 4 files, then `bothshape_ok` was reinstated once relying on
   sort-order alone proved insufficient, then `bothshape_removed` added alongside it for the
   same reason.)

## Ids-Vds bumps (gds = dIds/dVds)
Measured gds is monotonically decreasing in the knee (max at Vds→0). A model that rises-then-
falls is a bump.
- `neg_gds_energy` = `mean(relu(-gds)^2)` — Ids DIPS (gds actually goes negative). This is also
  the `--vds_loss` training penalty, reused here as a score.
- `gds_hump`/`gds_hump_max` = slope-hump SEVERITY per current-carrying trace in the STRONG-ON
  window (`--bump_vgs`, default `-1.6..0` — near-threshold traces hump in real data too, so
  excluded): `(max(gds) - gds_at_Vds→0) / max(gds)`. 0 = monotone like the data.

## gm3 shape — the compliance rules (this is what `gmshape_ok`/`gmshape_removed` is based on)

gm3 (`d³Ids/dVgs³`) is evaluated over `--extrema_vgs` × `--extrema_vds` (default
`-4..-1.7` × `5..28`, saturation only — different structure at low Vds). Extrema are found
per Vds slice with a ZigZag pivot detector (`_find_extrema`) that ignores the slice's own
endpoints (window cut points aren't real turning points) and requires a retrace of
`--extrema_prom` (default 0.02 × slice peak-to-peak) to confirm a pivot, so ripple below that
doesn't count.

Expected shape, checked per Vds slice, then aggregated as a **fraction of bad slices** (not a
median — a defect on any real slice matters, not just the "typical" one):

```
Vgs:   -4        -3.5           -2.2      -1.7                  0
        |----------|--------------|---------|--------------------|
         min window 2 (required) max window   tolerant zone        tail
       (-4..-3.5)  (-3.5..-1.7)  (-4..-2.2)   (-2.2..-1.7)         (>-1.7)
        optional     REQUIRED     REQUIRED    small OK (<1),       |gm3|<=1
                                               large=excess
```

- **`gm3_min_windows`** (default `-4..-3.5,-3.5..-1.7`): one expected minimum per window.
  - First window (`-4..-3.5`) is **optional** — 0 or 1 minimum there is fine. A model with only
    one real gm3 dip instead of two isn't spuriously shaped.
  - **Last window (`-3.5..-1.7`) is REQUIRED** — a real minimum must land there
    (`gm3_min_missing_frac == 0`).
  - A THIRD minimum (or a second one crowding an already-filled window) counts as excess.
- **`gm3_max_windows`** (default `-4..-2.2`): **REQUIRED** — a maximum must exist there
  (`gm3_max_missing_frac == 0`), at any amplitude.
- **`gm3_max_tolerant_window`** (default `-2.2..-1.7`, the gap between the max window's edge
  and the search boundary): an EXTRA, otherwise-unmatched maximum here is tolerated — not
  excess — as long as `|gm3| < --gm3_max_tolerant_amp` (default `1.0`). A large one there still
  counts as excess.
- **`gm3_bad_frac`** = fraction of judged Vds slices with any excess (unmatched-and-not-
  tolerated extremum). This is the primary shape-violation score; compliance requires `== 0`.
- **`gm3_tail_bad_frac`**: beyond the search boundary (`Vgs > extrema_vgs[1]`, i.e. past
  `-1.7`, where the window-match check doesn't even look), gm3 must stay within
  `[-1, 1]` (`--gm3_tail_amp`, default `1.0`). A large excursion out there is a real defect the
  window logic can't see — `gm3_tail_maxabs` reports the worst value.

## Start-of-curve checks (gm3 AND gm2)
The extrema check ignores window endpoints by design, so a spurious edge artifact right at the
very start of the evaluated curve (deep subthreshold, `gm_region`'s lowest Vgs) is otherwise
invisible. Checked against the REAL measured value there — not a fixed threshold, since gm3/gm2
genuinely vary with Vds (e.g. gm3 ranges -1.26 to +0.71 on `cg2h40010_new_2.4`):
- `measured_gm_start_range(csv, min_vgs, vgs_start, device, order)` computes `(min, max)` of the
  REAL gm{2,3} at the curve's start Vgs, across every measured Vds slice (`Step_Index` groups),
  via `create_gms_for_train` — the same ground truth the training pipeline itself uses.
- A model's Vds column passes if its value falls inside `[real_min - tol, real_max + tol]`.
  `gm3_start_tol`/`gm2_start_tol` both default `1.0`.
- `gm3_start_bad_frac == 0` and `gm2_start_bad_frac == 0` are both required for compliance.

## Global sanity check (gm3 only)
Beyond the localized checks above (start point, tail region), gm3 must also stay within a
broad envelope EVERYWHERE it's evaluated -- catches a runaway/unphysical mid-curve spike that
none of the other, more scoped checks happen to cover.
- `measured_gm_global_range(csv, min_vgs, device, order)` computes the REAL measured gm{2,3}'s
  global `(min, max)` across every measurement point (not scoped to one Vgs point like the
  start check) -- same `create_gms_for_train` ground truth.
- A model's value at any grid point passes if it falls inside
  `[real_min * gm3_global_min_mult, real_max * gm3_global_max_mult]` -- both multipliers
  default `3.0` (real_min is normally negative, real_max positive, so multiplying by 3 widens
  the allowed band 3x in each direction from the real observed range).
- `gm3_global_bad_frac == 0` is required for compliance; `gm3_global_excess_max` reports the
  worst violation's severity.

## Ids fit-error shape check (residual wobble)
A DIFFERENT kind of defect from everything above: the model's own gds/gm curves can be
individually smooth, yet the model can still fit real data BADLY in a structured, non-random
way -- the fit error (residual = model_Ids − real_Ids) gets WORSE then BETTER along a Vds
sweep instead of trending smoothly, i.e. it wobbles. Discovered by comparing a model's Ids-vs-
Vds curve directly against real data (not the model's self-referential derivatives): the
model can look monotonic and smooth on its own, but still visibly "peel away" from the data,
dip further, and partially recover -- exactly what `gds_hump` (which only looks at the
model's own curve) cannot see.
- `measured_vds_traces(csv, min_vgs, device, vgs_lo, vgs_hi, targets=DEFAULT_RESIDUAL_VGS_TARGETS)`
  pairs each of the 15 STANDARD Vgs sweep targets (the same list `plot_csv_row.py`'s
  `--plot_vgs_list` uses -- `-3.5,-3,...,-1.8,-1.6,...,0`) that falls in `--residual_vgs`
  (default `-3..0` -- wider than `--bump_vgs`, since this wobble isn't confined to the
  strong-on region) with its CLOSEST real measurement trace (a Vds sweep at fixed Vgs).
  **CRITICAL**: the model is evaluated at the TARGET Vgs, NOT the matched trace's own mean
  Vgs -- those differ by only ~0.03-0.05V, but using the trace's mean instead of the target
  was an early bug that changed the computed residual by ~2x, because it no longer matched
  what `plot_saved_state.py`'s Ids-vs-Vds panel actually plots/compares.
- For each target's matched trace, the model is evaluated at the trace's OWN (real,
  irregularly spaced) Vds points within `--residual_vds` (default `0..6`) -- interpolating
  the model onto real points, not the reverse, so no synthetic detail is invented. The
  trace's own first/last judged point is excluded (an "INTERIOR" convention, same idea as
  `_find_extrema`'s endpoint exclusion) -- EVERY trace showed a small positive residual blip
  at its very first point (~0.01-0.015) regardless of whether the model was otherwise good or
  bad, confirmed to be a boundary artifact, not a real signal.
- Split into two independent severities (they mean different things physically):
  - `gds_residual_neg_max` = worst UNDERSHOOT (model BELOW real data), i.e. how far negative
    the residual dips at its deepest interior point, across all targets.
  - `gds_residual_pos_max` = worst OVERSHOOT (model ABOVE real data), the mirror image.
- A target's trace counts as bad if `neg_max > --residual_neg_tol` (default `0.062`) OR
  `pos_max > --residual_pos_tol` (default `0.068`, widened from `0.062` -- see below).
  `gds_residual_bad_frac` = fraction of targets that are bad; `gds_residual_bad_frac == 0`
  required for compliance.

## Full compliance formulas
A row is `gmshape_ok` iff ALL of:
```
gm3_n_min >= 1               # some real minimum exists somewhere in the search range
gm3_n_max >= 1                # some real maximum exists somewhere in the search range
gm3_bad_frac == 0             # no excess/misplaced extrema on any judged slice
gm3_min_missing_frac == 0     # the LAST min window (-3.5..-1.7) has a real match
gm3_max_missing_frac == 0     # the max window (-4..-2.2) has a real match
gm3_start_bad_frac == 0       # gm3 at curve start matches the real measured range
gm2_start_bad_frac == 0       # gm2 at curve start matches the real measured range
gm3_tail_bad_frac == 0        # gm3 stays within [-1,1] past the search boundary (>-1.7)
gm3_global_bad_frac == 0      # gm3 stays within 3x the real global min/max, everywhere
```
A row is `gdsshape_ok` iff:
```
gds_residual_bad_frac == 0    # model Ids doesn't wobble away-from/back-toward real data
```
A row is `bothshape_ok` iff `gmshape_ok AND gdsshape_ok` (the intersection of the two sets
above; written to `_bothshape_ok_slim.xlsx`, sorted by `shape_rank`). Its complement within
the filtered input -- every row that's NOT `bothshape_ok` -- is written to
`_bothshape_removed_slim.xlsx`, same sort.

Rows that don't comply with `gmshape`/`gdsshape` individually simply aren't in that category's
`_ok*` files — the base `<name>_shape.csv` (kept as CSV+slim) has every evaluated row's
metrics regardless, so why any particular row is absent from an `_ok` file is visible by
looking it up there. `bothshape_removed` exists anyway since it's the AND of two independent
checks, so "why is this row missing from bothshape_ok" isn't always a single lookup.

Older `<ranked>_shape.csv` files computed before a given check existed simply don't have that
column — `write_gmshape_outputs()` skips a check rather than treat a missing column as failing
(so it doesn't silently exclude every row of an older file).

## Fast re-derivation without reloading models
`--from_shape_csv <ranked>_shape.csv` re-applies the compliance formula straight from an
already-computed `_shape.csv` (every row's metrics are already there) — no torch, no model
reload, no measurement CSV. Only valid if `--gm3_min_windows`/`--gm3_max_windows`/etc. match
what originally produced that file (the window-matched counts are read as-is, not recomputed).
Seconds instead of minutes on a large filtered set.

## Verified against real data (tanh_margin10_nogm, `filtered_quality` set, 24 rows)
Manually reviewed via `plot_csv_row.py` id-by-id against the final rule set above:
- `313,208,345,282,183,214,106` — only 1 of 2 min windows filled, correctly compliant (the
  relaxation from an earlier "exact count" rule to `>=1` fixed these as false negatives).
- `34,233,327` — gm3 genuinely swings past ±1 beyond Vgs=-1.7 (confirmed visually); correctly
  removed by the tail check.
- `278,248,213` — `gm3_start_bad_frac` clearly nonzero (0.78-0.80); `257,255` borderline
  (~0.02-0.03) but still nonzero — correctly removed by the start check.
- `298,169` — genuine `gm3_bad_frac > 0` (real excess extrema); `314,227` — severe
  `gm3_start_bad_frac = 1.0` — correctly removed, no ambiguity.

Result on that file: 12/24 compliant, matching the manually-reviewed set exactly.

## Verified against real data (sigmoid_margin10_nogm, filter #4, 12 rows) -- residual check
The 4 rows (204, 267, 294, 395) that survived every other check above still visually showed a
real "peak then dip" defect the other checks couldn't explain (model gm3/gds curves were
individually smooth). Traced via `generate_physics_plot_data` (the exact function
`plot_saved_state.py` itself uses) to rule out a data-mismatch, then confirmed the true
mechanism was a wobbling FIT ERROR, not a defect in the model's own curve shape -- see the
residual check section above.

An initial "flag any wobble at all" version (count-based, no magnitude) flagged all 12/12 --
too strict, since the discriminating factor turned out to be severity, not mere presence:
visual review confirmed `294`/`395` looked meaningfully BETTER than `267` despite all having
*some* wobble. Rebuilt around the neg/pos-max magnitude split above and calibrated
`--residual_neg_tol=0.062` directly against these 4 (`267`: `0.065` bad, `204`: `0.063` bad,
`294`: `0.061` ok, `395`: `0.054` ok) -- a narrow margin from only 4 examples, worth revisiting
with more data. `--residual_pos_tol` (same default, no calibration examples at the time) was
later validated independently: `294` unexpectedly failed via `gds_residual_pos_max=0.0675` at
`Vgs=-1.6` (a target neither of us had manually checked) -- visually confirmed as a real small
bump at Vds≈0.5, i.e. a genuine defect the automated per-target scan caught that a manual
spot-check of just 2 Vgs values (`-1.8`, `-2.2`) had missed. With the original
`--residual_pos_tol=0.062`: `gmshape=4/12`, `gdsshape=2/12` (`395`, `419`) -- their
intersection (`bothshape`) was `395` only.

`--residual_pos_tol` was then deliberately widened to `0.068` after a direct visual call: `294`
was judged "good enough" alongside `395` despite the confirmed small bump (a judgment call, not
a bug fix -- the defect is real, just accepted as tolerable). `0.068` clears `294`'s
`0.0675` with a small margin while leaving `204` (`pos_max=0.0704`) and `267`
(`neg_max=0.0711`, gated by `--residual_neg_tol` regardless) excluded. Final result on this
file with the widened tolerance: `gdsshape=3/12` (`395`, `294`, `419`), `bothshape=2/12`
(`395`, `294`), written to `_bothshape_ok_slim.xlsx`, with the other 10 rows in
`_bothshape_removed_slim.xlsx`.
