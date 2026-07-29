"""
analyze_shape.py -- rank models by PHYSICAL SHAPE, not just point-wise RMSE.

Motivation
----------
region_knee_combined_gm / region_knee_ids_rmse are point-wise fit metrics: a model
can score well yet still show Ids-Vds *bumps* in the knee or *peaks/wiggles* in gm2/gm3.
Those defects live in the DERIVATIVES and BETWEEN the measured points, so a point-wise
RMSE can't see them. This script evaluates each model on a DENSE (Vgs, Vds) grid and
measures shape directly:

  Ids-Vds monotonicity (knee bumps)   ->  gds = dIds/dVds should be >= 0
      neg_gds_frac    fraction of grid points with gds < 0
      neg_gds_energy  mean( relu(-gds)^2 )        (this is the --vds_loss penalty, as a score)
      min_gds         most-negative gds (worst bump)

  gm smoothness (gm2/gm3 peaks)       ->  gm2 = d2Ids/dVgs2, gm3 = d3Ids/dVgs3
      gm{2,3}_tv_norm total variation along Vgs / peak-to-peak  (1.0 = perfectly monotone;
                      higher = more wiggle, scale-free so models compare fairly)
      gm3_maxabs      largest |gm3| spike
      gm3_n_min/max   extrema count in the turn-on window; excess beyond --gm3_min_windows/
                      --gm3_max_windows (each a Vgs sub-range) -> gm3_bad_frac
      gm3_start_diff  how far the model's gm3 falls OUTSIDE the REAL measured RANGE (min..max
                      across Vds slices, padded by --gm3_start_tol) at the very start of the
                      evaluated curve (deep subthreshold); 0 = inside the padded range. The
                      real value genuinely varies with Vds (not just noise), so this checks
                      against the observed range rather than a single flattened reference
                      point. The extrema check above ignores window ENDPOINTS by design, so a
                      spurious edge artifact right at the start of the curve would otherwise
                      be invisible; this catches it by comparing against the actual
                      measurement (create_gms_for_train), not just the model against itself.
                      -> gm3_start_bad_frac (fraction of Vds columns outside the range)

All derivatives are exact (autograd), same convention as get_gm_rmse_metrics
(gm_n = d^n Ids/dVgs^n on input col 0; gds = dIds/dVds on col 1).

Workflow (matches how you judge them)
-------------------------------------
  1. FILTER first on the fit metrics (default: region_knee_combined_gm<0.9 AND
     region_knee_ids_rmse<0.013) -- keep only the already-good fits.
  2. Among survivors, rank by SHAPE (default: shape_rank = mean rank of
     region_knee_combined_gm (gmshape) + gds_residual_worst_max (gdshape)), so the best-fit /
     smoothest rise to the top.

Outputs
-------
  <ranked>_shape.csv                       every evaluated row, sorted by --sort_by.
  <ranked>_shape_gmshape_ok_by_combined_gm.csv   subset that COMPLIES with the expected gm3
                                            shape (one minimum per --gm3_min_windows window,
                                            one maximum per --gm3_max_windows window,
                                            gm3_bad_frac==0, AND gm3 at the curve start inside
                                            the real measured range, padded by
                                            --gm3_start_tol), ranked by region_knee_combined_gm
                                            (point-wise fit).
  <ranked>_shape_gmshape_ok_by_bump.csv    same compliant subset, ranked by gds_hump (the
                                            Ids-Vds slope-hump "bump" metric).
  <ranked>_shape_gmshape_removed.csv       the complement: rows from <ranked>_shape.csv that
                                            did NOT comply (same schema, so gm3_bad_frac /
                                            gm3_start_diff / gm3_n_min / gm3_n_max show why),
                                            ranked by shape_rank (worst-shaped first).
  (skip all three of the above with --no_gmshape_csvs)

Usage
-----
  python analyze_shape.py --ranked_csv <full ranked_by_*.csv with a file_path column>
  # defaults reproduce the requested test:
  #   --filter region_knee_combined_gm<0.9 --filter region_knee_ids_rmse<0.013
  #   --region vgs=-3..0,vds=0..15
  # NOTE: use the FULL ranked csv (has file_path), not the _slim one.

  # fast: re-derive gmshape_ok/gmshape_removed from an ALREADY-COMPUTED <ranked>_shape.csv --
  # no torch, no model reload (only valid if --gm3_min_windows/--gm3_max_windows match what
  # produced that file):
  python analyze_shape.py --from_shape_csv <ranked>_shape.csv
"""
from __future__ import annotations

import argparse
import contextlib
import csv as _csv
import io
import os
import re
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from plot_saved_state import _build_model_from_dir, _load_T_train
from per_neuron_simple_angelov_nn_test import create_gms_for_train


# --------------------------------------------------------------------------- #
# Filter (COL<op>VALUE), same spirit as filter_results.py
# --------------------------------------------------------------------------- #
_OPS = ("<=", ">=", "<", ">", "=")


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def apply_filters(rows, filters):
    for f in filters:
        op = next((o for o in _OPS if o in f), None)
        if op is None:
            sys.exit(f"[error] --filter must be COL<op>VALUE, got: {f!r}")
        col, val = (s.strip() for s in f.split(op, 1))
        if rows and col not in rows[0]:
            sys.exit(f"[error] --filter column {col!r} not in CSV. cols={list(rows[0].keys())}")
        thr = _num(val)
        cmp = {"<": lambda a: a < thr, ">": lambda a: a > thr, "<=": lambda a: a <= thr,
               ">=": lambda a: a >= thr, "=": lambda a: a == thr}[op]
        rows = [r for r in rows if _num(r.get(col)) is not None and cmp(_num(r.get(col)))]
    return rows


def parse_region(spec):
    def axis(name):
        m = re.search(rf"{name}\s*=\s*([-+.\d]+)\s*(?:\.\.|to|:)\s*([-+.\d]+)", spec, re.I)
        if not m:
            sys.exit(f"[error] --region {spec!r}: need '{name}=lo..hi'")
        lo, hi = float(m.group(1)), float(m.group(2))
        return (min(lo, hi), max(lo, hi))
    return {"vgs": axis("vgs"), "vds": axis("vds")}


# --------------------------------------------------------------------------- #
# Shape metrics on a dense (Vgs, Vds) grid
# --------------------------------------------------------------------------- #
def _derivs(model, region, vgs_n, vds_n, device):
    """(gds, gm2, gm3, vgs, vds): (vgs_n,vds_n) numpy grids + coord axes over `region`.
    gm_n = d^n/dVgs^n, gds = d/dVds."""
    vgs = torch.linspace(*region["vgs"], vgs_n, dtype=torch.float64, device=device)
    vds = torch.linspace(*region["vds"], vds_n, dtype=torch.float64, device=device)
    VG, VD = torch.meshgrid(vgs, vds, indexing="ij")
    X = torch.stack([VG.reshape(-1), VD.reshape(-1)], dim=1).requires_grad_(True)
    ids = model(X).reshape(-1)
    g1 = torch.autograd.grad(ids.sum(), X, create_graph=True)[0]
    gm1, gds = g1[:, 0], g1[:, 1]
    gm2 = torch.autograd.grad(gm1.sum(), X, create_graph=True)[0][:, 0]
    gm3 = torch.autograd.grad(gm2.sum(), X)[0][:, 0]
    grid = lambda t: t.detach().reshape(vgs_n, vds_n).cpu().numpy()
    return grid(gds), grid(gm2), grid(gm3), vgs.cpu().numpy(), vds.cpu().numpy()


def _tv_norm(A):   # total variation along Vgs / peak-to-peak (wiggliness; misses single spikes)
    tv = np.abs(np.diff(A, axis=0)).sum(axis=0)
    return float(np.mean(tv / (np.ptp(A, axis=0) + 1e-12)))


def _find_extrema(y, prom):
    """(min_indices, max_indices): indices of INTERIOR local extrema with swing >= prom.
    Endpoints are NOT counted -- a windowed curve's boundaries are usually just cut points of
    the full sweep, not real turning points (counting them creates a false extremum at the
    window edge). ZigZag: confirm a pivot only after the signal retraces by >= prom, so
    sub-prominence ripples are ignored but real small extrema are kept."""
    n = len(y)
    if n < 3:
        return [], []
    piv = [0]
    trend = 0
    hi = lo = y[0]
    hi_i = lo_i = 0
    for i in range(1, n):
        if y[i] > hi:
            hi, hi_i = y[i], i
        if y[i] < lo:
            lo, lo_i = y[i], i
        if trend >= 0 and y[i] <= hi - prom:          # rising -> confirm max
            if hi_i != piv[-1]:
                piv.append(hi_i)
            trend, lo, lo_i = -1, y[i], i
        elif trend <= 0 and y[i] >= lo + prom:        # falling -> confirm min
            if lo_i != piv[-1]:
                piv.append(lo_i)
            trend, hi, hi_i = 1, y[i], i
    if piv[-1] != n - 1:
        piv.append(n - 1)
    v = y[np.array(piv)]
    min_idx, max_idx = [], []
    for k in range(1, len(piv) - 1):                  # INTERIOR pivots only (skip both endpoints)
        if v[k] >= v[k - 1] and v[k] >= v[k + 1]:
            max_idx.append(piv[k])
        elif v[k] <= v[k - 1] and v[k] <= v[k + 1]:
            min_idx.append(piv[k])
    return min_idx, max_idx


def _match_windows(positions, windows, values=None, tolerant_window=None, tolerant_amp=None):
    """Greedy 1-per-window assignment: for each expected (lo,hi) window IN ORDER, claim the
    first not-yet-used detected position that falls inside it. Returns (excess, matched) --
    excess = detected positions that were extra crowding within an already-filled window, PLUS
    positions that never matched any window at all. matched = per-window bool, True if that
    window claimed a position. A window with NO match (missing extremum) does NOT add to
    excess -- fewer real extrema than expected isn't, on its own, a spurious-wiggle defect
    (callers that need a window to be non-optional check `matched` themselves). If
    tolerant_window/tolerant_amp are also given (with matching `values`, the extremum's
    y-value at each position), an unmatched position inside tolerant_window whose |value| <
    tolerant_amp is ALSO not penalized -- a small extra wiggle there is tolerated, only a
    large one counts as excess."""
    used = [False] * len(positions)
    matched = [False] * len(windows)
    for wi, (lo, hi) in enumerate(windows):
        cand = [i for i, v in enumerate(positions) if not used[i] and lo <= v <= hi]
        if cand:
            used[cand[0]] = True
            matched[wi] = True
    if tolerant_window is not None:
        tlo, thi = tolerant_window
        for i, v in enumerate(positions):
            if not used[i] and tlo <= v <= thi and abs(values[i]) < tolerant_amp:
                used[i] = True
    return sum(1 for u in used if not u), matched


def _tail_check(gm3_g, vgs, vds, tail_vgs_lo, vds_win, tail_amp):
    """Beyond the expected max window (Vgs > tail_vgs_lo, i.e. past --extrema_vgs's upper
    bound where the window-matching check above stops looking), gm3 must stay small
    (|gm3| <= tail_amp) -- a large excursion there is a shape defect the window check can't
    see, since it never searches that far. Judged per current-carrying Vds slice, same as the
    other gm3 checks."""
    dlo, dhi = min(vds_win), max(vds_win)
    vgs_mask = vgs > tail_vgs_lo
    vds_mask = (vds >= dlo) & (vds <= dhi)
    G = gm3_g[np.ix_(vgs_mask, vds_mask)]
    if G.shape[0] == 0 or G.shape[1] == 0:
        return {"gm3_tail_maxabs": 0.0, "gm3_tail_bad_frac": 0.0}
    per_slice_max = np.abs(G).max(axis=0)
    return {
        "gm3_tail_maxabs":   round(float(per_slice_max.max()), 3),
        "gm3_tail_bad_frac": round(float((per_slice_max > tail_amp).mean()), 3),
    }


def _extrema_counts(gm3_g, vgs, vds, vgs_win, vds_win, min_windows, max_windows, prom_frac,
                    max_tolerant_window=None, max_tolerant_amp=None):
    """Count gm3 minima/maxima vs Vgs in `vgs_win`, per Vds slice, and how many EXTRA beyond the
    expected shape -- one minimum per window in `min_windows`, one maximum per window in
    `max_windows` (each a (lo,hi) Vgs sub-range; e.g. min_windows=[(-4,-3.5),(-3.5,-1.7)] expects
    an early minimum before -3.5 and a second one between -3.5 and -1.7). Only Vds slices inside
    `vds_win` are judged -- this shape is a SATURATION expectation, so the low-Vds linear region
    (different gm3 structure, and not plotted) is excluded. Prominence = prom_frac * slice
    peak-to-peak ignores numerical ripple but catches real small spurious extrema.
    max_tolerant_window/max_tolerant_amp: an extra (unmatched) maximum inside that Vgs window
    is tolerated -- not counted as excess -- if its |value| is under the amplitude threshold.
    The LAST entry of min_windows (e.g. -3.5..-1.7) is REQUIRED (like the max window) --
    tracked separately from the others, which stay fully optional (missing not penalized)."""
    glo, ghi = min(vgs_win), max(vgs_win)
    dlo, dhi = min(vds_win), max(vds_win)
    vgs_mask = (vgs >= glo) & (vgs <= ghi)
    vgs_sub = vgs[vgs_mask]
    G = gm3_g[np.ix_(vgs_mask, (vds >= dlo) & (vds <= dhi))]
    zero = {"gm3_n_min": 0, "gm3_n_max": 0, "gm3_bad_frac": 0.0, "gm3_extra_mean": 0.0, "gm3_extra_max": 0,
            "gm3_max_missing_frac": 1.0, "gm3_min_missing_frac": 1.0}
    if G.shape[0] < 5 or G.shape[1] == 0:
        return zero
    ptp_all = np.ptp(G, axis=0)
    amp_gate = 0.10 * (ptp_all.max() if ptp_all.size else 0.0)   # skip near-zero-current slices
    nmins, nmaxs, excess = [], [], []
    max_window_missing = [[] for _ in max_windows]   # per max_windows entry: 1 slice not matched, else 0
    min_last_missing = []   # only the LAST min window (required) is tracked per-slice
    for j in range(G.shape[1]):
        col = G[:, j]
        pk = float(np.ptp(col))
        if pk < amp_gate or pk == 0.0:
            continue
        min_idx, max_idx = _find_extrema(col, prom_frac * pk)
        nmins.append(len(min_idx))
        nmaxs.append(len(max_idx))
        ex_min, min_matched = _match_windows(vgs_sub[min_idx], min_windows) if min_idx else \
                              (0, [False] * len(min_windows))
        ex_max, max_matched = _match_windows(vgs_sub[max_idx], max_windows, values=col[max_idx],
                                             tolerant_window=max_tolerant_window,
                                             tolerant_amp=max_tolerant_amp) if max_idx else \
                              (0, [False] * len(max_windows))
        excess.append(ex_min + ex_max)
        for wi, m in enumerate(max_matched):
            max_window_missing[wi].append(0 if m else 1)
        min_last_missing.append(0 if min_matched[-1] else 1)
    if not excess:
        return zero
    excess = np.asarray(excess)
    # A model is judged over ALL its current-carrying Vds curves, not the "typical" one:
    # gm3_bad_frac = fraction of Vds slices whose gm3 has extra extrema. (median hides defects
    # that occupy <50% of slices; max flags every model since ~all have >=1 bad slice.)
    # gm3_max_missing_frac: UNLIKE the OTHER min windows (missing is fine -- fewer real gm3
    # dips isn't a defect), the primary max window IS required -- worst (highest-missing) of
    # max_windows, since with the default single window that's just "how often is -4..-2.2
    # unmatched". gm3_min_missing_frac: same idea, but ONLY for the LAST min window
    # (-3.5..-1.7) -- the earlier one(s) stay fully optional.
    return {
        "gm3_n_min":     int(np.median(nmins)),
        "gm3_n_max":     int(np.median(nmaxs)),
        "gm3_bad_frac":  round(float((excess > 0).mean()), 3),   # <- primary shape-violation score
        "gm3_extra_mean": round(float(excess.mean()), 3),        # severity-weighted
        "gm3_extra_max": int(excess.max()),                      # worst single Vds slice
        "gm3_max_missing_frac": round(max(float(np.mean(m)) for m in max_window_missing), 3)
                                if max_window_missing else 0.0,
        "gm3_min_missing_frac": round(float(np.mean(min_last_missing)), 3) if min_last_missing else 0.0,
    }


def measured_gm_start_range(csv, min_vgs, vgs_start, device, order=3):
    """Real (measured) gm{order} range at the START of the evaluated Vgs range (vgs_start, the
    lowest/deep-subthreshold end): (min, max) across every measurement trace (Step_Index
    group = a transfer curve at ~fixed Vds) of that trace's gm{order} at its Vgs point nearest
    vgs_start. The real value genuinely varies with Vds (this isn't just noise -- e.g.
    -1.26 to +0.71 on cg2h40010_new_2.4 for gm3), so the model is checked against the whole
    observed RANGE (padded by a tolerance), not a single flattened point estimate -- a fixed
    Vgs=start slice compared to one median would otherwise fail models whose start value is
    simply at a different (but still physically real) Vds than the median trace. Computed
    ONCE per --ranked_csv (ground truth, model-independent), via create_gms_for_train -- the
    same gm-truth the training pipeline itself uses (same Step_Index grouping, same
    smooth_derivative smoothing). order=2 -> gm2, order=3 (default) -> gm3."""
    with contextlib.redirect_stdout(io.StringIO()):
        T = _load_T_train(csv, min_vgs, device)
    T = T.copy()
    T["Step_Index"] = T.groupby("TN").cumcount()
    T = T.sort_values(by=["Step_Index", "TN"]).reset_index(drop=True)
    _, gm2, gm3, _, _ = create_gms_for_train(T)
    T["gm"] = gm2 if order == 2 else gm3
    vals = [grp.loc[(grp["Vgs_meas"] - vgs_start).abs().idxmin(), "gm"]
            for _, grp in T.groupby("Step_Index")]
    return float(min(vals)), float(max(vals))


def measured_gm3_start_range(csv, min_vgs, vgs_start, device):
    """gm3 start range -- see measured_gm_start_range (order=3)."""
    return measured_gm_start_range(csv, min_vgs, vgs_start, device, order=3)


def _start_check(gm_g, real_start_range, tol, prefix):
    """Sanity-check the model's gm{prefix} at the VERY START of the evaluated curve (vgs[0],
    the lowest/deep-subthreshold Vgs) against the REAL measured RANGE there (min/max across
    Vds slices, padded by tol on each side -- see measured_gm_start_range). A Vds column
    passes if its model value falls inside [real_min-tol, real_max+tol]; {prefix}_start_diff
    is how far OUTSIDE that padded range the value sits (0 = inside). _find_extrema
    explicitly ignores window ENDPOINTS (cut points of the sweep, not real turning points) --
    so a model with a spurious edge artifact right at the start of the curve would otherwise
    be completely invisible to the extrema check above."""
    real_lo, real_hi = real_start_range
    lo, hi = real_lo - tol, real_hi + tol
    start_vals = gm_g[0, :]                        # model gm at Vgs=vgs[0], every Vds column
    out_of_range = np.clip(lo - start_vals, 0, None) + np.clip(start_vals - hi, 0, None)
    return {
        f"{prefix}_start_diff":     round(float(np.median(out_of_range)), 3),
        f"{prefix}_start_bad_frac": round(float((out_of_range > 0).mean()), 3),
    }


def measured_gm_global_range(csv, min_vgs, device, order=3):
    """Real (measured) gm{order} GLOBAL min/max across EVERY measurement point (every Vgs,
    every Vds slice) -- a broad sanity bound, unlike measured_gm_start_range which is scoped
    to just the curve's start Vgs. Same ground truth / same loading as measured_gm_start_range
    (create_gms_for_train), just not restricted to one Vgs point per trace. order=2 -> gm2,
    order=3 (default) -> gm3."""
    with contextlib.redirect_stdout(io.StringIO()):
        T = _load_T_train(csv, min_vgs, device)
    _, gm2, gm3, _, _ = create_gms_for_train(T)
    vals = gm2 if order == 2 else gm3
    return float(np.min(vals)), float(np.max(vals))


def _global_check(gm_g, real_global_range, max_mult, min_mult):
    """Sanity-check the model's gm{2,3} EVERYWHERE it was evaluated (not just the curve start)
    against the REAL measured GLOBAL min/max, each scaled by a multiplier -- catches runaway/
    unphysical excursions the start/tail/extrema checks don't cover (e.g. a mid-curve spike
    far beyond anything the real device ever does). A grid point passes if it falls inside
    [real_min*min_mult, real_max*max_mult] (real_min is normally negative and real_max
    positive, so multiplying by mult>1 widens the allowed band in each direction)."""
    real_min, real_max = real_global_range
    lo, hi = real_min * min_mult, real_max * max_mult
    out_of_range = np.clip(lo - gm_g, 0, None) + np.clip(gm_g - hi, 0, None)   # full grid
    per_slice_max = out_of_range.max(axis=0)                                   # worst per Vds column
    return {
        "gm3_global_excess_max": round(float(per_slice_max.max()), 3),
        "gm3_global_bad_frac":   round(float((per_slice_max > 0).mean()), 3),
    }


# The standard Vgs sweep targets used everywhere else this project plots Ids-vs-Vds (e.g.
# plot_csv_row.py's --plot_vgs_list default) -- reusing the SAME targets here means the
# residual check evaluates the model at the exact Vgs values a human actually looks at in
# plot_saved_state_full.png, not an arbitrary real trace's own (slightly off-target) mean Vgs.
DEFAULT_RESIDUAL_VGS_TARGETS = (-3.5, -3, -2.9, -2.8, -2.7, -2.5, -2.2, -2, -1.8, -1.6, -1.5,
                                -1.3, -1, -.5, 0)


def measured_vds_traces(csv, min_vgs, device, vgs_lo, vgs_hi, targets=DEFAULT_RESIDUAL_VGS_TARGETS):
    """[(actual_vgs, vds_sorted, ids_sorted), ...] -- one entry per TARGET Vgs in `targets`
    that falls inside [vgs_lo, vgs_hi], each paired with its CLOSEST real measurement trace
    (TN = a Vds sweep at a fixed Vgs setpoint). Ground truth for _residual_bump_check: real
    Ids at that trace's OWN (irregularly spaced) Vds points -- interpolating the model onto
    real points (rather than interpolating real data onto a model grid) avoids inventing
    detail the data doesn't have. The model is evaluated at the matched trace's own ACTUAL
    mean Vgs, NOT the nominal target -- those differ by only ~0.03-0.05V, but that's enough to
    change the computed residual by ~2x (confirmed empirically). Matches
    plot_saved_state.py's Ids-vs-Vds panel, which evaluates the model at the same actual
    matched Vgs for the same reason -- what's actually being compared, both here and visually,
    is the real trace, so the model must be evaluated at that trace's real Vgs, not the
    nominal grid value used only to pick WHICH trace to compare against. (Vds needs no
    analogous fix: `vds_sorted` below is already the trace's own real, irregularly-spaced Vds
    points -- the model is evaluated against those directly, never a nominal Vds grid.)"""
    with contextlib.redirect_stdout(io.StringIO()):
        T = _load_T_train(csv, min_vgs, device)
    trace_means = {tn: float(grp["Vgs"].mean()) for tn, grp in T.groupby("TN")}
    out = []
    for target in targets:
        if not (vgs_lo <= target <= vgs_hi):
            continue
        best_tn = min(trace_means, key=lambda tn: abs(trace_means[tn] - target))
        g = T[T["TN"] == best_tn].sort_values("Vds")
        out.append((trace_means[best_tn], g["Vds"].values.astype(float), g["Ids"].values.astype(float)))
    return out


def _model_ids_at(model, vgs_val, vds_array, device):
    """Forward-only model Ids (no grad needed) at a fixed Vgs over an arbitrary Vds array."""
    vd = torch.tensor(vds_array, dtype=torch.float64, device=device)
    vg = torch.full_like(vd, float(vgs_val))
    X = torch.stack([vg, vd], dim=1)
    with torch.no_grad():
        return model(X).reshape(-1).cpu().numpy()


def _residual_bump_check(model, real_traces, bump_vds, neg_tol, pos_tol, device):
    """For each REAL measured Vds trace, compare the model's Ids against the real Ids at the
    SAME (real, irregularly spaced) Vds points -- the fit error (residual = model - real)
    should trend smoothly (monotonically toward its steady-state offset) across a sweep, not
    wobble away and back. Split into two independent severities, since they mean different
    things physically:
      gds_residual_neg_max = worst UNDERSHOOT (model BELOW real data) -- how far negative the
                              residual dips, at its deepest INTERIOR point.
      gds_residual_pos_max = worst OVERSHOOT (model ABOVE real data) -- how far positive the
                              residual rises, at its highest INTERIOR point.
    "INTERIOR" excludes the trace's own first/last judged point -- the very edge of a Vds
    sweep is prone to boundary artifacts (confirmed empirically: EVERY trace shows a small
    positive blip at its first point, ~0.01, regardless of whether the model is otherwise
    good or bad -- not a real signal). A trace counts as bad if either severity exceeds its
    tolerance (--residual_neg_tol / --residual_pos_tol, both absolute Ids units)."""
    dlo, dhi = min(bump_vds), max(bump_vds)
    n_bad, worst_neg, worst_pos, n_traces = 0, 0.0, 0.0, 0
    for vgs_val, vds_arr, ids_arr in real_traces:
        m = (vds_arr >= dlo) & (vds_arr <= dhi)
        if m.sum() < 5:
            continue
        n_traces += 1
        vds_m, ids_m = vds_arr[m], ids_arr[m]
        pred = _model_ids_at(model, vgs_val, vds_m, device)
        resid = (pred - ids_m)[1:-1]                 # drop first/last (boundary artifact)
        neg_max = max(0.0, float(-resid.min())) if len(resid) else 0.0
        pos_max = max(0.0, float(resid.max())) if len(resid) else 0.0
        worst_neg = max(worst_neg, neg_max)
        worst_pos = max(worst_pos, pos_max)
        if neg_max > neg_tol or pos_max > pos_tol:
            n_bad += 1
    return {
        "gds_residual_bad_frac": round(n_bad / n_traces, 3) if n_traces else 0.0,
        "gds_residual_neg_max": round(worst_neg, 4),
        "gds_residual_pos_max": round(worst_pos, 4),
        "gds_residual_worst_max": round(max(worst_neg, worst_pos), 4),   # single sort key
    }


def _gds_bump(gds_g, vgs, vds, bump_vgs, bump_vds):
    """Ids-Vds 'slope hump' SEVERITY. The measured output conductance gds=dIds/dVds is
    MONOTONICALLY DECREASING in the knee (max at Vds->0), so a model bump makes gds rise then
    fall (an interior maximum) even while gds stays >0 -- which neg_gds misses.

    Restricted to the STRONG-ON Vgs window (bump_vgs): near-threshold traces hump in the real
    data too, so a hump there is physical, not a defect. Per current-carrying trace, severity =
    (max(gds) - gds_at_lowest_Vds)/max(gds): 0 if gds peaks at Vds->0 (monotone, like the data),
    grows as the hump gets bigger. gds_hump = mean severity, gds_hump_max = worst trace."""
    glo, ghi = min(bump_vgs), max(bump_vgs)
    dlo, dhi = min(bump_vds), max(bump_vds)
    G = gds_g[np.ix_((vgs >= glo) & (vgs <= ghi), (vds >= dlo) & (vds <= dhi))]
    zero = {"gds_hump": 0.0, "gds_hump_max": 0.0}
    if G.shape[0] == 0 or G.shape[1] < 3:
        return zero
    amp_gate = 0.10 * float(np.abs(G).max())            # skip subthreshold (near-zero gds) traces
    sev = []
    for i in range(G.shape[0]):
        row = G[i, :]
        mx = float(row.max())
        if float(np.abs(row).max()) < amp_gate or mx <= 0:
            continue
        sev.append(max(0.0, (mx - float(row[0])) / mx))  # 0 if peak at Vds->0, else the hump size
    if not sev:
        return zero
    return {"gds_hump": round(float(np.mean(sev)), 3), "gds_hump_max": round(float(np.max(sev)), 3)}


def shape_metrics(model, gds_region, gm_region, vgs_n, vds_n, device,
                  extrema_vgs, extrema_vds, gm3_min_windows, gm3_max_windows, prom_frac, bump_vgs, bump_vds,
                  real_gm3_start_range, gm3_start_tol, real_gm2_start_range, gm2_start_tol,
                  gm3_max_tolerant_window, gm3_max_tolerant_amp, gm3_tail_amp,
                  real_gm3_global_range, gm3_global_max_mult, gm3_global_min_mult,
                  real_vds_traces, residual_vds_window, residual_neg_tol, residual_pos_tol):
    # Ids-Vds bumps: gds inside the knee box (that's where the bumps are).
    gds_g, _, _, gds_vgs, gds_vds = _derivs(model, gds_region, vgs_n, vds_n, device)
    # gm smoothness: gm2/gm3 over the WIDER gm domain (turn-on peaks sit near Vgs<-3, deep sat Vds>15).
    _, gm2_g, gm3_g, vgs, vds = _derivs(model, gm_region, vgs_n, vds_n, device)

    neg = np.clip(-gds_g, 0, None)
    out = {
        "neg_gds_frac":   float((gds_g < 0).mean()),
        "neg_gds_energy": float((neg ** 2).mean()),     # Ids DIPS (gds<0)
        "min_gds":        float(gds_g.min()),
        "gm2_maxabs":     float(np.abs(gm2_g).max()),
        "gm3_maxabs":     float(np.abs(gm3_g).max()),   # PEAK severity (catches a single spike)
        "gm2_tv_norm":    _tv_norm(gm2_g),
        "gm3_tv_norm":    _tv_norm(gm3_g),              # wiggliness (multi-bump), not spikes
    }
    # Ids-Vds slope HUMP: gds rises-then-falls in the knee (a bump that stays gds>0).
    out.update(_gds_bump(gds_g, gds_vgs, gds_vds, bump_vgs, bump_vds))
    # gm shape: gm3 minima/maxima structure in the turn-on window.
    out.update(_extrema_counts(gm3_g, vgs, vds, extrema_vgs, extrema_vds, gm3_min_windows, gm3_max_windows,
                               prom_frac, max_tolerant_window=gm3_max_tolerant_window,
                               max_tolerant_amp=gm3_max_tolerant_amp))
    # gm3/gm2 at the very start of the curve vs the real measured value there (catches edge
    # artifacts the extrema check can't see, since it ignores window endpoints by design).
    out.update(_start_check(gm3_g, real_gm3_start_range, gm3_start_tol, "gm3"))
    out.update(_start_check(gm2_g, real_gm2_start_range, gm2_start_tol, "gm2"))
    # Beyond the max-window search range (Vgs > extrema_vgs's upper bound), gm3 must stay small.
    out.update(_tail_check(gm3_g, vgs, vds, max(extrema_vgs), extrema_vds, gm3_tail_amp))
    # gm3 anywhere on the evaluated grid vs the real measured GLOBAL min/max, scaled by a
    # multiplier -- catches runaway excursions the other, more localized checks don't cover.
    out.update(_global_check(gm3_g, real_gm3_global_range, gm3_global_max_mult, gm3_global_min_mult))
    # Ids fit-error SHAPE: does the model-vs-real residual wobble along a Vds sweep (gets worse
    # then better) instead of trending smoothly? Distinct from gds_hump (which only looks at
    # the model's OWN curve, self-referentially) -- this compares directly against real traces.
    out.update(_residual_bump_check(model, real_vds_traces, residual_vds_window,
                                    residual_neg_tol, residual_pos_tol, device))
    return out


def _run_dir(row):
    fp = row.get("file_path", "")
    return os.path.dirname(fp) if fp else None


def _parse_windows(spec):
    """'lo..hi,lo..hi' -> [(lo,hi), ...]."""
    out = []
    for piece in spec.split(","):
        lo, hi = re.split(r"\.\.|to|:", piece.strip())
        out.append((float(lo), float(hi)))
    return out


def write_gmshape_outputs(out_rows, cols, gm3_expect_min, gm3_expect_max,
                          gm3_min_windows, gm3_max_windows, base_path):
    """Writes SIX compliance/sort combinations from already-scored out_rows. Pure
    filtering/sorting on metrics already present -- no model evaluation, so this is cheap
    whether out_rows just came from the model-evaluation loop in main() or was loaded straight
    off an existing <ranked>_shape.csv (see --from_shape_csv).

      gmshape_ok_by_gmshape.csv   -- GM-compliant rows, sorted by region_knee_combined_gm.
      gmshape_ok_by_gdshape.csv   -- SAME GM-compliant rows, sorted by gds_residual_worst_max.
      gdsshape_ok_by_gmshape.csv  -- GDS-compliant rows (gds_residual_bad_frac==0), sorted by
                                      region_knee_combined_gm.
      gdsshape_ok_by_gdshape.csv  -- SAME GDS-compliant rows, sorted by gds_residual_worst_max.
      bothshape_ok.csv            -- rows compliant with BOTH gmshape AND gdsshape (the
                                      intersection), sorted by shape_rank (itself the mean
                                      rank of gmshape + gdshape, a natural fit here).
      bothshape_removed.csv       -- the complement within out_rows (every row that failed
                                      gmshape and/or gdsshape), same shape_rank sort -- unlike
                                      gmshape/gdsshape, bothshape gets an explicit "removed"
                                      file since it's the two-sided AND of both checks and a
                                      row can be absent from it for either reason, which isn't
                                      obvious from gmshape_ok/gdsshape_ok alone.

      gmshape = GM-only: gm3 window/tail/start/global checks (everything except the Ids
                fit-error check). A model can pass this with individually smooth gm2/gm3
                curves while still fitting Ids badly -- see gdsshape.
      gdsshape = GDS-only: gds_residual_bad_frac==0 (the Ids fit-error doesn't wobble
                 away-from/back-toward real data along any Vds sweep). Only written if that
                 column exists (older <ranked>_shape.csv files predate this check).

    No "_removed" files for gmshape/gdsshape individually -- the caller's own
    <ranked>_shape.csv already has every row's metrics (why something failed is visible
    there). bothshape is the one exception (see bothshape_removed.csv above).
    """
    def _eq(r, k, v):
        n = _num(r.get(k))
        return n is not None and n == v
    def _ge(r, k, v):
        n = _num(r.get(k))
        return n is not None and n >= v
    # gm2_start_bad_frac / gm3_tail_bad_frac / etc. are newer columns -- older <ranked>_shape.csv
    # files loaded via --from_shape_csv won't have them. Skip a check rather than treat
    # missing as failing (that would silently exclude every row of an older file).
    has_gm2_check = "gm2_start_bad_frac" in cols
    has_tail_check = "gm3_tail_bad_frac" in cols
    has_max_window_check = "gm3_max_missing_frac" in cols
    has_min_window_check = "gm3_min_missing_frac" in cols
    has_global_check = "gm3_global_bad_frac" in cols
    has_residual_check = "gds_residual_bad_frac" in cols
    # gm3_n_min only needs to hit >=1 (a real turn-on extremum exists somewhere) -- but the
    # LAST min window (gm3_min_windows[-1], e.g. -3.5..-1.7) is separately REQUIRED
    # (gm3_min_missing_frac==0); the earlier min window(s) stay fully optional, per
    # _match_windows's docstring (a missing match isn't, by itself, a defect). The max side is
    # similar: the primary max window (gm3_max_windows, e.g. -4..-2.2) is required
    # (gm3_max_missing_frac==0). gm3_bad_frac (EXCESS beyond window capacity, already tolerant
    # of a small extra max in --gm3_max_tolerant_window) and the gm3/gm2 start-vs-real checks
    # plus the gm3 tail check (large excursions past the search boundary) are the other
    # signals that distinguish good from bad GM shape.
    gm_ok = [r for r in out_rows
             if _ge(r, "gm3_n_min", 1)
             and _ge(r, "gm3_n_max", 1)
             and _eq(r, "gm3_bad_frac", 0)
             and _eq(r, "gm3_start_bad_frac", 0)
             and (not has_gm2_check or _eq(r, "gm2_start_bad_frac", 0))
             and (not has_tail_check or _eq(r, "gm3_tail_bad_frac", 0))
             and (not has_max_window_check or _eq(r, "gm3_max_missing_frac", 0))
             and (not has_min_window_check or _eq(r, "gm3_min_missing_frac", 0))
             and (not has_global_check or _eq(r, "gm3_global_bad_frac", 0))]
    print(f"\n{len(gm_ok)}/{len(out_rows)} rows comply with the expected GM (gm2/gm3) shape "
          f"(>=1 minimum found -- up to {gm3_expect_min} expected in {gm3_min_windows} -- "
          f"+ >=1 maximum -- up to {gm3_expect_max} expected in {gm3_max_windows} -- "
          f"no excess slices, gm3_start_bad_frac==0).")

    def _write_ranked(rows, sort_key, path):
        rows = sorted(rows, key=lambda r: (_num(r.get(sort_key)) is None, _num(r.get(sort_key))))
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = _csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {path}  ({len(rows)} rows, ranked by {sort_key})")

    _write_ranked(gm_ok, "region_knee_combined_gm", base_path + "_gmshape_ok_by_gmshape.csv")
    _write_ranked(gm_ok, "gds_residual_worst_max", base_path + "_gmshape_ok_by_gdshape.csv")

    if has_residual_check:
        gds_ok = [r for r in out_rows if _eq(r, "gds_residual_bad_frac", 0)]
        print(f"{len(gds_ok)}/{len(out_rows)} rows comply with the Ids fit-error (gds) shape "
              f"(gds_residual_bad_frac==0 -- no wobble away-from/back-toward real data).")
        _write_ranked(gds_ok, "region_knee_combined_gm", base_path + "_gdsshape_ok_by_gmshape.csv")
        _write_ranked(gds_ok, "gds_residual_worst_max", base_path + "_gdsshape_ok_by_gdshape.csv")

        gm_ok_ids = {r.get("id") for r in gm_ok}
        both_ok_ids = {r.get("id") for r in gds_ok if r.get("id") in gm_ok_ids}
        both_ok = [r for r in out_rows if r.get("id") in both_ok_ids]
        both_removed = [r for r in out_rows if r.get("id") not in both_ok_ids]
        print(f"{len(both_ok)}/{len(out_rows)} rows comply with BOTH gmshape and gdsshape.")
        _write_ranked(both_ok, "shape_rank", base_path + "_bothshape_ok.csv")
        _write_ranked(both_removed, "shape_rank", base_path + "_bothshape_removed.csv")


def rederive_from_shape_csv(path: str, gm3_min_windows_spec: str, gm3_max_windows_spec: str):
    """Fast path for --from_shape_csv: re-derive gmshape_ok/gmshape_removed from an ALREADY-
    COMPUTED <ranked>_shape.csv -- no torch, no model reload, no measurement CSV. Only valid
    when gm3_min_windows/gm3_max_windows match what produced that file (gm3_n_min/gm3_n_max
    are read as-is, not recomputed)."""
    with open(path, newline="", encoding="utf-8") as f:
        out_rows = list(_csv.DictReader(f))
    if not out_rows:
        sys.exit(f"no rows in {path}")
    cols = list(out_rows[0].keys())
    needed = ("id", "gm3_n_min", "gm3_n_max", "gm3_bad_frac", "gm3_start_bad_frac", "shape_rank")
    missing = [c for c in needed if c not in cols]
    if missing:
        sys.exit(f"[error] {path} is missing column(s) {missing} -- not a <ranked>_shape.csv "
                 "produced by this script's normal (non---from_shape_csv) path.")
    gm3_min_windows = _parse_windows(gm3_min_windows_spec)
    gm3_max_windows = _parse_windows(gm3_max_windows_spec)
    print(f"[from_shape_csv] {len(out_rows)} rows loaded from {path} -- reusing already-computed "
          "gm3_n_min/gm3_n_max/gm3_bad_frac/gm3_start_bad_frac (no model reload).")
    base = os.path.splitext(path)[0]
    write_gmshape_outputs(out_rows, cols, len(gm3_min_windows), len(gm3_max_windows),
                          gm3_min_windows, gm3_max_windows, base)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ranked_csv", default=None,
                    help="FULL ranked_by_*.csv (must have a file_path column; not the _slim). "
                         "Required unless --from_shape_csv is given instead.")
    ap.add_argument("--from_shape_csv", default=None,
                    help="Fast path: re-derive _gmshape_ok_by_*.csv / _gmshape_removed.csv from "
                         "an ALREADY-COMPUTED <ranked>_shape.csv (e.g. from a prior run of this "
                         "script) -- no torch, no model reload, no measurement CSV needed. Only "
                         "valid when --gm3_min_windows/--gm3_max_windows match what was used to "
                         "produce that file (gm3_n_min/gm3_n_max are read as-is, not "
                         "recomputed). Mutually exclusive with --ranked_csv.")
    ap.add_argument("--csv", default=None,
                    help="Measurement CSV, for the gm3-start-vs-real check. If omitted, "
                         "auto-detected from the ranked CSV's results folder "
                         "(base_files/_run_meta.json).")
    ap.add_argument("--min_vgs", type=float, default=-4.0,
                    help="min_vgs passed to the data loader for the gm3-start-vs-real check "
                         "(extrapolation floor, matches training's --min_vgs).")
    ap.add_argument("--gm3_start_tol", type=float, default=1.0,
                    help="Padding added on each side of the REAL measured gm3 RANGE (min..max "
                         "across Vds slices) at the START of the evaluated curve (gm_region's "
                         "lowest Vgs, deep subthreshold). A Vds column counts as OK if the "
                         "model's value there falls inside [real_min-tol, real_max+tol] "
                         "(default tol=1.0). The extrema check ignores window endpoints by "
                         "design, so this catches spurious edge artifacts it can't see.")
    ap.add_argument("--gm2_start_tol", type=float, default=1.0,
                    help="Same as --gm3_start_tol, but for gm2 (second derivative) at the "
                         "curve start -- an independent edge-artifact check.")
    ap.add_argument("--filter", dest="filters", action="append", default=[],
                    help="COL<op>VALUE, repeatable. Not passed -> NO filter is applied (every row "
                         "gets the -- expensive -- shape evaluation). Pre-filter for fit quality "
                         "first (e.g. via filter_results.py) or pass this explicitly, e.g. "
                         "--filter region_knee_combined_gm<0.9 --filter region_knee_ids_rmse<0.013.")
    ap.add_argument("--region", default="vgs=-3..0,vds=0..15",
                    help="Knee window for the Ids-Vds (gds) bump metrics.")
    ap.add_argument("--gm_region", default="vgs=-4..0,vds=0..28",
                    help="WIDER window for gm2/gm3 smoothness -- must cover the turn-on region "
                         "(Vgs<-3) and deep saturation where gm peaks appear, or they're missed.")
    ap.add_argument("--vgs_n", type=int, default=121, help="grid points along Vgs.")
    ap.add_argument("--vds_n", type=int, default=151, help="grid points along Vds.")
    ap.add_argument("--extrema_vgs", default="-4..-1.7",
                    help="Vgs turn-on window in which gm3's INTERIOR extrema are counted "
                         "(default -4..-1.7 -- wide enough to cover --gm3_min_windows/"
                         "--gm3_max_windows below; the extra gm3 wiggles appear near turn-on).")
    ap.add_argument("--extrema_vds", default="5..28",
                    help="SATURATION Vds window for the gm3 extrema check (default 5..28). The "
                         "expected gm3 shape is a saturation expectation; the low-Vds linear "
                         "region is excluded (different gm3 structure, and not plotted).")
    ap.add_argument("--gm3_min_windows", default="-4..-3.5,-3.5..-1.7",
                    help="Comma-separated Vgs sub-windows ('lo..hi'), one per expected INTERIOR "
                         "gm3 minimum, in order (default: an early minimum before -3.5, then a "
                         "second between -3.5 and -1.7 -- calibrated against the measured gm3 "
                         "shape, see plot_gm_derivatives.py). A minimum found outside every "
                         "window, or a second one crowding an already-filled window, counts as "
                         "excess; a window with no match (missing) is NOT penalized.")
    ap.add_argument("--gm3_max_windows", default="-4..-2.2",
                    help="Comma-separated Vgs sub-windows, one per expected INTERIOR gm3 maximum "
                         "(default: before -2.2). Same matching rule as --gm3_min_windows.")
    ap.add_argument("--gm3_max_tolerant_window", default="-2.2..-1.7",
                    help="Vgs window (default: the gap between --gm3_max_windows's edge and "
                         "--extrema_vgs's upper bound) where an EXTRA, otherwise-unmatched gm3 "
                         "maximum is tolerated -- not counted as excess -- as long as it's small "
                         "(see --gm3_max_tolerant_amp). A small secondary bump there isn't a "
                         "defect; a large one still is.")
    ap.add_argument("--gm3_max_tolerant_amp", type=float, default=1.0,
                    help="Amplitude threshold for --gm3_max_tolerant_window: an extra max there "
                         "with |gm3| under this is tolerated (default 1.0).")
    ap.add_argument("--gm3_tail_amp", type=float, default=1.0,
                    help="Beyond --extrema_vgs's upper bound (i.e. past where the window-match "
                         "check even looks), gm3 must stay under this amplitude (default 1.0) -- "
                         "a large excursion there is a shape defect the window check can't see, "
                         "since it never searches that far. -> gm3_tail_bad_frac.")
    ap.add_argument("--gm3_global_max_mult", type=float, default=3.0,
                    help="Model gm3 must not exceed (real measured GLOBAL max) * this multiplier "
                         "(default 3), checked EVERYWHERE on the evaluated grid, not just the "
                         "curve start/tail -- catches a runaway/unphysical excursion anywhere. "
                         "-> gm3_global_bad_frac.")
    ap.add_argument("--gm3_global_min_mult", type=float, default=3.0,
                    help="Same as --gm3_global_max_mult, for the (real measured GLOBAL min) "
                         "side (real_min is normally negative, so this widens the LOWER bound). "
                         "Default 3.")
    ap.add_argument("--residual_vgs", default="-3..0",
                    help="Vgs window for the Ids fit-error (residual) shape check -- wider than "
                         "--bump_vgs by default, since this wobble can show up well outside the "
                         "strong-on region (e.g. Vgs=-1.8 or -2.2). Every REAL measurement trace "
                         "whose Vgs falls in this window is checked.")
    ap.add_argument("--residual_vds", default="0..6",
                    help="Vds window for the residual shape check (default 0..6, same as "
                         "--bump_vds).")
    ap.add_argument("--residual_neg_tol", type=float, default=0.062,
                    help="Absolute Ids tolerance for the WORST UNDERSHOOT (model BELOW real "
                         "data) at any interior point of a real Vds trace's fit-error residual "
                         "-> gds_residual_neg_max. Default 0.062, calibrated against "
                         "sigmoid_margin10_nogm ids 267(0.065,bad)/204(0.063,bad)/"
                         "294(0.061,ok)/395(0.054,ok) -- razor-thin margin from just 4 examples, "
                         "verify against more data before trusting broadly.")
    ap.add_argument("--residual_pos_tol", type=float, default=0.068,
                    help="Same as --residual_neg_tol, for the WORST OVERSHOOT (model ABOVE "
                         "real data) -> gds_residual_pos_max. Widened from the original 0.062 "
                         "(same as --residual_neg_tol) to 0.068 after visual review: id=294's "
                         "pos_max=0.0675 (a small Vgs=-1.6 bump near Vds=0.5) was judged 'good "
                         "enough' alongside 395, so the tolerance was relaxed just past it -- "
                         "204 (pos_max=0.0704) and 267 (neg_max=0.0711, still over "
                         "--residual_neg_tol regardless) remain excluded.")
    ap.add_argument("--extrema_prom", type=float, default=0.02,
                    help="Min prominence (fraction of a slice's gm3 peak-to-peak) for an extremum "
                         "to count -- filters numerical ripple, keeps real small extrema.")
    ap.add_argument("--bump_vds", default="0..6",
                    help="Knee Vds window for the gds slope-hump check (default 0..6). gds should "
                         "decrease monotonically here; a rise-then-fall = an Ids-Vds bump.")
    ap.add_argument("--bump_vgs", default="-1.6..0",
                    help="STRONG-ON Vgs window for the gds hump check (default -1.6..0). Near-"
                         "threshold traces hump in the real data too, so they're excluded.")
    ap.add_argument("--sort_by", default="shape_rank",
                    help="Column to sort ascending (lower=smoother). Default shape_rank -- "
                         "mean rank of region_knee_combined_gm (gmshape) and "
                         "gds_residual_worst_max (gdshape).")
    ap.add_argument("--top", type=int, default=None, help="Only print the top N after sorting.")
    ap.add_argument("--out", default=None, help="Output CSV (default: <ranked>_shape.csv).")
    ap.add_argument("--no_gmshape_csvs", action="store_true",
                    help="Skip writing the two extra gm-shape-compliant CSVs (see below).")
    args = ap.parse_args()

    if bool(args.from_shape_csv) == bool(args.ranked_csv):
        sys.exit("[error] give exactly one of --ranked_csv (full evaluation) or "
                 "--from_shape_csv (fast re-derive, no model reload).")
    if args.from_shape_csv:
        rederive_from_shape_csv(args.from_shape_csv, args.gm3_min_windows, args.gm3_max_windows)
        return

    rows = list(_csv.DictReader(open(args.ranked_csv, newline="", encoding="utf-8")))
    if rows and "file_path" not in rows[0]:
        sys.exit("[error] no 'file_path' column -- use the FULL ranked csv, not the _slim one.")
    print(f"loaded {len(rows)} rows")
    if args.filters:
        kept = apply_filters(rows, args.filters)
        print(f"after filter {args.filters}: {len(kept)} rows")
    else:
        kept = rows
        print("no --filter given: evaluating ALL rows (this loads/evaluates every model -- "
              "expensive for a large sweep; consider pre-filtering).")
    if not kept:
        sys.exit("no rows passed the filter.")

    gds_region = parse_region(args.region)
    gm_region = parse_region(args.gm_region)
    ev = re.split(r"\.\.|to|:", args.extrema_vgs)
    extrema_vgs = (float(ev[0]), float(ev[1]))
    ed = re.split(r"\.\.|to|:", args.extrema_vds)
    extrema_vds = (float(ed[0]), float(ed[1]))
    eb = re.split(r"\.\.|to|:", args.bump_vds)
    bump_vds = (float(eb[0]), float(eb[1]))
    ebg = re.split(r"\.\.|to|:", args.bump_vgs)
    bump_vgs = (float(ebg[0]), float(ebg[1]))
    gm3_min_windows = _parse_windows(args.gm3_min_windows)
    gm3_max_windows = _parse_windows(args.gm3_max_windows)
    gm3_max_tolerant_window = _parse_windows(args.gm3_max_tolerant_window)[0]
    gm3_expect_min = len(gm3_min_windows)
    gm3_expect_max = len(gm3_max_windows)
    print(f"gds/Ids-Vds window: {gds_region}   |   gm2/gm3 window: {gm_region}")
    print(f"gm3 extrema window: Vgs in {extrema_vgs}, Vds in {extrema_vds} (saturation), expect "
          f"{gm3_expect_min} minima in {gm3_min_windows} + {gm3_expect_max} maximum in {gm3_max_windows}")
    print(f"gm3 tolerant zone: {gm3_max_tolerant_window} allows an extra max under "
          f"|{args.gm3_max_tolerant_amp}|; beyond {extrema_vgs[1]} (the search boundary), gm3 "
          f"must stay under |{args.gm3_tail_amp}| (-> gm3_tail_bad_frac)")

    device = torch.device("cpu")

    # --- measurement CSV, for the gm3-start-vs-real check (auto-detect if not given) ---
    csv = args.csv
    if not csv:
        import run_artifacts
        ranked_folder = os.path.dirname(os.path.abspath(args.ranked_csv))
        for base in (os.path.dirname(ranked_folder), ranked_folder,
                     os.path.dirname(os.path.dirname(ranked_folder))):
            csv = run_artifacts.run_meta(base).get("csv")
            if csv:
                print(f"[auto] measurement CSV: {csv}")
                break
    if not csv or not os.path.isfile(csv):
        sys.exit("[error] measurement CSV not found for the gm3-start check; pass --csv.")
    vgs_start = gm_region["vgs"][0]
    real_lo, real_hi = measured_gm_start_range(csv, args.min_vgs, vgs_start, device, order=3)
    real_lo2, real_hi2 = measured_gm_start_range(csv, args.min_vgs, vgs_start, device, order=2)
    real_glo, real_ghi = measured_gm_global_range(csv, args.min_vgs, device, order=3)
    print(f"real gm3 at Vgs={vgs_start:.2f} (curve start) ranges {real_lo:.3f} to {real_hi:.3f} "
          f"across the measured Vds slices; model must fall within "
          f"[{real_lo - args.gm3_start_tol:.3f}, {real_hi + args.gm3_start_tol:.3f}] "
          f"(that range, padded by +-{args.gm3_start_tol})")
    print(f"real gm2 at Vgs={vgs_start:.2f} (curve start) ranges {real_lo2:.3f} to {real_hi2:.3f}; "
          f"model must fall within "
          f"[{real_lo2 - args.gm2_start_tol:.3f}, {real_hi2 + args.gm2_start_tol:.3f}]")
    print(f"real gm3 GLOBAL range: {real_glo:.3f} to {real_ghi:.3f}; model must fall within "
          f"[{real_glo * args.gm3_global_min_mult:.3f}, {real_ghi * args.gm3_global_max_mult:.3f}] "
          f"everywhere on the evaluated grid ({real_glo:.3f}x{args.gm3_global_min_mult}, "
          f"{real_ghi:.3f}x{args.gm3_global_max_mult})")

    erv = re.split(r"\.\.|to|:", args.residual_vgs)
    residual_vgs_window = (float(erv[0]), float(erv[1]))
    erd = re.split(r"\.\.|to|:", args.residual_vds)
    residual_vds_window = (float(erd[0]), float(erd[1]))
    real_vds_traces = measured_vds_traces(csv, args.min_vgs, device, *residual_vgs_window)
    print(f"residual-shape check: {len(real_vds_traces)} real trace(s) with Vgs in "
          f"{residual_vgs_window}, Vds in {residual_vds_window}, neg_tol={args.residual_neg_tol} "
          f"pos_tol={args.residual_pos_tol}")

    out_rows, fail = [], 0
    for i, r in enumerate(kept, 1):
        rd = _run_dir(r)
        if not rd or not os.path.isdir(rd):
            print(f"[{i}/{len(kept)}] id={r.get('id')} SKIP: run dir not found ({rd})")
            fail += 1
            continue
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                model = _build_model_from_dir(os.path.abspath(rd), device)
            m = shape_metrics(model, gds_region, gm_region, args.vgs_n, args.vds_n, device,
                              extrema_vgs, extrema_vds, gm3_min_windows, gm3_max_windows,
                              args.extrema_prom, bump_vgs, bump_vds,
                              (real_lo, real_hi), args.gm3_start_tol,
                              (real_lo2, real_hi2), args.gm2_start_tol,
                              gm3_max_tolerant_window, args.gm3_max_tolerant_amp, args.gm3_tail_amp,
                              (real_glo, real_ghi), args.gm3_global_max_mult, args.gm3_global_min_mult,
                              real_vds_traces, residual_vds_window,
                              args.residual_neg_tol, args.residual_pos_tol)
        except Exception as e:
            print(f"[{i}/{len(kept)}] id={r.get('id')} FAILED: {e}")
            fail += 1
            continue
        keep = {k: r.get(k) for k in ("id", "run_id", "arch_id", "arch_hash", "vds_loss",
                                      "region_knee_combined_gm", "region_knee_ids_rmse",
                                      "file_path") if k in r}
        keep.update(m)   # full precision -- neg_gds_energy is ~1e-7, don't round it away
        out_rows.append(keep)
        print(f"[{i}/{len(kept)}] id={r.get('id')}  neg_gds_E={m['neg_gds_energy']:.2e} "
              f"gds_hump={m['gds_hump']:.2f}  gm3_maxabs={m['gm3_maxabs']:.2f} "
              f"gm3_bad={m['gm3_bad_frac']:.2f}  gm3_start_diff={m['gm3_start_diff']:.2f} "
              f"gm2_start_diff={m['gm2_start_diff']:.2f}", flush=True)

    if not out_rows:
        sys.exit(f"nothing computed (fail={fail}).")

    # shape_rank = mean of ascending ranks of the two independent shape dimensions this file
    # is actually organized around: gmshape (region_knee_combined_gm -- point-wise fit, also
    # the basis of every gm3/gm2 shape check) and gdshape (gds_residual_worst_max -- the Ids
    # fit-error severity). Replaces an older 4-metric blend (neg_gds_energy/gds_hump/
    # gm3_maxabs/gm3_bad_frac) that predated the gmshape/gdshape split and mixed both
    # dimensions into one number without a clear meaning.
    def add_rank(key):
        order = sorted(range(len(out_rows)), key=lambda j: out_rows[j][key])
        for rank, j in enumerate(order, 1):
            out_rows[j].setdefault("_r", {})[key] = rank
    add_rank("region_knee_combined_gm")
    add_rank("gds_residual_worst_max")
    for r in out_rows:
        r["shape_rank"] = round((r["_r"]["region_knee_combined_gm"]
                                 + r["_r"]["gds_residual_worst_max"]) / 2, 1)
        del r["_r"]

    if args.sort_by not in out_rows[0]:
        sys.exit(f"[error] --sort_by {args.sort_by!r} not a column. have={list(out_rows[0])}")
    out_rows.sort(key=lambda r: (r[args.sort_by] is None, r[args.sort_by]))

    out = args.out or os.path.splitext(args.ranked_csv)[0] + "_shape.csv"
    cols = list(out_rows[0].keys())
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(out_rows)

    # Three extra CSVs: two for configs that COMPLY with the expected gm3 extrema shape --
    # >=1 real minimum and >=1 real maximum found (not a strict count match -- a missing
    # window match isn't a defect), no slice shows extra extrema beyond window capacity
    # (gm3_bad_frac==0), and gm3/gm2 at the curve start both fall within the real measured
    # range (gm3_start_bad_frac==0, gm2_start_bad_frac==0). One CSV ranked by point-wise fit
    # (region_knee_combined_gm), one by the Ids-Vds slope-hump severity (gds_hump -- the
    # "bump metric") -- plus the removed complement. See write_gmshape_outputs() -- shared
    # with the --from_shape_csv fast path.
    if not args.no_gmshape_csvs:
        print(f"(gm3 at the curve start must additionally fall within "
              f"[{real_lo - args.gm3_start_tol:.3f}, {real_hi + args.gm3_start_tol:.3f}], "
              f"gm2 within [{real_lo2 - args.gm2_start_tol:.3f}, {real_hi2 + args.gm2_start_tol:.3f}])")
        base = os.path.splitext(out)[0]
        write_gmshape_outputs(out_rows, cols, gm3_expect_min, gm3_expect_max,
                              gm3_min_windows, gm3_max_windows, base)

    show = out_rows[: args.top] if args.top else out_rows
    print(f"\n=== ranked by {args.sort_by} (lower = smoother / more monotone), {len(show)} rows ===")
    hdr = ("id", "vds_loss", "region_knee_combined_gm", "region_knee_ids_rmse",
           "neg_gds_energy", "gds_hump", "gm3_maxabs", "gm3_bad_frac", "gm3_start_diff", "shape_rank")

    def fmt(v):
        n = _num(v)
        return f"{n:.4g}" if n is not None else str(v)
    print("  ".join(f"{h:>13}" for h in hdr))
    for r in show:
        print("  ".join(f"{fmt(r.get(h, '')):>13}" for h in hdr))
    print(f"\nwrote {out}   (ok={len(out_rows)} fail={fail})")


if __name__ == "__main__":
    main()
