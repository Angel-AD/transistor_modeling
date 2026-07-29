"""
Find architectures (by stable arch_hash) that are CONSISTENT across a <root>/<csv_name>/
<folder>/... sweep's measurement csvs, three different ways:

  A) fit-quality-only min-max search (no shape data needed): for every arch_hash present in
     EVERY csv's own combined_gm<=--gm_ceiling pool, compute its ids_rmse PERCENTILE within
     that csv's pool, take the WORST (max) percentile across all csvs, sort ascending (best
     worst-case first). Finds the single most reliably-good-everywhere architecture. Threshold
     -free by design -- a narrow top-20-per-csv repeat-search misses real consistent
     performers (this session's concrete lesson: e.g. vdsgate_aeff_quad_tanhm's 93fd5b66-style
     picks only show up this way, not via top-20 overlap).

  B) gmshape_ok / bothshape_ok consistency, goal=all_csvs: reads each csv's full-population
     <hash>_ranked_by_region_knee_combined_gm_shape.csv (written by run_full_pop_shape.py /
     analyze_shape.py -- NOT the narrow filtered variant), ranks by (count of csvs where the
     check passes) desc, tie-break avg ids_rmse asc across all csvs present.

  C) gmshape_ok / bothshape_ok consistency, goal=pair: same shape data, but restricted to
     architectures where the check passes on BOTH of --pair_csvs specifically (regardless of
     the other csvs), ranked by avg ids_rmse asc across just that pair.

Prints the top --top of each of the 5 result sets (A, B-gm, B-both, C-gm, C-both) as a table.
Pass --json to also dump machine-readable results (for driving plot_arch_hash.py programmatically).

Usage:
    python find_consistent_archs.py --root ../runs/best200ids_of_9069_byloss --folder tanh_margin10
    python find_consistent_archs.py --root ../runs/bothshapeok_of_9069_byshape --folder softplus_nogm \\
        --gm_ceiling 1.4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from plot_csv_row import _read_rows  # noqa: E402

DEFAULT_PAIR = ("cg2h40010_new_2.5_20_2_70W_center9", "cg2h40010_new_2.9_28_2_70W_center9")

# Canonical (json_key, heading, scope) for all 6 method result-sets a search run produces --
# ONE shared source of truth so write_findings.py / write_tanh_only_comparison.py /
# write_overall_findings.py / write_findings_with_plots.py / write_overall_best_archs.py don't
# each maintain their own slightly-drifting copy of this list. scope is "all" (every csv) or
# "pair" (just --pair_csvs).
METHOD_INFO = [
    ("fit_quality_minmax", "A) Fit-quality min-max search", "all"),
    ("gmshape_ok_all_csvs", "B) `gmshape_ok` consistency, goal=all_csvs", "all"),
    ("bothshape_ok_all_csvs", "B) `bothshape_ok` consistency, goal=all_csvs", "all"),
    ("gmshape_ok_pair", "C) `gmshape_ok` consistency, goal=pair", "pair"),
    ("bothshape_ok_pair", "C) `bothshape_ok` consistency, goal=pair", "pair"),
    ("fit_quality_minmax_pair", "D) Fit-quality min-max search, goal=pair", "pair"),
]


def method_sort_key(json_key: str, r: dict):
    """Sort key such that SMALLER = BETTER. Used ONLY to compare two rows that already each won
    their OWN search (rank #1 within their own population) against each other -- e.g. a
    tanh-only #1 pick vs a heterogeneous #1 pick, one base_config's #1 pick vs another's, or one
    folder's best pick vs another folder's -- never to decide who wins within a single search
    (each search function has its own internal sort for that, unaffected by this).

    For fit_quality_minmax/_pair this is avg_ids_rmse first, NOT worst_percentile: worst_percentile
    is computed against each search's own combined_gm-gated pool, and those pools can differ in
    size by 1-2 orders of magnitude across the populations being compared here (tanh-only: ~4-9
    candidates; heterogeneous: ~100-1147; different base_configs similarly), so the SAME-looking
    percentile isn't the same statistical thing in each -- comparing it directly is misleading.
    avg_ids_rmse is an absolute, pool-size-independent quantity (average fit error across the same
    csvs), so it's the fair basis for "is THIS pick actually better than THAT pick." Within a
    single search, worst_percentile remains the primary criterion (that search's own sort, not
    this function) since everyone compared there does share the same pool."""
    if json_key in ("fit_quality_minmax", "fit_quality_minmax_pair"):
        return (r["avg_ids_rmse"], r["worst_percentile"])
    if json_key in ("gmshape_ok_all_csvs", "bothshape_ok_all_csvs"):
        return (-r["count"], r["avg_ids_rmse"])
    if json_key in ("gmshape_ok_pair", "bothshape_ok_pair"):
        return (r["avg_ids_rmse"],)
    raise ValueError(f"unknown method json_key: {json_key!r}")


def _csv_subdirs(root: Path, folder: str) -> list[Path]:
    out = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        ranked_dir = p / folder / "ranked_region_knee_vgs-3to0_vds0to15"
        if ranked_dir.is_dir():
            out.append(p)
    return out


def arch_summary(arch_str: str) -> str:
    """'[tanh×7][swish×4]...' -- compact display of a layer-activation JSON string like
    '[["tanh","tanh"],["swish","swish","swish","swish"]]'. Falls back to the raw string if it
    doesn't parse."""
    try:
        layers = json.loads(arch_str)
    except (json.JSONDecodeError, TypeError):
        return arch_str
    parts = []
    for layer in layers:
        if not layer:
            continue
        counts = {}
        order = []
        for eq in layer:
            if eq not in counts:
                order.append(eq)
            counts[eq] = counts.get(eq, 0) + 1
        parts.append(",".join(f"{eq}×{counts[eq]}" for eq in order))
    return " ".join(f"[{p}]" for p in parts)


def load_arch_by_hash(root: Path, folder: str) -> dict[str, str]:
    """{arch_hash: architecture JSON string} for every architecture in this (root, folder)'s
    population. Architecture composition is a property of the arch itself, independent of which
    measurement csv it was trained against (same reasoning as tanh_only_hashes below), so reading
    just the FIRST available csv's base ranked file is sufficient."""
    out = {}
    subdirs = _csv_subdirs(root, folder)
    if not subdirs:
        return out
    ranked_dir = subdirs[0] / folder / "ranked_region_knee_vgs-3to0_vds0to15"
    base_matches = sorted(ranked_dir.glob("*_ranked_by_region_knee_combined_gm.csv"))
    if not base_matches:
        return out
    for r in _read_rows(base_matches[0]):
        h = r.get("arch_hash")
        if h and h not in out:
            out[h] = r.get("architecture", "")
    return out


def find_arch_plot_dir(out_root: Path, folder: str, arch_hash: str) -> Path | None:
    """<out_root>/<folder>/<category>/<goal>/<arch_hash>/ -- wherever plot_arch_hash.py wrote
    this arch_hash's plots + compliance.json + all_csvs.md, or None if it hasn't been plotted
    yet. If this arch_hash was picked under more than one <goal> (common -- an architecture can
    be both the fit-quality winner and a shape-consistency winner), returns whichever sorts
    first; doesn't matter which, since all copies for the same arch_hash hold identical plots
    and per-csv numbers (same architecture, same measurement csvs) -- verified by diff, the only
    difference between copies is a single cosmetic 'Search goal:' label line in all_csvs.md."""
    matches = sorted((out_root / folder).glob(f"*/*/{arch_hash}/compliance.json"))
    return matches[0].parent if matches else None


def arch_link_cell(arch_hash: str, arch_by_hash: dict[str, str], arch_dir: Path | None,
                    from_dir: Path, is_tanh_only: bool = False) -> str:
    """'[`c617950a`](../both/best_fit_quality/c617950a/all_csvs.md) [tanh×7][swish×4]
    **(tanh-only)**' if arch_dir/all_csvs.md exists, else plain '`c617950a` [tanh×7][swish×4]'.
    from_dir = the directory the CALLING .md file itself lives in, used to compute the relative
    link path (Ctrl+Click in VS Code opens it in a new editor tab). is_tanh_only appends a plain
    bold '(tanh-only)' marker -- plain bold in raw .md/VS Code, but render_html_view.py
    specifically recognizes this exact marker text and renders it in red in the .html view, so
    callers should only pass True in UNRESTRICTED (heterogeneous-population) contexts where a
    reader would find it notable -- redundant (and should be omitted) anywhere already scoped to
    tanh-only architectures, e.g. inside a folder's own tanh_only/ output."""
    summary = arch_summary(arch_by_hash.get(arch_hash, ""))
    marker = " **(tanh-only)**" if is_tanh_only else ""
    if arch_dir is not None:
        all_csvs = arch_dir / "all_csvs.md"
        if all_csvs.is_file():
            rel = os.path.relpath(all_csvs, start=from_dir).replace(os.sep, "/")
            return f"[`{arch_hash}`]({rel}) {summary}{marker}"
    return f"`{arch_hash}` {summary}{marker}"


def _f(row, key):
    try:
        return float(row.get(key))
    except (TypeError, ValueError):
        return None


def _compliance(row) -> dict:
    gmshape_ok = (
        (_f(row, "gm3_n_min") or 0) >= 1 and (_f(row, "gm3_n_max") or 0) >= 1 and
        _f(row, "gm3_bad_frac") == 0 and _f(row, "gm3_min_missing_frac") == 0 and
        _f(row, "gm3_max_missing_frac") == 0 and _f(row, "gm3_start_bad_frac") == 0 and
        _f(row, "gm2_start_bad_frac") == 0 and _f(row, "gm3_tail_bad_frac") == 0 and
        _f(row, "gm3_global_bad_frac") == 0
    )
    gdsshape_ok = _f(row, "gds_residual_bad_frac") == 0
    return {"gmshape_ok": gmshape_ok, "gdsshape_ok": gdsshape_ok, "bothshape_ok": gmshape_ok and gdsshape_ok}


def load_population(root: Path, folder: str) -> dict[str, dict]:
    """{csv_name: {"fit": {arch_hash: {ids_rmse, combined_gm}}, "shape": {arch_hash: {...}} or None}}"""
    pop = {}
    for csv_subdir in _csv_subdirs(root, folder):
        csv_name = csv_subdir.name
        ranked_dir = csv_subdir / folder / "ranked_region_knee_vgs-3to0_vds0to15"
        base_matches = sorted(ranked_dir.glob("*_ranked_by_region_knee_combined_gm.csv"))
        if not base_matches:
            continue
        base = base_matches[0]
        fit = {}
        for r in _read_rows(base):
            h = r.get("arch_hash")
            if not h:
                continue
            fit[h] = {"ids_rmse": _f(r, "region_knee_ids_rmse"), "combined_gm": _f(r, "region_knee_combined_gm")}

        shape_path = base.with_name(base.stem + "_shape.csv")
        shape = None
        if shape_path.is_file():
            shape = {}
            for r in _read_rows(shape_path):
                h = r.get("arch_hash")
                if not h:
                    continue
                shape[h] = _compliance(r)
        pop[csv_name] = {"fit": fit, "shape": shape}
    return pop


def _layer_activations(arch_str: str) -> set[str] | None:
    """Set of every activation name used anywhere in an architecture string (e.g.
    '[["tanh","tanh"],["swish","mish"]]' -> {"tanh","swish","mish"}), or None if unparseable."""
    try:
        layers = json.loads(arch_str)
    except (json.JSONDecodeError, TypeError):
        return None
    acts = set()
    for layer in layers:
        acts.update(layer)
    return acts


def tanh_only_hashes(root: Path, folder: str) -> set[str]:
    """arch_hash set for every architecture whose every layer uses ONLY 'tanh' (no swish/mish/
    other mixed in). Architecture composition is a property of the arch itself, independent of
    which measurement csv it was trained against, so reading just the FIRST available csv
    subfolder's base ranked csv is sufficient -- every csv subfolder shares the same arch_hash
    -> architecture mapping."""
    for csv_subdir in _csv_subdirs(root, folder):
        ranked_dir = csv_subdir / folder / "ranked_region_knee_vgs-3to0_vds0to15"
        base_matches = sorted(ranked_dir.glob("*_ranked_by_region_knee_combined_gm.csv"))
        if not base_matches:
            continue
        hashes = set()
        for r in _read_rows(base_matches[0]):
            h = r.get("arch_hash")
            arch_str = r.get("architecture", "")
            if not h or not arch_str:
                continue
            acts = _layer_activations(arch_str)
            if acts == {"tanh"}:
                hashes.add(h)
        return hashes
    return set()


def is_tanh_only(root: Path, folder: str, arch_hash: str, cache: dict[tuple[str, str], set[str]]) -> bool:
    """arch_hash membership in tanh_only_hashes(root, folder), cached per (root, folder) since
    callers typically check many arch_hashes against the same (root, folder) in a row (one
    tanh_only_hashes() call/csv-read per (root, folder) instead of one per row)."""
    key = (str(root), folder)
    if key not in cache:
        cache[key] = tanh_only_hashes(root, folder)
    return arch_hash in cache[key]


def filter_population(pop: dict, keep_hashes: set[str]) -> dict:
    """New population dict (same shape as load_population's return), each csv's fit/shape
    sub-dicts restricted to keep_hashes."""
    out = {}
    for csv_name, d in pop.items():
        fit = {h: v for h, v in d["fit"].items() if h in keep_hashes}
        shape = {h: v for h, v in d["shape"].items() if h in keep_hashes} if d["shape"] is not None else None
        out[csv_name] = {"fit": fit, "shape": shape}
    return out


def _full_metric_pools(pop: dict, csv_names: list[str], metric: str) -> dict[str, tuple[dict, int]]:
    """Per-csv {arch_hash: 0-indexed rank} + total row count, over the FULL population in that
    csv's base *_ranked_by_region_knee_combined_gm.csv file (every arch_hash with a valid value
    for `metric`, one of 'ids_rmse'/'combined_gm') -- NO filtering by the other metric. Purely
    descriptive context (e.g. 'this architecture is #3 of 200 on ids_rmse, #47 of 200 on
    combined_gm'), never used to decide which architecture qualifies or wins -- that's still
    governed by whatever gate (combined_gm ceiling, shape compliance, ...) each method uses."""
    pools = {}
    for csv_name in csv_names:
        rows = [(h, v[metric]) for h, v in pop[csv_name]["fit"].items() if v[metric] is not None]
        rows.sort(key=lambda t: t[1])
        rank = {h: i for i, (h, _) in enumerate(rows)}
        pools[csv_name] = (rank, len(rows))
    return pools


def _worst_best_full_rank(full_ids_pools: dict, full_gm_pools: dict, csv_names: list[str], h: str) -> dict:
    """This arch's worst/best ids_rmse rank (against the FULL per-csv population, see
    _full_metric_pools) across csv_names -- same shape of fields search_fit_quality_minmax uses
    (worst_percentile/best_percentile + _rank/_pool_size/_csv), so format_worst_pct/
    format_best_pct work unchanged. Also attaches this arch's combined_gm rank on that SAME
    (worst/best) csv, for context. {} if h isn't present in any of csv_names' ids_rmse files."""
    percentiles, ranks, ns, present_csvs = [], [], [], []
    for csv_name in csv_names:
        rank, n = full_ids_pools[csv_name]
        if h not in rank or n == 0:
            continue
        percentiles.append(rank[h] / n)
        ranks.append(rank[h] + 1)
        ns.append(n)
        present_csvs.append(csv_name)
    if not percentiles:
        return {}
    worst_i = max(range(len(percentiles)), key=lambda i: percentiles[i])
    best_i = min(range(len(percentiles)), key=lambda i: percentiles[i])
    out = {
        "worst_percentile": percentiles[worst_i],
        "worst_percentile_pool_size": ns[worst_i],
        "worst_percentile_rank": ranks[worst_i],
        "worst_percentile_csv": present_csvs[worst_i],
        "best_percentile": percentiles[best_i],
        "best_percentile_pool_size": ns[best_i],
        "best_percentile_rank": ranks[best_i],
        "best_percentile_csv": present_csvs[best_i],
    }
    _attach_combined_gm_rank(out, full_gm_pools, h)
    return out


def _attach_combined_gm_rank(out: dict, full_gm_pools: dict, h: str) -> None:
    """Mutates `out` (already carrying worst_percentile_csv/best_percentile_csv from an
    ids_rmse-based selection) with this arch's combined_gm rank on those SAME two csvs."""
    for prefix, csv_key in (("worst", "worst_percentile_csv"), ("best", "best_percentile_csv")):
        csv_name = out.get(csv_key)
        if csv_name is None:
            continue
        gm_rank, gm_n = full_gm_pools.get(csv_name, ({}, 0))
        if h in gm_rank and gm_n:
            out[f"{prefix}_combined_gm_rank"] = gm_rank[h] + 1
            out[f"{prefix}_combined_gm_pool_size"] = gm_n


def search_fit_quality_minmax(pop: dict, gm_ceiling: float, top: int,
                               csv_names: list[str] | None = None) -> list[dict]:
    """Worst-case ids_rmse percentile across csv_names (default: every csv in pop). Passing a
    2-csv subset gives the same 'goal=pair' restriction the shape searches use, instead of the
    default all-6-csv worst case."""
    csv_names = list(csv_names) if csv_names is not None else list(pop)
    # per-csv gm-capped pool, sorted by ids_rmse ascending -- this is what actually decides
    # qualification (an arch must be in EVERY csv_names' pool to not be disqualified) and the
    # worst/best PERCENTILE value used to rank results (the sort key below). Untouched by the
    # full-file rank change: combined_gm still gates who wins, exactly as before.
    pools = {}
    for csv_name in csv_names:
        d = pop[csv_name]
        rows = [(h, v["ids_rmse"]) for h, v in d["fit"].items()
                if v["combined_gm"] is not None and v["ids_rmse"] is not None
                and v["combined_gm"] <= gm_ceiling]
        rows.sort(key=lambda t: t[1])
        rank = {h: i for i, (h, _) in enumerate(rows)}
        pools[csv_name] = (rank, len(rows))

    # per-csv FULL (unfiltered) pools, purely to report a qualifying arch's ids_rmse AND
    # combined_gm rank against the whole config's population (e.g. "rank 3/200") instead of the
    # much smaller, less intuitive gm-gated pool size -- display-only, never affects who qualifies
    # or which arch ranks #1 above.
    full_ids_pools = _full_metric_pools(pop, csv_names, "ids_rmse")
    full_gm_pools = _full_metric_pools(pop, csv_names, "combined_gm")

    all_hashes = set()
    for csv_name in csv_names:
        all_hashes |= set(pop[csv_name]["fit"])

    results = []
    for h in all_hashes:
        percentiles = []
        pool_sizes = []
        rmses = []
        combined_gms = []
        disqualified = False
        for csv_name in csv_names:
            rank, n = pools[csv_name]
            if h not in rank or n == 0:
                disqualified = True
                break
            percentiles.append(rank[h] / n)
            pool_sizes.append(n)
            rmses.append(pop[csv_name]["fit"][h]["ids_rmse"])
            combined_gms.append(pop[csv_name]["fit"][h]["combined_gm"])
        if disqualified:
            continue
        # index (within csv_names) of the csv that produced the worst/best gm-gated percentile --
        # this decides the arch's rank in the results list below, unchanged.
        worst_i = max(range(len(percentiles)), key=lambda i: percentiles[i])
        best_i = min(range(len(percentiles)), key=lambda i: percentiles[i])
        # this arch's ids_rmse rank against the FULL population of that SAME csv (not the
        # gm-gated pool) -- e.g. "rank 3/200" instead of "rank 3/25".
        worst_full_rank, worst_full_n = full_ids_pools[csv_names[worst_i]]
        best_full_rank, best_full_n = full_ids_pools[csv_names[best_i]]
        result = {
            "arch_hash": h,
            "worst_percentile": percentiles[worst_i],
            "worst_percentile_pool_size": worst_full_n,
            "worst_percentile_rank": worst_full_rank[h] + 1,
            "worst_percentile_csv": csv_names[worst_i],
            "best_percentile": percentiles[best_i],
            "best_percentile_pool_size": best_full_n,
            "best_percentile_rank": best_full_rank[h] + 1,
            "best_percentile_csv": csv_names[best_i],
            "avg_ids_rmse": sum(rmses) / len(rmses),
            "min_ids_rmse": min(rmses),
            "max_ids_rmse": max(rmses),
            "avg_combined_gm": sum(combined_gms) / len(combined_gms),
            "min_combined_gm": min(combined_gms),
            "max_combined_gm": max(combined_gms),
            "per_csv_percentile": dict(zip(csv_names, percentiles)),
            "per_csv_pool_size": dict(zip(csv_names, pool_sizes)),
        }
        _attach_combined_gm_rank(result, full_gm_pools, h)
        results.append(result)
    results.sort(key=lambda r: (r["worst_percentile"], r["avg_ids_rmse"]))
    return results[:top]


def format_worst_pct(r: dict) -> str:
    """'X.X% (ids_rmse rank R/N, combined_gm rank R2/N2)' -- the leading X.X% is whatever
    percentile/criterion actually ranked this arch (gm-gated for A/D; not used for sorting at
    all for B/C, see _worst_best_full_rank). The parenthetical is always this arch's plain
    ids_rmse rank AND combined_gm rank against the FULL per-csv population (N = total
    architectures in that csv's base ranked file, e.g. 200/100/52) -- purely descriptive
    context, not the same computation as the percentage, so the numbers won't necessarily
    correspond to each other."""
    n = r.get("worst_percentile_pool_size")
    rank = r.get("worst_percentile_rank")
    if n is None or rank is None:
        return f"{r['worst_percentile']*100:.1f}%"
    s = f"{r['worst_percentile']*100:.1f}% (ids_rmse rank {rank}/{n}"
    gm_n, gm_rank = r.get("worst_combined_gm_pool_size"), r.get("worst_combined_gm_rank")
    if gm_n is not None and gm_rank is not None:
        s += f", combined_gm rank {gm_rank}/{gm_n}"
    return s + ")"


def format_best_pct(r: dict) -> str:
    n = r.get("best_percentile_pool_size")
    rank = r.get("best_percentile_rank")
    if n is None or rank is None:
        return f"{r['best_percentile']*100:.1f}%"
    s = f"{r['best_percentile']*100:.1f}% (ids_rmse rank {rank}/{n}"
    gm_n, gm_rank = r.get("best_combined_gm_pool_size"), r.get("best_combined_gm_rank")
    if gm_n is not None and gm_rank is not None:
        s += f", combined_gm rank {gm_rank}/{gm_n}"
    return s + ")"


def format_ids_rmse(r: dict) -> str:
    """'avg=X (range lo-hi)' if per-csv min/max is available, else just 'avg=X'."""
    if r.get("min_ids_rmse") is None:
        return f"avg={r['avg_ids_rmse']:.5f}"
    return f"avg={r['avg_ids_rmse']:.5f} (range {r['min_ids_rmse']:.5f}-{r['max_ids_rmse']:.5f})"


def format_combined_gm(r: dict) -> str:
    if r.get("avg_combined_gm") is None:
        return "n/a"
    if r.get("min_combined_gm") is None:
        return f"avg={r['avg_combined_gm']:.3f}"
    return f"avg={r['avg_combined_gm']:.3f} (range {r['min_combined_gm']:.3f}-{r['max_combined_gm']:.3f})"


def search_shape_all_csvs(pop: dict, metric: str, top: int) -> list[dict]:
    csv_names = [c for c, d in pop.items() if d["shape"] is not None]
    full_ids_pools = _full_metric_pools(pop, csv_names, "ids_rmse")
    full_gm_pools = _full_metric_pools(pop, csv_names, "combined_gm")
    all_hashes = set()
    for c in csv_names:
        all_hashes |= set(pop[c]["shape"])

    results = []
    for h in all_hashes:
        count = sum(1 for c in csv_names if h in pop[c]["shape"] and pop[c]["shape"][h][metric])
        if count == 0:
            continue
        rmses = [pop[c]["fit"][h]["ids_rmse"] for c in csv_names
                  if h in pop[c]["fit"] and pop[c]["fit"][h]["ids_rmse"] is not None]
        gms = [pop[c]["fit"][h]["combined_gm"] for c in csv_names
               if h in pop[c]["fit"] and pop[c]["fit"][h]["combined_gm"] is not None]
        avg_rmse = sum(rmses) / len(rmses) if rmses else float("inf")
        results.append({
            "arch_hash": h, "count": count, "of": len(csv_names),
            "avg_ids_rmse": avg_rmse,
            "min_ids_rmse": min(rmses) if rmses else None,
            "max_ids_rmse": max(rmses) if rmses else None,
            "avg_combined_gm": (sum(gms) / len(gms)) if gms else None,
            "min_combined_gm": min(gms) if gms else None,
            "max_combined_gm": max(gms) if gms else None,
            **_worst_best_full_rank(full_ids_pools, full_gm_pools, csv_names, h),
        })
    results.sort(key=lambda r: (-r["count"], r["avg_ids_rmse"]))
    return results[:top]


def search_shape_pair(pop: dict, metric: str, pair: tuple[str, str], top: int) -> list[dict]:
    a, b = pair
    if a not in pop or b not in pop or pop[a]["shape"] is None or pop[b]["shape"] is None:
        return []
    hashes_a = {h for h, c in pop[a]["shape"].items() if c[metric]}
    hashes_b = {h for h, c in pop[b]["shape"].items() if c[metric]}
    common = hashes_a & hashes_b
    full_ids_pools = _full_metric_pools(pop, list(pair), "ids_rmse")
    full_gm_pools = _full_metric_pools(pop, list(pair), "combined_gm")
    results = []
    for h in common:
        rmses = [pop[c]["fit"][h]["ids_rmse"] for c in pair
                  if h in pop[c]["fit"] and pop[c]["fit"][h]["ids_rmse"] is not None]
        gms = [pop[c]["fit"][h]["combined_gm"] for c in pair
               if h in pop[c]["fit"] and pop[c]["fit"][h]["combined_gm"] is not None]
        avg_rmse = sum(rmses) / len(rmses) if rmses else float("inf")
        results.append({
            "arch_hash": h,
            "avg_ids_rmse": avg_rmse,
            "min_ids_rmse": min(rmses) if rmses else None,
            "max_ids_rmse": max(rmses) if rmses else None,
            "avg_combined_gm": (sum(gms) / len(gms)) if gms else None,
            "min_combined_gm": min(gms) if gms else None,
            "max_combined_gm": max(gms) if gms else None,
            **_worst_best_full_rank(full_ids_pools, full_gm_pools, list(pair), h),
        })
    results.sort(key=lambda r: r["avg_ids_rmse"])
    return results[:top]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--folder", required=True)
    ap.add_argument("--pair_csvs", default=",".join(DEFAULT_PAIR),
                    help="comma-separated pair of csv_names for the goal=pair searches")
    ap.add_argument("--gm_ceiling", type=float, default=None,
                    help="default: 1.4 if folder name contains '_nogm', else 1.1")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--json", default=None, help="also write full results to this json path")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    gm_ceiling = args.gm_ceiling
    if gm_ceiling is None:
        gm_ceiling = 1.4 if "_nogm" in args.folder else 1.1
    pair = tuple(c.strip() for c in args.pair_csvs.split(","))

    pop = load_population(root, args.folder)
    n_shape = sum(1 for d in pop.values() if d["shape"] is not None)
    print(f"=== {args.folder} ({root.name}) === {len(pop)} csvs, {n_shape} with full-pop shape data, "
          f"gm_ceiling={gm_ceiling}, pair={pair}\n")

    out = {}

    fq = search_fit_quality_minmax(pop, gm_ceiling, args.top)
    used_ceiling = gm_ceiling
    while not fq and used_ceiling < 5.0:
        used_ceiling += 0.4
        fq = search_fit_quality_minmax(pop, used_ceiling, args.top)
    if used_ceiling != gm_ceiling:
        print(f"(no arch present in every csv's combined_gm<={gm_ceiling} pool -- "
              f"auto-escalated gm_ceiling to {used_ceiling:.1f} for search A)\n")
    out["fit_quality_minmax"] = fq
    out["fit_quality_gm_ceiling_used"] = used_ceiling
    print(f"-- A) fit-quality min-max (worst-case percentile across all {len(pop)} csvs, "
          f"gm_ceiling={used_ceiling:.1f}) --")
    for r in fq:
        print(f"  {r['arch_hash']}  worst_pct={format_worst_pct(r)}  "
              f"best_pct={format_best_pct(r)}  ids_rmse: {format_ids_rmse(r)}  "
              f"combined_gm: {format_combined_gm(r)}")
    print()

    for metric in ("gmshape_ok", "bothshape_ok"):
        res = search_shape_all_csvs(pop, metric, args.top)
        out[f"{metric}_all_csvs"] = res
        print(f"-- B) {metric} goal=all_csvs (of {n_shape} csvs w/ shape data) --")
        for r in res:
            print(f"  {r['arch_hash']}  {r['count']}/{r['of']}  ids_rmse: {format_ids_rmse(r)}  "
                  f"combined_gm: {format_combined_gm(r)}  worst_pct={format_worst_pct(r)}  "
                  f"best_pct={format_best_pct(r)}")
        print()

    for metric in ("gmshape_ok", "bothshape_ok"):
        res = search_shape_pair(pop, metric, pair, args.top)
        out[f"{metric}_pair"] = res
        print(f"-- C) {metric} goal=pair ({pair[0]} AND {pair[1]}) --")
        if not res:
            print("  (none -- either no shape data for this pair, or zero archs pass on both)")
        for r in res:
            print(f"  {r['arch_hash']}  ids_rmse: {format_ids_rmse(r)}  combined_gm: {format_combined_gm(r)}  "
                  f"worst_pct={format_worst_pct(r)}  best_pct={format_best_pct(r)}")
        print()

    fq_pair = search_fit_quality_minmax(pop, gm_ceiling, args.top, csv_names=list(pair))
    used_ceiling_pair = gm_ceiling
    while not fq_pair and used_ceiling_pair < 5.0:
        used_ceiling_pair += 0.4
        fq_pair = search_fit_quality_minmax(pop, used_ceiling_pair, args.top, csv_names=list(pair))
    if used_ceiling_pair != gm_ceiling:
        print(f"(no arch present in both pair csvs' combined_gm<={gm_ceiling} pool -- "
              f"auto-escalated gm_ceiling to {used_ceiling_pair:.1f} for search D)\n")
    out["fit_quality_minmax_pair"] = fq_pair
    out["fit_quality_gm_ceiling_used_pair"] = used_ceiling_pair
    print(f"-- D) fit-quality min-max, goal=pair ({pair[0]} AND {pair[1]}, "
          f"gm_ceiling={used_ceiling_pair:.1f}) --")
    if not fq_pair:
        print("  (none -- no arch present in both pair csvs' gm-capped pool even after escalating)")
    for r in fq_pair:
        print(f"  {r['arch_hash']}  worst_pct={format_worst_pct(r)}  "
              f"best_pct={format_best_pct(r)}  ids_rmse: {format_ids_rmse(r)}  "
              f"combined_gm: {format_combined_gm(r)}")
    print()

    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
