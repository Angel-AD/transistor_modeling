#!/usr/bin/env python3
"""
gen_config_from_rows.py  —  Generate a multi_experiment_runner config from CSV rows.

Reads a compiled/ranked CSV (from compile_results_csv.py or compile_ranked.py),
selects rows by id, loads their JSON files for the full hyperparameter set, groups
by unique non-architecture params, and writes a ready-to-run config JSON.

Usage
-----
    python gen_config_from_rows.py \\
        --csv refine3/refine3_ranked_by_ids_rmse.csv \\
        --ids 106-140 \\
        --set gm_vds_mins=[1,2,3,4] \\
        --out physics_nn_configs/opt_configs_refine5_vdsmin.json \\
        --prefix refine5_vdsmin \\
        --max_workers 42

--ids
    106          single row (by 'id' column value)
    106-140      inclusive range
    106,108,112  explicit list
    all          every row in the CSV

--set KEY=VALUE  (repeatable, JSON-parsed)
    Override or add any config field.  Value is JSON-parsed, so lists, numbers,
    bools, and strings all work.  Applied after the per-row params are extracted,
    so these WIN over the source values.

    Examples:
        --set gm_vds_mins=[1,2,3,4]
        --set epochs=1000
        --set ids_constraint=false
        --set seeds=[27,42]

Output
------
Rows are grouped by their unique non-architecture hyperparameter tuple (read from
the run JSON, so params absent from the CSV are still captured).  Each group
becomes one EXPERIMENT with nn_architectures = all unique archs in that group.
base_config is auto-detected from equation_type and use_gm.
"""

import argparse
import json
import csv
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Hyperparameter keys extracted from each run's JSON.
# architecture is handled separately (collected per group).
# ---------------------------------------------------------------------------

# Keys that become single-element *lists* in the config (swept dimensions).
# format: json_key -> config_key
SWEPT_MAP = {
    "gm1_weight":       "gm1_weights",
    "gm2_weight":       "gm2_weights",
    "gm3_weight":       "gm3_weights",
    "gm_surgery_mode":  "gm_surgery_modes",
    "lr":               "learning_rates",
    "output_activation":"output_activations",
    "knee_alpha_scale": "knee_alpha_scales",
    "knee_combiner":    "knee_combiners",
    "seed":             "seeds",
    "gm_max_ratio":     "gm_max_ratios",
    "gm_vds_min":       "gm_vds_mins",
    "gm_vgs_min":       "gm_vgs_mins",
}

# Keys that are scalars in the config (not swept).
# format: json_key -> config_key
SCALAR_MAP = {
    "epochs":           "epochs",
    "lbfgs_epochs":     "lbfgs_epochs",
    "lbfgs_max_iter":   "lbfgs_max_iter",
    "lbfgs_gm_aware":   "lbfgs_gm_aware",
    "loss_norm":        "loss_norm",
    "deterministic":    "deterministic",
    "mixed_init":       "mixed_init",
    "gm_warmup_epochs": "gm_warmup_epochs",
    "gm_warmup_lr":     "gm_warmup_lr",
    "ids_constraint":   "ids_constraint",
    "ids_target":       "ids_target",
    "ids_lambda":       "ids_lambda",
}

# Fields used to form the group key (must exclude architecture).
# equation_type/use_gm/use_opt_params are included so mixed pure-NN + physics-NN
# CSVs never collapse into the same group with the wrong base_config.
GROUP_KEY_FIELDS = (
    list(SWEPT_MAP.keys())
    + list(SCALAR_MAP.keys())
    + ["equation_type", "use_gm", "use_opt_params"]
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_bool(v):
    if isinstance(v, bool):
        return v
    if v in (None, "", "None"):
        return False
    return str(v).strip().lower() in ("true", "1", "yes")


def _coerce(v):
    """Try int → float → bool → str for a JSON/CSV string value."""
    if v is None or v == "" or v == "None":
        return None
    if isinstance(v, (int, float, bool)):
        return v
    s = str(v).strip()
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    try:
        i = int(s)
        return i
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return s


def parse_set_value(v: str):
    """Parse a --set value. Prefer strict JSON; fall back to a lenient parse for
    bracketed lists whose quotes were stripped or mangled by the shell — a common
    Windows PowerShell issue where ["a","b"] arrives as [a,b] or [\\a\\,\\b\\].
    Each element is de-quoted/de-slashed and coerced (int/float/bool/str)."""
    v = v.strip()
    try:
        return json.loads(v)
    except json.JSONDecodeError:
        pass
    if v.startswith("[") and v.endswith("]"):
        inner = v[1:-1].strip()
        if not inner:
            return []
        junk = " \t\"'\\"          # strip stray quotes/backslashes/space per element
        return [_coerce(p.strip(junk)) for p in inner.split(",")]
    return _coerce(v)


def load_csv(path: Path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _reroot_path(path: str):
    """Compiled CSVs store absolute file_paths that may point at an OLD project
    root (e.g. ...\\new_optimizations_claude\\physics_nn_pipeline\\... before the
    tree was moved up a level). Find the longest tail of `path` that exists under
    the current ROOT, so the run JSON is still located after a relocation."""
    parts = Path(path).parts
    # Try several base roots so paths resolve whether the data sits alongside this
    # script (ROOT) or one level up (ROOT.parent) — e.g. the scripts were moved into
    # a physics_nn_pipeline/ subfolder while base_arch/ stayed at the repo root.
    for base in (ROOT, ROOT.parent):
        for i in range(len(parts)):
            cand = base.joinpath(*parts[i:])
            if cand.exists():
                return cand
    return None


def load_json(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        alt = _reroot_path(path)
        if alt is not None:
            p = alt
    if not p.exists():
        return {}
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"  [warn] could not read {path}: {e}", file=sys.stderr)
        return {}


def parse_ids(spec: str, rows: list) -> set:
    """Return the set of id strings matching the spec."""
    if spec.strip().lower() == "all":
        return {r["id"] for r in rows}
    ids = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            ids.update(str(i) for i in range(int(a), int(b) + 1))
        else:
            ids.add(part)
    return ids


def _values_match(a, b) -> bool:
    """Compare two coerced values: numerically when both are numbers, else
    case-insensitive string compare."""
    ca, cb = _coerce(a), _coerce(b)
    if isinstance(ca, (int, float)) and not isinstance(ca, bool) \
       and isinstance(cb, (int, float)) and not isinstance(cb, bool):
        return float(ca) == float(cb)
    return str(ca).strip().lower() == str(cb).strip().lower()


def apply_filters(rows: list, filters: list) -> list:
    """Keep rows matching each filter COL<op>VALUE, op in {=, <, >, <=, >=}.
    '=' is numeric-or-string equality; the inequalities are numeric (rows whose
    column is missing/non-numeric are dropped). E.g. region_knee_ids_rmse<0.01."""
    for f in filters:
        op = next((o for o in ("<=", ">=", "<", ">", "=") if o in f), None)
        if op is None:
            sys.exit(f"[error] --filter must be COL<op>VALUE (op in =,<,>,<=,>=), got: {f!r}")
        col, val = (s.strip() for s in f.split(op, 1))
        if rows and col not in rows[0]:
            sys.exit(f"[error] --filter column {col!r} not in CSV. "
                     f"Available: {list(rows[0].keys())}")
        before = len(rows)
        if op == "=":
            rows = [r for r in rows if _values_match(r.get(col, ""), val)]
        else:
            try:
                thr = float(val)
            except ValueError:
                sys.exit(f"[error] --filter {f!r}: {val!r} not numeric for operator {op!r}.")
            cmp = {"<": lambda x: x < thr, ">": lambda x: x > thr,
                   "<=": lambda x: x <= thr, ">=": lambda x: x >= thr}[op]
            def _num(r):
                try:
                    return float(r.get(col, ""))
                except (TypeError, ValueError):
                    return None
            rows = [r for r in rows if _num(r) is not None and cmp(_num(r))]
        print(f"  --filter {col}{op}{val}: kept {len(rows)}/{before} rows")
    return rows


def params_from_json(jdata: dict) -> dict:
    """Extract all relevant hyperparameters from a run JSON."""
    p = {}

    # gm weights: prefer direct JSON fields; fall back to run_info if nested
    for w in ("gm1_weight", "gm2_weight", "gm3_weight"):
        v = jdata.get(w)
        if v is None:
            v = jdata.get("run_info", {}).get(w)
        p[w] = _coerce(v) if v is not None else 0.0

    p["gm_surgery_mode"] = jdata.get("gm_surgery_mode", "none") or "none"

    lr = jdata.get("lr") or jdata.get("learning_rate")
    p["lr"] = _coerce(lr) if lr is not None else 0.01

    p["output_activation"] = jdata.get("output_activation", "linear") or "linear"
    p["knee_alpha_scale"]  = _coerce(jdata.get("knee_alpha_scale", 1.0))
    p["knee_combiner"]     = jdata.get("knee_combiner", "sum") or "sum"
    p["seed"]              = _coerce(jdata.get("seed", 27))
    p["gm_max_ratio"]      = _coerce(jdata.get("gm_max_ratio", 1.0))

    p["epochs"]          = _coerce(jdata.get("epochs", 800))
    p["lbfgs_epochs"]    = _coerce(jdata.get("lbfgs_epochs", 20))
    p["lbfgs_max_iter"]  = _coerce(jdata.get("lbfgs_max_iter", 100))
    p["lbfgs_gm_aware"]  = _parse_bool(jdata.get("lbfgs_gm_aware", True))
    p["loss_norm"]        = jdata.get("loss_norm", "nmse") or "nmse"
    p["deterministic"]    = _parse_bool(jdata.get("deterministic", True))
    p["mixed_init"]       = jdata.get("mixed_init", "per_activation")

    p["gm_warmup_epochs"] = _coerce(jdata.get("gm_warmup_epochs", 0)) or 0
    p["gm_warmup_lr"]     = _coerce(jdata.get("gm_warmup_lr"))
    p["ids_constraint"]   = _parse_bool(jdata.get("ids_constraint", False))
    p["ids_target"]       = _coerce(jdata.get("ids_target"))
    p["ids_lambda"]       = _coerce(jdata.get("ids_lambda", 100.0))

    p["gm_vds_min"]    = _coerce(jdata.get("gm_vds_min", None))
    p["gm_vgs_min"]    = _coerce(jdata.get("gm_vgs_min", None))

    p["equation_type"] = jdata.get("equation_type", "pure") or "pure"
    p["use_gm"]        = _parse_bool(jdata.get("use_gm", True))
    p["use_opt_params"]= _parse_bool(jdata.get("use_opt_params", False))

    # Always stringify: some JSONs store architecture as a list [64,64,64]; str() normalises
    # it to "[64, 64, 64]" which is hashable and matches the CSV representation.
    p["architecture"] = str(jdata.get("architecture", ""))

    return p


def group_key_of(params: dict, exclude=frozenset()) -> tuple:
    """Stable hashable key from all non-architecture hyperparameters.

    `exclude` is a set of json_keys to drop from the key — used when a dimension
    is being overridden/swept uniformly (e.g. gm_vds_min set via --set, or gm
    weights fanned out via --gm_weight_tuples), so rows are not split on it."""
    return tuple(str(params.get(k, "")) for k in GROUP_KEY_FIELDS
                 if k != "architecture" and k not in exclude)


# Override keys that select/configure the base_config (not swept dims). Setting
# these via --set must reach detect_base_config, e.g. --set use_gm=true to turn on
# gm training for rows whose source run was pure-Ids.
BASE_CONFIG_OVERRIDE_KEYS = {"use_gm", "equation_type", "use_opt_params"}


def swept_keys_overridden(overrides: dict) -> set:
    """json_keys whose swept config_key is given a list value in --set. These are
    dropped from the group key (rows don't split on them) and the override list is
    used directly as the swept dimension — i.e. --set REPLACES the source value."""
    swept_rev = {v: k for k, v in SWEPT_MAP.items()}  # cfg_key -> json_key
    return {swept_rev[ck] for ck, v in overrides.items()
            if ck in swept_rev and isinstance(v, list)}


def parse_weight_tuples(spec: str) -> list:
    """Parse '0.1,0,0 ; 0.1,0.1,0 ; 0.1,0.1,0.1' into [(0.1,0.0,0.0), ...].
    Each tuple becomes its own experiment (gm1/gm2/gm3 fixed), so the curated set
    is used verbatim — NOT the Cartesian product of independent weight lists."""
    tuples = []
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        try:
            nums = [float(x) for x in part.split(",")]
        except ValueError:
            sys.exit(f"[error] --gm_weight_tuples entry not numeric: {part!r}")
        if len(nums) != 3:
            sys.exit(f"[error] --gm_weight_tuples entry must be g1,g2,g3 (3 values): {part!r}")
        tuples.append(tuple(nums))
    if not tuples:
        sys.exit("[error] --gm_weight_tuples parsed to empty.")
    return tuples


def _eq_label(eq: str) -> str:
    """Short label from equation_type for use in experiment names.
    E.g. 'pure:vdsgate_vdsk_quad' -> 'vdsk_quad', 'pure:softplus' -> 'softplus'."""
    if not eq:
        return ""
    base = eq.split(":")[-1]           # drop 'pure:' / 'product:' prefix
    base = base.replace("vdsgate_", "")  # vdsgate_vdsk_quad -> vdsk_quad
    return base


def make_exp_name(prefix: str, params: dict, eq_label: str = "") -> str:
    """Build an experiment name from the group's key characteristics."""
    def fmt_w(w):
        try:
            v = float(w)
            return str(int(v)) if v == int(v) else str(v).replace(".", "p")
        except Exception:
            return str(w)

    w1 = fmt_w(params.get("gm1_weight", 0))
    w2 = fmt_w(params.get("gm2_weight", 0))
    w3 = fmt_w(params.get("gm3_weight", 0))
    mode = str(params.get("gm_surgery_mode", "none"))
    m_tag = mode.replace("element-wise-", "ew").replace("-", "")
    wu = int(params.get("gm_warmup_epochs", 0) or 0)
    cst = params.get("ids_constraint", False)

    parts = [prefix]
    if eq_label:
        parts.append(eq_label)
    parts.extend([f"w{w1}-{w2}-{w3}", m_tag])
    if wu:
        parts.append(f"wu{wu}")
    if cst:
        parts.append("cst")
    return "__".join(parts)


def detect_base_config(params: dict) -> dict:
    eq = params.get("equation_type", "pure")
    use_gm = params.get("use_gm", True)
    use_opt = params.get("use_opt_params", False)
    label = "PureNN" if eq == "pure" else "PhysNN_" + eq.replace(":", "-")  # ':' is illegal in Windows filenames
    return {label: {"use_opt_params": use_opt, "use_gm": use_gm, "equation_type": eq}}


def build_experiment(group_params_list: list, overrides: dict,
                     base_overrides: dict = None, weight_tuple=None) -> dict:
    """Build one experiment config dict from a list of (arch, params) tuples.

    base_overrides : use_gm/equation_type/use_opt_params overrides that must reach
                     detect_base_config (so --set use_gm=true actually enables gm).
    weight_tuple   : optional (g1,g2,g3) that fixes this experiment's gm weights,
                     overriding the source rows' weights."""
    base_overrides = base_overrides or {}
    archs = list(dict.fromkeys(p["architecture"] for p in group_params_list))
    rep = dict(group_params_list[0])
    if weight_tuple is not None:
        rep["gm1_weight"], rep["gm2_weight"], rep["gm3_weight"] = weight_tuple

    exp = {"nn_architectures": archs}

    # Swept dimensions → single-element lists
    for json_key, cfg_key in SWEPT_MAP.items():
        v = rep.get(json_key)
        if v is not None:
            exp[cfg_key] = [v]

    # Scalar dimensions → plain values (omit None / zero-defaults that are implicit)
    for json_key, cfg_key in SCALAR_MAP.items():
        v = rep.get(json_key)
        if v is None:
            continue
        # Omit zero-value warmup (no warmup = default)
        if json_key == "gm_warmup_epochs" and v == 0:
            continue
        # Omit ids fields when constraint is off
        if json_key in ("ids_target", "ids_lambda") and not rep.get("ids_constraint"):
            continue
        exp[cfg_key] = v

    # base_config picks up use_gm/equation_type overrides so gm can be enabled fresh
    rep_for_base = dict(rep)
    rep_for_base.update(base_overrides)
    exp["base_configs"] = detect_base_config(rep_for_base)

    # Swept/scalar overrides win over everything
    exp.update(overrides)

    return exp


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Generate a multi_experiment_runner config from ranked CSV rows.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--csv", required=True,
                    help="Path to compiled/ranked CSV (absolute or relative to physics_nn_pipeline/).")
    ap.add_argument("--ids", default=None,
                    help="Row id(s): single (106), range (106-140), list (106,108), or 'all'. "
                         "Matches the 'id' column (overall rank). Mutually exclusive with --top.")
    ap.add_argument("--top", type=int, default=None,
                    help="Select the first N rows after filtering (CSV is ranked best-first). "
                         "Use with --filter to get e.g. the N best softplus rows.")
    ap.add_argument("--filter", dest="filters", action="append", default=[],
                    metavar="COL=VALUE",
                    help="Keep only rows where column COL == VALUE (applied BEFORE --ids/--top). "
                         "Repeatable. Numeric or string compare. "
                         "E.g. --filter output_activation=softplus")
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    metavar="KEY=VALUE",
                    help="Override/add a config key (JSON-parsed value).  Repeatable. "
                         "A list value for a swept dim (e.g. gm_vds_mins=[2.6,3.0]) REPLACES "
                         "the source value and is swept by the runner. use_gm/equation_type "
                         "reach the base_config (e.g. --set use_gm=true enables gm on pure-Ids rows).")
    ap.add_argument("--gm_weight_tuples", default=None,
                    metavar="'g1,g2,g3; g1,g2,g3; ...'",
                    help="Curated gm-weight combinations, one experiment each (NOT a cross-product). "
                         "E.g. '0.1,0,0; 0.1,0.1,0; 0.1,0.1,0.1; 0.3,0,0'. Implies use_gm=true.")
    ap.add_argument("--out", required=True,
                    help="Output JSON config path (absolute or relative to physics_nn_pipeline/).")
    ap.add_argument("--prefix", default="exp",
                    help="Experiment name prefix (default: 'exp').")
    ap.add_argument("--max_workers", type=int, default=42,
                    help="Pool size in DEFAULTS.max_workers (default: 42).")
    ap.add_argument("--has_gm", action="store_true",
                    help="Skip rows where all gm weights (gm1/gm2/gm3) are 0.")
    args = ap.parse_args()

    # Parse --set overrides
    overrides = {}
    for item in args.overrides:
        if "=" not in item:
            sys.exit(f"[error] --set must be KEY=VALUE, got: {item!r}")
        k, v = item.split("=", 1)
        overrides[k.strip()] = parse_set_value(v)

    # Safety net: a swept-dimension override must be a list — otherwise the runner
    # iterates a bare string character-by-character. Wrap stray scalars.
    _swept_cfg_keys = set(SWEPT_MAP.values())
    for ck, val in list(overrides.items()):
        if ck in _swept_cfg_keys and not isinstance(val, list):
            overrides[ck] = [val]
            print(f"  [note] --set {ck}: wrapped scalar {val!r} into a 1-element list")

    # Resolve paths
    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = ROOT / csv_path
    if not csv_path.exists():
        sys.exit(f"[error] CSV not found: {csv_path}")

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path

    # Validate selection mode
    if (args.ids is None) == (args.top is None):
        sys.exit("[error] specify exactly one of --ids or --top.")

    # Load CSV
    rows = load_csv(csv_path)
    print(f"Loaded {len(rows)} rows from {csv_path.name}")

    # Column filters (e.g. output_activation=softplus) applied BEFORE id/top selection
    if args.filters:
        rows = apply_filters(rows, args.filters)
        if not rows:
            sys.exit("[error] --filter removed all rows; nothing to select.")

    # Select rows: by id (overall rank) or by top-N of the filtered, ranked list
    if args.top is not None:
        if args.top <= 0:
            sys.exit("[error] --top must be a positive integer.")
        selected = rows[:args.top]
        print(f"Selected {len(selected)} rows (--top {args.top}"
              f"{' after --filter' if args.filters else ''})")
    else:
        wanted = parse_ids(args.ids, rows)
        selected = [r for r in rows if r.get("id", "") in wanted]
        if not selected:
            id_vals = [r.get("id", "?") for r in rows[:3]]
            sys.exit(
                f"[error] No rows matched --ids {args.ids!r}.  "
                f"id column sample: {id_vals} ... (total {len(rows)} rows)"
            )
        print(f"Selected {len(selected)} rows (--ids {args.ids})")

    # Optional: filter out rows with no gm (all weights == 0)
    if args.has_gm:
        before = len(selected)
        def _has_gm(r):
            try:
                return not (float(r.get("gm1_weight", 0) or 0) == 0
                            and float(r.get("gm2_weight", 0) or 0) == 0
                            and float(r.get("gm3_weight", 0) or 0) == 0)
            except Exception:
                return False
        selected = [r for r in selected if _has_gm(r)]
        print(f"  --has_gm: dropped {before - len(selected)} zero-gm rows -> {len(selected)} remaining")
        if not selected:
            sys.exit("[error] --has_gm filtered out all rows; no experiments to generate.")

    # Load JSON for each row and extract full params
    all_params = []
    missing_json = 0
    for row in selected:
        fp = row.get("file_path", "")
        jdata = load_json(fp)
        if not jdata:
            missing_json += 1
        params = params_from_json(jdata)
        # Architecture safety net: if the JSON was unreadable, params["architecture"] is empty,
        # which would collapse every such row into one group. Fall back to the CSV column, which
        # stores the same normalised string form.
        if not params.get("architecture") and row.get("architecture"):
            params["architecture"] = str(row["architecture"])
        # CSV gm weights come from the run-dir name (reliable); prefer them over JSON whenever
        # the CSV has a value, including 0.0 (e.g. gm3_weight=0 in a w1-1-0 run).
        for w_col in ("gm1_weight", "gm2_weight", "gm3_weight"):
            csv_val = _coerce(row.get(w_col, ""))
            if csv_val is not None:
                params[w_col] = csv_val
        # If the CSV has a gm_vds_min column, use it — each row gets its own exact value,
        # which becomes the group key so no cross-row sweep is created.
        if "gm_vds_min" in row:
            csv_vds = _coerce(row.get("gm_vds_min", ""))
            if csv_vds is not None:
                params["gm_vds_min"] = csv_vds
        if "gm_vgs_min" in row:
            csv_vgs = _coerce(row.get("gm_vgs_min", ""))
            if csv_vgs is not None:
                params["gm_vgs_min"] = csv_vgs
        all_params.append(params)

    if missing_json:
        print(f"  [warn] {missing_json} JSON files could not be read; using CSV values for those rows.")

    # Split overrides: base_config selectors (use_gm/equation_type/...) vs the rest.
    base_overrides = {k: v for k, v in overrides.items() if k in BASE_CONFIG_OVERRIDE_KEYS}
    other_overrides = {k: v for k, v in overrides.items() if k not in BASE_CONFIG_OVERRIDE_KEYS}

    # equation_types fan-out: --set equation_types=[X,Y] or --set equation_type=X creates one
    # experiment (with its own base_config) per equation_type. Each type is combined with every
    # group × weight_tuple, and gets a short label in the experiment name to disambiguate.
    eq_overrides = None
    if "equation_types" in other_overrides and isinstance(other_overrides["equation_types"], list):
        eq_overrides = other_overrides.pop("equation_types")
    elif "equation_type" in base_overrides and isinstance(base_overrides["equation_type"], list):
        eq_overrides = base_overrides.pop("equation_type")
    elif "equation_type" in base_overrides:
        eq_overrides = [base_overrides.pop("equation_type")]

    # Curated gm-weight tuples (one experiment each). Imply use_gm.
    weight_tuples = parse_weight_tuples(args.gm_weight_tuples) if args.gm_weight_tuples else None
    if weight_tuples:
        base_overrides.setdefault("use_gm", True)

    # Dimensions that are overridden uniformly are dropped from the group key so
    # rows are not split on them (the override list / tuple drives the sweep instead).
    exclude = swept_keys_overridden(other_overrides)
    if weight_tuples:
        exclude |= {"gm1_weight", "gm2_weight", "gm3_weight"}
    if eq_overrides:
        exclude |= {"equation_type"}   # source eq_type irrelevant; we fan out explicitly

    # Group by non-arch hyperparameter tuple (minus excluded dims)
    groups: dict[tuple, list] = defaultdict(list)
    for params in all_params:
        groups[group_key_of(params, exclude)].append(params)

    n_eq = len(eq_overrides) if eq_overrides else 1
    print(f"\n{len(groups)} group(s)"
          + (f", x {n_eq} equation_types" if eq_overrides else "")
          + (f", x {len(weight_tuples)} weight tuples" if weight_tuples else "") + ":")

    # Resolve duplicate experiment names by appending a counter
    name_counts: dict[str, int] = defaultdict(int)
    EXPERIMENTS = {}
    total_runs = 0
    skipped_zero_gm = 0

    def _emit(group, weight_tuple, cur_base_overrides):
        nonlocal total_runs, skipped_zero_gm
        rep = dict(group[0])
        if weight_tuple is not None:
            rep["gm1_weight"], rep["gm2_weight"], rep["gm3_weight"] = weight_tuple

        # Skip all-zero gm when use_gm is True — inert configs that waste compute.
        eff_use_gm = cur_base_overrides.get("use_gm", rep.get("use_gm", False))
        if eff_use_gm:
            all_zero = all(float(rep.get(w, 0) or 0) == 0
                           for w in ("gm1_weight", "gm2_weight", "gm3_weight"))
            if all_zero:
                skipped_zero_gm += 1
                return

        eq_label = _eq_label(cur_base_overrides.get("equation_type", rep.get("equation_type", "")))
        base_name = make_exp_name(args.prefix, rep, eq_label)
        name_counts[base_name] += 1
        exp_name = base_name if name_counts[base_name] == 1 else f"{base_name}_{name_counts[base_name]}"

        exp = build_experiment(group, other_overrides, cur_base_overrides, weight_tuple)

        n_archs = len(exp["nn_architectures"])
        # Count runs: archs × product of all list-valued swept dims (dedup keys)
        n_sweep = 1
        for key in set(list(SWEPT_MAP.values()) + list(other_overrides.keys())):
            v = exp.get(key)
            if isinstance(v, list):
                n_sweep *= len(v)
        n_runs = n_archs * n_sweep
        total_runs += n_runs

        ug = list(exp["base_configs"].values())[0].get("use_gm")
        print(f"  {exp_name}: {n_archs} archs x {n_sweep} = {n_runs} runs  [use_gm={ug}]")
        EXPERIMENTS[exp_name] = exp

    for gk, group in groups.items():
        for eq in (eq_overrides or [None]):
            cur_bov = dict(base_overrides)
            if eq is not None:
                cur_bov["equation_type"] = eq
            if weight_tuples:
                for wt in weight_tuples:
                    _emit(group, wt, cur_bov)
            else:
                _emit(group, None, cur_bov)

    if skipped_zero_gm:
        print(f"  [skip] {skipped_zero_gm} all-zero-gm experiment(s) suppressed (use_gm=True)")

    DEFAULTS = {
        "knee_alpha_scales":  [1.0],
        "knee_combiners":     ["sum"],
        "learning_rates":     [0.01],
        "output_activations": ["linear"],
        "gm1_weights":        [0.0],
        "gm2_weights":        [0.0],
        "gm3_weights":        [0.0],
        "gm_surgery_modes":   ["none"],
        "gm_max_ratios":      [1.0],
        "gm_vds_mins":        [0],
        "gm_vgs_mins":        [None],
        "epochs":             800,
        "lbfgs_epochs":       20,
        "lbfgs_max_iter":     100,
        "lbfgs_gm_aware":     True,
        "loss_norm":          "nmse",
        "seeds":              [27],
        "deterministic":      True,
        "mixed_init":         "per_activation",
        "max_workers":        args.max_workers,
    }

    cfg = {
        "_comment": (
            f"Generated by gen_config_from_rows.py  |  "
            f"source: {csv_path.name}  |  ids: {args.ids}  |  overrides: {overrides}"
        ),
        "DEFAULTS": DEFAULTS,
        "EXPERIMENTS": EXPERIMENTS,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    print(f"\nexperiments : {len(EXPERIMENTS)}")
    print(f"total runs  : {total_runs}")
    print(f"wrote       : {out_path}")


if __name__ == "__main__":
    main()
