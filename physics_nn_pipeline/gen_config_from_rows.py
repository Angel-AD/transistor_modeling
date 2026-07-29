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

import run_artifacts

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
    "region_weight":    "region_weights",
    "vds_loss":         "vds_losses",
    "ids_out_margin":   "ids_out_margins",
    "vdsgate_output_activation": "vdsgate_output_activations",
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
    # Region box bounds (2-D Vgs x Vds weighting). Scalars per experiment; paired
    # with the swept region_weight above. Read from the ranked CSV (see below).
    "region_vgs_lo":    "region_vgs_lo",
    "region_vgs_hi":    "region_vgs_hi",
    "region_vds_lo":    "region_vds_lo",
    "region_vds_hi":    "region_vds_hi",
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
    if v.startswith("@"):                 # @path.json -> load the value from a JSON file
        with open(v[1:].strip(), encoding="utf-8") as fh:   # (e.g. a big nn_architectures list)
            return json.load(fh)
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
    # script (ROOT), one level up (ROOT.parent) — e.g. the scripts were moved into
    # a physics_nn_pipeline/ subfolder while base_arch/ stayed at the repo root —
    # or under a dedicated runs/ tree (experiment dirs moved out of the scripts dir).
    for base in (ROOT, ROOT.parent, ROOT.parent / "runs"):
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


def params_from_json(jdata: dict, run_dir=None) -> dict:
    """Extract all relevant hyperparameters from a run JSON.

    For fields an OLDER JSON is missing (e.g. gm_max_ratio, lbfgs_*, loss_norm, and
    on very old runs gm2/gm3_weight), fall back to the exact invocation preserved in
    that run's run_log.txt.gz CMD line BEFORE using a hardcoded default -- so a
    reproduced config stays faithful to what actually ran instead of guessing.
    `run_dir` is the run directory (or any file inside it, e.g. the JSON path)."""
    cmd = run_artifacts.cli_args_from_log(run_dir) if run_dir else {}

    def g(key, default, cmd_flag=None):
        """JSON value if present, else run_info nested, else run_log CMD, else default (raw)."""
        v = jdata.get(key)
        if v is None:
            v = jdata.get("run_info", {}).get(key)
        if v is None:
            v = cmd.get(cmd_flag or key)
        return default if v is None else v

    p = {}
    for w in ("gm1_weight", "gm2_weight", "gm3_weight"):
        p[w] = _coerce(g(w, 0.0))

    p["gm_surgery_mode"] = g("gm_surgery_mode", "none") or "none"

    lr = jdata.get("lr") or jdata.get("learning_rate") or cmd.get("lr")
    p["lr"] = _coerce(lr) if lr is not None else 0.01

    p["output_activation"] = g("output_activation", "linear") or "linear"
    p["knee_alpha_scale"]  = _coerce(g("knee_alpha_scale", 1.0))
    p["knee_combiner"]     = g("knee_combiner", "sum") or "sum"
    p["seed"]              = _coerce(g("seed", 27))
    p["gm_max_ratio"]      = _coerce(g("gm_max_ratio", 1.0))

    p["epochs"]          = _coerce(g("epochs", 800))
    p["lbfgs_epochs"]    = _coerce(g("lbfgs_epochs", 20))
    p["lbfgs_max_iter"]  = _coerce(g("lbfgs_max_iter", 100))
    p["lbfgs_gm_aware"]  = _parse_bool(g("lbfgs_gm_aware", True))
    p["loss_norm"]        = g("loss_norm", "nmse") or "nmse"
    p["deterministic"]    = _parse_bool(g("deterministic", True))
    p["mixed_init"]       = g("mixed_init", "per_activation")

    p["gm_warmup_epochs"] = _coerce(g("gm_warmup_epochs", 0)) or 0
    p["gm_warmup_lr"]     = _coerce(g("gm_warmup_lr", None))
    p["ids_constraint"]   = _parse_bool(g("ids_constraint", False))
    p["ids_target"]       = _coerce(g("ids_target", None))
    p["ids_lambda"]       = _coerce(g("ids_lambda", 100.0))

    p["gm_vds_min"]    = _coerce(g("gm_vds_min", None))
    p["gm_vgs_min"]    = _coerce(g("gm_vgs_min", None))

    # Region box (2-D Vgs x Vds weighting). Usually absent from the run JSON; the
    # ranked CSV carries them and overrides these (see CSV loop in main()). The run_log
    # fallback recovers them too when the CSV lacks the columns (older compiles).
    for rk in ("region_weight", "region_vgs_lo", "region_vgs_hi",
               "region_vds_lo", "region_vds_hi"):
        p[rk] = _coerce(g(rk, None))
    p["vds_loss"] = _coerce(g("vds_loss", None))   # gds>=0 penalty weight (CSV overrides below)

    p["equation_type"] = g("equation_type", "pure") or "pure"
    p["use_gm"]        = _parse_bool(g("use_gm", True))
    p["use_opt_params"]= _parse_bool(g("use_opt_params", False))

    # Output-scale margin for bounded output activations (sigmoid/tanh/tanhm) and the
    # vdsgate gate mode -- both post-date this tool's original field list, so an OLDER
    # JSON without them means "legacy default" (no margin / softplus gate), same fallback
    # multi_experiment_runner.py itself uses (see its ids_out_margins/vdsgate_output_activations
    # merged.get(...) defaults) -- NOT silently dropped, which would retrain margin/tanhm
    # rows with the wrong (unmargined / softplus-gated) behavior.
    p["ids_out_margin"] = _coerce(g("ids_out_margin", 0.0))
    p["vdsgate_output_activation"] = g("vdsgate_output_activation", "softplus") or "softplus"

    # Always stringify: some JSONs store architecture as a list [64,64,64]; str() normalises
    # it to "[64, 64, 64]" which is hashable and matches the CSV representation.
    p["architecture"] = str(g("architecture", ""))

    return p


def _canon_key(v):
    """Canonical string for a group-key value so numerically-equal values match
    regardless of type/representation: 0, 0.0 and "0" all map to "0"; -3 and -3.0
    to "-3". Without this, two source sweeps that stored the same number with
    different types (e.g. vds_loss as int 0 vs float 0.0) split into duplicate
    groups. Non-numeric values (and bools) keep their str() form."""
    if v is None or v == "":
        return ""
    if isinstance(v, bool):
        return str(v)
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return str(int(f)) if f.is_integer() else repr(f)


def group_key_of(params: dict, exclude=frozenset()) -> tuple:
    """Stable hashable key from all non-architecture hyperparameters.

    `exclude` is a set of json_keys to drop from the key — used when a dimension
    is being overridden/swept uniformly (e.g. gm_vds_min set via --set, or gm
    weights fanned out via --gm_weight_tuples), so rows are not split on it.
    Values are canonicalized (see _canon_key) so type/representation differences
    (0 vs 0.0) don't split otherwise-identical rows."""
    return tuple(_canon_key(params.get(k, "")) for k in GROUP_KEY_FIELDS
                 if k != "architecture" and k not in exclude)


# Override keys that select/configure the base_config (not swept dims). Setting
# these via --set must reach detect_base_config, e.g. --set use_gm=true to turn on
# gm training for rows whose source run was pure-Ids.
BASE_CONFIG_OVERRIDE_KEYS = {"use_gm", "equation_type", "use_opt_params"}


def overridden_group_keys(overrides: dict) -> set:
    """json_keys (group-key fields) being overridden via --set, so rows are not split
    on them.

    Any field given a uniform --set value is dropped from the group key, whether it's
    a swept dimension or a scalar:
      * a list-valued swept override (e.g. region_weights=[2,4]) drives the sweep, and
      * a scalar override (e.g. region_vgs_lo=-3) sets every experiment alike.
    In both cases two source rows that differ ONLY on the overridden field must collapse
    into one group — otherwise they yield duplicate experiments once build_experiment
    homogenizes the field. (Before this covered only list-valued SWEPT_MAP overrides,
    so scalar SCALAR_MAP overrides like the region box bounds left rows split — e.g. a
    folder mixing region_weight 0 rows (empty box) and 1 rows (box set) produced 2x
    identical configs.) Covers both SWEPT_MAP (cfg key is plural) and SCALAR_MAP."""
    cfg_to_json = {v: k for k, v in SWEPT_MAP.items()}   # cfg_key -> json_key
    cfg_to_json.update({v: k for k, v in SCALAR_MAP.items()})
    return {cfg_to_json[ck] for ck in overrides if ck in cfg_to_json}


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
        if v is None:
            continue
        # vds_loss=0 means the penalty is off (its implicit default), so don't clutter
        # every generated config with vds_losses=[0]. Use --set vds_losses=[...] to turn it on.
        if json_key == "vds_loss" and _coerce(v) in (0, 0.0):
            continue
        # Same for ids_out_margin=0 (no margin, the legacy default) and
        # vdsgate_output_activation="softplus" (the legacy gate mode, also the default
        # multi_experiment_runner.py falls back to when the flag is omitted) -- omitting
        # these reproduces exact legacy command lines instead of cluttering every config.
        if json_key == "ids_out_margin" and _coerce(v) in (0, 0.0):
            continue
        if json_key == "vdsgate_output_activation" and str(v).strip().lower() == "softplus":
            continue
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

    # nn_architectures must be a list of JSON STRINGS: the runner passes each entry
    # straight to `--architecture <str>`. A --set nn_architectures=@file.json may bring
    # raw nested lists (e.g. [["tanh","tanh"], ...]) -> stringify so the runner doesn't
    # choke on a list where it expects a string.
    if isinstance(exp.get("nn_architectures"), list):
        exp["nn_architectures"] = [a if isinstance(a, str) else json.dumps(a)
                                   for a in exp["nn_architectures"]]

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
        params = params_from_json(jdata, run_dir=fp)
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
        # Region box columns live in the ranked CSV (not the run JSON); prefer them so
        # the generated config reproduces the source region weighting automatically.
        for r_col in ("region_weight", "region_vgs_lo", "region_vgs_hi",
                      "region_vds_lo", "region_vds_hi", "vds_loss"):
            if r_col in row:
                csv_rv = _coerce(row.get(r_col, ""))
                if csv_rv is not None:
                    params[r_col] = csv_rv
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
    exclude = overridden_group_keys(other_overrides)
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
        # Count runs: archs × product of all list-valued swept dims. Exclude
        # nn_architectures itself (it's the arch axis, already counted as n_archs;
        # otherwise a --set nn_architectures=[...] would be double-counted).
        n_sweep = 1
        for key in set(list(SWEPT_MAP.values()) + list(other_overrides.keys())) - {"nn_architectures"}:
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

    # Provenance (Option A): let each generated run trace back to its base row.
    # arch_hash -> base row id. The runner writes this next to the runs; compile
    # fills base_exp / base_id / base_csv from it (no --base_csv needed later).
    prov_map = {}
    for row in selected:
        h = run_artifacts.arch_id(row.get("architecture", ""))
        if h and h not in prov_map:
            bid = _coerce(row.get("id"))
            prov_map[h] = bid if bid is not None else row.get("id")
    provenance = {
        "base_exp": run_artifacts._base_exp_from_csv_path(str(csv_path)),
        "base_csv": str(csv_path.resolve()),
        "arch_to_base_id": prov_map,
    }

    cfg = {
        "_comment": (
            f"Generated by gen_config_from_rows.py  |  "
            f"source: {csv_path.name}  |  ids: {args.ids}  |  overrides: {overrides}"
        ),
        "_provenance": provenance,
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
