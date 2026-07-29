"""
run_artifacts.py -- helpers to recover run metadata from a run directory.

Newer runs record every hyperparameter in run_loss_*.json. Older runs recorded a
subset there but their FULL invocation is preserved in run_log.txt.gz as the first
line: `CMD: ['python', 'script.py', '--flag', 'value', '--bool_flag', ...]`.

`cli_args_from_log(run_dir)` parses that CMD line into a {flag: value} dict so the
compile scripts can back-fill fields the older JSONs are missing (gm_max_ratio,
lbfgs_*, loss_norm, csv, ...). Store-true flags map to True. Returns {} if there is
no log (e.g. runs pruned down to compiled CSVs).
"""
import ast
import csv as _csv
import glob as _glob
import gzip
import hashlib
import json
import os

_cache = {}
_prov_cache = {}
_basecsv_cache = {}
_meta_cache = {}


def run_meta(root):
    """{'csv', 'base_csv', 'base_exp', 'configs'} for a results folder, as absolute usable
    paths, so tools can auto-detect the inputs instead of the user retyping them.

    The runner stashes copies of the config, measurement CSV and base CSV under
    <root>/base_files/ and records them in <root>/_run_meta.json (both the local copy and
    the original source). This prefers the local copy (portable -- survives moving the
    folder), falls back to the original absolute source, then for legacy folders reads the
    measurement csv from a run's JSON 'csv' (or run_log --csv) and base_* from a
    _provenance.json sidecar. Values may be None. Cached per root."""
    key = os.path.abspath(root) if root else ""
    if key in _meta_cache:
        return _meta_cache[key]
    meta = {"csv": None, "base_csv": None, "base_exp": None, "configs": []}

    def _resolve(rel, src):
        """Absolute path to the local copy if it exists, else the source if it exists, else None."""
        if rel:
            p = rel if os.path.isabs(rel) else os.path.join(root, rel)
            if os.path.isfile(p):
                return os.path.abspath(p)
        if src and os.path.isfile(src):
            return os.path.abspath(src)
        return None

    raw = {}
    mp = os.path.join(root, "_run_meta.json") if root else ""
    if mp and os.path.isfile(mp):
        try:
            raw = json.load(open(mp, encoding="utf-8"))
        except Exception:
            raw = {}
    if raw and root:
        meta["csv"] = _resolve(raw.get("csv"), raw.get("csv_src"))
        meta["base_csv"] = _resolve(raw.get("base_csv"), raw.get("base_csv_src"))
        meta["base_exp"] = raw.get("base_exp")
        meta["configs"] = [c if os.path.isabs(c) else os.path.abspath(os.path.join(root, c))
                           for c in (raw.get("configs") or [])]

    if root and (not meta.get("csv") or not meta.get("base_csv")):
        js = _glob.glob(os.path.join(root, "*", "*", "run_loss_*.json"))
        if js:
            jp = js[0]
            if not meta.get("csv"):
                try:
                    meta["csv"] = json.load(open(jp, encoding="utf-8")).get("csv") \
                                  or cli_args_from_log(jp).get("csv")
                except Exception:
                    pass
            if not meta.get("base_csv"):
                prov = provenance_for(jp)
                meta["base_csv"] = meta["base_csv"] or prov.get("base_csv")
                meta["base_exp"] = meta["base_exp"] or prov.get("base_exp")
    _meta_cache[key] = meta
    return meta


def run_hash(json_path):
    """Stable, globally-unique id for a run, from its directory identity
    (<exp_name>/<run_dir_name>) -- the same in every ranked file/folder/window, and
    unaffected by re-ranking. Uses the RELATIVE dir identity so it survives moving the
    master root. (Distinct from arch_hash, which is shared by all runs of one arch.)"""
    rd = os.path.dirname(os.path.abspath(json_path))
    exp = os.path.basename(os.path.dirname(rd))
    return hashlib.sha1(f"{exp}/{os.path.basename(rd)}".encode("utf-8")).hexdigest()[:10]


def arch_id(arch_str):
    """Short stable id for an architecture. All runs sharing an arch (e.g. a vds_loss or
    region_weight sweep over the same net) get the SAME arch_id, so you can group / pivot
    by it (arch_id x swept-dim) to isolate the effect of the swept parameter. Canonicalizes
    quote style / spacing before hashing so the id is format-independent."""
    try:
        canon = json.dumps(ast.literal_eval(str(arch_str)))
    except Exception:
        canon = str(arch_str)
    return hashlib.sha1(canon.encode("utf-8")).hexdigest()[:8]


def cli_args_from_log(run_dir):
    """{cli_flag: value} parsed from the run_log.txt.gz CMD line, or {} if absent.

    `run_dir` may be a run directory or the path to a file inside it (e.g. the JSON).
    Results are cached per directory."""
    d = run_dir if os.path.isdir(run_dir) else os.path.dirname(run_dir)
    if d in _cache:
        return _cache[d]
    out = {}
    log = os.path.join(d, "run_log.txt.gz")
    try:
        with gzip.open(log, "rt", encoding="utf-8", errors="replace") as f:
            first = f.readline().strip()
        if first.startswith("CMD:"):
            argv = ast.literal_eval(first[len("CMD:"):].strip())
            i = 0
            while i < len(argv):
                tok = str(argv[i])
                if tok.startswith("--"):
                    key = tok[2:]
                    if "=" in key:                       # --min_vgs=-4.0 style
                        k, v = key.split("=", 1)
                        out[k] = v
                    elif i + 1 < len(argv) and not str(argv[i + 1]).startswith("--"):
                        out[key] = str(argv[i + 1]); i += 1
                    else:
                        out[key] = True                  # store-true flag
                i += 1
    except (OSError, ValueError, SyntaxError):
        pass
    _cache[d] = out
    return out


def field(json_data, run_dir, json_key, cmd_flag=None, default=""):
    """Value for a field: prefer the run JSON, fall back to the run_log CMD line,
    else `default`. `cmd_flag` defaults to `json_key` (they usually match)."""
    v = json_data.get(json_key)
    if v is not None:
        return v
    return cli_args_from_log(run_dir).get(cmd_flag or json_key, default)


# --------------------------------------------------------------------------- #
# Lineage: link a run back to the BASE experiment/row it was generated from.
#   Option A: gen_config_from_rows embeds provenance in the config; the runner
#             writes it as <exp_dir>/_provenance.json next to the runs.
#   Option B: compile is given --base_csv and matches by arch_hash.
# --------------------------------------------------------------------------- #
def _base_exp_from_csv_path(csv_path):
    """Experiment folder name from a base ranked/master_best CSV path."""
    parent = os.path.dirname(os.path.abspath(csv_path))
    if os.path.basename(parent).startswith("ranked_"):   # runs/<exp>/ranked_*/<file>.csv
        return os.path.basename(os.path.dirname(parent))
    return os.path.basename(parent)                       # runs/<exp>/<file>.csv


def provenance_for(path):
    """Nearest _provenance.json for a run (walks up a few levels from its dir).
    Returns the dict (base_exp / base_csv / arch_to_base_id) or {}. Cached."""
    cur = path if os.path.isdir(path) else os.path.dirname(path)
    for _ in range(4):
        p = os.path.join(cur, "_provenance.json")
        if p in _prov_cache:
            return _prov_cache[p]
        if os.path.isfile(p):
            try:
                _prov_cache[p] = json.load(open(p, encoding="utf-8"))
            except Exception:
                _prov_cache[p] = {}
            return _prov_cache[p]
        nxt = os.path.dirname(cur)
        if nxt == cur:
            break
        cur = nxt
    return {}


def base_csv_index(base_csv, base_exp=None):
    """(arch_hash -> base_id, base_exp, abspath) built from a base ranked CSV. First
    (best-ranked) row wins on an arch_hash collision. Cached per csv path."""
    key = os.path.abspath(base_csv)
    if key in _basecsv_cache:
        return _basecsv_cache[key]
    amap = {}
    try:
        rows = list(_csv.DictReader(open(base_csv, newline="", encoding="utf-8")))
    except Exception:
        rows = []
    for r in rows:
        h = r.get("arch_hash") or arch_id(r.get("architecture", ""))
        if h and h not in amap:
            _id = r.get("id")
            try:
                _id = int(_id)
            except (TypeError, ValueError):
                pass
            amap[h] = _id
    res = (amap, base_exp or _base_exp_from_csv_path(base_csv), key)
    _basecsv_cache[key] = res
    return res


def base_lineage(file_path, arch_hash, base_index=None):
    """(base_exp, base_id, base_csv) for a run. Prefers the provenance sidecar (A),
    falls back to the --base_csv index (B). Returns ("NA", "NA", "NA") -- not applicable /
    unresolved -- when neither resolves (matches the eq/knee_combiner 'NA' convention;
    pandas reads 'NA' as NaN, so numeric use of base_id is unaffected)."""
    prov = provenance_for(file_path)
    amap = prov.get("arch_to_base_id") or {}
    if arch_hash in amap:
        return prov.get("base_exp", "NA"), amap[arch_hash], prov.get("base_csv", "NA")
    if base_index is not None:
        bmap, bexp, bcsv = base_index
        if arch_hash in bmap:
            return bexp, bmap[arch_hash], bcsv
    return "NA", "NA", "NA"
