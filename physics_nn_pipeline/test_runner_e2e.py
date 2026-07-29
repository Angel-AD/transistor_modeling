"""
test_runner_e2e.py -- END-TO-END test of the whole pipeline through the REAL runner.

For a set of scenarios that each toggle different options the runner can pass (region
weight, vds_loss, gm mask, gm surgery, arch/lr/output-activation/seed, ids-region,
loss_norm), it builds ONE config with one experiment per scenario, runs
`multi_experiment_runner.py` ONCE (parallel), then for every scenario verifies:

  * the run_loss JSON records the intended option value  -> proves the runner PLUMBED it, and
  * the training's [trace]/[region]/[ids_region] log shows the intended EFFECT -> proves it ACTED.

So it answers "does the runner have the effect we want for each option?" -- not just that it ran.
Each run is tiny (1 arch, 2 epochs, 1 L-BFGS step). Run:  python test_runner_e2e.py   (~1-3 min)
"""
import os, sys, re, json, glob, gzip, copy, subprocess, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CSV  = os.path.join(REPO, "csvs", "cg2h40010_new_2.4_5_2_70W_center9.csv")
CFG_SRC = os.path.join(HERE, "physics_nn_configs", "refine_vdsk_freelam_vdsloss_gm_1_regionwt_2_4.json")
OUT  = os.path.join(HERE, "_e2e_tmp")
BOX  = dict(region_vgs_lo=-3, region_vgs_hi=0, region_vds_lo=0, region_vds_hi=15)


def _num(x):
    try: return float(x)
    except (TypeError, ValueError): return None

def jeq(actual, expected):
    a, e = _num(actual), _num(expected)
    return (abs(a - e) < 1e-9) if (a is not None and e is not None) else (actual == expected)


def region_wsum_ok(log, jd, name):
    m = re.search(r"region_w built: n=(\d+) in_box=(\d+) out_box=\d+ weight=([\d.]+) w\.sum=([\d.]+)", log)
    if not m: return False, "no 'region_w built' trace"
    n, inbox, wt, wsum = int(m[1]), int(m[2]), float(m[3]), float(m[4])
    exp = n + wt * inbox
    return (inbox > 0 and abs(wsum - exp) <= 1e-2 * max(1.0, exp),
            f"w.sum={wsum:.1f} == n+weight*in_box={exp:.1f} (in_box={inbox})")

def gm_mask_shrinks(log, jd, name):
    m = re.search(r"gm_loss_mask: set -> (\d+)/(\d+) points kept", log)
    if not m: return False, "no 'gm_loss_mask: set' trace"
    kept, tot = int(m[1]), int(m[2])
    return kept < tot, f"gm mask kept {kept}/{tot} (excluded {tot-kept})"

def loss_norm_none(log, jd, name):
    m = re.search(r"norms \(none\): ids_norm=([\d.eE+-]+)", log)
    if not m: return False, "no 'norms (none)' trace"
    return abs(float(m[1]) - 1.0) < 1e-6, f"loss_norm=none -> ids_norm={float(m[1]):.3f} (==1.0)"


# (key, experiment overrides, JSON asserts, trace-substrings-present, trace-substrings-absent, custom)
SCENARIOS = [
    ("baseline", dict(region_weights=[0], vds_losses=[0]),
     {"use_gm": True, "region_weight": 0.0, "vds_loss": 0.0, "gm1_weight": 0.1},
     ["gm_loss_mask:", "ids_loss PLAIN (region off)", "gm_losses: region_w applied=False", "vds_mono_loss: OFF"],
     ["region_w built:"], None),

    ("region", dict(region_weights=[5], vds_losses=[0], **BOX),
     {"region_weight": 5.0, "region_vgs_lo": -3, "region_vds_hi": 15},
     ["region_w built:", "ids_loss WEIGHTED", "gm_losses: region_w applied=True"], [], region_wsum_ok),

    ("vds_loss", dict(region_weights=[0], vds_losses=[5], **BOX),
     {"vds_loss": 5.0, "region_weight": 0.0},
     ["vds_mono_loss: ON weight=5"], ["ids_loss WEIGHTED"], None),

    ("gm_mask", dict(region_weights=[0], vds_losses=[0], gm_vds_mins=[8]),
     {"gm_vds_min": 8.0},
     ["gm_loss_mask: set ->"], [], gm_mask_shrinks),

    ("gm_surgery", dict(region_weights=[0], vds_losses=[0], gm_surgery_modes=["soft"]),
     {"gm_surgery_mode": "soft"},
     ["surgery_mode=soft"], [], None),

    ("arch_lr_act_seed", dict(region_weights=[0], vds_losses=[0],
                              learning_rates=[0.005], output_activations=["softplus"], seeds=[42]),
     {"learning_rate": 0.005, "output_activation": "softplus", "seed": 42},
     ["seed=42", "lr=0.005", "model built:"], [], None),

    ("ids_region", dict(region_weights=[0], vds_losses=[0],
                        ids_region_weights=[3], ids_region_lo=-2, ids_region_hi=0),
     {"ids_region_weight": 3.0, "ids_region_lo": -2, "ids_region_hi": 0},
     ["[ids_region] band"], [], None),

    ("loss_norm_none", dict(region_weights=[0], vds_losses=[0], loss_norm="none"),
     {},
     ["norms (none):"], [], loss_norm_none),
]


def build_config(path):
    d = json.load(open(CFG_SRC, encoding="utf-8"))
    _, base = next(iter(d["EXPERIMENTS"].items()))
    archs = sorted(base["nn_architectures"], key=lambda a: sum(len(x) for x in a))
    base = dict(base)
    base["nn_architectures"] = [archs[0]]           # 1 tiny arch by default
    base["seeds"] = [27]
    base.setdefault("gm_vds_mins", [0]); base.setdefault("gm_vgs_mins", [None])
    exps = {}
    for key, over, *_ in SCENARIOS:
        e = copy.deepcopy(base)
        if key == "arch_lr_act_seed":            # give this scenario a *different* arch
            e["nn_architectures"] = [archs[1] if len(archs) > 1 else archs[0]]
        e.update(over)
        exps["e2e_" + key] = e
    d["EXPERIMENTS"] = exps
    d["DEFAULTS"].update(epochs=2, lbfgs_epochs=1, lbfgs_max_iter=5, seeds=[27],
                         loss_norm="nmse", max_workers=len(SCENARIOS))
    json.dump(d, open(path, "w"), indent=1)


def read_log(run_dir):
    p = glob.glob(os.path.join(run_dir, "run_log.txt.gz"))
    if not p: return ""
    with gzip.open(p[0], "rt", encoding="utf-8", errors="replace") as f:
        return f.read()


def main():
    if not os.path.isfile(CSV):
        print(f"[SKIP] measurement CSV not found: {CSV}"); return 0
    if os.path.isdir(OUT): shutil.rmtree(OUT)
    os.makedirs(OUT)
    cfg = os.path.join(OUT, "e2e.json"); build_config(cfg)
    master = os.path.join(OUT, "runs")

    print(f"running the REAL pipeline over {len(SCENARIOS)} option scenarios ...")
    env = {**os.environ, "PNN_DEBUG": "1"}
    r = subprocess.run([sys.executable, "multi_experiment_runner.py", "--config", cfg,
                        "--csv", CSV, "--master_root_path", master,
                        "--max_workers", str(len(SCENARIOS))],
                       cwd=HERE, capture_output=True, text=True, env=env)
    ok = True
    def check(cond, msg):
        nonlocal ok; ok = ok and cond
        print(("  [ok] " if cond else "  [FAIL] ") + msg)

    check(r.returncode == 0, f"runner exit == 0 (got {r.returncode})")
    if r.returncode != 0:
        print("---- runner stderr tail ----\n" + (r.stderr or "")[-1500:])

    for key, over, jchecks, has, absent, custom in SCENARIOS:
        print(f"\n-- scenario: {key} --")
        rundirs = glob.glob(os.path.join(master, "e2e_" + key, "exp_*"))
        if not rundirs:
            check(False, f"{key}: run dir produced"); continue
        rd = rundirs[0]
        js = glob.glob(os.path.join(rd, "run_loss_*.json"))
        check(bool(js), f"{key}: run_loss JSON exists")
        if not js: continue
        jd = json.load(open(js[0], encoding="utf-8"))
        log = read_log(rd)
        for k, v in jchecks.items():
            check(jeq(jd.get(k), v), f"{key}: JSON {k}={jd.get(k)!r} (expected {v!r})")
        for s in has:
            check(s in log, f"{key}: effect traced -> {s!r}")
        for s in absent:
            check(s not in log, f"{key}: correctly absent -> {s!r}")
        if custom:
            cond, msg = custom(log, jd, key)
            check(cond, f"{key}: {msg}")

    print("\n" + "=" * 64)
    print("RUNNER E2E: " + ("ALL PASSED" if ok else "FAILED"))
    if ok: shutil.rmtree(OUT, ignore_errors=True)
    else:  print(f"(artifacts kept: {OUT})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
