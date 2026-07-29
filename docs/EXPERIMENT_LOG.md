# Experiment Log — Ids / gm architecture & physics+NN campaign

Living record of every optimization round: its **objective**, the config that ran
it, scale, key result, and the decision it drove. Newest rounds at the bottom.
Data CSV throughout: `csvs/cg2h40010_new_2.4_5_2_70W_center9.csv`. All pure-NN runs
use `per_activation` init, 5 seeds, `deterministic`. Metrics are seed-averaged RMSE.

## Baselines (best so far, lower = better)
| metric | best | where | config |
|---|---|---|---|
| `ids_rmse` | **0.00318** | pure NN | `6×10 tanh`, 600 ep (round 4) |
| `gm1_rmse` | **0.04661** | pure NN | `2×4 swish/mish`, surgery=none, W0.3 (round 3/4) |
| `gm2_rmse` | **0.20352** | pure NN | `2×4 swish`, surgery=none, W0.3 (round 3/4) |
| `gm3_rmse` | **0.87980** | **physics+NN** | classic, trained, `sum`, a5.0, W3, none (round 6) |

Pure-NN gm3 floor = 0.94910 (`2×4 sin`, **soft PCGrad**, W1.0). Physics+NN beats it by −7%.

## Standing conclusions
- **Ids wants large nets; gm wants small nets** (round 3) — the tension physics+NN targets.
- **Pure-NN Ids plateaus ~0.0032** (round 4); capacity past `6×10` gives <10%.
- **PCGrad is marginal for pure NN** (only gm3, via `soft`) but **helps in physics+NN** (`no-bounded`).
- **Physics+NN: train params within ±10% leash > frozen** (round 7, 9/12 cells).
- **Best physics eq: 6-term for gm1/gm2, classic for gm3.** Wider knee gate (a3–a5) helps gm.
- Physics+NN **wins gm3**, **ties gm1**, **loses gm2** vs pure NN (rounds 6–7).

---

## Rounds

### R1 / R1b — weight-init study (pre-existing)
`round1_init/`, `round1b_init/` — het_xavier vs het_per_activation. Outcome:
`per_activation` adopted as default init.

### R2 — pure-NN architecture search, small (pre-existing)
`round2_arch/` — curated archs, ids+gm1 only. Best ids `3×4 swish` 0.0066. Showed
swish/mish > tanh at small scale; ranking still climbing at the grid edge.

### R3 — pure-NN capacity-extended search, all 4 objectives
- **Objective:** best arch per {ids, gm1, gm2, gm3}, pure NN.
- **Config:** `physics_nn_configs/opt_configs_round3_arch.json` (gen: `temp/gen_round3_config.py`). 52 archs × 4 obj × 5 seeds = 1040 runs, 400 ep.
- **Result:** ids `4×8 tanh` 0.00363 (at capacity ceiling); gm1 `2×4 swish/mish` 0.04661; gm2 `2×4 swish` 0.20352; gm3 `2×6 sin` 0.98470.
- **Drove:** Ids needs more capacity (R4); gm plateaued small → gm lever is PCGrad/physics.

### R4 — Ids capacity extension + PCGrad on pure NN
- **Objective:** find Ids plateau; test whether PCGrad lowers pure-NN gm.
- **Config:** `opt_configs_round4.json` (gen: `temp/gen_round4_config.py`). ids_capacity (180) + gm{1,2,3}_pcgrad (360) = 540 runs, 400–600 ep.
- **Result:** ids `6×10 tanh` 0.00318 (near plateau). PCGrad marginal: gm1 none best; gm2 none best; gm3 `soft` 0.94910 (−3.6% vs none, halves variance).
- **Drove:** lock pure-NN baselines; move gm work to physics+NN.

### R-SLSQP — physics seeds
- **Config:** `slsqp_scripts/slsqp_configs/opt_configs.json` → `runs/slsqp/`. 8 seeds: {mod1, classic, 6-term, 9-term} × {no_gm, gm1}.
- **Used:** the `_no_gm` seeds (pure-Ids physics fits). `_gm1` seeds reserved as a lever.

### R5 — physics+NN (noNN_knee) screening
- **Objective:** does any physics+NN config beat pure-NN gm? Map the region.
- **Config:** `opt_configs_round5_phys_screen.json` (gen: `temp/gen_round5_phys_screen.py`). 4 eq × {frozen,trained} × {sum,product,residual} × knee-scale{1,3} × 3 gm obj = 432 runs, gm W0.5, surgery none, 3 seeds, 400 ep.
- **Result:** trained>frozen; a3.0>a1.0; 6-term (gm1/gm2), classic (gm3). Only gm3 ~tied baseline. (Note: `residual` CLI choice added during this round.)
- **Drove:** R6 with the two held-back levers (high gm weight, PCGrad).

### R6 — physics+NN refinement (high gm weight + PCGrad)
- **Objective:** push gm weight (1/3/10) + PCGrad (none/no-bounded/soft) + wider gate (a3/a5), trained.
- **Config:** `opt_configs_round6_phys_refine.json` (gen: `temp/gen_round6_phys_refine.py`). 540 runs, 800 ep, 5 seeds.
- **Result:** **gm3 0.87980 (−7%, beats pure NN)** (classic sum a5 W3 none; no-bounded close 2nd). gm1 0.04730 (+1% tie). gm2 0.27839 (+37% loss). High weight (W10) did NOT help; W1–W3 best. PCGrad `no-bounded` best mode for gm2/gm3.
- **Drove:** confirmed gm3 win; freeze question → R7.

### R7 / R7b — frozen vs trained × 4 eqs
- **Objective:** fixed vs trained (±10%) physics params, per eq family.
- **Config:** `opt_configs_round7_freeze.json` (+ `opt_configs_round7b_soft.json` for soft). 480 (+240) runs, 800 ep, 5 seeds.
- **Result:** **trained wins 9/12 (eq,gm) cells.** Best: gm1 6-term trained 0.04730, gm2 6-term trained 0.27839, gm3 classic trained 0.87980.
- **Drove:** adopt trained (±10%); use 6-term (gm1/gm2), classic (gm3) in R8.

### R8 — multi-gm combination + PCGrad search  *(in progress)*
- **Objective:** which gm / combination of gms (penalized together, with PCGrad) gives the best **overall** loss across ALL gms (ranked by a normalized combined-gm score)?
- **Config:** `opt_configs_round8_multigm.json` (gen: `temp/gen_round8_multigm.py`). 7 combos × 2 model types × 4 surgery × 5 seeds = 280 runs, 500 ep. Pure arch `2×4 [sin,swish,mish,tanh]`; phys 6-term/product/a3.0/trained, arch `2×4 swish/mish`. Score = `mean(gm1/0.04661, gm2/0.20352, gm3/0.94910)`.
- **Result:** best overall = **pure gm2+gm3, none = 1.130** (gm1 0.0435 *beats* baseline, gm2 0.2205, gm3 1.303). Best all-three = pure gm123 **element-wise-bounded 1.272**.
  - **PCGrad on the triple (gm123) is decisive:** pure none 1.539 → ewb **1.272 (−17%)**; physics none **diverged (~11775)** → no-bounded 1.505. Marginal for single gm, essential for multi-gm — matches the PCGrad design intent.
  - **Throttle:** combined score limited by gm3 (round-8 arch was mixed, weak at gm3; gm3 wants `sin`). Physics used 6-term/product (not gm3's best classic/sum).
- **Drove:** R9 — multi-gm with gm3-capable (`sin`) archs + classic/sum physics, on the winning combos {gm23, gm123} × PCGrad.

### R9 — multi-gm with gm3-capable archs
- **Objective:** lift the all-around combined score by giving gm3 a `sin` architecture and physics its gm3-best (classic/sum), on the round-8 winning combos.
- **Config:** `opt_configs_round9_multigm2.json` (gen: `temp/gen_round9_multigm2.py`). 180 runs, 600 ep.
- **Result:** sin archs **fixed gm3** — `pure gm23, 2×4 sin` hit **gm3=0.810** (−15% vs baseline). Best combined = `pure gm23, 2×6 sin, element-wise-bounded = 1.128` (barely past R8's 1.130). All-around **plateaus ~1.13** due to the activation tradeoff (sin→great gm3/weak gm1-2; swish/mish→reverse). **gm2+gm3 is the best all-around recipe.**
- **Drove:** R10 (gm_max_ratio lever) + the magnitude-balanced-weights idea.

### R-CODE — wired gm_max_ratio + L-BFGS controls into config
- `multi_experiment_runner.py` now forwards `gm_max_ratios` (swept), `lbfgs_epochs`, `lbfgs_max_iter`, `lbfgs_gm_aware`. Trainer: gm-aware L-BFGS closure + **combined-objective best-selection** when `use_gm`. Smoke test PASS; README documents all. Defaults reproduce old behavior.

### R10 — gm_max_ratio sweep + gm-aware L-BFGS
- **Objective:** does `gm_max_ratio>1` (amplify via element-wise-percent / un-cap via element-wise-bounded) + gm-aware L-BFGS lower gm without raising Ids? gm123, 5 seeds, lbfgs_epochs=15.
- **Config:** `opt_configs_round10_gmratio.json` (gen: `temp/gen_round10_gmratio.py`). 100 runs.
- **Result:**
  - **`gm_max_ratio>1` improves gm** (amplifier strongest): pure `element-wise-percent` gm3 0.664→0.476, gm2 0.274→0.208 as ratio 1→10; `phys_cap` improves all gms monotonically.
  - **Pure NN trades Ids** (ids 0.147→0.172 with ratio); **physics+NN holds Ids ~0.03–0.04** → physics+NN is the right setting for "good gm without hurting Ids."
  - **gm-aware L-BFGS HURT**: Ids-only polish (`lbfgs_gm_aware=False`) gave lower Ids (0.105–0.117 vs 0.147–0.172) AND better combined. The useful part of the change is the **combined-best-selection safeguard**, not the gm-aware closure.
  - **Equal-weight gm123 wrecks pure-NN Ids** (gm3 magnitude dominates) → use magnitude-balanced gm weights.
- **Drove:** next — physics+NN + magnitude-balanced gm weights + moderate `gm_max_ratio` + `lbfgs_gm_aware=False`.

### R-CODE2 — loss_norm=nmse + auto gm-aware L-BFGS
- `--loss_norm {none,nmse}`: nmse divides each MSE term (Ids + every gm) by `mean(target^2)` → dimensionless/scale-balanced so equal weights balance the gms (no manual per-gm tuning; normalize by global per-signal scale, not per-point). `--lbfgs_gm_aware` now tri-state, **AUTO-ON when use_gm** (AdamW + L-BFGS both apply surgery); `--no-lbfgs_gm_aware` to disable. Runner forwards both. Smoke PASS; README documented.

### R11 — NMSE loss balancing vs none
- **Objective:** does `loss_norm=nmse` lower the combined gm score (balance gms) without hurting Ids? gm123 equal w0.3, element-wise-percent, gm_max_ratio {1,5}, auto gm-aware, pure + physics(6-term).
- **Config:** `opt_configs_round11_nmse.json` (gen: `temp/gen_round11_nmse.py`). 40 runs.
- **Result:** **NMSE helps pure NN** — combined 1.578→**1.453** (ratio 5, −8%); 1.727→1.566 (ratio 1); gm1/gm2/gm3 all slightly better; ids ~flat (~0.17). **Slightly worse for physics+NN** (1.32→1.36): there the gm3 ceiling is the *equation* (6-term/product), not loss balance, so nmse only perturbs.
- **Standing tradeoff unchanged:** all-around combined floor ~1.45 (pure) — driven by the activation tradeoff (gm1 stuck ~0.135 on the sin-heavy arch) and, in physics, gm3's eq choice. Pure NN gets gm3 great (0.47) but ids ~0.17; physics+NN holds ids ~0.04 but gm3 ~1.3.
- **Takeaway:** NMSE is the right tool where *scale imbalance* is the problem (pure NN); it doesn't override architecture/equation limits.

### R12 — BASE arch search (1 seed, 2984 runs) + 4 FOLLOW-UPS (3135 runs)  ★ breakthrough
- **Base:** pure NN, no gm, 1000+50 ep, total<=16 / depth1-3 / >=2per-layer, homog tanh/mish/swish + mixed tanh+mish/tanh+swish (full ratio) + **non-uniform (funnel) widths**, out_act {linear,softplus}. (`opt_configs_base_arch.json`)
  - Best Ids = **0.00338** (wide tanh/swish, gm-poor: gm3~5). Best *natural* gm = funnel `[swish 3,2,6]` combined 0.626 (gm1 0.041/gm2 0.113/gm3 0.415, ids 0.014) — non-uniform widths were the unlock.
- **Follow-ups:** best-Ids & best-gm archs × multi-gm combos {gm12,gm13,gm23,gm123} × surgery {none,no-bounded,soft,ew-percent,ew-bounded} × gm_max_ratio {1,5}; physics {classic,9-term,mod1}×{sum,product}. loss_norm=nmse, gm-aware L-BFGS auto. (`opt_configs_followups.json`)
- **RESULT — best all-around model of the whole campaign:**
  - **pureNN_gms_ids: combined 0.312** — gm12, **surgery none**, pure, high-capacity tanh arch: **gm1 0.022 / gm2 0.032 / gm3 0.285 / ids 0.020**. gm123 variant → **gm3 0.149** at ids 0.021.
  - Crushes everything prior (gm1 ~2x, gm2 ~3-6x, gm3 ~2-5x better than old bests; combined 0.31 vs old 1.07).
- **Key conclusions:**
  1. **Multi-gm training + NMSE balancing on HIGH-capacity (best-Ids) archs is the winner** — reverses the old "gm wants small nets": once gm is actively trained with balanced losses, capacity helps ALL objectives. pureNN_gms_ids (0.312) < pureNN_gms_gms (0.347).
  2. **PCGrad surgery did NOT help — `none` (plain weighted sum) won.** NMSE loss balancing was the real lever, not gradient surgery.
  3. **gm12 gives the best all-around** (and gm3 falls to 0.285 for free); gm123 pushes gm3 to 0.149.
  4. **Pure NN beat physics+NN** on combined gm (0.31 vs 0.63); physics did not hold ids lower (0.038 vs 0.020). classic was the best physics eq.
  5. **Ids cost:** best model ids 0.020 (vs 0.0034 pure-Ids floor) — the gm↔Ids frontier, now at a far better operating point.
- **Known bug (FIXED):** gm-aware L-BFGS + *bounded* surgery made strong-Wolfe line search thrash (surgery set a gradient inconsistent with the returned loss) → 1 run hung (`physicsNN_gms_gms/gm123/ew-bounded`), killed. **Fix applied:** gradient surgery is now applied ONLY in AdamW; the L-BFGS closure polishes the *plain* combined loss (`Ids + Σ gm`) with its true gradient, so strong-Wolfe stays consistent for all modes. Verified: the previously-hanging case now completes. (Surgery still does its conflict-resolution job during AdamW.)
