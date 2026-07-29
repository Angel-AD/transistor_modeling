# bestpicks10 — run_id reference

`avkf2_id21677_bestpicks10_gmvds3.json` trains 10 hand-picked, good-shape architectures
(via `analyze_shape.py`'s `gds_hump`/`gm3_maxabs`/`gm3_bad_frac` metrics) against all 6
measurement CSVs, using `gm_vds_min=3`, `adamw_avoid_localmin=true`, `epochs=2000`,
`vds_loss=0`. Results live in `runs\bestpicks10\<csv-name>\`.

**`equation_type: "pure:vdsgate_aeff_quad"`** — all 10 architectures are trained with the
vdsgate structured output, NOT a bare NN:

    Ids = softplus(NN(Vgs,Vds)) * tanh(a_eff(Vgs)*Vds) * (1 + l_eff(Vgs)*Vds)
    a_eff(Vgs) = softplus(a0 + a1*vgs_norm + a2*vgs_norm^2)   [always > 0]
    l_eff(Vgs) = softplus(l0 + l1*vgs_norm)

`a_eff(Vgs) > 0` for any Vgs, so `tanh(a_eff*Vds)` guarantees `sign(Ids) = sign(Vds)` and
exact 0 at `Vds=0`, regardless of what the tanh/swish/mish pattern below does inside the NN —
the odd-symmetry guarantee and the architecture-pattern finding are orthogonal, not competing,
mechanisms. (`a_eff`/`l_eff` are Vgs-*dependent* polynomials, not the constant `alpha`/`lamb`
of the plain `vdsgate` wrapper — this is the more flexible `vdsgate_aeff_quad` variant.)
Verified fitted numbers for run_id 7 (id 1, `cg2h40010_new_2.4_...` CSV):
`a0=1.207330, a1=-1.379142, a2=1.116913`; `l0=3.964194, l1=3.908457`
(equation string vs. torch model RMSE = 1.75e-07 A after the `plot_saved_state.py`
equation-export fix — see below).

## Sources
- **8 architectures** from `avkf2_id21677_483pct_gmvds3` (percentage-mixed activations,
  depth ≤3, 16-neuron cap): ids 1, 55, 301, 128, 233, 333, 287, 292.
- **2 architectures** from `avkf2_id21677_8586mixed_gmvds3` (per-layer single-activation
  choice, depth ≤3): ids 35, 75.

## `run_id` vs `id` — why run_id is the stable cross-CSV key
`id` is re-assigned per ranking/sort order and differs between files. `run_id` is derived
from the run's directory identity (`<exp_name>/<run_dir_name>`), which is the same for a
given architecture across all 6 CSV training runs (same config → same task order → same
`exp_NNN` index). **Verified**: the `run_id → architecture` mapping is byte-identical
between the `cg2h40010_new_2.4_...` and `cg2h40010_new_2.9_0_...` result folders — so
`run_id` is the correct key for comparing one architecture across measurement CSVs (use
`plot_run_id_all_csvs.ps1 -RunId <N>` for this).

## Full run_id → source → architecture mapping

| run_id | source | src id | depth | widths | layer 1 (tanh/swish/mish) | layer 2 (tanh/swish/mish) |
|---|---|---|---|---|---|---|
| 1  | 483pct    | 55  | 2 | 8,8 | 4/1/3 | 1/4/3 |
| 2  | 8586mixed | 75  | 2 | 7,4 | 7/0/0 (pure tanh) | 0/4/0 (pure swish) |
| 3  | 483pct    | 301 | 2 | 8,8 | 3/2/3 | 1/4/3 |
| 4  | 8586mixed | 35  | 2 | 5,5 | 0/0/5 (pure mish) | 0/5/0 (pure swish) |
| 5  | 483pct    | 292 | 2 | 8,8 | 5/2/1 | 6/1/1 |
| 6  | 483pct    | 128 | 2 | 8,8 | 4/1/3 | 3/4/1 |
| 7  | 483pct    | 1   | 2 | 8,8 | 4/1/3 | 1/3/4 |
| 8  | 483pct    | 333 | 2 | 8,8 | 4/3/1 | 2/4/2 |
| 9  | 483pct    | 233 | 2 | 8,8 | 4/2/2 | 2/5/1 |
| 10 | 483pct    | 287 | 2 | 8,8 | 4/3/1 | 4/2/2 |

**All 10 are depth=2** — universal across every pick, from both source sweeps.

## run_id 6 & 7 — the most structurally similar pair
- **run_id 7** (src `483pct` id 1) and **run_id 6** (src `483pct` id 128) share an
  **identical layer 1**: tanh=4, swish=1, mish=3 — byte-for-byte the same composition.
- They differ only in layer 2: run_id 7 leans mish (1/3/4), run_id 6 leans swish (3/4/1).
- Both still land in the "good shape" bucket regardless of which non-tanh activation
  dominates layer 2 — reinforcing that the specific swish-vs-mish choice in layer 2
  matters less than *having* a non-tanh-dominant layer 2.

## The pattern across all 10 (majority tendency, not a hard rule)
- **Universal (10/10)**: depth = 2 layers.
- **Dominant (7/10)**: tanh is the plurality activation in layer 1, and layer 2 shifts
  away from tanh toward swish or mish.
  - tanh→mish: run_id 7 (id 1)
  - tanh→swish: run_id 1, 6, 9, 8 (ids 55, 128, 233, 333); run_id 2 (id 75, pure)
  - tanh/mish tied→swish: run_id 3 (id 301, borderline)
- **Exceptions (3/10)**:
  - run_id 5, 10 (ids 292, 287): tanh dominant in **both** layers — no shift away from tanh.
  - run_id 4 (id 35): **mish**-dominant layer 1 → swish layer 2 — not tanh-first at all.

**Caveat**: this is 10 hand-picked architectures from an already shape-filtered subset,
not a controlled/statistical comparison — it describes what these 10 have in common, not
a proven general rule for the full architecture space.

## Fit quality (region_knee_combined_gm / region_knee_ids_rmse) across all 6 measurement CSVs

| run_id | 2.2_5 | 2.4_5 | 2.5_0 | 2.5_20 | 2.9_0 | 2.9_28 |
|---|---|---|---|---|---|---|
| 1  | 0.876 / 0.0141 | 0.830 / 0.0127 | 0.871 / 0.0133 | 0.859 / 0.0118 | 0.808 / 0.0118 | 1.032 / 0.0095 |
| 2  | 0.923 / 0.0194 | 0.895 / 0.0140 | 1.003 / 0.0125 | 0.883 / 0.0121 | 0.841 / 0.0128 | 1.056 / 0.0098 |
| 3  | 0.948 / 0.0150 | 0.889 / 0.0138 | 0.916 / 0.0131 | 0.869 / 0.0127 | 0.812 / 0.0122 | 1.167 / 0.0108 |
| 4  | 0.916 / 0.0176 | 0.859 / 0.0138 | 0.847 / 0.0121 | 0.929 / 0.0125 | 0.827 / 0.0134 | 0.983 / 0.0113 |
| 5  | 0.920 / 0.0158 | 0.887 / 0.0138 | 0.864 / 0.0121 | 0.862 / 0.0134 | 0.816 / 0.0119 | 1.037 / 0.0090 |
| 6  | 0.917 / 0.0161 | 0.852 / 0.0139 | 0.882 / 0.0128 | 0.868 / 0.0118 | 0.752 / 0.0114 | 1.006 / 0.0089 |
| 7  | 0.866 / 0.0138 | 0.780 / 0.0136 | 0.841 / 0.0121 | 0.829 / 0.0122 | 0.792 / 0.0114 | 1.059 / 0.0088 |
| 8  | 0.898 / 0.0152 | 0.896 / 0.0137 | 0.875 / 0.0124 | 0.902 / 0.0119 | 0.812 / 0.0114 | 1.050 / 0.0089 |
| 9  | 0.864 / 0.0157 | 0.877 / 0.0135 | 0.874 / 0.0121 | 0.876 / 0.0118 | 0.839 / 0.0117 | 1.037 / 0.0088 |
| 10 | 0.891 / 0.0181 | 0.887 / 0.0137 | 0.882 / 0.0127 | 0.856 / 0.0117 | 0.810 / 0.0117 | 1.024 / 0.0086 |

Format: `region_knee_combined_gm / region_knee_ids_rmse`. Columns are the 6 CSVs
(`cg2h40010_new_2.2_5_2_70W_center9`, etc., abbreviated by their distinguishing numbers).

Notable: **every run_id's `combined_gm` jumps above 1.0 on `2.9_28`** — that CSV is
harder to fit for all 10 architectures uniformly, suggesting a dataset-level effect
(not an architecture-specific weakness) worth investigating separately.

**Best `combined_gm` per CSV** (verified by direct min() over all 10, not eyeballed):

| CSV | best run_id | combined_gm |
|---|---|---|
| 2.2_5 | 9 | 0.864 |
| 2.4_5 | 7 | 0.780 |
| 2.5_0 | 7 | 0.841 |
| 2.5_20 | 7 | 0.829 |
| 2.9_0 | 6 | 0.752 |
| 2.9_28 | 4 | 0.983 |

**run_id 7 (id 1) wins on 3 of 6 CSVs** (2.4, 2.5_0, 2.5_20) — the most consistent
standout, but not a universal winner: run_id 9 (2.2), run_id 6 (2.9_0), and run_id 4
(2.9_28) each win on one CSV apiece. No single architecture dominates every dataset.

## Equation-export bug (fixed) — the `.md`/`.va` files in `plotted_configs/` were wrong

`plot_saved_state.py`'s equation extraction only recognized the plain `vdsgate`/`vdsgatelin`
wrapper (scalar alpha/lamb). It silently dropped the *entire* gating term for
`vdsgate_aeff*`/`vdsgate_vdsk*` — the variant every run in this file actually uses — so every
previously-generated `plot_saved_state_full.md`/`.va` for these 10 architectures (and for
`refine_vdsgate_gm_1`, the original id74/id159 `2000archs` sweep, etc.) showed only the bare
NN output, missing the `tanh(a_eff(Vgs)*Vds)*(1+l_eff(Vgs)*Vds)` factor entirely. Confirmed via
the file's own self-check (equation string vs. torch model, should be ~1e-6): it was showing
**12.8 A** of error. Fixed in `plot_saved_state.py` (Vgs-dependent `a_eff`/`l_eff` polynomials
now correctly expanded into the exported equation and numpy verification); after the fix the
same self-check reads **1.75e-07 A**. Only the exported equation *text* was wrong — trained
weights and predictions were always correct, so no retraining was needed, just regenerating
the `.md`/`.va` files (done for all affected `plotted_configs` folders project-wide).
