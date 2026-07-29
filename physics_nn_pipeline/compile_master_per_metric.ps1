# Best-N of EACH experiment, sorted separately by each metric.
# Just calls compile_master_best.py once per metric (single-key sort), producing:
#   master_best_ids_rmse_top<N>_<root>.csv
#   master_best_gm1_rmse_top<N>_<root>.csv
#   master_best_gm2_rmse_top<N>_<root>.csv
#   master_best_gm3_rmse_top<N>_<root>.csv
# Each file = the top-N runs of every sub-experiment, ranked by that one metric,
# tagged with an 'experiment' column.
#
# Run from physics_nn_pipeline/. compile_results_csv must have been run first
# (the per-sub-experiment compiled_results_*.csv must already exist).
#
# Usage:
#   .\compile_master_per_metric.ps1 -Root refine2
#   .\compile_master_per_metric.ps1 -Root refine2 -TopN 10
param(
    [string] $Root = "refine2",
    [int]    $TopN = 15
)

foreach ($metric in "ids_rmse", "gm1_rmse", "gm2_rmse", "gm3_rmse") {
    Write-Host "==> master_best by $metric (top $TopN per experiment)" -ForegroundColor Cyan
    python compile_master_best.py --root $Root --sort_by $metric --top_n $TopN
}
