# Runs the region_weight = {2, 4} x vds_loss = {0, 5} sweeps for avkf2 and freelam.
# Companion to run_new_sweeps.ps1. Run from anywhere: it cd's to its own folder first.
#   .\run_regionwt_2_4_sweeps.ps1            # 46 workers (default)
#   .\run_regionwt_2_4_sweeps.ps1 -Workers 60
#
# NOTE on size: each experiment sweeps region_weights=[2,4] x vds_losses=[0,5] (4 combos),
#   so:  avkf2  : 208 exp -> ~17472 runs        freelam: 56 exp -> ~4704 runs
# Output goes into the ORIGINAL master roots alongside the existing region_weight 0/1 runs.
# Run dirs are suffixed with _rw<weight> (multi_experiment_runner.py), so the new 2/4 runs
# get their own dirs and never collide with / overwrite / get skipped by the 0/1 runs.
#
# -Workers overrides each config's max_workers (capped at os.cpu_count() by the runner).
param([int]$Workers = 46)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# Measurement CSV (device IV data). Same default as run_new_sweeps.ps1 / compile_overall.ps1.
$csv = "..\csvs\cg2h40010_new_2.4_5_2_70W_center9.csv"

Write-Host "=== [1/2] refine_avkf2_gm_1 : region_weights=[2,4], box vgs -3..0 / vds 0..15 ===" -ForegroundColor Cyan
python multi_experiment_runner.py `
    --config physics_nn_configs\refine_avkf2_gm_1_regionwt_2_4.json `
    --csv $csv `
    --master_root_path ..\runs\refine_avkf2_gm_1 `
    --max_workers $Workers
if ($LASTEXITCODE -ne 0) {
    Write-Host "First sweep failed (exit $LASTEXITCODE); stopping before the second." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "=== [2/2] refine_vdsk_freelam_vdsloss_gm_1 : region_weights=[2,4] ===" -ForegroundColor Cyan
python multi_experiment_runner.py `
    --config physics_nn_configs\refine_vdsk_freelam_vdsloss_gm_1_regionwt_2_4.json `
    --csv $csv `
    --master_root_path ..\runs\refine_vdsk_freelam_vdsloss_gm_1 `
    --max_workers $Workers
if ($LASTEXITCODE -ne 0) {
    Write-Host "Second sweep failed (exit $LASTEXITCODE)." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "Both region_weight 2/4 sweeps finished." -ForegroundColor Green
