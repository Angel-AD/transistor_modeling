# Runs ONLY the avkf2 region_weight={2,4} x vds_loss={0,5} sweep (208 exp -> ~17472 runs).
# Use on the machine handling avkf2 while freelam runs elsewhere (run_freelam_only.ps1).
# Resumable: finished runs (dir has run_loss_* AND weights_loss_*) are skipped.
# Run from anywhere: it cd's to its own folder first.
#   .\run_avkf2_only.ps1                # 46 workers (default)
#   .\run_avkf2_only.ps1 -Workers 60
#
# -Workers overrides the config's max_workers (capped at os.cpu_count() by the runner).
param([int]$Workers = 46)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$csv = "..\csvs\cg2h40010_new_2.4_5_2_70W_center9.csv"

Write-Host "=== refine_avkf2_gm_1 : region_weights=[2,4] x vds_losses=[0,5] ===" -ForegroundColor Cyan
python multi_experiment_runner.py `
    --config physics_nn_configs\refine_avkf2_gm_1_regionwt_2_4.json `
    --csv $csv `
    --master_root_path ..\runs\refine_avkf2_gm_1 `
    --max_workers $Workers
if ($LASTEXITCODE -ne 0) { Write-Host "avkf2 sweep failed (exit $LASTEXITCODE)." -ForegroundColor Red; exit $LASTEXITCODE }
Write-Host "avkf2 sweep finished." -ForegroundColor Green
