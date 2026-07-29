# Runs the two new sweeps one after the other.
# Run from anywhere: it cd's to its own folder (physics_nn_pipeline) first.
#   .\run_new_sweeps.ps1
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

# Measurement CSV (device IV data). Default taken from compile_overall.ps1 — change if needed.
$csv = "..\csvs\cg2h40010_new_2.4_5_2_70W_center9.csv"

Write-Host "=== [1/2] refine_avkf2_gm_1 : gm_vds_min=3.8  (208 exp / 4368 runs) ===" -ForegroundColor Cyan
python multi_experiment_runner.py `
    --config ..\physics_nn_configs\refine_avkf2_gm_1_gmvdsmin_3p8.json `
    --csv $csv `
    --master_root_path ..\runs\refine_avkf2_gm_1
if ($LASTEXITCODE -ne 0) {
    Write-Host "First sweep failed (exit $LASTEXITCODE); stopping before the second." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "=== [2/2] refine_vdsk_freelam_vdsloss_gm_1 : gm_vds_min=4, region_vgs_lo=-3  (56 exp / 1176 runs) ===" -ForegroundColor Cyan
python multi_experiment_runner.py `
    --config ..\physics_nn_configs\refine_vdsk_freelam_vdsloss_gm_1_gmvds4_vgslo3.json `
    --csv $csv `
    --master_root_path ..\runs\refine_vdsk_freelam_vdsloss_gm_1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Second sweep failed (exit $LASTEXITCODE)." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "Both sweeps finished." -ForegroundColor Green
