# Runs ONLY the freelam region_weight={2,4} x vds_loss={0,5} sweep (56 exp -> ~4704 runs).
# Intended for offloading freelam to a second machine while another runs avkf2.
# Run from anywhere: it cd's to its own folder first.
#   .\run_freelam_only.ps1                # 46 workers (default)
#   .\run_freelam_only.ps1 -Workers 32    # match the other machine's core count
#
# -Workers overrides the config's max_workers (capped at os.cpu_count() by the runner).
param([int]$Workers = 46)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$csv = "..\csvs\cg2h40010_new_2.4_5_2_70W_center9.csv"

Write-Host "=== refine_vdsk_freelam_vdsloss_gm_1 : region_weights=[2,4] x vds_losses=[0,5] ===" -ForegroundColor Cyan
python multi_experiment_runner.py `
    --config ..\physics_nn_configs\refine_vdsk_freelam_vdsloss_gm_1_regionwt_2_4.json `
    --csv $csv `
    --master_root_path ..\runs\refine_vdsk_freelam_vdsloss_gm_1 `
    --max_workers $Workers
if ($LASTEXITCODE -ne 0) { Write-Host "freelam sweep failed (exit $LASTEXITCODE)." -ForegroundColor Red; exit $LASTEXITCODE }
Write-Host "freelam sweep finished." -ForegroundColor Green
