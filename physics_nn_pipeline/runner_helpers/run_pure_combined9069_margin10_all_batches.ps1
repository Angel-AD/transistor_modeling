# Re-runs the SAME combined 9069-architecture set (8586 from avkf2_id21677_8586mixed_gmvds3 +
# 483 from avkf2_id21677_483pct_gmvds3), pure NN equation, sigmoid + tanh output activations
# ONLY (softplus doesn't need this fix -- it's unbounded, its earlier results already worked
# correctly and are untouched here). Adds --ids_out_margin 0.1: the model now predicts
# ids_scale*activation(NN(...)) with ids_scale=1.1*max(Ids), so sigmoid's (0,1) and tanh's
# (-1,1) output can actually reach the data's real range (up to ~2.7A) instead of saturating
# at 1.0 -- see bestpicks10_run_ids.md / this session's discussion for why the ORIGINAL
# sigmoid/tanh sweep (runs\pure_combined9069\sigmoid|tanh\) produced bad results.
#
# Same batching/accumulation pattern as run_pure_combined9069_all_batches.ps1: all 3 batches
# per activation land in ONE shared root (distinct "_margin10_batchN" experiment-name suffix
# avoids collision), compiled after every batch so results grow to the full 9069 as batches
# finish:
#   runs\pure_combined9069\sigmoid_margin10\
#   runs\pure_combined9069\tanh_margin10\
#
#   .\run_pure_combined9069_margin10_all_batches.ps1
#   .\run_pure_combined9069_margin10_all_batches.ps1 -Workers 60 -Csv ..\csvs\cg2h40010_new_2.4_5_2_70W_center9.csv
param(
    [int]$Workers = 46,
    [string]$Csv = "..\csvs\cg2h40010_new_2.4_5_2_70W_center9.csv"
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

$roots = @("..\runs\pure_combined9069\sigmoid_margin10", "..\runs\pure_combined9069\tanh_margin10")

foreach ($batch in 1, 2, 3) {
    foreach ($act in @("sigmoid", "tanh")) {
        $config = "..\physics_nn_configs\pure_combined9069_${act}_margin10_batch${batch}_gmvds3.json"
        $root   = "..\runs\pure_combined9069\${act}_margin10"
        Write-Host "=== batch$batch output_activation=$act (margin=0.1) -> $root ===" -ForegroundColor Cyan
        python multi_experiment_runner.py `
            --config $config `
            --csv $Csv `
            --master_root_path $root `
            --max_workers $Workers
        if ($LASTEXITCODE -ne 0) {
            Write-Host "FAILED on batch$batch/$act (exit $LASTEXITCODE)." -ForegroundColor Red
            exit $LASTEXITCODE
        }
        Write-Host "done: batch$batch/$act" -ForegroundColor Green
    }
    Write-Host "batch$batch finished. compiling..." -ForegroundColor Green
    foreach ($root in $roots) {
        Write-Host "=== compile $root ===" -ForegroundColor Cyan
        & .\compile_helpers\compile_overall.ps1 -root $root
        Write-Host "compiled: $root" -ForegroundColor Green
    }
}
Write-Host "all batches, sigmoid+tanh (margin=0.1) done." -ForegroundColor Green
