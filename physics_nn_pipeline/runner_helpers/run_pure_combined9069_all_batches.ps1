# Runs all 3 batches (3023 archs each, 9069 total) x 3 output_activations (sigmoid, tanh,
# softplus) in one go. Each batch's experiment name has a distinct "_batchN" suffix, so all
# 3 batches accumulate into ONE shared root per activation (no collision) -- compiled after
# every batch, so the ranked files grow to cover more archs as each batch finishes:
#   runs\pure_combined9069\sigmoid\   (grows: 3023 -> 6046 -> 9069 archs covered)
#   runs\pure_combined9069\tanh\
#   runs\pure_combined9069\softplus\
#
#   .\run_pure_combined9069_all_batches.ps1
#   .\run_pure_combined9069_all_batches.ps1 -Workers 60 -Csv ..\csvs\cg2h40010_new_2.4_5_2_70W_center9.csv
param(
    [int]$Workers = 46,
    [string]$Csv = "..\csvs\cg2h40010_new_2.4_5_2_70W_center9.csv"
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

$roots = @("..\runs\pure_combined9069\sigmoid", "..\runs\pure_combined9069\tanh", "..\runs\pure_combined9069\softplus")

foreach ($batch in 1, 2, 3) {
    foreach ($act in @("sigmoid", "tanh", "softplus")) {
        $config = "..\physics_nn_configs\pure_combined9069_${act}_batch${batch}_gmvds3.json"
        $root   = "..\runs\pure_combined9069\$act"
        Write-Host "=== batch$batch output_activation=$act -> $root ===" -ForegroundColor Cyan
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
Write-Host "all batches, all output_activations done." -ForegroundColor Green
