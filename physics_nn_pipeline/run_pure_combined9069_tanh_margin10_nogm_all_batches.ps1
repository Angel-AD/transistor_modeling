# Re-runs the SAME combined 9069-architecture set (8586 from avkf2_id21677_8586mixed_gmvds3 +
# 483 from avkf2_id21677_483pct_gmvds3), pure NN equation, tanh output activation, --ids_out_margin
# 0.1 (same as tanh_margin10) -- but with use_gm=False, i.e. Ids-only training (no gm1/gm2/gm3
# loss terms, no gm-aware L-BFGS polish). Configs are byte-identical to
# pure_combined9069_tanh_margin10_batch{1,2,3}_gmvds3.json except base_configs.PureNN.use_gm
# flipped true->false; see pure_combined9069_tanh_margin10_nogm_batch{1,2,3}.json.
#
# Same batching/accumulation pattern as run_pure_combined9069_margin10_all_batches.ps1: all 3
# batches land in ONE shared root (distinct "_nogm_batchN" experiment-name suffix avoids
# collision), compiled after every batch so results grow to the full 9069 as batches finish:
#   runs\pure_combined9069\tanh_margin10_nogm\
#
#   .\run_pure_combined9069_tanh_margin10_nogm_all_batches.ps1
#   .\run_pure_combined9069_tanh_margin10_nogm_all_batches.ps1 -Workers 60 -Csv ..\csvs\cg2h40010_new_2.4_5_2_70W_center9.csv
param(
    [int]$Workers = 46,
    [string]$Csv = "..\csvs\cg2h40010_new_2.4_5_2_70W_center9.csv"
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$root = "..\runs\pure_combined9069\tanh_margin10_nogm"

foreach ($batch in 1, 2, 3) {
    $config = "..\physics_nn_configs\pure_combined9069_tanh_margin10_nogm_batch${batch}.json"
    Write-Host "=== batch$batch output_activation=tanh (margin=0.1, use_gm=False) -> $root ===" -ForegroundColor Cyan
    python multi_experiment_runner.py `
        --config $config `
        --csv $Csv `
        --master_root_path $root `
        --max_workers $Workers
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED on batch$batch (exit $LASTEXITCODE)." -ForegroundColor Red
        exit $LASTEXITCODE
    }
    Write-Host "done: batch$batch" -ForegroundColor Green
    Write-Host "batch$batch finished. compiling..." -ForegroundColor Green
    Write-Host "=== compile $root ===" -ForegroundColor Cyan
    & .\compile_overall.ps1 -root $root
    Write-Host "compiled: $root" -ForegroundColor Green
}
Write-Host "all batches, tanh_margin10_nogm done." -ForegroundColor Green
