# Re-runs the SAME combined 9069-architecture set (8586 from avkf2_id21677_8586mixed_gmvds3 +
# 483 from avkf2_id21677_483pct_gmvds3), pure NN equation, with use_gm=False (Ids-only training,
# no gm1/gm2/gm3 loss terms, no gm-aware L-BFGS polish) -- mirrors
# run_pure_combined9069_tanh_margin10_nogm_all_batches.ps1, for the other two output activations:
#   - sigmoid, with --ids_out_margin 0.1 (same margin fix as sigmoid_margin10/tanh_margin10,
#     needed because sigmoid's (0,1) range can't reach the data's real Ids without it)
#   - softplus, NO margin needed (unbounded, (0,inf) already covers the data range)
# Configs are byte-identical to their non-nogm sources except base_configs.PureNN.use_gm
# flipped true->false; see pure_combined9069_sigmoid_margin10_nogm_batch{1,2,3}.json and
# pure_combined9069_softplus_nogm_batch{1,2,3}.json.
#
# Same batching/accumulation pattern as the other pure_combined9069 runners: all 3 batches per
# activation land in ONE shared root (distinct "_nogm_batchN" experiment-name suffix avoids
# collision), compiled after every batch so results grow to the full 9069 as batches finish:
#   runs\pure_combined9069\sigmoid_margin10_nogm\
#   runs\pure_combined9069\softplus_nogm\
#
#   .\run_pure_combined9069_sigmoid_margin10_softplus_nogm_all_batches.ps1
#   .\run_pure_combined9069_sigmoid_margin10_softplus_nogm_all_batches.ps1 -Workers 60 -Csv ..\csvs\cg2h40010_new_2.4_5_2_70W_center9.csv
param(
    [int]$Workers = 46,
    [string]$Csv = "..\csvs\cg2h40010_new_2.4_5_2_70W_center9.csv"
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$configPrefixes = @{
    "sigmoid_margin10_nogm" = "pure_combined9069_sigmoid_margin10_nogm_batch"
    "softplus_nogm"         = "pure_combined9069_softplus_nogm_batch"
}
$roots = $configPrefixes.Keys | ForEach-Object { "..\runs\pure_combined9069\$_" }

foreach ($batch in 1, 2, 3) {
    foreach ($key in $configPrefixes.Keys) {
        $config = "..\physics_nn_configs\$($configPrefixes[$key])${batch}.json"
        $root   = "..\runs\pure_combined9069\$key"
        Write-Host "=== batch$batch $key -> $root ===" -ForegroundColor Cyan
        python multi_experiment_runner.py `
            --config $config `
            --csv $Csv `
            --master_root_path $root `
            --max_workers $Workers
        if ($LASTEXITCODE -ne 0) {
            Write-Host "FAILED on batch$batch/$key (exit $LASTEXITCODE)." -ForegroundColor Red
            exit $LASTEXITCODE
        }
        Write-Host "done: batch$batch/$key" -ForegroundColor Green
    }
    Write-Host "batch$batch finished. compiling..." -ForegroundColor Green
    foreach ($root in $roots) {
        Write-Host "=== compile $root ===" -ForegroundColor Cyan
        & .\compile_overall.ps1 -root $root
        Write-Host "compiled: $root" -ForegroundColor Green
    }
}
Write-Host "all batches, sigmoid_margin10_nogm+softplus_nogm done." -ForegroundColor Green
