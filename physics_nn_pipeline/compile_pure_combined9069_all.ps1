# Runs compile_overall.ps1 on all 6 pure_combined9069 output-activation sweeps -- the 3
# gm-trained roots and their 3 Ids-only ("nogm") counterparts:
#   runs\pure_combined9069\tanh_margin10        runs\pure_combined9069\tanh_margin10_nogm
#   runs\pure_combined9069\sigmoid_margin10      runs\pure_combined9069\sigmoid_margin10_nogm
#   runs\pure_combined9069\softplus              runs\pure_combined9069\softplus_nogm
#
# Each compile_overall.ps1 call does the full pipeline: region metrics -> compile/rank/xlsx ->
# drop-all-zero filter (auto-skipped for "nogm" roots) -> quality filter
# (region_knee_combined_gm<=maxCombinedGm AND region_knee_ids_rmse<=maxIdsRmse,
# default 0.9/0.009) -> shape analysis on the quality-filtered results -> slim xlsx.
#
#   .\compile_pure_combined9069_all.ps1
#   .\compile_pure_combined9069_all.ps1 -MaxIdsRmse 0.012      # relax ids_rmse for sigmoid/softplus
#   .\compile_pure_combined9069_all.ps1 -SkipRegionMetrics     # re-derive/re-filter/re-shape-analyze
#                                                               # only -- skips the slow per-model step
param(
    [double]$MaxCombinedGm = 0.9,
    [double]$MaxIdsRmse    = 0.009,
    [switch]$SkipRegionMetrics
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$roots = @(
    "..\runs\pure_combined9069\tanh_margin10",
    "..\runs\pure_combined9069\tanh_margin10_nogm",
    "..\runs\pure_combined9069\sigmoid_margin10",
    "..\runs\pure_combined9069\sigmoid_margin10_nogm",
    "..\runs\pure_combined9069\softplus",
    "..\runs\pure_combined9069\softplus_nogm"
)

foreach ($root in $roots) {
    Write-Host "=== compile_overall: $root ===" -ForegroundColor Cyan
    & .\compile_overall.ps1 -root $root -maxCombinedGm $MaxCombinedGm -maxIdsRmse $MaxIdsRmse -SkipRegionMetrics:$SkipRegionMetrics
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED on $root (exit $LASTEXITCODE)." -ForegroundColor Red
        exit $LASTEXITCODE
    }
    Write-Host "done: $root" -ForegroundColor Green
}
Write-Host "all 6 pure_combined9069 roots compiled." -ForegroundColor Green
