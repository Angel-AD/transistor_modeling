# Runs compile_overall.ps1's -TopN/-SortBy best-N filter (instead of the threshold-based
# -maxCombinedGm/-maxIdsRmse quality filter) on all 6 pure_combined9069 sweeps, once per
# --sort_by metric (default: region_knee_combined_gm and region_knee_ids_rmse) -- so each
# sweep ends up with TWO best-N filter results, auto-numbered (via filter_readme.md) so they
# coexist instead of overwriting each other, plus shape analysis run on each (compile_overall.ps1's
# step 3c always shape-analyzes whatever filter number step 3b just assigned/reused).
#
# -SkipRegionMetrics is always passed: region metrics are assumed already computed by a prior
# full compile (e.g. compile_pure_combined9069_all.ps1) -- this only re-derives/re-filters/
# re-shape-analyzes the ranked CSVs. -ShortName is always passed too, matching the rest of this
# sweep family (long filter -> shape -> gmshape_ok -> slim chains hit Excel's/Windows' path-length
# limit otherwise -- tanh_margin10/tanh_margin10_nogm already use it; this will additionally mint
# short-named ranked files for sigmoid_margin10(_nogm)/softplus(_nogm), which don't have them yet
# -- their old long-named ranked files are left in place, not deleted).
#
#   .\run_topn_filter_all.ps1
#   .\run_topn_filter_all.ps1 -TopN 30
#   .\run_topn_filter_all.ps1 -SortByMetrics region_knee_combined_gm     # just one metric
param(
    [int]$TopN = 20,
    [string[]]$SortByMetrics = @("region_knee_combined_gm", "region_knee_ids_rmse")
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

$roots = @(
    "..\runs\pure_combined9069\tanh_margin10",
    "..\runs\pure_combined9069\tanh_margin10_nogm",
    "..\runs\pure_combined9069\sigmoid_margin10",
    "..\runs\pure_combined9069\sigmoid_margin10_nogm",
    "..\runs\pure_combined9069\softplus",
    "..\runs\pure_combined9069\softplus_nogm"
)

foreach ($root in $roots) {
    foreach ($metric in $SortByMetrics) {
        Write-Host "=== compile_overall: $root  (top $TopN by $metric) ===" -ForegroundColor Cyan
        & .\compile_helpers\compile_overall.ps1 -root $root -SkipRegionMetrics -ShortName -TopN $TopN -SortBy $metric
        if ($LASTEXITCODE -ne 0) {
            Write-Host "FAILED on $root / $metric (exit $LASTEXITCODE)." -ForegroundColor Red
            exit $LASTEXITCODE
        }
    }
    Write-Host "done: $root" -ForegroundColor Green
}
Write-Host "all 6 pure_combined9069 roots: top-$TopN filter + shape analysis done for $($SortByMetrics -join ', ')." -ForegroundColor Green
