# Re-runs ONLY the shape-analysis step (analyze_shape.py + its slim xlsx) on the 6
# pure_combined9069 folders' already-compiled quality-filtered results -- without redoing
# region metrics / ranking / filtering (those don't change when analyze_shape.py's own
# defaults, e.g. --gm3_min_windows/--gm3_max_windows, are tuned). Mirrors step 3c of
# compile_overall.ps1 but as a standalone, targeted re-run.
#
#   .\run_shape_analysis_all.ps1
#   .\run_shape_analysis_all.ps1 -GmMinWindows "-4..-3.5,-3.5..-1.7" -GmMaxWindows "-4..-2.2"
#   .\run_shape_analysis_all.ps1 -GmStartTol 1.5
param(
    [string]$GmMinWindows = "",
    [string]$GmMaxWindows = "",
    [double]$GmStartTol = 0
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

$shapeKeep = @("id","run_id","arch_id","arch_hash","vds_loss","region_knee_combined_gm",
               "region_knee_ids_rmse","neg_gds_energy","gds_hump","gds_hump_max",
               "gm3_maxabs","gm3_bad_frac","gm3_start_diff","shape_rank")

$extraArgs = @()
if ($GmMinWindows) { $extraArgs += @("--gm3_min_windows", $GmMinWindows) }
if ($GmMaxWindows) { $extraArgs += @("--gm3_max_windows", $GmMaxWindows) }
if ($GmStartTol -gt 0) { $extraArgs += @("--gm3_start_tol", $GmStartTol) }

foreach ($root in $roots) {
    if (-not (Test-Path $root)) {
        Write-Host "skip (not found): $root" -ForegroundColor Yellow
        continue
    }
    Write-Host "=== $root ===" -ForegroundColor Cyan
    $shapeInputs = Get-ChildItem -Path $root -Directory -Filter "ranked_region_*" |
        Where-Object { $_.Name -match "_vgs.*_vds" } |
        ForEach-Object { Get-ChildItem -Path $_.FullName -Filter "*_combined_gm_filtered_quality.csv" }
    if (-not $shapeInputs) {
        Write-Host "  no *_combined_gm_filtered_quality.csv found (run compile_overall.ps1 first)." -ForegroundColor Yellow
        continue
    }
    foreach ($f in $shapeInputs) {
        $nLines = (Get-Content $f.FullName | Measure-Object -Line).Lines
        if ($nLines -le 1) {
            Write-Host "  skip shape analysis (0 rows): $($f.FullName)" -ForegroundColor Yellow
            continue
        }
        Write-Host "  --- shape analysis: $($f.FullName) ---" -ForegroundColor Cyan
        python analyze_shape.py --ranked_csv $f.FullName @extraArgs
        if ($LASTEXITCODE -ne 0) {
            Write-Host "FAILED on $($f.FullName) (exit $LASTEXITCODE)." -ForegroundColor Red
            exit $LASTEXITCODE
        }
        $shapeOutputs = Get-ChildItem -Path $f.Directory.FullName -Filter "*_shape*.csv" | ForEach-Object { $_.FullName }
        if ($shapeOutputs) { python keep_columns.py --csv $shapeOutputs --keep $shapeKeep }
    }
}
Write-Host "done: shape analysis refreshed for all 6 pure_combined9069 folders." -ForegroundColor Green
