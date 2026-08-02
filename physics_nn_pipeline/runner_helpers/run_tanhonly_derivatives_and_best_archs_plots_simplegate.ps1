# Dedicated tanh-only derivation + training + best_archs_plots, for the SIMPLEGATE folders
# (vdsgate_v3, vdsgate_tanhm). Mirrors run_tanhonly_derivatives_and_best_archs_plots.ps1
# (the original/old-mechanism version) but uses the *_simplegate.py/*_simplegate.ps1 scripts
# throughout, since vdsgate_v3/vdsgate_tanhm were trained with the SIMPLEGATE convention
# (output_activation IS the gate, no separate vdsgate_output_activation key) -- the ORIGINAL
# scripts would silently reconstruct their models wrong (double-squash/mismatch).
#
# Includes the "compile the base sweep" step (like
# run_vdsgate_v3_tanhm_derivatives_and_best_archs_plots.ps1) since that's a prerequisite for
# extract_derived_configs_simplegate.py's --tanh_only. Safe/cheap to re-run even if already done
# -- skip with -SkipCompileBaseSweep if you know it's current.
#
# Usage:
#   .\run_tanhonly_derivatives_and_best_archs_plots_simplegate.ps1
#   .\run_tanhonly_derivatives_and_best_archs_plots_simplegate.ps1 -Workers 16
#   .\run_tanhonly_derivatives_and_best_archs_plots_simplegate.ps1 -SkipCompileBaseSweep -SkipDerive -SkipTrainAll6Csvs

param(
    [string]$RunsDir     = "C:\Users\acost\repos\new_opts_2\runs",
    [string]$CsvsDir     = "C:\Users\acost\repos\new_opts_2\csvs",
    [string]$SourceRoot  = "C:\Users\acost\repos\new_opts_2\runs\pure_combined9069_rw4_2.5_20",
    [string]$CsvBaseRoot = "C:\Users\acost\repos\new_opts_2\runs\csv_base_2.5_20_rw4",
    [string]$Suffix       = "rw4_2.5_20",
    [string]$OutputParent = "csv_base_2.5_20_rw4",
    [string[]]$SourceFolders = @("vdsgate_v3", "vdsgate_tanhm"),
    [string[]]$BaseConfigs = @(
        "best200ids_of_9069_byloss_rw4_2.5_20",
        "bothshapeok_of_9069_byshape_rw4_2.5_20",
        "best100_gmshapeok_of_9069_byloss_rw4_2.5_20"
    ),
    [int]$Workers = 16,
    [switch]$SkipCompileBaseSweep,
    [switch]$SkipDerive,
    [switch]$SkipTrainAll6Csvs,
    [switch]$SkipShapeAnalysis,
    [switch]$SkipPlotConsistentArchs,
    [switch]$SkipHtml
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)   # -> physics_nn_pipeline\

$tanhonlyFolders = $SourceFolders | ForEach-Object { "$_`_tanhonly" }
$sourceFolderList = $SourceFolders -join ","
$tanhonlyFolderList = $tanhonlyFolders -join ","
$baseConfigList = $BaseConfigs -join ","

if (-not $SkipCompileBaseSweep) {
    Write-Host "`n=== [1/5] compile the base 9069 sweep itself (region metrics -> ranked csv, simplegate) ===" -ForegroundColor Cyan
    foreach ($folder in $SourceFolders) {
        $root = Join-Path $SourceRoot $folder
        Write-Host "  compile_overall_simplegate: $root" -ForegroundColor Cyan
        & .\compile_helpers\compile_overall_simplegate.ps1 -root $root -ShortName
        if ($LASTEXITCODE -ne 0) { throw "compile_overall_simplegate.ps1 failed for $root (exit $LASTEXITCODE)" }
    }
} else {
    Write-Host "`n=== [1/5] SKIPPED (-SkipCompileBaseSweep) ===" -ForegroundColor Yellow
}

if (-not $SkipDerive) {
    Write-Host "`n=== [2/5] extract_derived_configs_simplegate.py --tanh_only ===" -ForegroundColor Cyan
    $best200Out = Join-Path $RunsDir "best200ids_of_9069_byloss_$Suffix\_configs"
    $bothshapeokOut = Join-Path $RunsDir "bothshapeok_of_9069_byshape_$Suffix\_configs"
    $best100Out = Join-Path $RunsDir "best100_gmshapeok_of_9069_byloss_$Suffix\_configs"
    python extract_derived_configs_simplegate.py --source_root $SourceRoot --folders $sourceFolderList --tanh_only `
        --best200_configs_out $best200Out --bothshapeok_configs_out $bothshapeokOut `
        --best100_gmshapeok_configs_out $best100Out
    if ($LASTEXITCODE -ne 0) { throw "extract_derived_configs_simplegate.py failed (exit $LASTEXITCODE)" }
} else {
    Write-Host "`n=== [2/5] SKIPPED (-SkipDerive) ===" -ForegroundColor Yellow
}

if (-not $SkipTrainAll6Csvs) {
    Write-Host "`n=== [3/5] train derived configs across all 6 csvs + compile (simplegate) ===" -ForegroundColor Cyan
    & .\runner_helpers\run_derived_configs_from_pure_combined9069_rw0_2.5_20_all6csvs_simplegate.ps1 `
        -Suffix $Suffix -OutputParent $OutputParent -RunsDir $RunsDir -CsvsDir $CsvsDir `
        -Folders $tanhonlyFolders -Workers $Workers
    if ($LASTEXITCODE -ne 0) { throw "all6csvs training driver (simplegate) failed (exit $LASTEXITCODE)" }
} else {
    Write-Host "`n=== [3/5] SKIPPED (-SkipTrainAll6Csvs) ===" -ForegroundColor Yellow
}

if (-not $SkipShapeAnalysis) {
    Write-Host "`n=== [4/5] run_shape_analysis_csv_base_simplegate.py ===" -ForegroundColor Cyan
    python run_shape_analysis_csv_base_simplegate.py --csv_base_root $CsvBaseRoot --base_configs $baseConfigList --folders $tanhonlyFolderList
    if ($LASTEXITCODE -ne 0) { throw "run_shape_analysis_csv_base_simplegate.py failed (exit $LASTEXITCODE)" }
} else {
    Write-Host "`n=== [4/5] SKIPPED (-SkipShapeAnalysis) ===" -ForegroundColor Yellow
}

if (-not $SkipPlotConsistentArchs) {
    Write-Host "`n=== [5/5] run_plot_consistent_archs_simplegate.py ===" -ForegroundColor Cyan
    python run_plot_consistent_archs_simplegate.py --csv_base_root $CsvBaseRoot --base_configs $baseConfigList --folders $tanhonlyFolderList --workers $Workers
    if ($LASTEXITCODE -ne 0) { throw "run_plot_consistent_archs_simplegate.py failed (exit $LASTEXITCODE)" }
} else {
    Write-Host "`n=== [5/5] SKIPPED (-SkipPlotConsistentArchs) ===" -ForegroundColor Yellow
}

if (-not $SkipHtml) {
    Write-Host "`n=== HTML ===" -ForegroundColor Cyan
    python render_html_view.py --out_root (Join-Path $CsvBaseRoot "best_archs_plots")
    if ($LASTEXITCODE -ne 0) { throw "render_html_view.py failed (exit $LASTEXITCODE)" }
} else {
    Write-Host "`n=== HTML SKIPPED (-SkipHtml) ===" -ForegroundColor Yellow
}

Write-Host "`nall done." -ForegroundColor Green
