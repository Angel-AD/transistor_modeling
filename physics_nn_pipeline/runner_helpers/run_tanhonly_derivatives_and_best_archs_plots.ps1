# Dedicated tanh-only derivation + training + best_archs_plots, for the 8 ORIGINAL folders
# (sigmoid_margin10, sigmoid_margin10_nogm, softplus, softplus_nogm, tanh_margin10,
# tanh_margin10_nogm, vdsgate_aeff_quad_tanhm, vdsgate_aeff_quad_v3).
#
# Uses the ORIGINAL (non-simplegate) scripts throughout -- these 8 folders were never trained
# with the SIMPLEGATE convention, so no *_simplegate.py forks are needed here.
#
# Unlike the vdsgate_v3/vdsgate_tanhm pipeline, there's no "compile the base sweep" step: these
# 8 folders' base 9069 sweep was already compiled long ago (that's how csv_base_2.5_20_rw4 was
# originally built) -- extract_derived_configs.py's --tanh_only just reads that existing data.
#
# Output folder names get a "_tanhonly" suffix throughout (e.g. sigmoid_margin10_tanhonly), so
# nothing here ever collides with the existing heterogeneous derived configs/training/analysis.
#
# Usage:
#   .\run_tanhonly_derivatives_and_best_archs_plots.ps1
#   .\run_tanhonly_derivatives_and_best_archs_plots.ps1 -Workers 16
#   .\run_tanhonly_derivatives_and_best_archs_plots.ps1 -SkipDerive -SkipTrainAll6Csvs   # just rerun shape+plot

param(
    [string]$RunsDir     = "C:\Users\acost\repos\new_opts_2\runs",
    [string]$CsvsDir     = "C:\Users\acost\repos\new_opts_2\csvs",
    [string]$SourceRoot  = "C:\Users\acost\repos\new_opts_2\runs\pure_combined9069_rw4_2.5_20",
    [string]$CsvBaseRoot = "C:\Users\acost\repos\new_opts_2\runs\csv_base_2.5_20_rw4",
    [string]$Suffix       = "rw4_2.5_20",
    [string]$OutputParent = "csv_base_2.5_20_rw4",
    [string[]]$SourceFolders = @(
        "sigmoid_margin10", "sigmoid_margin10_nogm", "softplus", "softplus_nogm",
        "tanh_margin10", "tanh_margin10_nogm", "vdsgate_aeff_quad_tanhm", "vdsgate_aeff_quad_v3"
    ),
    [string[]]$BaseConfigs = @(
        "best200ids_of_9069_byloss_rw4_2.5_20",
        "bothshapeok_of_9069_byshape_rw4_2.5_20",
        "best100_gmshapeok_of_9069_byloss_rw4_2.5_20"
    ),
    [int]$Workers = 16,
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

if (-not $SkipDerive) {
    Write-Host "`n=== [1/4] extract_derived_configs.py --tanh_only ===" -ForegroundColor Cyan
    $best200Out = Join-Path $RunsDir "best200ids_of_9069_byloss_$Suffix\_configs"
    $bothshapeokOut = Join-Path $RunsDir "bothshapeok_of_9069_byshape_$Suffix\_configs"
    $best100Out = Join-Path $RunsDir "best100_gmshapeok_of_9069_byloss_$Suffix\_configs"
    python extract_derived_configs.py --source_root $SourceRoot --folders $sourceFolderList --tanh_only `
        --best200_configs_out $best200Out --bothshapeok_configs_out $bothshapeokOut `
        --best100_gmshapeok_configs_out $best100Out
    if ($LASTEXITCODE -ne 0) { throw "extract_derived_configs.py failed (exit $LASTEXITCODE)" }
} else {
    Write-Host "`n=== [1/4] SKIPPED (-SkipDerive) ===" -ForegroundColor Yellow
}

if (-not $SkipTrainAll6Csvs) {
    Write-Host "`n=== [2/4] train derived configs across all 6 csvs + compile (original, non-simplegate) ===" -ForegroundColor Cyan
    & .\runner_helpers\run_derived_configs_from_pure_combined9069_rw0_2.5_20_all6csvs.ps1 `
        -Suffix $Suffix -OutputParent $OutputParent -RunsDir $RunsDir -CsvsDir $CsvsDir `
        -Folders $tanhonlyFolders -Workers $Workers
    if ($LASTEXITCODE -ne 0) { throw "all6csvs training driver failed (exit $LASTEXITCODE)" }
} else {
    Write-Host "`n=== [2/4] SKIPPED (-SkipTrainAll6Csvs) ===" -ForegroundColor Yellow
}

if (-not $SkipShapeAnalysis) {
    Write-Host "`n=== [3/4] run_shape_analysis_csv_base.py ===" -ForegroundColor Cyan
    python run_shape_analysis_csv_base.py --csv_base_root $CsvBaseRoot --base_configs $baseConfigList --folders $tanhonlyFolderList
    if ($LASTEXITCODE -ne 0) { throw "run_shape_analysis_csv_base.py failed (exit $LASTEXITCODE)" }
} else {
    Write-Host "`n=== [3/4] SKIPPED (-SkipShapeAnalysis) ===" -ForegroundColor Yellow
}

if (-not $SkipPlotConsistentArchs) {
    Write-Host "`n=== [4/4] run_plot_consistent_archs.py ===" -ForegroundColor Cyan
    python run_plot_consistent_archs.py --csv_base_root $CsvBaseRoot --base_configs $baseConfigList --folders $tanhonlyFolderList --workers $Workers
    if ($LASTEXITCODE -ne 0) { throw "run_plot_consistent_archs.py failed (exit $LASTEXITCODE)" }
} else {
    Write-Host "`n=== [4/4] SKIPPED (-SkipPlotConsistentArchs) ===" -ForegroundColor Yellow
}

if (-not $SkipHtml) {
    Write-Host "`n=== HTML ===" -ForegroundColor Cyan
    python render_html_view.py --out_root (Join-Path $CsvBaseRoot "best_archs_plots")
    if ($LASTEXITCODE -ne 0) { throw "render_html_view.py failed (exit $LASTEXITCODE)" }
} else {
    Write-Host "`n=== HTML SKIPPED (-SkipHtml) ===" -ForegroundColor Yellow
}

Write-Host "`nall done." -ForegroundColor Green
