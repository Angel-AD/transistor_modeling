# End-to-end pipeline for the two SIMPLEGATE-trained folders (vdsgate_v3, vdsgate_tanhm):
# derive the 3 "best ids" config sets from the base 9069 sweep -> train them across all 6
# measurement csvs (+ compile) -> shape analysis -> search/plot/summarize into best_archs_plots
# -> HTML twins.
#
# Every step here uses the *_simplegate variant of the relevant script, because these two
# folders were trained with equation_type: "pure:vdsgate"/"pure:vdsgate_aeff_quad" +
# output_activation directly encoding the gate (softplus/tanh), with NO
# vdsgate_output_activation key in their run_loss_*.json. The ORIGINAL scripts
# (plot_saved_state.py et al.) would default the missing key to 'softplus' and apply it on TOP
# of output_activation, double-squashing vdsgate_v3 and producing a mismatched
# softplus(tanh(...)) for vdsgate_tanhm -- silently wrong region metrics/shape/plots. See the
# *_simplegate.py forks this script calls (compute_region_metrics_simplegate.py,
# analyze_shape_simplegate.py, plot_arch_hash_simplegate.py, extract_derived_configs_simplegate.py,
# run_shape_analysis_csv_base_simplegate.py, run_plot_consistent_archs_simplegate.py,
# compile_overall_simplegate.ps1) for the exact redirected imports.
#
# The other 8 folders (sigmoid_margin10, ..., vdsgate_aeff_quad_v3/tanhm) are NOT touched by
# this script at all -- they were trained with the original mechanism and must keep going
# through the original (non-simplegate) scripts, which is what the existing
# run_derived_configs_from_pure_combined9069_rw0_2.5_20_all6csvs.ps1 / run_shape_analysis_csv_base.py /
# run_plot_consistent_archs.py already do for them.
#
# Every step is resumable (skips already-done leaves/exp dirs), so safe to re-run after adding
# more architectures or if it's interrupted partway.
#
# Usage (defaults match this project's actual new_opts_2 setup):
#   .\run_vdsgate_v3_tanhm_derivatives_and_best_archs_plots.ps1
#   .\run_vdsgate_v3_tanhm_derivatives_and_best_archs_plots.ps1 -Workers 16
#   .\run_vdsgate_v3_tanhm_derivatives_and_best_archs_plots.ps1 -SkipDerive -SkipTrainAll6Csvs   # just rerun shape+plot

param(
    [string]$RunsDir       = "C:\Users\acost\repos\new_opts_2\runs",
    [string]$CsvsDir       = "C:\Users\acost\repos\new_opts_2\csvs",
    [string]$SourceRoot    = "C:\Users\acost\repos\new_opts_2\runs\pure_combined9069_rw4_2.5_20",
    [string]$CsvBaseRoot   = "C:\Users\acost\repos\new_opts_2\runs\csv_base_2.5_20_rw4",
    [string]$Suffix        = "rw4_2.5_20",
    [string]$OutputParent  = "csv_base_2.5_20_rw4",
    [string[]]$Folders     = @("vdsgate_v3", "vdsgate_tanhm"),
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

$folderList = $Folders -join ","
$baseConfigList = $BaseConfigs -join ","

if (-not $SkipCompileBaseSweep) {
    Write-Host "`n=== [1/6] compile the base 9069 sweep itself (region metrics -> ranked csv) ===" -ForegroundColor Cyan
    # extract_derived_configs_simplegate.py (next step) needs a
    # ranked_region_knee_vgs-3to0_vds0to15/*_ranked_by_region_knee_combined_gm.csv under each
    # folder -- this is what produces it. The raw exp_NNN training results alone are not enough.
    foreach ($folder in $Folders) {
        $root = Join-Path $SourceRoot $folder
        Write-Host "  compile_overall_simplegate: $root" -ForegroundColor Cyan
        & .\compile_helpers\compile_overall_simplegate.ps1 -root $root -ShortName
        if ($LASTEXITCODE -ne 0) { throw "compile_overall_simplegate.ps1 failed for $root (exit $LASTEXITCODE)" }
    }
} else {
    Write-Host "`n=== [1/6] SKIPPED (-SkipCompileBaseSweep) ===" -ForegroundColor Yellow
}

if (-not $SkipDerive) {
    Write-Host "`n=== [2/6] extract_derived_configs_simplegate.py ===" -ForegroundColor Cyan
    # Explicit --*_configs_out: this script's own default output root is <repo>/runs/... (relative
    # to ITS OWN location, i.e. transistor_modeling\runs\), NOT derived from --source_root or
    # -RunsDir -- without these three flags it writes _configs\ to the wrong repo entirely, and
    # step 3 below (which DOES read from $RunsDir) would never find them.
    $best200Out = Join-Path $RunsDir "best200ids_of_9069_byloss_$Suffix\_configs"
    $bothshapeokOut = Join-Path $RunsDir "bothshapeok_of_9069_byshape_$Suffix\_configs"
    $best100Out = Join-Path $RunsDir "best100_gmshapeok_of_9069_byloss_$Suffix\_configs"
    python extract_derived_configs_simplegate.py --source_root $SourceRoot --folders $folderList `
        --best200_configs_out $best200Out --bothshapeok_configs_out $bothshapeokOut `
        --best100_gmshapeok_configs_out $best100Out
    if ($LASTEXITCODE -ne 0) { throw "extract_derived_configs_simplegate.py failed (exit $LASTEXITCODE)" }
} else {
    Write-Host "`n=== [2/6] SKIPPED (-SkipDerive) ===" -ForegroundColor Yellow
}

if (-not $SkipTrainAll6Csvs) {
    Write-Host "`n=== [3/6] train derived configs across all 6 csvs + compile (simplegate) ===" -ForegroundColor Cyan
    & .\runner_helpers\run_derived_configs_from_pure_combined9069_rw0_2.5_20_all6csvs_simplegate.ps1 `
        -Suffix $Suffix -OutputParent $OutputParent -RunsDir $RunsDir -CsvsDir $CsvsDir `
        -Folders $Folders -Workers $Workers
    if ($LASTEXITCODE -ne 0) { throw "all6csvs training driver failed (exit $LASTEXITCODE)" }
} else {
    Write-Host "`n=== [3/6] SKIPPED (-SkipTrainAll6Csvs) ===" -ForegroundColor Yellow
}

if (-not $SkipShapeAnalysis) {
    Write-Host "`n=== [4/6] run_shape_analysis_csv_base_simplegate.py ===" -ForegroundColor Cyan
    python run_shape_analysis_csv_base_simplegate.py --csv_base_root $CsvBaseRoot --base_configs $baseConfigList --folders $folderList
    if ($LASTEXITCODE -ne 0) { throw "run_shape_analysis_csv_base_simplegate.py failed (exit $LASTEXITCODE)" }
} else {
    Write-Host "`n=== [4/6] SKIPPED (-SkipShapeAnalysis) ===" -ForegroundColor Yellow
}

if (-not $SkipPlotConsistentArchs) {
    Write-Host "`n=== [5/6] run_plot_consistent_archs_simplegate.py ===" -ForegroundColor Cyan
    python run_plot_consistent_archs_simplegate.py --csv_base_root $CsvBaseRoot --base_configs $baseConfigList --folders $folderList --workers $Workers
    if ($LASTEXITCODE -ne 0) { throw "run_plot_consistent_archs_simplegate.py failed (exit $LASTEXITCODE)" }
} else {
    Write-Host "`n=== [5/6] SKIPPED (-SkipPlotConsistentArchs) ===" -ForegroundColor Yellow
}

if (-not $SkipHtml) {
    Write-Host "`n=== [6/6] render_html_view.py ===" -ForegroundColor Cyan
    python render_html_view.py --out_root (Join-Path $CsvBaseRoot "best_archs_plots")
    if ($LASTEXITCODE -ne 0) { throw "render_html_view.py failed (exit $LASTEXITCODE)" }
} else {
    Write-Host "`n=== [6/6] SKIPPED (-SkipHtml) ===" -ForegroundColor Yellow
}

Write-Host "`nall done." -ForegroundColor Green
