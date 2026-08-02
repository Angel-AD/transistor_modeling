# Creates the 10 <folder>\tanh_only junctions pointing at <folder>_tanhonly, so
# write_tanh_only_comparison.py's hardcoded lookup path
# (<out_root>/<folder>/tanh_only/consistency_summary_<base_config>.json) resolves to the
# dedicated tanh-only data produced by run_tanhonly_derivatives_and_best_archs_plots.ps1 (8
# original folders) and run_tanhonly_derivatives_and_best_archs_plots_simplegate.ps1
# (vdsgate_v3/vdsgate_tanhm). Safe to re-run -- skips any junction that already exists.
#
# Usage:
#   .\create_tanhonly_junctions.ps1
#   .\create_tanhonly_junctions.ps1 -BestArchsPlotsRoot "D:\some\other\best_archs_plots"

param(
    [string]$BestArchsPlotsRoot = "C:\Users\acost\repos\new_opts_2\runs\csv_base_2.5_20_rw4\best_archs_plots",
    [string[]]$Folders = @(
        "sigmoid_margin10", "sigmoid_margin10_nogm", "softplus", "softplus_nogm",
        "tanh_margin10", "tanh_margin10_nogm", "vdsgate_aeff_quad_tanhm", "vdsgate_aeff_quad_v3",
        "vdsgate_v3", "vdsgate_tanhm"
    )
)
$ErrorActionPreference = "Stop"

foreach ($f in $Folders) {
    $link = Join-Path $BestArchsPlotsRoot "$f\tanh_only"
    $target = Join-Path $BestArchsPlotsRoot "${f}_tanhonly"

    if (-not (Test-Path $target)) {
        Write-Host "SKIP $f`: target does not exist yet: $target" -ForegroundColor Yellow
        continue
    }
    if (Test-Path $link) {
        Write-Host "SKIP $f`: $link already exists" -ForegroundColor Yellow
        continue
    }

    New-Item -ItemType Junction -Path $link -Target $target | Out-Null
    Write-Host "created: $link -> $target" -ForegroundColor Green
}
Write-Host "done." -ForegroundColor Green
