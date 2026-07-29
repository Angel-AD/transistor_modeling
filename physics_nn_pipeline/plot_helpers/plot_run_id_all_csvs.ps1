# Plots ONE run_id across every runs\bestpicks10\<csv-name>\ folder, so you can compare how the
# SAME architecture (same run_id -- same exp_NNN index across csvs, since they all trained from
# the identical avkf2_id21677_bestpicks10_gmvds3.json config) fits each measurement csv.
#
# Uses the FULL (unfiltered) ranked_by_region_knee_combined_gm.csv in each folder, so the run_id
# resolves even if that architecture didn't pass the fit-quality filter on a particular csv.
#
#   .\plot_run_id_all_csvs.ps1 -RunId 3              # plot + open (first time: trains the plot)
#   .\plot_run_id_all_csvs.ps1 -RunId 3 -ShowOnly     # just reopen already-plotted images
param(
    [Parameter(Mandatory=$true)][string]$RunId,
    [switch]$ShowOnly
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

$mode = if ($ShowOnly) { "--show_only" } else { "--open" }

foreach ($csv in Get-ChildItem ..\csvs\*.csv) {
    $ranked = "..\runs\bestpicks10\$($csv.BaseName)\ranked_region_knee_vgs-3to0_vds0to15\$($csv.BaseName)_ranked_by_region_knee_combined_gm.csv"

    if (-not (Test-Path $ranked)) {
        Write-Host "skip (not compiled yet): $ranked" -ForegroundColor Yellow
        continue
    }
    Write-Host "=== $($csv.BaseName) ===" -ForegroundColor Cyan
    python plot_csv_row.py --ranked_csv $ranked --run_id $RunId $mode
}
