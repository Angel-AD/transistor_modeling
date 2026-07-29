# Runs plot_gm_derivatives.py against all 6 cg2h40010 measurement CSVs, one PNG per CSV
# (written next to each CSV as <csv>_gm_derivs[_idN].png). Measured-only by default; pass
# -RankedCsv/-Id to overlay the same architecture's model gm on every CSV (only meaningful
# if that exact run exists for each CSV, e.g. a bestpicks10-style sweep trained on all 6 --
# most single-CSV sweeps like tanh_margin10 only have one).
#
#   .\plot_gm_derivatives_all_csvs.ps1
#   .\plot_gm_derivatives_all_csvs.ps1 -Vds "0,5,10,15,20,28"
#   .\plot_gm_derivatives_all_csvs.ps1 -RankedCsv ..\runs\bestpicks10\<csv>\ranked_region_knee_vgs-3to0_vds0to15\..._combined_gm.csv -Id 7
param(
    [string]$Vds = "",
    [string]$RankedCsv = "",
    [string]$Id = "",
    [switch]$Open
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

$csvs = Get-ChildItem -Path "..\csvs" -Filter "cg2h40010_new_*.csv"
$written = @()

foreach ($csv in $csvs) {
    Write-Host "=== $($csv.Name) ===" -ForegroundColor Cyan
    $cmd = @("plot_gm_derivatives.py", "--csv", $csv.FullName)
    if ($Vds) { $cmd += @("--vds", $Vds) }
    if ($RankedCsv -and $Id) { $cmd += @("--ranked_csv", $RankedCsv, "--id", $Id) }
    $out = python @cmd | Tee-Object -Variable lines
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED on $($csv.Name) (exit $LASTEXITCODE)." -ForegroundColor Red
        exit $LASTEXITCODE
    }
    $wroteLine = $lines | Where-Object { $_ -like "wrote *" } | Select-Object -Last 1
    if ($wroteLine) { $written += ($wroteLine -replace "^wrote ", "") }
}

# Open sequentially with a short delay between each -- Windows Photos is single-instance,
# so firing os.startfile()/Invoke-Item back-to-back just redirects one window instead of
# opening N (same issue plot_csv_row.py works around).
if ($Open) {
    foreach ($png in $written) {
        Invoke-Item $png
        Start-Sleep -Milliseconds 700
    }
}
Write-Host "done: gm derivatives plotted for all 6 measurement CSVs." -ForegroundColor Green
