# Runs plot_ids_vds_derivatives.py against all 6 cg2h40010 measurement CSVs, one PNG per CSV
# (written next to each CSV as <csv>_ids_vds_derivs[_idN].png). Measured-only by default --
# pass -RankedCsv/-Id to overlay a specific architecture's model Ids-Vds derivatives on every
# CSV (only meaningful if that exact run exists for each CSV).
#
#   .\plot_ids_vds_derivatives_all_csvs.ps1
#   .\plot_ids_vds_derivatives_all_csvs.ps1 -VdsMax 6 -Vgs "-3,-2,-1.6,-1,-0.5,0"
#   .\plot_ids_vds_derivatives_all_csvs.ps1 -RankedCsv ..\runs\...\..._combined_gm_4.csv -Id 267
param(
    [string]$Vgs = "",
    [double]$VdsMax = 0,
    [string]$RankedCsv = "",
    [string]$Id = "",
    [switch]$Open
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$csvs = Get-ChildItem -Path "..\csvs" -Filter "cg2h40010_new_*.csv"
$written = @()

foreach ($csv in $csvs) {
    Write-Host "=== $($csv.Name) ===" -ForegroundColor Cyan
    $cmd = @("plot_ids_vds_derivatives.py", "--csv", $csv.FullName)
    if ($Vgs) { $cmd += "--vgs=$Vgs" }
    if ($VdsMax -gt 0) { $cmd += @("--vds_max", $VdsMax) }
    if ($RankedCsv -and $Id) { $cmd += @("--ranked_csv", $RankedCsv, "--id", $Id) }
    $lines = python @cmd
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED on $($csv.Name) (exit $LASTEXITCODE)." -ForegroundColor Red
        exit $LASTEXITCODE
    }
    $wroteLine = $lines | Where-Object { $_ -like "wrote *" } | Select-Object -Last 1
    if ($wroteLine) { $written += ($wroteLine -replace "^wrote ", "") }
}

if ($Open) {
    foreach ($png in $written) {
        Invoke-Item $png
        Start-Sleep -Milliseconds 700
    }
}
Write-Host "done: Ids-Vds derivatives plotted for all 6 measurement CSVs." -ForegroundColor Green
