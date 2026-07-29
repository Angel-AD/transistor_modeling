# Compiles every runs\bestpicks10\<csv-name>\ folder (one per csv in ..\csvs) without
# re-running training. Companion to run_bestpicks10_all_csvs.ps1, which already compiles as
# its second phase -- use this script on its own to (re)compile independently, e.g. after
# stopping training early or re-running compile after a manual fix.
#   .\compile_bestpicks10_all_csvs.ps1
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

foreach ($csv in Get-ChildItem ..\csvs\*.csv) {
    $root = "..\runs\bestpicks10\$($csv.BaseName)"
    if (-not (Test-Path $root)) {
        Write-Host "skip (not found): $root" -ForegroundColor Yellow
        continue
    }
    Write-Host "=== compile $root ===" -ForegroundColor Cyan
    & .\compile_helpers\compile_overall.ps1 -root $root
    Write-Host "compiled: $root" -ForegroundColor Green
}
Write-Host "all done." -ForegroundColor Green
