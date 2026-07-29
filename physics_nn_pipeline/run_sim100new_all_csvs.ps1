# Runs avkf2_id21677_sim100new_vdssweep.json against EVERY csv in ..\csvs, one after another,
# then compiles each resulting folder (compile_overall.ps1).
# All results are grouped under ..\runs\sim100new\<csv-name>\
# Resumable: finished runs (dir has run_loss_* AND weights_loss_*) are skipped.
#   .\run_sim100new_all_csvs.ps1                # 46 workers (default)
#   .\run_sim100new_all_csvs.ps1 -Workers 60
param([int]$Workers = 46)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$config = "..\physics_nn_configs\avkf2_id21677_sim100new_vdssweep.json"
$roots  = @()

foreach ($csv in Get-ChildItem ..\csvs\*.csv) {
    $root = "..\runs\sim100new\$($csv.BaseName)"
    $roots += $root
    Write-Host "=== $($csv.Name) -> $root ===" -ForegroundColor Cyan
    python multi_experiment_runner.py `
        --config $config `
        --csv $csv.FullName `
        --master_root_path $root `
        --max_workers $Workers
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED on $($csv.Name) (exit $LASTEXITCODE)." -ForegroundColor Red
        exit $LASTEXITCODE
    }
    Write-Host "done: $($csv.Name)" -ForegroundColor Green
}
Write-Host "all csvs finished. compiling..." -ForegroundColor Green

# Compile every resulting folder (csv/base auto-detected from each folder's base_files).
foreach ($root in $roots) {
    Write-Host "=== compile $root ===" -ForegroundColor Cyan
    & .\compile_overall.ps1 -root $root
    Write-Host "compiled: $root" -ForegroundColor Green
}
Write-Host "all done." -ForegroundColor Green
