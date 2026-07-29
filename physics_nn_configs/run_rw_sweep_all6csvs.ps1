# Retrains best200ids_of_9069_byloss's and bothshapeok_of_9069_byshape's architecture pools
# against EVERY csv in repo_root\csvs, under two region_weight variants (0 = off, 2 = half the
# original 4x emphasis) -- to see how much the knee-box weighting in the original best200/
# bothshapeok sweeps (region_weight=4, see run_best200_all6csvs.ps1 / run_bothshapeok_all6csvs.ps1)
# actually mattered for fit quality and shape consistency.
#
# Same architecture pools / hyperparameters as the original best200/bothshapeok configs --
# every EXPERIMENTS entry in _configs\<folder>_<best200|bothshapeok>.json is byte-identical to
# its source in runs\best200ids_of_9069_byloss\_configs\ / runs\bothshapeok_of_9069_byshape\_configs\
# except region_weights, which is overridden to [0.0] or [2.0] (was [4.0] in both sources).
#
# Results land in 4 separate root dirs under runs\ (kept fully separate from the original
# region_weight=4 results, so nothing there is touched or overwritten):
#   runs\best200ids_of_9069_byloss_rw0\<csv-name>\<folder>\      (region_weight=0)
#   runs\best200ids_of_9069_byloss_rw2\<csv-name>\<folder>\      (region_weight=2)
#   runs\bothshapeok_of_9069_byshape_rw0\<csv-name>\<folder>\    (region_weight=0)
#   runs\bothshapeok_of_9069_byshape_rw2\<csv-name>\<folder>\    (region_weight=2)
#
# Resumable: finished runs (dir has run_loss_* AND weights_loss_*) are skipped by
# multi_experiment_runner.py itself.
#   .\run_rw_sweep_all6csvs.ps1                              # all 4 roots x 8 folders x 6 csvs, 46 workers
#   .\run_rw_sweep_all6csvs.ps1 -Workers 60
#   .\run_rw_sweep_all6csvs.ps1 -Folders tanh_margin10,softplus            # just a subset of folders
#   .\run_rw_sweep_all6csvs.ps1 -BaseRoots best200ids_of_9069_byloss -RwValues 0   # just one combo
#
# This script lives in repo_root\physics_nn_configs\ (a sibling of runs\ and physics_nn_pipeline\,
# NOT inside either) -- every path below is built from $repoRoot (one level up from this script),
# never from $PSScriptRoot directly, so it doesn't matter that its own location differs from the
# runs\/physics_nn_pipeline\ dirs it reads from and writes into.
# ABSOLUTE paths only for --config/--master_root_path: multi_experiment_runner.py resolves a
# RELATIVE path against its OWN file location (HERE = physics_nn_pipeline\), not against
# whatever directory this script (or PowerShell) is sitting in. --csv is already absolute
# (Get-ChildItem's .FullName), so it isn't affected.

param(
    [int]$Workers = 46,
    [string[]]$Folders = @(
        "sigmoid_margin10", "sigmoid_margin10_nogm",
        "softplus", "softplus_nogm",
        "tanh_margin10", "tanh_margin10_nogm",
        "vdsgate_aeff_quad_tanhm", "vdsgate_aeff_quad_v3"
    ),
    [string[]]$BaseRoots = @("best200ids_of_9069_byloss", "bothshapeok_of_9069_byshape"),
    [int[]]$RwValues = @(0, 2)
)
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path $PSScriptRoot -Parent
$runsDir = Join-Path $repoRoot "runs"
$physicsPipelineDir = Join-Path $repoRoot "physics_nn_pipeline"
$pipeline  = Join-Path $physicsPipelineDir "multi_experiment_runner.py"
$csvsDir   = Join-Path $repoRoot "csvs"

# config filename suffix per base root (matches each root's own _configs\<folder><suffix>.json
# convention -- see run_best200_all6csvs.ps1 / run_bothshapeok_all6csvs.ps1).
$suffixMap = @{
    "best200ids_of_9069_byloss"   = "_best200.json"
    "bothshapeok_of_9069_byshape" = "_bothshapeok.json"
}

$allRoots = @()
$csvFiles = Get-ChildItem (Join-Path $csvsDir "*.csv")

foreach ($baseRoot in $BaseRoots) {
    $suffix = $suffixMap[$baseRoot]
    foreach ($rw in $RwValues) {
        $resultRoot = Join-Path $runsDir "${baseRoot}_rw${rw}"
        $configDir = Join-Path $resultRoot "_configs"
        foreach ($csv in $csvFiles) {
            foreach ($folder in $Folders) {
                $config = Join-Path $configDir "$folder$suffix"
                if (-not (Test-Path $config)) {
                    Write-Host "SKIP: config not found: $config" -ForegroundColor Yellow
                    continue
                }
                $root = Join-Path $resultRoot "$($csv.BaseName)\$folder"
                $allRoots += $root
                Write-Host "=== [$baseRoot rw=$rw] $($csv.Name) / $folder -> $root ===" -ForegroundColor Cyan
                python $pipeline `
                    --config $config `
                    --csv $csv.FullName `
                    --master_root_path $root `
                    --max_workers $Workers
                if ($LASTEXITCODE -ne 0) {
                    Write-Host "FAILED on [$baseRoot rw=$rw] $($csv.Name) / $folder (exit $LASTEXITCODE)." -ForegroundColor Red
                    exit $LASTEXITCODE
                }
                Write-Host "done: [$baseRoot rw=$rw] $($csv.Name) / $folder" -ForegroundColor Green
            }
        }
    }
}
Write-Host "all csvs/folders finished. compiling..." -ForegroundColor Green

# Compile every resulting folder (csv/base auto-detected from each folder's base_files).
# compile_overall.ps1 calls its OWN downstream python scripts (compute_region_metrics.py,
# compile_results_csv.py, ...) by BARE filename, assuming the caller's working directory is
# physics_nn_pipeline\ -- so it must be invoked from there via Push-Location/Pop-Location.
foreach ($root in $allRoots) {
    Write-Host "=== compile $root ===" -ForegroundColor Cyan
    Push-Location $physicsPipelineDir
    try {
        & .\compile_helpers\compile_overall.ps1 -root $root -ShortName
    } finally {
        Pop-Location
    }
    Write-Host "compiled: $root" -ForegroundColor Green
}
Write-Host "all done: $($allRoots.Count) ($($BaseRoots.Count) roots x $($RwValues.Count) rw-values x $($Folders.Count) folders x $($csvFiles.Count) csvs) runs compiled." -ForegroundColor Green
