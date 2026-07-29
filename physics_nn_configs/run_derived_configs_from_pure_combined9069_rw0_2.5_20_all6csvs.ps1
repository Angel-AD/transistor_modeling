# Retrains the 3 derived populations (best200, bothshapeok, best100_gmshapeok -- see
# extract_derived_configs.py) extracted from runs\pure_combined9069_rw0_2.5_20\ against EVERY
# csv in repo_root\csvs, same pattern as run_best200_all6csvs.ps1 / run_bothshapeok_all6csvs.ps1
# but covering 3 derivations x 8 folders x 6 csvs instead of 1 x 8 x 6.
#
# Prerequisite: extract_derived_configs.py --source_root ..\runs\pure_combined9069_rw0_2.5_20
# must have already been run (produces the 24 config files this script reads):
#   runs\best200ids_of_9069_byloss_rw0_2.5_20\_configs\<folder>_best200.json
#   runs\bothshapeok_of_9069_byshape_rw0_2.5_20\_configs\<folder>_bothshapeok.json
#   runs\best100_gmshapeok_of_9069_byloss_rw0_2.5_20\_configs\<folder>_best100_gmshapeok.json
#
# Results land nested by csv name under each derivation's own root, all 3 derivation roots
# themselves nested under one shared -OutputParent folder (default: csv_base_2.5_20):
#   runs\csv_base_2.5_20\best200ids_of_9069_byloss_rw0_2.5_20\<csv-name>\<folder>\
#   runs\csv_base_2.5_20\bothshapeok_of_9069_byshape_rw0_2.5_20\<csv-name>\<folder>\
#   runs\csv_base_2.5_20\best100_gmshapeok_of_9069_byloss_rw0_2.5_20\<csv-name>\<folder>\
# Config lookup (_configs\) is UNCHANGED (still under each derivation root directly, e.g.
# runs\best200ids_of_9069_byloss_rw0_2.5_20\_configs\ -- where extract_derived_configs.py wrote
# them) -- -OutputParent only affects where the actual training results/compiled csvs go, not
# where the configs are read from. Pass -OutputParent '' to restore the old flat layout (results
# directly under runs\<derivation-root>\<csv-name>\<folder>\, no shared parent).
#
# Resumable: finished runs (dir has run_loss_* AND weights_loss_*) are skipped by
# multi_experiment_runner.py itself.
#   .\run_derived_configs_from_pure_combined9069_rw0_2.5_20_all6csvs.ps1
#   .\run_derived_configs_from_pure_combined9069_rw0_2.5_20_all6csvs.ps1 -Workers 60
#   .\run_derived_configs_from_pure_combined9069_rw0_2.5_20_all6csvs.ps1 -Derivations best200
#   .\run_derived_configs_from_pure_combined9069_rw0_2.5_20_all6csvs.ps1 -Folders tanh_margin10,softplus
#   .\run_derived_configs_from_pure_combined9069_rw0_2.5_20_all6csvs.ps1 -OutputParent ''   # old flat layout
#
# This script lives in repo_root\physics_nn_configs\ (a sibling of runs\ and
# physics_nn_pipeline\, NOT inside either) -- every path is built from $repoRoot, never from
# $PSScriptRoot directly, same convention as run_rw_sweep_all6csvs.ps1.

param(
    [int]$Workers = 46,
    [string[]]$Folders = @(
        "sigmoid_margin10", "sigmoid_margin10_nogm",
        "softplus", "softplus_nogm",
        "tanh_margin10", "tanh_margin10_nogm",
        "vdsgate_aeff_quad_tanhm", "vdsgate_aeff_quad_v3"
    ),
    [string[]]$Derivations = @("best200", "bothshapeok", "best100_gmshapeok"),
    [string]$OutputParent = "csv_base_2.5_20"
)
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path $PSScriptRoot -Parent
$runsDir = Join-Path $repoRoot "runs"
$physicsPipelineDir = Join-Path $repoRoot "physics_nn_pipeline"
$pipeline = Join-Path $physicsPipelineDir "multi_experiment_runner.py"
$csvsDir = Join-Path $repoRoot "csvs"

# derivation -> (root dir name under runs\, config filename suffix in that root's _configs\)
$derivationMap = @{
    "best200"           = @{ RootName = "best200ids_of_9069_byloss_rw0_2.5_20";     Suffix = "_best200.json" }
    "bothshapeok"        = @{ RootName = "bothshapeok_of_9069_byshape_rw0_2.5_20";   Suffix = "_bothshapeok.json" }
    "best100_gmshapeok" = @{ RootName = "best100_gmshapeok_of_9069_byloss_rw0_2.5_20"; Suffix = "_best100_gmshapeok.json" }
}

$allRoots = @()
$csvFiles = Get-ChildItem (Join-Path $csvsDir "*.csv")

foreach ($derivation in $Derivations) {
    $info = $derivationMap[$derivation]
    if (-not $info) {
        Write-Host "SKIP: unknown derivation '$derivation'" -ForegroundColor Yellow
        continue
    }
    # Config lookup always stays at the original (flat) derivation root, regardless of
    # -OutputParent -- that's where extract_derived_configs.py actually wrote the _configs\.
    $configRoot = Join-Path $runsDir $info.RootName
    $configDir = Join-Path $configRoot "_configs"
    if (-not (Test-Path $configDir)) {
        Write-Host "SKIP derivation '$derivation': config dir not found: $configDir " -ForegroundColor Yellow
        Write-Host "  (run extract_derived_configs.py first)" -ForegroundColor Yellow
        continue
    }
    # Output root: nested under -OutputParent when set, else same as $configRoot (old flat layout).
    # Windows PowerShell 5.1's Join-Path only takes two path segments -- chain two calls, not
    # three positional args.
    $outputRoot = if ($OutputParent) { Join-Path (Join-Path $runsDir $OutputParent) $info.RootName } else { $configRoot }
    foreach ($csv in $csvFiles) {
        foreach ($folder in $Folders) {
            $config = Join-Path $configDir "$folder$($info.Suffix)"
            if (-not (Test-Path $config)) {
                Write-Host "SKIP: config not found: $config" -ForegroundColor Yellow
                continue
            }
            $root = Join-Path $outputRoot "$($csv.BaseName)\$folder"
            $allRoots += $root
            Write-Host "=== [$derivation] $($csv.Name) / $folder -> $root ===" -ForegroundColor Cyan
            python $pipeline `
                --config $config `
                --csv $csv.FullName `
                --master_root_path $root `
                --max_workers $Workers
            if ($LASTEXITCODE -ne 0) {
                Write-Host "FAILED on [$derivation] $($csv.Name) / $folder (exit $LASTEXITCODE)." -ForegroundColor Red
                exit $LASTEXITCODE
            }
            Write-Host "done: [$derivation] $($csv.Name) / $folder" -ForegroundColor Green
        }
    }
}
Write-Host "all csvs/folders/derivations finished. compiling..." -ForegroundColor Green

# Compile every resulting folder. compile_overall.ps1 calls its OWN downstream python scripts
# by BARE filename, assuming CWD = physics_nn_pipeline\ -- invoked from there via
# Push-Location/Pop-Location. -ShortName matches best200/bothshapeok's own convention (this
# layout has the same extra path depth: <root>\<csv-name>\<folder>\...).
foreach ($root in $allRoots) {
    Write-Host "=== compile $root ===" -ForegroundColor Cyan
    Push-Location $physicsPipelineDir
    try {
        & .\compile_overall.ps1 -root $root -ShortName
    } finally {
        Pop-Location
    }
    Write-Host "compiled: $root" -ForegroundColor Green
}
Write-Host "all done: $($allRoots.Count) ($($Derivations.Count) derivations x $($Folders.Count) folders x $($csvFiles.Count) csvs) runs compiled." -ForegroundColor Green
