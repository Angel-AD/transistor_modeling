# Retrains the 3 derived populations (best200, bothshapeok, best100_gmshapeok -- see
# extract_derived_configs.py) extracted from ..\runs\pure_combined9069_<Suffix>\ against EVERY
# csv in repo_root\csvs, same pattern as run_best200_all6csvs.ps1 / run_bothshapeok_all6csvs.ps1
# but covering 3 derivations x 8 folders x 6 csvs instead of 1 x 8 x 6.
#
# Prerequisite: extract_derived_configs.py --source_root ..\runs\pure_combined9069_<Suffix>
# must have already been run (produces the 24 config files this script reads):
#   ..\runs\best200ids_of_9069_byloss_<Suffix>\_configs\<folder>_best200.json
#   ..\runs\bothshapeok_of_9069_byshape_<Suffix>\_configs\<folder>_bothshapeok.json
#   ..\runs\best100_gmshapeok_of_9069_byloss_<Suffix>\_configs\<folder>_best100_gmshapeok.json
# -Suffix (default rw0_2.5_20) must match whatever extract_derived_configs.py derived its own
# output dir names from (source_root's name minus the "pure_combined9069_" prefix) -- e.g.
# -Suffix rw4_2.5_20 for a source_root of ..\runs\pure_combined9069_rw4_2.5_20.
#
# Results land nested by csv name under each derivation's own root, all 3 derivation roots
# themselves nested under one shared -OutputParent folder (default: csv_base_2.5_20):
#   ..\runs\<OutputParent>\best200ids_of_9069_byloss_<Suffix>\<csv-name>\<folder>\
#   ..\runs\<OutputParent>\bothshapeok_of_9069_byshape_<Suffix>\<csv-name>\<folder>\
#   ..\runs\<OutputParent>\best100_gmshapeok_of_9069_byloss_<Suffix>\<csv-name>\<folder>\
# Config lookup (_configs\) is UNCHANGED (still under each derivation root directly, e.g.
# ..\runs\best200ids_of_9069_byloss_<Suffix>\_configs\ -- where extract_derived_configs.py wrote
# them) -- -OutputParent only affects where the actual training results/compiled csvs go, not
# where the configs are read from. Pass -OutputParent '' to restore the old flat layout (results
# directly under ..\runs\<derivation-root>\<csv-name>\<folder>\, no shared parent).
#
# Resumable: finished runs (dir has run_loss_* AND weights_loss_*) are skipped by
# multi_experiment_runner.py itself.
#   .\run_derived_configs_from_pure_combined9069_rw0_2.5_20_all6csvs.ps1
#   .\run_derived_configs_from_pure_combined9069_rw0_2.5_20_all6csvs.ps1 -Workers 60
#   .\run_derived_configs_from_pure_combined9069_rw0_2.5_20_all6csvs.ps1 -Derivations best200
#   .\run_derived_configs_from_pure_combined9069_rw0_2.5_20_all6csvs.ps1 -Folders tanh_margin10,softplus
#   .\run_derived_configs_from_pure_combined9069_rw0_2.5_20_all6csvs.ps1 -OutputParent ''   # old flat layout
#   .\run_derived_configs_from_pure_combined9069_rw0_2.5_20_all6csvs.ps1 -Suffix rw4_2.5_20 -OutputParent csv_base_2.5_20_rw4
#   .\run_derived_configs_from_pure_combined9069_rw0_2.5_20_all6csvs.ps1 -Suffix rw4_2.5_20 -OutputParent csv_base_2.5_20_rw4 `
#       -RunsDir C:\Users\acost\repos\new_opts_2\runs -CsvsDir C:\Users\acost\repos\new_opts_2\csvs
#       # ^ reads configs AND writes output against new_opts_2's OWN runs\ tree (its real
#       #   ..\runs\best200ids_of_9069_byloss_rw4_2.5_20\_configs\ etc., and its real csvs\) --
#       #   i.e. this IS how to actually build new_opts_2's csv_base_2.5_20_rw4 from here,
#       #   with nothing copied either direction.
#
# This script lives in repo_root\physics_nn_pipeline\runner_helpers\ -- Set-Location moves CWD
# to repo_root\physics_nn_pipeline\ (its parent), matching every other runner_helpers\*.ps1
# script's convention, so multi_experiment_runner.py / compile_helpers\compile_overall.ps1 can
# be called by a short relative path, and runs\/csvs\ are reached via a single "..\" (one level
# up from physics_nn_pipeline\ = repo root -- same depth physics_nn_configs\ used to be at, so
# the number of "..\" needed here is unchanged from before the move).

param(
    [int]$Workers = 46,
    [string[]]$Folders = @(
        "sigmoid_margin10", "sigmoid_margin10_nogm",
        "softplus", "softplus_nogm",
        "tanh_margin10", "tanh_margin10_nogm",
        "vdsgate_aeff_quad_tanhm", "vdsgate_aeff_quad_v3",
        "vdsgate_v3", "vdsgate_tanhm"
    ),
    [string[]]$Derivations = @("best200", "bothshapeok", "best100_gmshapeok"),
    [string]$OutputParent = "csv_base_2.5_20",
    [string]$Suffix = "rw0_2.5_20",
    [string]$RunsDir = "",
    [string]$CsvsDir = ""
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

# -RunsDir/-CsvsDir let this script read configs/csvs from (and write output to -- $runsDir
# drives BOTH config lookup and -OutputParent below) a DIFFERENT repo/clone's runs\/csvs\ tree
# entirely, e.g. to run this repo's script against new_opts_2's real data without copying
# anything either direction. Absolute paths recommended when overriding (relative paths would
# be resolved against THIS repo's physics_nn_pipeline\, per the Set-Location above).
$runsDir = if ($RunsDir) { $RunsDir } else { "..\runs" }
$csvsDir = if ($CsvsDir) { $CsvsDir } else { "..\csvs" }

# derivation -> (root dir name under ..\runs\, config filename suffix in that root's _configs\).
# RootName is built from -Suffix so this same script works for any pure_combined9069_<suffix>
# sweep (e.g. rw0_2.5_20, rw4_2.5_20) that extract_derived_configs.py --source_root
# ..\runs\pure_combined9069_<suffix> has already been run against -- that script derives its
# OWN output dir names the same way (source_root's name minus the "pure_combined9069_" prefix),
# so -Suffix here should match whatever that produced.
$derivationMap = @{
    "best200"           = @{ RootName = "best200ids_of_9069_byloss_$Suffix";     Suffix = "_best200.json" }
    "bothshapeok"        = @{ RootName = "bothshapeok_of_9069_byshape_$Suffix";   Suffix = "_bothshapeok.json" }
    "best100_gmshapeok" = @{ RootName = "best100_gmshapeok_of_9069_byloss_$Suffix"; Suffix = "_best100_gmshapeok.json" }
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
            python multi_experiment_runner.py `
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

# Compile every resulting folder. compile_helpers\compile_overall.ps1 itself calls its OWN
# downstream python scripts by BARE filename, assuming CWD = physics_nn_pipeline\ -- already
# true here (Set-Location above), so no Push-Location/Pop-Location needed around it.
foreach ($root in $allRoots) {
    Write-Host "=== compile $root ===" -ForegroundColor Cyan
    & .\compile_helpers\compile_overall.ps1 -root $root -ShortName
    Write-Host "compiled: $root" -ForegroundColor Green
}
Write-Host "all done: $($allRoots.Count) ($($Derivations.Count) derivations x $($Folders.Count) folders x $($csvFiles.Count) csvs) runs compiled." -ForegroundColor Green
