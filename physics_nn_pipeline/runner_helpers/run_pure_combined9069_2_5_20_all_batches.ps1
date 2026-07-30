# Re-runs the exact pure_combined9069 architecture sweep (8 folders, 3 batches each, 9068-9069
# architectures per folder) against a DIFFERENT measurement csv -- default cg2h40010_new_2.5_20_
# 2_70W_center9.csv -- instead of the original cg2h40010_new_2.4_5_2_70W_center9.csv, AND with
# region_weight overridden to whatever -ConfigDir's configs use (default: 0, i.e. off, instead
# of the original sweep's 4.0).
#
# Uses the config copies in -ConfigDir (default: ..\runs\pure_combined9069_rw0_2.5_20\_configs\,
# region_weight=0) -- byte-identical to the 24 originals archived in each folder's own
# ..\runs\pure_combined9069\<folder>\base_files\ (region_weight=4, verified via deep-diff) except
# region_weights itself. Everything else (nn_architectures, gm weights, epochs, base_configs,
# ...) is unchanged. To reproduce the ORIGINAL region_weight=4 against this different csv (e.g.
# ..\runs\pure_combined9069_rw4_2.5_20\_configs\, populated by copying each folder's own
# base_files\*.json directly -- they're already region_weight=4), pass
# -ConfigDir ..\runs\pure_combined9069_rw4_2.5_20\_configs -OutSuffix rw4_2.5_20.
#
# Same batching/accumulation pattern as the original scripts: all 3 batches per folder land in
# ONE shared root (distinct "_batchN" experiment-name suffix avoids collision), compiled once
# after all 3 batches finish. Results land in a SEPARATE root (named after -OutSuffix, or
# exactly -OutRoot if given) so neither the original 2.4_5/rw4 sweep in
# ..\runs\pure_combined9069\ nor any other -OutSuffix's/-OutRoot's results are touched:
#   ..\runs\pure_combined9069_$OutSuffix\<folder>\    (default, -OutRoot not given)
#   $OutRoot\<folder>\                                (when -OutRoot IS given -- any absolute
#                                                        path, e.g. to read/write against a
#                                                        DIFFERENT repo/clone's runs\ tree for
#                                                        a quick test without copying data)
#
# Resumable: finished runs (dir has run_loss_* AND weights_loss_*) are skipped by
# multi_experiment_runner.py itself.
#   .\run_pure_combined9069_2_5_20_all_batches.ps1
#   .\run_pure_combined9069_2_5_20_all_batches.ps1 -Workers 60
#   .\run_pure_combined9069_2_5_20_all_batches.ps1 -Csv ..\csvs\cg2h40010_new_2.9_28_2_70W_center9.csv -OutSuffix rw0_2_9_28
#   .\run_pure_combined9069_2_5_20_all_batches.ps1 -Folders tanh_margin10,softplus   # just a subset
#   .\run_pure_combined9069_2_5_20_all_batches.ps1 -ConfigDir ..\runs\pure_combined9069_rw4_2.5_20\_configs -OutSuffix rw4_2.5_20
#   .\run_pure_combined9069_2_5_20_all_batches.ps1 -Csv C:\Users\acost\repos\new_opts_2\csvs\cg2h40010_new_2.5_20_2_70W_center9.csv `
#       -ConfigDir C:\Users\acost\repos\new_opts_2\runs\pure_combined9069_rw4_2.5_20\_configs `
#       -OutRoot C:\Users\acost\repos\new_opts_2\runs\pure_combined9069_rw4_2.5_20_TEST_FROM_TM -Folders tanh_margin10
#       # ^ reads input (csv+config) from new_opts_2's real tree, but writes output to a
#       #   DISTINCT, clearly-test-labeled directory there -- NEVER reuse the real rw4_2.5_20
#       #   output root itself for -OutRoot, since that would write/mix new test runs into
#       #   the already-completed real training results (multi_experiment_runner.py resumes/
#       #   skips existing runs, but any NOT-yet-done or differently-selected architecture would
#       #   still write into that live, valuable directory).
#
# This script lives in repo_root\physics_nn_pipeline\runner_helpers\ -- Set-Location moves CWD
# to repo_root\physics_nn_pipeline\ (its parent), matching every other runner_helpers\*.ps1
# script's convention (see e.g. run_pure_combined9069_all_batches.ps1), so
# multi_experiment_runner.py / compile_helpers\compile_overall.ps1 can be called by a short
# relative path, and -Csv/-ConfigDir default to "..\"-prefixed paths (one level up from
# physics_nn_pipeline\ = repo root -- same depth physics_nn_configs\ used to be at, so the
# number of "..\" needed here is unchanged from before the move).

param(
    [int]$Workers = 46,
    [string]$Csv = "..\csvs\cg2h40010_new_2.5_20_2_70W_center9.csv",
    [string]$OutSuffix = "rw0_2.5_20",
    [string]$OutRoot = "",
    [string]$ConfigDir = "",
    [string[]]$Folders = @(
        "sigmoid_margin10", "sigmoid_margin10_nogm",
        "softplus", "softplus_nogm",
        "tanh_margin10", "tanh_margin10_nogm",
        "vdsgate_aeff_quad_tanhm", "vdsgate_aeff_quad_v3"
    )
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

# folder -> config filename PREFIX (before "_batchN...json"). Not a simple derivation from
# folder name -- some have a "_gmvds3" suffix, some don't, and vdsgate_aeff_quad_v3's config
# file is named "..._vdsgate_aeff_quad_batchN.json" (no "_v3"). Verified directly against
# ..\runs\pure_combined9069\<folder>\base_files\ contents (same filenames used in the rw0
# configs dir below, just region_weights overridden).
$configPrefixMap = @{
    "sigmoid_margin10"        = "pure_combined9069_sigmoid_margin10"
    "sigmoid_margin10_nogm"   = "pure_combined9069_sigmoid_margin10_nogm"
    "softplus"                = "pure_combined9069_softplus"
    "softplus_nogm"           = "pure_combined9069_softplus_nogm"
    "tanh_margin10"           = "pure_combined9069_tanh_margin10"
    "tanh_margin10_nogm"      = "pure_combined9069_tanh_margin10_nogm"
    "vdsgate_aeff_quad_tanhm" = "pure_combined9069_vdsgate_aeff_quad_tanhm"
    "vdsgate_aeff_quad_v3"    = "pure_combined9069_vdsgate_aeff_quad"
}
# some configs have a "_gmvds3" filename suffix, some don't -- checked per folder.
$configSuffixMap = @{
    "sigmoid_margin10"        = "_gmvds3"
    "sigmoid_margin10_nogm"   = ""
    "softplus"                = "_gmvds3"
    "softplus_nogm"           = ""
    "tanh_margin10"           = "_gmvds3"
    "tanh_margin10_nogm"      = ""
    "vdsgate_aeff_quad_tanhm" = ""
    "vdsgate_aeff_quad_v3"    = ""
}

$rw0ConfigDir = if ($ConfigDir) { $ConfigDir } else { "..\runs\pure_combined9069_rw0_2.5_20\_configs" }
$outRoot = if ($OutRoot) { $OutRoot } else { "..\runs\pure_combined9069_$OutSuffix" }

$roots = @()
foreach ($folder in $Folders) {
    $roots += (Join-Path $outRoot $folder)
}

foreach ($batch in 1, 2, 3) {
    foreach ($folder in $Folders) {
        $prefix = $configPrefixMap[$folder]
        $suffix = $configSuffixMap[$folder]
        $config = Join-Path $rw0ConfigDir "${prefix}_batch${batch}${suffix}.json"
        if (-not (Test-Path $config)) {
            Write-Host "SKIP: config not found: $config" -ForegroundColor Yellow
            continue
        }
        $root = Join-Path $outRoot $folder
        Write-Host "=== batch$batch / $folder -> $root ===" -ForegroundColor Cyan
        python multi_experiment_runner.py `
            --config $config `
            --csv $Csv `
            --master_root_path $root `
            --max_workers $Workers
        if ($LASTEXITCODE -ne 0) {
            Write-Host "FAILED on batch$batch / $folder (exit $LASTEXITCODE)." -ForegroundColor Red
            exit $LASTEXITCODE
        }
        Write-Host "done: batch$batch / $folder" -ForegroundColor Green
    }
}
Write-Host "all batches finished. compiling..." -ForegroundColor Green
foreach ($root in $roots) {
    Write-Host "=== compile $root ===" -ForegroundColor Cyan
    & .\compile_helpers\compile_overall.ps1 -root $root
    Write-Host "compiled: $root" -ForegroundColor Green
}
Write-Host "all done: $($roots.Count) folders x 3 batches compiled into $outRoot" -ForegroundColor Green
