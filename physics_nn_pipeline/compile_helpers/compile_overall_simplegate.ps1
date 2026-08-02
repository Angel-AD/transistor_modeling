param(
    [Parameter(Mandatory=$true)][string]$root,
    [string]$csv    = "",      # measurement CSV; blank -> auto-detected from <root>\_run_meta.json / run records
    [string]$region = "knee:vgs=-3..0,vds=0..15",
    [string[]]$filter = @(),   # e.g. -filter "region_knee_combined_gm<0.8","region_knee_ids_rmse<0.01"
    [string[]]$dropAllZero = @("gm1_weight","gm2_weight","gm3_weight"),  # drop inert all-zero-gm rows; -dropAllZero @() to disable
    [string]$baseCsv = "",     # base ranked CSV; blank -> auto-detected (from _run_meta.json / provenance)
    [double]$maxCombinedGm = 0.9,   # quality-filter threshold: region_knee_combined_gm <= this
    [double]$maxIdsRmse    = 0.009, # quality-filter threshold: region_knee_ids_rmse <= this
    [int]$TopN = 0,           # ALTERNATIVE to the maxCombinedGm/maxIdsRmse threshold filter in
        # step 3b: when > 0, keep only the best TopN rows by -SortBy instead (e.g. -TopN 20
        # -SortBy region_knee_combined_gm). Combines with --auto_number the same way -- a
        # different -TopN/-SortBy combo gets its own number, doesn't overwrite a prior one.
    [string]$SortBy = "region_knee_combined_gm",  # metric for -TopN (lower=better by default)
    [switch]$Descending,      # with -TopN: rank highest-first instead of lowest-first
    [switch]$RegionRangeOnly = $true,  # only write the range-tagged ranked_region_<name>_vgs..to.._
        # vds..to../ folder (combined_gm + ids_rmse); skips ranked_global/, the plain
        # ranked_region_<name>/ folder, and region gm1/gm2/gm3-only rankings -- cuts ranked-CSV
        # count from ~12 to 2 for a large sweep, which cascades through xlsx conversion and the
        # filter steps too. Pass -RegionRangeOnly:$false to restore the full set.
    [switch]$SkipRegionMetrics,  # skip step 1 (compute_region_metrics.py) -- the slow, per-model
        # step. Use when the run_loss_*.json files already have region metrics from a prior
        # compile and you only want to re-derive/re-filter/re-shape-analyze the ranked CSVs
        # (e.g. after a code change to compile_ranked.py / filter_results.py / analyze_shape.py).
    [switch]$ShortName  # use a short hash instead of the full sweep name as the ranked-file
        # filename prefix (<hash>_ranked_by_<metric>.csv). OFF by default -- OPT IN, since some
        # tooling (plot_run_id_all_csvs.ps1, pipeline_gui.py) depends on the full-name pattern
        # for OTHER sweeps (e.g. bestpicks10). Turn on for sweeps whose downstream filenames get
        # long enough (filter -> shape -> gmshape_ok -> slim) to risk hitting Excel's/Windows'
        # path-length limits -- pure_combined9069 hit this in practice.
)

# CSV args default to auto-detection: only pass them when explicitly given (the Python steps
# read the missing ones from <root>\_run_meta.json / run records).
$csvArgs  = @(); if ($csv)     { $csvArgs  = @("--csv", $csv) }
$baseArgs = @(); if ($baseCsv) { $baseArgs = @("--base_csv", $baseCsv) }
$rankArgs = @(); if ($RegionRangeOnly) { $rankArgs = @("--region_range_only") }
if ($ShortName) { $rankArgs += "--short_name" }

# 1. region metrics -> written into each run_loss_*.json  (slow: loads each model)
if ($SkipRegionMetrics) {
    Write-Host "-SkipRegionMetrics set -- skipping compute_region_metrics.py (reusing existing region_* fields in run_loss_*.json)." -ForegroundColor Yellow
} else {
    python compute_region_metrics_simplegate.py --root $root @csvArgs --region $region
}

# Shape-output columns are a different, narrower schema than the wide ranked/filtered/
# master_best files (DEFAULT_KEEP in keep_columns.py) -- it wouldn't even keep the metrics
# that make a shape file useful (shape_rank, gds_hump, ...), so it needs its own --keep list.
$shapeKeep = @("id","run_id","arch_id","arch_hash","vds_loss","region_knee_combined_gm",
               "region_knee_ids_rmse","neg_gds_energy","gds_hump","gds_hump_max",
               "gm3_maxabs","gm3_bad_frac","gm3_start_diff","gm2_start_diff",
               "gm3_tail_maxabs","gm3_tail_bad_frac","gm3_max_missing_frac","gm3_min_missing_frac",
               "gm3_global_excess_max","gm3_global_bad_frac",
               "gds_residual_bad_frac","gds_residual_neg_max","gds_residual_pos_max",
               "gds_residual_worst_max","shape_rank")

# 2. compile: per-experiment CSVs -> master-best -> grouped ranked CSVs.
#    No xlsx is written for compile_ranked's own output (the range-tagged combined_gm /
#    ids_rmse CSVs) -- those two are the only files this pipeline keeps as plain CSV, by
#    design (they're read as-is by later steps and by hand). master_best_*.csv (the top-15
#    lists) gets a slim xlsx only, no full one.
python compile_results_csv.py --root $root @baseArgs
python compile_master_best.py --root $root --sort_by gm1_rmse,gm2_rmse,gm3_rmse --top_n 15
python compile_master_best.py --root $root --sort_by ids_rmse --top_n 15
python compile_master_best.py --root $root --sort_by combined_gm --top_n 15
python compile_ranked.py      --root $root @baseArgs @rankArgs
$masterBest = Get-ChildItem -Path $root -File -Filter "master_best_*.csv" |
    Where-Object { $_.Name -notlike "*_filtered*" } | ForEach-Object { $_.FullName }
if ($masterBest) { python keep_columns.py --csv $masterBest }

# 3. filter the ranked CSVs -> *_filtered.csv (no full xlsx -- slimmed below).
#    Runs when there's a metric -filter OR a -dropAllZero set (the latter is on by default,
#    removing the inert gm1=gm2=gm3=0 rows). Pass -dropAllZero @() to keep them.
#    Auto-skipped for "nogm" (use_gm=False) roots: EVERY row in those sweeps has
#    gm1_weight=gm2_weight=gm3_weight=0 (gm loss was never used, by construction), so
#    drop-all-zero would wipe every row to an empty file instead of filtering anything
#    meaningfully -- see the quality filter in step 3b for the useful filter on nogm runs.
$effectiveDropAllZero = $dropAllZero
if ($root -match "(?i)nogm") {
    if ($dropAllZero.Count -gt 0) {
        Write-Host "root matches 'nogm' -- skipping -dropAllZero (every row is gm-inert by construction; would empty the file)." -ForegroundColor Yellow
    }
    $effectiveDropAllZero = @()
}
if ($filter.Count -gt 0 -or $effectiveDropAllZero.Count -gt 0) {
    $fargs = @()
    foreach ($f in $filter) { $fargs += "--filter"; $fargs += $f }
    if ($effectiveDropAllZero.Count -gt 0) { $fargs += "--drop-all-zero"; $fargs += $effectiveDropAllZero }
    python filter_results.py --root $root @fargs
    $filtered = Get-ChildItem -Path $root -Recurse -Filter "*_filtered.csv" | ForEach-Object { $_.FullName }
    if ($filtered) { python keep_columns.py --csv $filtered }
}

# 3b. quality filter: EITHER region_knee_combined_gm<=maxCombinedGm AND region_knee_ids_rmse<=
#     maxIdsRmse (defaults 0.9 / 0.009, the default), OR -- when -TopN is set -- the best -TopN
#     rows by -SortBy instead. Uses --auto_number: a short "_<N>" suffix instead of a fixed
#     name, with a filter_readme.md (in $root) mapping N -> the exact criteria used, so
#     re-running compile_overall.ps1 with DIFFERENT thresholds/-TopN doesn't overwrite a
#     previous run's results -- multiple filter criteria can coexist in the same folder (same
#     criteria as before just reuses its number). No full xlsx is written -- slimmed below (and
#     step 3c reads the .csv directly, which still has file_path -- the slim xlsx never does).
$qArgs = @()
if ($TopN -gt 0) {
    $qArgs += @("--top_n", $TopN, "--sort_by", $SortBy)
    if ($Descending) { $qArgs += "--descending" }
} else {
    $qArgs += @("--filter", "region_knee_combined_gm<=$maxCombinedGm",
                "--filter", "region_knee_ids_rmse<=$maxIdsRmse")
}
python filter_results.py --root $root @qArgs --auto_number | Tee-Object -Variable fqLines
$filterNum = ($fqLines | Select-String -Pattern 'filter #(\d+)').Matches[0].Groups[1].Value
$filteredQ = Get-ChildItem -Path $root -Recurse -Filter "*_$filterNum.csv" | ForEach-Object { $_.FullName }
if ($filteredQ) { python keep_columns.py --csv $filteredQ }

# 3c. shape analysis (analyze_shape.py) on the quality-filtered "already good fits" combined_gm
#     ranking from step 3b (filter #$filterNum, just assigned/reused above), in every
#     range-tagged region folder (ranked_region_<name>_vgs..to.._vds..to../). analyze_shape.py's
#     own workflow assumes a pre-filtered "good fit" input (see its docstring) -- since it's
#     already filtered here, no --filter of its own is passed (it evaluates every row of the
#     already-filtered file). Writes <name>_shape.csv plus the gm-shape-compliant CSVs alongside
#     (CSV only; slimmed below, no full xlsx). Skips a filtered file with 0 rows (analyze_shape.py
#     would otherwise just exit with "no rows passed the filter").
$shapeInputs = Get-ChildItem -Path $root -Directory -Filter "ranked_region_*" |
    Where-Object { $_.Name -match "_vgs.*_vds" } |
    ForEach-Object { Get-ChildItem -Path $_.FullName -Filter "*_combined_gm_$filterNum.csv" }
foreach ($f in $shapeInputs) {
    $nLines = (Get-Content $f.FullName | Measure-Object -Line).Lines
    if ($nLines -le 1) {
        Write-Host "skip shape analysis (0 rows): $($f.FullName)" -ForegroundColor Yellow
        continue
    }
    Write-Host "=== shape analysis: $($f.FullName) ===" -ForegroundColor Cyan
    python analyze_shape_simplegate.py --ranked_csv $f.FullName
    $shapeOutputs = Get-ChildItem -Path $f.Directory.FullName -Filter "*_shape*.csv" | ForEach-Object { $_.FullName }
    if ($shapeOutputs) { python keep_columns.py --csv $shapeOutputs --keep $shapeKeep }
    # The 6 gmshape/gdsshape/bothshape "_ok*"/"_removed" sort/compliance files are
    # slim-xlsx-only -- delete their plain CSV now that the slim exists (the base
    # <name>_shape.csv itself, with every row's metrics, stays as CSV+slim).
    $slimOnlyCsvs = $shapeOutputs | Where-Object { $_ -notmatch "_shape\.csv$" }
    if ($slimOnlyCsvs) { Remove-Item -Path $slimOnlyCsvs -Force }
}
