#!/usr/bin/env python3
"""
pipeline_gui.py — One GUI to drive the physics_nn_pipeline scripts.

Wraps the command-line tools in this folder so they can be run without typing
long commands:

  * compare_architecture.py      (find matching ids)
  * compare_column_values.py     (per-column value diff)
  * gen_config_from_rows.py      (build a runner config)
  * multi_experiment_runner.py   (run the optimizations)
  * compile_results_csv.py / compile_ranked.py / compile_master_best.py
    / compile_results_to_xlsx.py (compile + rank results)
  * plot_best_configs.py         (plot the best runs)

Shared context at the top (experiment / master_root, measurement CSV, and a
metric -> ranked-file resolver) feeds every tab, so e.g. picking
ranking=region_knee + metric=combined_gm auto-fills
  refine_vdsk_gm_1/ranked_region_knee/refine_vdsk_gm_1_ranked_by_region_knee_combined_gm.xlsx

Pure standard library (tkinter) — no extra installs. Run with:
  python pipeline_gui.py
"""
import os
import sys
import re
import glob
import threading
import subprocess
import tkinter as tk
from tkinter import ttk, filedialog

HERE = os.path.dirname(os.path.abspath(__file__))      # physics_nn_pipeline/
REPO = os.path.dirname(HERE)                            # repo root
PY = sys.executable
CSV_DIR = os.path.join(REPO, "csvs")
CONFIG_DIR = os.path.join(HERE, "physics_nn_configs")

# Ranking flavour -> (subfolder, filename infix). Files are named
#   {exp}_ranked_by_{infix}{metric}.{ext}
RANKINGS = {"region_knee": ("ranked_region_knee", "region_knee_"),
            "global":      ("ranked_global", "")}
METRICS = ["combined_gm", "ids_rmse", "gm1_rmse", "gm2_rmse", "gm3_rmse"]
VARIANTS = {"full": "", "filtered": "_filtered", "slim": "_slim"}

# Defaults for the full compile pipeline (mirror compile_overall.ps1).
DEFAULT_REGION = "knee:vgs=-3..0,vds=0..15"
DROP_ALL_ZERO = ["gm1_weight", "gm2_weight", "gm3_weight"]   # inert all-zero-gm rows
DEFAULT_MEAS_CSV = os.path.join(CSV_DIR, "cg2h40010_new_2.4_5_2_70W_center9.csv")


# ----------------------------------------------------------------------------
# Filesystem discovery helpers
# ----------------------------------------------------------------------------
RUNS_DIR = os.path.join(REPO, "runs")


def experiment_dirs(runs_root=None):
    """Master-root experiment folders: anything holding a ranked_* subdir.

    Scans the configured runs root (where experiment dirs live after being moved
    out of the scripts folder), plus the scripts dir and repo root for backward
    compat. `runs_root` overrides the default runs/ location."""
    bases = [runs_root or RUNS_DIR, HERE, REPO]
    found = []
    for base in bases:
        if not base or not os.path.isdir(base):
            continue
        for d in sorted(glob.glob(os.path.join(base, "*"))):
            if os.path.isdir(d) and (
                os.path.isdir(os.path.join(d, "ranked_region_knee"))
                or os.path.isdir(os.path.join(d, "ranked_global"))
            ):
                found.append(d)
    # de-dupe while keeping order
    seen, out = set(), []
    for d in found:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def measurement_csvs():
    return sorted(glob.glob(os.path.join(CSV_DIR, "*.csv")))


def config_files():
    return sorted(glob.glob(os.path.join(CONFIG_DIR, "*.json")))


def _ranking_paths(ranking):
    """(subfolder, filename_infix) for a ranking label. Handles 'global', 'region_<name>',
    and range-tagged 'region_<name>_vgs..to.._vds..to..' (files INSIDE a range folder still
    use the region_<name>_ infix; only the FOLDER carries the range)."""
    if ranking in RANKINGS:
        return RANKINGS[ranking]
    if ranking == "global":
        return "ranked_global", ""
    if ranking.startswith("region_"):
        name = ranking[len("region_"):].split("_vgs", 1)[0]   # region_knee_vgs.. -> knee
        return "ranked_" + ranking, f"region_{name}_"
    return "ranked_" + ranking, ""


def discover_rankings(exp_dir):
    """Ranking labels available under exp_dir (for the 'Ranked file' dropdown): 'global',
    'region_<name>', and any range-tagged 'region_<name>_vgs..to.._vds..to..'."""
    labels = []
    if exp_dir and os.path.isdir(exp_dir):
        for d in sorted(os.listdir(exp_dir)):
            if not os.path.isdir(os.path.join(exp_dir, d)):
                continue
            if d == "ranked_global":
                labels.append("global")
            elif d.startswith("ranked_region_"):
                labels.append(d[len("ranked_"):])            # region_knee[_vgs..]
    order = {"global": 0, "region_knee": 1}
    labels = sorted(set(labels), key=lambda x: (order.get(x, 2), x))
    return labels or list(RANKINGS)


def resolve_ranked(exp_dir, ranking, metric, ext, variant):
    """Build the ranked CSV/xlsx path from the shared selectors."""
    if not exp_dir:
        return ""
    sub, infix = _ranking_paths(ranking)
    name = os.path.basename(exp_dir.rstrip("/\\"))
    fname = f"{name}_ranked_by_{infix}{metric}{VARIANTS[variant]}.{ext}"
    return os.path.join(exp_dir, sub, fname)


# ----------------------------------------------------------------------------
# Small widget helpers
# ----------------------------------------------------------------------------
def labeled_combo(parent, label, values, width=46, default=None):
    row = ttk.Frame(parent)
    row.pack(fill="x", pady=2)
    ttk.Label(row, text=label, width=18).pack(side="left")
    var = tk.StringVar(value=default or (values[0] if values else ""))
    cb = ttk.Combobox(row, textvariable=var, values=values, width=width)
    cb.pack(side="left", fill="x", expand=True)
    return var, cb


def labeled_entry(parent, label, default="", width=46):
    row = ttk.Frame(parent)
    row.pack(fill="x", pady=2)
    ttk.Label(row, text=label, width=18).pack(side="left")
    var = tk.StringVar(value=default)
    ttk.Entry(row, textvariable=var, width=width).pack(side="left", fill="x", expand=True)
    return var


def labeled_path(parent, label, default="", is_dir=False):
    row = ttk.Frame(parent)
    row.pack(fill="x", pady=2)
    ttk.Label(row, text=label, width=18).pack(side="left")
    var = tk.StringVar(value=default)
    ttk.Entry(row, textvariable=var).pack(side="left", fill="x", expand=True)

    def browse():
        p = (filedialog.askdirectory(initialdir=HERE) if is_dir
             else filedialog.askopenfilename(initialdir=HERE))
        if p:
            var.set(p)
    ttk.Button(row, text="...", width=3, command=browse).pack(side="left", padx=2)
    return var


# ----------------------------------------------------------------------------
# Main application
# ----------------------------------------------------------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("physics_nn_pipeline — control panel")
        self.geometry("1080x960")
        self.minsize(900, 600)
        self.proc = None

        self._build_shared()
        # Tabs and log share a draggable vertical split, so the log can be resized.
        paned = ttk.PanedWindow(self, orient="vertical")
        paned.pack(fill="both", expand=True, padx=8, pady=4)
        nb = ttk.Notebook(paned)
        paned.add(nb, weight=0)
        self._tab_plot_row(nb)
        self._tab_compare(nb)
        self._tab_generate(nb)
        self._tab_run(nb)
        self._tab_compile(nb)
        self._tab_plot(nb)
        self._build_log(paned)
        self._refresh_resolved()

    # --- shared context -----------------------------------------------------
    def _build_shared(self):
        box = ttk.LabelFrame(self, text="Shared context")
        box.pack(fill="x", padx=8, pady=6)

        # Runs root: where experiment dirs live (default runs/). Browsable so an
        # external master root works too. Drives experiment discovery + --master_root_path.
        rr = ttk.Frame(box)
        rr.pack(fill="x", pady=2)
        ttk.Label(rr, text="Runs root", width=18).pack(side="left")
        self.runs_root_var = tk.StringVar(value=RUNS_DIR)
        ttk.Entry(rr, textvariable=self.runs_root_var).pack(side="left", fill="x", expand=True)
        ttk.Button(rr, text="...", width=3,
                   command=lambda: self._set_runs_root(filedialog.askdirectory(initialdir=RUNS_DIR))).pack(side="left", padx=2)
        ttk.Button(rr, text="Rescan", width=7, command=self._rescan_experiments).pack(side="left", padx=2)

        self.exp_var, self.exp_cb = labeled_combo(box, "Experiment (root)", [])
        self._exp_map = {}

        csvs = measurement_csvs()
        self.meas_var, _ = labeled_combo(box, "Measurement CSV", [os.path.relpath(c, REPO) for c in csvs])

        # ranking + metric + format resolver
        rrow = ttk.Frame(box)
        rrow.pack(fill="x", pady=2)
        ttk.Label(rrow, text="Ranked file", width=18).pack(side="left")
        self.rank_var = tk.StringVar(value="region_knee")
        self.rank_cb = ttk.Combobox(rrow, textvariable=self.rank_var, values=list(RANKINGS), width=26)
        self.rank_cb.pack(side="left", padx=2)   # wide: range-window folders have long labels
        self.metric_var = tk.StringVar(value="combined_gm")
        ttk.Combobox(rrow, textvariable=self.metric_var, values=METRICS, width=14).pack(side="left", padx=2)
        self.fmt_var = tk.StringVar(value="xlsx")
        ttk.Combobox(rrow, textvariable=self.fmt_var, values=["xlsx", "csv"], width=7).pack(side="left", padx=2)
        self.variant_var = tk.StringVar(value="full")
        ttk.Combobox(rrow, textvariable=self.variant_var, values=list(VARIANTS), width=9).pack(side="left", padx=2)

        prow = ttk.Frame(box)
        prow.pack(fill="x", pady=2)
        ttk.Label(prow, text="-> resolves to", width=18).pack(side="left")
        self.resolved_var = tk.StringVar(value="")
        # Read-only Entry (not a Label) so the path is selectable / copyable with Ctrl+C.
        self.resolved_lbl = tk.Entry(prow, textvariable=self.resolved_var, relief="flat",
                                     readonlybackground=self.cget("background"), foreground="gray")
        self.resolved_lbl.configure(state="readonly")
        self.resolved_lbl.pack(side="left", fill="x", expand=True)
        self.resolved_status = ttk.Label(prow, text="", width=10, foreground="#b00")
        self.resolved_status.pack(side="left")

        # Quick-open buttons for the current ranked file (the run being analyzed).
        orow = ttk.Frame(box)
        orow.pack(fill="x", pady=2)
        ttk.Label(orow, text="Open current file", width=18).pack(side="left")
        ttk.Button(orow, text="Open xlsx", command=lambda: self._open_path(self.resolved_path("xlsx"))).pack(side="left", padx=2)
        ttk.Button(orow, text="Open csv", command=lambda: self._open_path(self.resolved_path("csv"))).pack(side="left", padx=2)
        ttk.Button(orow, text="Reveal in Explorer", command=lambda: self._reveal_path(self.resolved_path())).pack(side="left", padx=2)
        ttk.Button(orow, text="Copy path", command=lambda: self._copy_path(self.resolved_path())).pack(side="left", padx=2)
        ttk.Button(orow, text="Open ranked folder",
                   command=lambda: self._open_path(os.path.dirname(self.resolved_path()))).pack(side="left", padx=2)

        for v in (self.exp_var, self.rank_var, self.metric_var, self.fmt_var, self.variant_var):
            v.trace_add("write", lambda *_: self._refresh_resolved())

        self._rescan_experiments()

    def _set_runs_root(self, path):
        if path:
            self.runs_root_var.set(path)
            self._rescan_experiments()

    def _rescan_experiments(self):
        """Re-discover experiment dirs under the current Runs root and refill the combo."""
        root = self.runs_root_var.get()

        def under_root(d):
            try:
                return os.path.commonpath([os.path.abspath(d), os.path.abspath(root)]) == os.path.abspath(root)
            except ValueError:           # different drives
                return False

        def label(d):
            # Relative to the Runs root when inside it (clean: "refine_vdsk_gm_1"),
            # else relative to the repo so out-of-tree experiments stay distinguishable.
            return os.path.relpath(d, root) if root and under_root(d) else os.path.relpath(d, REPO)

        exps = experiment_dirs(root)
        labels = [label(d) for d in exps]
        self._exp_map = dict(zip(labels, exps))
        self.exp_cb["values"] = labels
        cur = self.exp_var.get()
        if cur not in labels:
            self.exp_var.set(next((l for l in labels if "vdsk_gm_1" in l),
                                  labels[0] if labels else ""))
        self._refresh_resolved()

    def exp_path(self):
        return self._exp_map.get(self.exp_var.get(), "")

    def meas_path(self):
        v = self.meas_var.get()
        return os.path.join(REPO, v) if v else ""

    def resolved_path(self, ext=None):
        return resolve_ranked(self.exp_path(), self.rank_var.get(), self.metric_var.get(),
                              ext or self.fmt_var.get(), self.variant_var.get())

    def _refresh_resolved(self):
        # Keep the 'Ranked file' dropdown in sync with the current experiment's folders
        # (adds range-window folders like region_knee_vgs-2.5to0_vds0to6 when present).
        if hasattr(self, "rank_cb"):
            labels = discover_rankings(self.exp_path())
            if list(self.rank_cb["values"]) != labels:
                self.rank_cb["values"] = labels
        self._autofill_from_meta()
        p = self.resolved_path()
        exists = os.path.isfile(p)
        self.resolved_var.set(p or "(pick an experiment)")   # clean path only, for copying
        self.resolved_lbl.configure(foreground="#1a7f37" if exists else "#b00")
        self.resolved_status.configure(text="" if exists else "[missing]")
        # Keep the Plot config (row) tab's CSV display in sync (forced to .csv).
        if hasattr(self, "pr_csv_var"):
            self.pr_csv_var.set(self.resolved_path("csv"))

    def _autofill_from_meta(self):
        """When the selected experiment changes, fill Measurement CSV and base CSV from the
        experiment's base_files/_run_meta.json so the user doesn't retype them.

        Overwrites a field only when it is blank OR still holds the value WE auto-filled for
        the previous experiment -- so switching experiments refreshes both, but a value the
        user typed is never clobbered."""
        exp = self.exp_path()
        if exp == getattr(self, "_last_meta_exp", None):
            return
        self._last_meta_exp = exp
        if not exp:
            return
        try:
            import run_artifacts
            meta = run_artifacts.run_meta(exp)
        except Exception:
            return

        def _apply(var, new_val, remembered_attr):
            cur = var.get().strip()
            if cur and cur != getattr(self, remembered_attr, None):
                return                      # user typed something -> leave it
            var.set(new_val or "")
            setattr(self, remembered_attr, new_val or "")

        csv = meta.get("csv")
        if csv:
            try:
                csv = os.path.relpath(csv, REPO)
            except ValueError:
                pass
        if hasattr(self, "meas_var"):
            _apply(self.meas_var, csv, "_autofilled_meas")
        if hasattr(self, "comp_base_csv"):
            _apply(self.comp_base_csv, meta.get("base_csv"), "_autofilled_base")

    # --- open / reveal / copy helpers --------------------------------------
    def _open_path(self, p):
        """Open a file or folder in its default app (Excel for xlsx, etc.)."""
        if not p or not os.path.exists(p):
            self._append(f"\n[open] not found: {p}\n")
            return
        try:
            os.startfile(p)  # Windows: launches the associated app / Explorer
        except Exception as e:
            self._append(f"\n[open error: {e}]\n")

    def _reveal_path(self, p):
        """Open Explorer with the file selected."""
        if not p or not os.path.exists(p):
            self._append(f"\n[reveal] not found: {p}\n")
            return
        subprocess.Popen(f'explorer /select,"{os.path.normpath(p)}"')

    def _copy_path(self, p):
        self.clipboard_clear()
        self.clipboard_append(p or "")
        self._append(f"\n[copied] {p}\n")

    def _help(self, parent, text):
        """A gray, read-only but selectable/copyable help block (drop-in for a Label)."""
        lines = text.split("\n")
        w = tk.Text(parent, height=len(lines),
                    width=max((len(l) for l in lines), default=10) + 1,
                    wrap="none", relief="flat", borderwidth=0, highlightthickness=0,
                    background=self.cget("background"), foreground="gray", takefocus=0)
        w.insert("1.0", text)

        def _ro(e):
            if (e.state & 0x4) and e.keysym.lower() in ("c", "a"):
                return          # allow Ctrl+C (copy) / Ctrl+A (select all)
            return "break"      # block edits — read-only, but mouse-selectable
        w.bind("<Key>", _ro)
        return w

    PLOT_PNG = "plot_saved_state_full.png"

    def _selected_run_pngs(self):
        """Map the Plot-row selection (id/row/config_name) to each run's output PNG.

        Reuses plot_csv_row's row->run_dir logic (with reroot) so it matches exactly
        what the script plotted."""
        import csv as _csv
        import plot_csv_row as pcr
        csv_path = self.resolved_path("csv")
        if not os.path.isfile(csv_path):
            return []
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(_csv.DictReader(f))
        if not rows:
            return []
        mode, raw = self.pr_mode.get(), self.pr_value.get()
        vals = [x.strip() for x in raw.split(",") if x.strip()]
        picked = []
        if mode == "id":
            by = {str(r.get("id")): r for r in rows}
            picked = [by[v] for v in vals if v in by]
        elif mode == "row":
            picked = [rows[int(v) - 1] for v in vals if v.isdigit() and 1 <= int(v) <= len(rows)]
        else:  # config_name
            want = set(vals)
            picked = [r for r in rows if r.get("config_name") in want]
        # plot_csv_row.py copies each plotted config's PNG into
        #   <csv_dir>/plotted_configs/<csv_stem>_id<id>/plot_saved_state_full.png
        # Prefer that curated copy; fall back to the run dir's own PNG if it's not there yet.
        csv_dir = os.path.dirname(csv_path)
        csv_stem = os.path.splitext(os.path.basename(csv_path))[0]
        pngs = []
        for r in picked:
            id_val = str(r.get("id", r.get("config_name", ""))).strip()
            plotted = os.path.join(csv_dir, "plotted_configs", f"{csv_stem}_id{id_val}", self.PLOT_PNG)
            if os.path.isfile(plotted):
                pngs.append(plotted)
                continue
            rd = pcr._run_dir_from_row(r)
            if rd is not None:
                pngs.append(os.path.join(str(rd), self.PLOT_PNG))
        return pngs

    def _open_selected_plots(self, rc=None):
        """Open the PNG(s) for the current selection (skips when a run failed)."""
        if rc not in (None, 0):
            return
        pngs = self._selected_run_pngs()
        if not pngs:
            self._append("\n[show] no matching run dirs for the current selection\n")
            return
        opened = 0
        for p in pngs:
            if os.path.isfile(p):
                self._open_path(p)
                opened += 1
            else:
                self._append(f"\n[show] plot not found (yet): {p}\n")
        if opened:
            self._append(f"\n[show] opened {opened} plot(s)\n")

    # --- tab: plot config from a CSV row -----------------------------------
    def _tab_plot_row(self, nb):
        t = self._scroll_tab(nb, "Plot config (row)")
        ttk.Label(t, text="plot_csv_row.py — plot a SPECIFIC config picked from a ranked CSV "
                          "(by id / row / config_name).", foreground="gray").pack(anchor="w", pady=(6, 2))
        # Always uses the shared-context selection, forced to .csv (the script needs
        # CSV; csv/xlsx hold identical data). Tracks the shared selectors live.
        crow = ttk.Frame(t)
        crow.pack(fill="x", pady=2)
        ttk.Label(crow, text="Ranked CSV (shared)", width=18).pack(side="left")
        self.pr_csv_var = tk.StringVar(value="")
        pr_csv_entry = tk.Entry(crow, textvariable=self.pr_csv_var, relief="flat",
                                readonlybackground=self.cget("background"))
        pr_csv_entry.configure(state="readonly")
        pr_csv_entry.pack(side="left", fill="x", expand=True)

        srow = ttk.Frame(t)
        srow.pack(fill="x", pady=2)
        ttk.Label(srow, text="Select by", width=18).pack(side="left")
        self.pr_mode = tk.StringVar(value="id")
        ttk.Combobox(srow, textvariable=self.pr_mode, values=["id", "row", "config_name"],
                     width=14, state="readonly").pack(side="left", padx=2)
        self.pr_value = tk.StringVar(value="1")
        ttk.Entry(srow, textvariable=self.pr_value, width=40).pack(side="left", fill="x", expand=True, padx=2)
        ttk.Label(t, text="   value(s): comma-separated, e.g. id = 1,2,3   (id survives Excel re-sorting)",
                  foreground="gray").pack(anchor="w")

        self.pr_val = labeled_entry(t, "--val (blank=off)", "")
        self.pr_minvgs = labeled_entry(t, "--min_vgs", "-4.0", width=10)
        cr = ttk.Frame(t)
        cr.pack(anchor="w", pady=2)
        self.pr_zero = tk.BooleanVar(value=False)
        self.pr_dry = tk.BooleanVar(value=False)
        self.pr_open = tk.BooleanVar(value=True)
        ttk.Checkbutton(cr, text="--add_zero_vds", variable=self.pr_zero).pack(side="left", padx=4)
        ttk.Checkbutton(cr, text="--dry_run", variable=self.pr_dry).pack(side="left", padx=4)
        ttk.Checkbutton(cr, text="open plot when done", variable=self.pr_open).pack(side="left", padx=4)

        br = ttk.Frame(t)
        br.pack(anchor="w", pady=6)
        ttk.Button(br, text="Plot selected config", command=self._run_plot_row).pack(side="left", padx=3)
        ttk.Button(br, text="Show plot(s)", command=self._open_selected_plots).pack(side="left", padx=3)

    def _run_plot_row(self):
        cmd = [PY, "plot_csv_row.py", "--ranked_csv", self.resolved_path("csv"), "--csv", self.meas_path(),
               f"--{self.pr_mode.get()}", self.pr_value.get().strip()]
        if self.pr_val.get().strip():
            cmd += ["--val", self.pr_val.get().strip()]
        if self.pr_minvgs.get().strip() not in ("", "-4.0"):
            cmd += ["--min_vgs", self.pr_minvgs.get().strip()]
        if self.pr_zero.get():
            cmd += ["--add_zero_vds"]
        if self.pr_dry.get():
            cmd += ["--dry_run"]
        # After a successful (non-dry) run, open each selected run's plot PNG.
        on_done = None if self.pr_dry.get() or not self.pr_open.get() else self._open_selected_plots
        self.run(cmd, on_done=on_done)

    # --- tab: compare -------------------------------------------------------
    def _tab_compare(self, nb):
        t = self._scroll_tab(nb, "Compare")
        ttk.Label(t, text="Compare a ranked file against another (architecture / value diff).",
                  foreground="gray").pack(anchor="w", pady=(6, 2))
        self.cmp_f1 = labeled_path(t, "File 1 (xlsx/csv)")
        self.cmp_f2 = labeled_path(t, "File 2 (xlsx/csv)")
        self.cmp_id = labeled_entry(t, "Ref id (file1)", "70", width=10)
        self.cmp_excl = labeled_entry(t, "Extra --exclude", "region_weight")
        self.cmp_tol = labeled_entry(t, "Tolerance", "0", width=10)

        ttk.Button(t, text="Use resolved file as File 1",
                   command=lambda: self.cmp_f1.set(self.resolved_path())).pack(anchor="w", pady=2)

        br = ttk.Frame(t)
        br.pack(anchor="w", pady=6)
        ttk.Button(br, text="Find matching ids", command=self._run_compare_arch).pack(side="left", padx=3)
        ttk.Button(br, text="Per-column value diff", command=self._run_compare_vals).pack(side="left", padx=3)

    def _run_compare_arch(self):
        cmd = [PY, "compare_architecture.py", self.cmp_f1.get(), self.cmp_id.get(), self.cmp_f2.get()]
        if self.cmp_excl.get().strip():
            cmd += ["--exclude"] + self.cmp_excl.get().split()
        if self.cmp_tol.get().strip() not in ("", "0"):
            cmd += ["--tol", self.cmp_tol.get()]
        self.run(cmd)

    def _run_compare_vals(self):
        cmd = [PY, "compare_column_values.py", self.cmp_f1.get(), self.cmp_f2.get()]
        if self.cmp_excl.get().strip():
            cmd += ["--exclude"] + self.cmp_excl.get().split()
        self.run(cmd)

    # --- tab: generate config ----------------------------------------------
    def _tab_generate(self, nb):
        t = self._scroll_tab(nb, "Generate config")
        ttk.Label(t, text="gen_config_from_rows.py — build a runner config from base-CSV rows.",
                  foreground="gray").pack(anchor="w", pady=(6, 2))
        self.gen_csv = labeled_path(t, "Base CSV")
        ttk.Button(t, text="Use resolved (as .csv)",
                   command=lambda: self.gen_csv.set(self.resolved_path("csv"))).pack(anchor="w", pady=2)

        self.gen_ids = labeled_entry(t, "--ids", "all")
        self.gen_top = labeled_entry(t, "--top (blank=off)", "", width=10)
        self.gen_filter = labeled_entry(t, "--filter (blank=off)", "")
        self._help(t,
"   Select base rows (--filter first, then --ids OR --top). Drag-select an example to copy:\n"
                       "     --ids:  all | 70 | 70-140 | 70,72,80            --top:  20\n"
                       "     --filter (COL<op>VAL, op = < > <= >=; combine with 'and'/commas):\n"
                       "        region_knee_ids_rmse<=0.1 and region_knee_combined_gm<=0.8"
                  ).pack(anchor="w")

        ttk.Label(t, text="--set  (overrides; space- or newline-separated)").pack(anchor="w", pady=(6, 0))
        self.gen_set = tk.Text(t, height=4, wrap="word")
        self.gen_set.insert("1.0", "region_weights=[2,4]")
        self.gen_set.pack(fill="x", pady=2)
        self._help(t,
"   KEY=VALUE, space/newline-separated, JSON-parsed; a list for a swept dim replaces & sweeps it.\n"
                       "   Only --set what you want to CHANGE; the rest (incl. the region box) comes from the base CSV.\n"
                       "   Examples -- copy a line, paste, edit:  region_weights=[2,4]   gm_vds_mins=[2.6,3.0]\n"
                       "     seeds=[27,42]   epochs=1000   use_gm=true   equation_type=pure:vdsgate_vdsk_quad"
                  ).pack(anchor="w")

        self.gen_prefix = labeled_entry(t, "--prefix", "refine_vdsk_gm_1_rw")
        self._help(t,
"   Names the experiment(s): <prefix>__<combo>__<weights>__<mode>, which is also the run\n"
                       "   SUBFOLDER under the master root. A DIFFERENT prefix -> a different subfolder, so\n"
                       "   add-on runs (e.g. an extra vds_loss) land beside existing runs without colliding.\n"
                       "   Reuse the SAME prefix only to resume/extend the same experiment."
                  ).pack(anchor="w")
        self.gen_out = labeled_entry(t, "Output name (.json)", "refine_vdsk_gm_1_regionwt_2_4.json")
        self.gen_workers = labeled_entry(t, "--max_workers", "42", width=10)
        ttk.Button(t, text="Generate config", command=self._run_generate).pack(anchor="w", pady=6)

    def _run_generate(self):
        out = self.gen_out.get().strip()
        out_path = out if os.path.isabs(out) else os.path.join("physics_nn_configs", out)
        cmd = [PY, "gen_config_from_rows.py", "--csv", self.gen_csv.get(),
               "--out", out_path, "--prefix", self.gen_prefix.get(),
               "--max_workers", self.gen_workers.get()]
        if self.gen_top.get().strip():
            cmd += ["--top", self.gen_top.get().strip()]
        else:
            cmd += ["--ids", self.gen_ids.get().strip() or "all"]
        # --filter: multiple COL<op>VAL conditions (combine with 'and'/commas), each a
        # repeatable --filter (AND) -- same parser as the Compile tab's Filter button.
        for cond in self._parse_filter_conditions(self.gen_filter.get()):
            cmd += ["--filter", cond]
        # --set: split into KEY=VALUE overrides only at whitespace that precedes a new
        # "key=", so list values containing spaces (e.g. [2, 4]) stay intact.
        set_text = self.gen_set.get("1.0", "end").strip()
        if set_text:
            import re
            for kv in re.split(r"\s+(?=[A-Za-z_]\w*=)", set_text):
                kv = kv.strip()
                if kv:
                    cmd += ["--set", kv]
        self.run(cmd)

    # --- tab: run optimizations --------------------------------------------
    def _tab_run(self, nb):
        t = self._scroll_tab(nb, "Run optimizations")
        ttk.Label(t, text="multi_experiment_runner.py — launch the sweep.",
                  foreground="gray").pack(anchor="w", pady=(6, 2))
        cfgs = [os.path.relpath(c, HERE) for c in config_files()]
        self.run_cfg, _ = labeled_combo(t, "Config (.json)", cfgs)
        self.run_master = labeled_entry(t, "--master_root_path", RUNS_DIR)
        brow = ttk.Frame(t)
        brow.pack(anchor="w", pady=2)
        ttk.Button(brow, text="Use Runs root",
                   command=lambda: self.run_master.set(self.runs_root_var.get())).pack(side="left", padx=2)
        ttk.Button(brow, text="Use experiment dir",
                   command=lambda: self.run_master.set(self.exp_path())).pack(side="left", padx=2)
        ttk.Label(t, text="   Runs root -> new experiment folders created directly under runs/.\n"
                          "   experiment dir -> runs nested inside the chosen experiment (old layout).",
                  foreground="gray").pack(anchor="w")
        self.run_workers = labeled_entry(t, "--max_workers (blank=cfg)", "", width=10)
        ttk.Button(t, text="Run sweep", command=self._run_runner).pack(anchor="w", pady=6)

    def _run_runner(self):
        cmd = [PY, "multi_experiment_runner.py", "--config", self.run_cfg.get(),
               "--csv", self.meas_path()]
        if self.run_master.get().strip():
            cmd += ["--master_root_path", self.run_master.get().strip()]
        if self.run_workers.get().strip():
            cmd += ["--max_workers", self.run_workers.get().strip()]
        self.run(cmd)

    # --- tab: compile -------------------------------------------------------
    def _tab_compile(self, nb):
        t = self._scroll_tab(nb, "Compile / rank")
        ttk.Label(t, text="Turn the raw training runs into the files you analyze. Uses the Experiment root above.",
                  foreground="gray").pack(anchor="w", pady=(6, 2))

        # --- branch A: the ranked files you analyze ------------------------
        main = ttk.LabelFrame(t, text="Make the ranked files you analyze")
        main.pack(fill="x", padx=2, pady=4)

        self.comp_metrics = labeled_entry(main, "metrics to rank by", "region_knee_combined_gm,region_knee_ids_rmse")
        self._help(main,
"   comma-separated; writes ONE ranked file (csv + xlsx) per metric listed."
                  ).pack(anchor="w", pady=(0, 4))
        self.comp_region = labeled_entry(main, "region (for region_* metrics)", DEFAULT_REGION)
        self._help(main,
"   Analysis window where region_<name>_* metrics are evaluated (used by 'Rank runs + make Excel').\n"
                       "   'name:vgs=lo..hi,vds=lo..hi' -> columns region_<name>_*.  e.g. knee:vgs=-3..0,vds=0..15\n"
                       "   (This is the METRIC window, separate from the training-time region box.)"
                  ).pack(anchor="w", pady=(0, 4))
        self.comp_base_csv = labeled_path(main, "base CSV (optional)")
        self._help(main,
"   --base_csv: the base ranked CSV this sweep was generated from -> fills base_exp/base_id/base_csv\n"
                       "   (by matching arch_hash) for LEGACY folders. Blank = rely on the run's _provenance.json\n"
                       "   (auto for sweeps made after that feature). Used by the 3 buttons below."
                  ).pack(anchor="w", pady=(0, 4))

        # one-click: rank, then convert every ranked csv to xlsx
        ttk.Button(main, text="Rank runs + make Excel  (recommended)",
                   command=self._run_rank_and_xlsx).pack(anchor="w", pady=(2, 0))
        self._help(main,
"   Full compile pipeline (same as compile_overall.ps1), in order:\n"
                       "   region metrics -> raw tables -> leaderboards -> ranked csv -> xlsx ->\n"
                       "   *_filtered.xlsx (drops inert all-zero-gm rows) -> *_slim.xlsx. THESE\n"
                       "   ranked files are what the Plot / Generate / Compare tabs use.\n"
                       "   Uses the Experiment root + Measurement CSV from Shared context above."
                  ).pack(anchor="w", pady=(0, 4))

        ttk.Label(main, text="...or run a single step (Filter/Slim act ONLY on the file resolved in Shared context):",
                  foreground="gray").pack(anchor="w")
        self.comp_filter = labeled_entry(main, "filter conditions", "")
        self.comp_keep = labeled_entry(main, "slim: keep columns", "")
        self._help(main,
"   'filter conditions' (Filter btn): COL<op>VAL (op: <= >= < > =), combine with 'and'/commas.\n"
                       "   Blank = just drop inert all-zero-gm rows. Copy an example into the box above:\n"
                       "     region_knee_ids_rmse<=0.1 and region_knee_combined_gm<=0.8\n"
                       "     combined_gm<=0.5, ids_rmse<=0.02, gm2_weight>0\n"
                       "\n"
                       "   'slim: keep columns' (Slim btn): space/comma-separated columns, in order.\n"
                       "   Blank = the default set below (includes run_id, arch_id, arch_hash, base_id, vds_loss). Copy to edit:\n"
                       "     id run_id arch_id arch_hash base_id eq combined_gm ids_rmse region_knee_combined_gm region_knee_ids_rmse region_weight vds_loss region_vds_hi region_vds_lo region_vgs_hi region_vgs_lo gm_vds_min gm_vgs_min"
                  ).pack(anchor="w", pady=(0, 4))
        sbr = ttk.Frame(main)
        sbr.pack(anchor="w", pady=(0, 4))
        ttk.Button(sbr, text="Rank runs only", command=self._run_ranked).pack(side="left", padx=3)
        ttk.Button(sbr, text="Selected ranked -> Excel", command=self._run_to_xlsx).pack(side="left", padx=3)
        ttk.Button(sbr, text="Filter", command=self._run_filter).pack(side="left", padx=3)
        ttk.Button(sbr, text="Slim", command=self._run_slim).pack(side="left", padx=3)
        self._help(main,
"   'Rank runs only' -> just the csv files.   'Selected ranked -> Excel' -> xlsx of the resolved file.\n"
                       "   'Filter' -> <resolved>_filtered.csv/.xlsx: applies the filter conditions above AND drops\n"
                       "   inert all-zero-gm rows.   'Slim' -> <resolved>_slim.xlsx keeping the columns above.\n"
                       "   All three act on the ONE file resolved in Shared context (pick ranking/metric/format there)."
                  ).pack(anchor="w", pady=(0, 4))

        # --- branch B: optional leaderboard (separate, 2 steps) ------------
        opt = ttk.LabelFrame(t, text="Optional: best-of-each leaderboard  (independent of the above)")
        opt.pack(fill="x", padx=2, pady=4)
        self._help(opt,
"   Builds ONE table with the best few runs from EACH experiment folder.\n"
                       "   Skip this unless you want such a summary. Two steps, a then b:"
                  ).pack(anchor="w", pady=(2, 2))

        ttk.Button(opt, text="a. Collect raw tables", command=self._run_results_csv).pack(anchor="w", pady=(2, 0))
        self._help(opt,
"   Writes compiled_results*.csv (one raw per-run table per experiment).\n"
                       "   Required input for step b."
                  ).pack(anchor="w")
        self.comp_metric1 = labeled_entry(opt, "best judged by metric", "combined_gm", width=16)
        self.comp_topn = labeled_entry(opt, "how many per folder", "15", width=10)
        ttk.Button(opt, text="b. Build leaderboard (master_best)", command=self._run_master_best).pack(anchor="w", pady=(2, 0))
        self._help(opt,
"   Takes the N best runs (by the metric above) from each experiment's raw table\n"
                       "   and concatenates them into one master_best.csv."
                  ).pack(anchor="w", pady=(0, 4))

    def _base_csv_args(self):
        """['--base_csv', <path>] when the 'base CSV (optional)' field is set, else []."""
        b = self.comp_base_csv.get().strip()
        return ["--base_csv", b] if b else []

    def _run_results_csv(self):
        self.run([PY, "compile_results_csv.py", "--root", self.exp_path(), *self._base_csv_args()])

    def _run_ranked(self):
        cmd = [PY, "compile_ranked.py", "--root", self.exp_path(), *self._base_csv_args()]
        if self.comp_metrics.get().strip():
            cmd += ["--metrics", self.comp_metrics.get().strip()]
        self.run(cmd)

    def _run_rank_and_xlsx(self):
        """Full compile pipeline -- the GUI equivalent of compile_overall.ps1.

        Runs, in order and stopping on the first failure:
          1. compute_region_metrics  (writes region_knee_* into each run_loss JSON)
          2. compile_results_csv      (per-experiment raw tables)
          3. compile_master_best x3   (gm / ids / combined leaderboards)
          4. compile_ranked           (the *_ranked_by_*.csv files)
          5. compile_results_to_xlsx  (matching .xlsx for every ranked csv)
          6. filter_results           (drop inert all-zero-gm rows -> *_filtered.xlsx)
          7. keep_columns             (slim copies of the combined_gm/ids_rmse rankings)
        """
        root = self.exp_path()
        if not root:
            self._append("\n[compile] pick an Experiment root first\n")
            return
        # measurement CSV: the field (if set), else auto-detect from the experiment's
        # base_files/_run_meta.json, else the hardcoded default.
        csv = self.meas_path()
        if not csv:
            try:
                import run_artifacts
                csv = run_artifacts.run_meta(root).get("csv") or ""
            except Exception:
                csv = ""
        csv = csv or DEFAULT_MEAS_CSV
        if not os.path.isfile(csv):
            self._append(f"\n[compile] measurement CSV not found: {csv}\n"
                         "          pick one in 'Measurement CSV' above (region metrics need it)\n")
            return

        base = self._base_csv_args()   # fills base_exp/base_id/base_csv when a base CSV is given
        ranked = [PY, "compile_ranked.py", "--root", root, *base]
        if self.comp_metrics.get().strip():
            ranked += ["--metrics", self.comp_metrics.get().strip()]

        steps = [
            [PY, "compute_region_metrics.py", "--root", root, "--csv", csv,
             "--region", (self.comp_region.get().strip() or DEFAULT_REGION)],
            [PY, "compile_results_csv.py", "--root", root, *base],
            [PY, "compile_master_best.py", "--root", root, "--sort_by", "gm1_rmse,gm2_rmse,gm3_rmse", "--top_n", "15"],
            [PY, "compile_master_best.py", "--root", root, "--sort_by", "ids_rmse", "--top_n", "15"],
            [PY, "compile_master_best.py", "--root", root, "--sort_by", "combined_gm", "--top_n", "15"],
            ranked,
            [PY, "compile_results_to_xlsx.py", "--root", root],
            [PY, "filter_results.py", "--root", root, "--drop-all-zero", *DROP_ALL_ZERO, "--xlsx"],
            lambda: self._slim_cmd(root),   # lazy: globs the xlsx the steps above just wrote
        ]
        self.run_sequence(steps)

    def _slim_cmd(self, root):
        """Build the keep_columns.py command for the combined_gm/ids_rmse rankings
        (full + filtered) in EVERY ranked_* folder under `root` -- ranked_global,
        ranked_region_<name>, AND range-tagged ranked_region_<name>_vgs..to.. -- so the
        range-window folders are slimmed too. Returns None to skip when none exist."""
        cands = []
        for sub in glob.glob(os.path.join(root, "ranked_*")):
            if not os.path.isdir(sub):
                continue
            for pat in ("*_ranked_by_*combined_gm.xlsx", "*_ranked_by_*combined_gm_filtered.xlsx",
                        "*_ranked_by_*ids_rmse.xlsx", "*_ranked_by_*ids_rmse_filtered.xlsx"):
                cands += glob.glob(os.path.join(sub, pat))
        cands = sorted(set(cands))
        if not cands:
            self._append("\n[slim] no combined_gm/ids_rmse rankings found; skipping\n")
            return None
        return [PY, "keep_columns.py", "--xlsx", *cands]

    def _run_master_best(self):
        self.run([PY, "compile_master_best.py", "--root", self.exp_path(),
                  "--metric", self.comp_metric1.get().strip(),
                  "--top_n", self.comp_topn.get().strip()])

    def _run_to_xlsx(self):
        # convert the currently-resolved ranked CSV to xlsx
        self.run([PY, "compile_results_to_xlsx.py", "--csv", self.resolved_path("csv")])

    def _run_filter(self):
        """Filter ONLY the file resolved in Shared context -> <name>_filtered.csv/.xlsx.
        Applies the 'filter conditions' (COL<op>VAL, AND-combined) plus the inert
        all-zero-gm drop. Blank conditions = just the all-zero drop."""
        csv_path = self.resolved_path("csv")
        if not os.path.isfile(csv_path):
            self._append(f"\n[filter] resolved ranked CSV not found: {csv_path}\n"
                         "         pick an existing ranking/metric in Shared context above.\n")
            return
        cmd = [PY, "filter_results.py", "--csv", csv_path, "--drop-all-zero", *DROP_ALL_ZERO, "--xlsx"]
        for cond in self._parse_filter_conditions(self.comp_filter.get()):
            cmd += ["--filter", cond]
        self.run(cmd)

    @staticmethod
    def _parse_filter_conditions(text):
        """Split a filter string into individual COL<op>VAL conditions. Separators:
        'and' (case-insensitive), commas, semicolons, or newlines. filter_results.py's
        parse_filter tolerates spaces around the operator, so 'a <= 0.1' is fine."""
        import re
        parts = re.split(r"(?i)\s+and\s+|[,;\n]+", (text or "").strip())
        return [p.strip() for p in parts if p.strip()]

    def _run_slim(self):
        """Write <name>_slim.xlsx for ONLY the file resolved in Shared context, keeping
        the 'slim: keep columns' (blank = keep_columns.py's default set)."""
        xlsx_path = self.resolved_path("xlsx")
        if not os.path.isfile(xlsx_path):
            self._append(f"\n[slim] resolved xlsx not found: {xlsx_path}\n"
                         "        pick an existing ranking/metric in Shared context (format = xlsx).\n")
            return
        cmd = [PY, "keep_columns.py", "--xlsx", xlsx_path]
        cols = [c for c in re.split(r"[,\s]+", self.comp_keep.get().strip()) if c]
        if cols:
            cmd += ["--keep", *cols]
        self.run(cmd)

    # --- tab: plot ----------------------------------------------------------
    def _tab_plot(self, nb):
        t = self._scroll_tab(nb, "Plot")
        ttk.Label(t, text="plot_best_configs.py — plot the top runs for the resolved ranked file.",
                  foreground="gray").pack(anchor="w", pady=(6, 2))
        self.plot_dir = labeled_path(t, "--dir (experiment)", "", is_dir=True)
        ttk.Button(t, text="Use experiment root",
                   command=lambda: self.plot_dir.set(self.exp_path())).pack(anchor="w", pady=2)
        self.plot_results = labeled_path(t, "--results_csv (ranked)")
        ttk.Button(t, text="Use resolved (as .csv)",
                   command=lambda: self.plot_results.set(self.resolved_path("csv"))).pack(anchor="w", pady=2)
        self.plot_top = labeled_entry(t, "--top", "5", width=10)
        self.plot_sort = labeled_entry(t, "--sort_by", "region_knee_combined_gm")
        self.plot_regen = tk.BooleanVar(value=False)
        self.plot_dry = tk.BooleanVar(value=True)
        cr = ttk.Frame(t)
        cr.pack(anchor="w", pady=2)
        ttk.Checkbutton(cr, text="--regen", variable=self.plot_regen).pack(side="left", padx=4)
        ttk.Checkbutton(cr, text="--dry_run", variable=self.plot_dry).pack(side="left", padx=4)
        ttk.Button(t, text="Plot best", command=self._run_plot).pack(anchor="w", pady=6)

    def _run_plot(self):
        cmd = [PY, "plot_best_configs.py", "--dir", self.plot_dir.get() or self.exp_path(),
               "--csv", self.meas_path(), "--top", self.plot_top.get().strip(),
               "--sort_by", self.plot_sort.get().strip()]
        if self.plot_results.get().strip():
            cmd += ["--results_csv", self.plot_results.get().strip()]
        if self.plot_regen.get():
            cmd += ["--regen"]
        if self.plot_dry.get():
            cmd += ["--dry_run"]
        self.run(cmd)

    # --- log + process runner ----------------------------------------------
    def _scroll_tab(self, nb, title, height=430):
        """Add a notebook tab whose content scrolls vertically. Tall tabs
        (compile/generate) otherwise demand so much height in the vertical split
        that the log pane below gets squeezed to nothing. Returns the inner frame
        to pack widgets into (use it exactly like the old ttk.Frame(nb))."""
        outer = ttk.Frame(nb)
        nb.add(outer, text=title)
        canvas = tk.Canvas(outer, height=height, highlightthickness=0, bd=0,
                           background=self.cget("background"))
        vsb = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        inner = ttk.Frame(canvas)
        win = canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(win, width=e.width))
        # Wheel scrolls only while the pointer is over this tab's canvas.
        def _wheel(e): canvas.yview_scroll(int(-e.delta / 120), "units")
        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _wheel))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))
        return inner

    def _build_log(self, parent):
        # Log pane lives in the vertical split (weight=1 -> it grows with the window
        # and the sash above it can be dragged to enlarge it).
        frame = ttk.Frame(parent)
        parent.add(frame, weight=1)

        self.log2 = None          # mirror Text in the popped-out window (if open)
        self.log_win = None

        bar = ttk.Frame(frame)
        bar.pack(fill="x")
        self.run_btn_state = ttk.Label(bar, text="idle", foreground="gray")
        self.run_btn_state.pack(side="left")
        ttk.Button(bar, text="Stop", command=self._stop).pack(side="right")
        ttk.Button(bar, text="Clear log", command=self._clear_log).pack(side="right", padx=4)
        ttk.Button(bar, text="Pop out", command=self._popout_log).pack(side="right", padx=4)

        logwrap = ttk.Frame(frame)
        logwrap.pack(fill="both", expand=True, pady=(4, 0))
        sb = ttk.Scrollbar(logwrap, orient="vertical")
        sb.pack(side="right", fill="y")
        self.log = tk.Text(logwrap, height=18, bg="#0f1115", fg="#d6deeb",
                           insertbackground="#d6deeb", yscrollcommand=sb.set)
        self.log.pack(side="left", fill="both", expand=True)
        sb.config(command=self.log.yview)

    def _popout_log(self):
        """Open (or focus) a separate window mirroring the log live."""
        if self.log_win is not None and self.log_win.winfo_exists():
            self.log_win.lift()
            return
        win = tk.Toplevel(self)
        win.title("Log — physics_nn_pipeline")
        win.geometry("960x640")
        wrap = ttk.Frame(win)
        wrap.pack(fill="both", expand=True, padx=6, pady=6)
        sb = ttk.Scrollbar(wrap, orient="vertical")
        sb.pack(side="right", fill="y")
        txt = tk.Text(wrap, bg="#0f1115", fg="#d6deeb", insertbackground="#d6deeb",
                      yscrollcommand=sb.set)
        txt.pack(side="left", fill="both", expand=True)
        sb.config(command=txt.yview)
        txt.insert("1.0", self.log.get("1.0", "end"))   # seed with current contents
        txt.see("end")
        self.log_win, self.log2 = win, txt

        def _on_close():
            self.log2 = None
            self.log_win = None
            win.destroy()
        win.protocol("WM_DELETE_WINDOW", _on_close)

    def _clear_log(self):
        self.log.delete("1.0", "end")
        if self.log2 is not None and self.log_win is not None and self.log_win.winfo_exists():
            self.log2.delete("1.0", "end")

    def _stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            self._append("\n[terminated]\n")

    def _append(self, text):
        self.log.insert("end", text)
        self.log.see("end")
        if self.log2 is not None:                # mirror to the pop-out window if open
            try:
                self.log2.insert("end", text)
                self.log2.see("end")
            except tk.TclError:                  # window was closed underneath us
                self.log2 = None
                self.log_win = None

    def run(self, cmd, on_done=None):
        """Show the command, then run it from HERE streaming output to the log.

        on_done(returncode) is invoked on the main thread after the process exits."""
        if self.proc and self.proc.poll() is None:
            self._append("\n[busy — stop the running job first]\n")
            return
        # validate path-ish args minimally
        printable = " ".join(f'"{c}"' if " " in c else c for c in cmd)
        self._append(f"\n$ {printable}\n")
        self.run_btn_state.configure(text="running...", foreground="#b58900")
        threading.Thread(target=self._worker, args=(cmd, on_done), daemon=True).start()

    def run_sequence(self, steps):
        """Run a list of steps in order, stopping on the first non-zero exit.

        Each step is either a command (list of str) or a zero-arg callable that
        returns a command -- callables are evaluated just before they run, so a
        later step can glob files produced by earlier ones. A step that is (or
        returns) None/[] is skipped."""
        if self.proc and self.proc.poll() is None:
            self._append("\n[busy — stop the running job first]\n")
            return
        self._run_step(steps, 0)

    def _run_step(self, steps, i):
        if i >= len(steps):
            self._append("\n[sequence done]\n")
            return
        cmd = steps[i]() if callable(steps[i]) else steps[i]
        if not cmd:
            self._run_step(steps, i + 1)          # nothing to do for this step
            return

        def cont(rc):
            if rc == 0:
                self._run_step(steps, i + 1)
            else:
                self._append(f"\n[sequence] step {i + 1} failed (exit {rc}); stopping\n")
        self.run(cmd, on_done=cont)

    def _worker(self, cmd, on_done=None):
        rc = None
        try:
            self.proc = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in self.proc.stdout:
                self.after(0, self._append, line)
            self.proc.wait()
            rc = self.proc.returncode
            self.after(0, self._append, f"\n[exit {rc}]\n")
        except Exception as e:
            self.after(0, self._append, f"\n[error: {e}]\n")
        finally:
            self.after(0, lambda: self.run_btn_state.configure(text="idle", foreground="gray"))
            if on_done is not None:
                self.after(0, on_done, rc)


if __name__ == "__main__":
    App().mainloop()
