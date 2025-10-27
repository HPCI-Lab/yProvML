#!/usr/bin/env python3
"""
analysis/cli.py

Single entry CLI with subcommands:
  all, enrich, pareto, heatmaps, cluster, correlations
"""
import argparse
from pathlib import Path
import sys

# make sure this directory is importable when run directly
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from io_utils_ml import (
    load_runs_csv, save_csv, choose_columns, ensure_outdir
)
from enrich_mod import merge_lr_min_max, merge_compute_json, add_emissions_cumsum, add_co2_from_energy
from pareto_mod import pareto_and_plots
from heatmaps_mod import heatmaps_lr_batch
from clustering_corr_mod import cluster_outputs, export_correlations


def cmd_enrich(args):
    outdir = ensure_outdir(args.outdir)
    runs = load_runs_csv(args.runs_csv)
    runs = merge_lr_min_max(runs, args.lr_trace_dir)
    runs = merge_compute_json(runs, args.compute_dirs)
    runs = add_emissions_cumsum(runs, args.emissions_cols or [])
    if args.carbon_intensity_kg_per_kwh is not None:
        runs = add_co2_from_energy(runs, args.carbon_intensity_kg_per_kwh)
    save_csv(runs, outdir / "runs_enriched.csv")


def cmd_pareto(args):
    outdir = ensure_outdir(args.outdir)
    runs = load_runs_csv(args.runs_csv)
    acc_col, em_col, cost_col = choose_columns(runs, args.accuracy_col, args.emissions_cols, args.cost_priority)
    pareto_and_plots(runs, outdir, acc_col, em_col, cost_col)


def cmd_heatmaps(args):
    outdir = ensure_outdir(args.outdir)
    runs = load_runs_csv(args.runs_csv)
    acc_col, _, cost_col = choose_columns(runs, args.accuracy_col, args.emissions_cols, args.cost_priority)
    heatmaps_lr_batch(runs, outdir, acc_col, cost_col, x_param="param_lr", y_param="param_batch_size")


def cmd_cluster(args):
    outdir = ensure_outdir(args.outdir)
    runs = load_runs_csv(args.runs_csv)
    cluster_outputs(runs, outdir, n_clusters=args.n_clusters)


def cmd_correlations(args):
    outdir = ensure_outdir(args.outdir)
    runs = load_runs_csv(args.runs_csv)
    export_correlations(runs, outdir)


def cmd_all(args):
    outdir = ensure_outdir(args.outdir)
    runs = load_runs_csv(args.runs_csv)

    # enrichment
    runs = merge_lr_min_max(runs, args.lr_trace_dir)
    runs = merge_compute_json(runs, args.compute_dirs)
    runs = add_emissions_cumsum(runs, args.emissions_cols or [])
    if args.carbon_intensity_kg_per_kwh is not None:
        runs = add_co2_from_energy(runs, args.carbon_intensity_kg_per_kwh)
    save_csv(runs, outdir / "runs_enriched.csv")

    # choose cols after enrichment
    acc_col, em_col, cost_col = choose_columns(runs, args.accuracy_col, args.emissions_cols, args.cost_priority)

    # pareto, heatmaps, clustering, correlations
    pareto_and_plots(runs, outdir, acc_col, em_col, cost_col)
    heatmaps_lr_batch(runs, outdir, acc_col, cost_col, x_param="param_lr", y_param="param_batch_size")
    cluster_outputs(runs, outdir, n_clusters=args.n_clusters)
    if args.export_correlations:
        export_correlations(runs, outdir)


def main():
    ap = argparse.ArgumentParser(prog="analysis", description="Modular analysis toolkit")
    sub = ap.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--runs_csv", required=True, type=str)
    common.add_argument("--outdir", required=True, type=str)
    common.add_argument("--accuracy_col", type=str, default=None)
    common.add_argument("--emissions_cols", type=str, nargs="*", default=None)
    common.add_argument("--cost_priority", type=str, nargs="*", default=None)

    p_enrich = sub.add_parser("enrich", parents=[common])
    p_enrich.add_argument("--lr_trace_dir", type=str, default=None)
    p_enrich.add_argument("--compute_dirs", type=str, nargs="*", default=None)
    p_enrich.add_argument("--carbon_intensity_kg_per_kwh", type=float, default=None)
    p_enrich.set_defaults(func=cmd_enrich)

    p_pareto = sub.add_parser("pareto", parents=[common])
    p_pareto.set_defaults(func=cmd_pareto)

    p_heat = sub.add_parser("heatmaps", parents=[common])
    p_heat.set_defaults(func=cmd_heatmaps)

    p_cluster = sub.add_parser("cluster", parents=[common])
    p_cluster.add_argument("--n_clusters", type=int, default=3)
    p_cluster.set_defaults(func=cmd_cluster)

    p_corr = sub.add_parser("correlations", parents=[common])
    p_corr.set_defaults(func=cmd_correlations)

    p_all = sub.add_parser("all", parents=[common])
    p_all.add_argument("--lr_trace_dir", type=str, default=None)
    p_all.add_argument("--compute_dirs", type=str, nargs="*", default=None)
    p_all.add_argument("--carbon_intensity_kg_per_kwh", type=float, default=None)
    p_all.add_argument("--n_clusters", type=int, default=3)
    p_all.add_argument("--export_correlations", action="store_true")
    p_all.set_defaults(func=cmd_all)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
