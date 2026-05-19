"""CLI entry point: python -m heatup  or  heatup."""

from __future__ import annotations

import argparse
import sys

from heatup import config
from heatup.pipeline import run_stability_pipeline, run_stability_pipeline_batch


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="heatup",
        description=(
            "Sequential stability analysis for solid-state electrolyte candidates.\n"
            "Pass a symmetry directory to analyse one material, or 'batch' to run "
            "the entire database."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "sym_dir",
        help=(
            "Path to database/<material>/<symmetry>/, "
            "or 'batch' to process the entire database."
        ),
    )
    parser.add_argument(
        "--operating-T", type=float, default=1200.0,
        metavar="K",
        help="Temperature (K) at which to evaluate thermodynamic stability.",
    )
    parser.add_argument(
        "--database", default=config.DATABASE_ROOT,
        help="Root of the simulation database.",
    )
    parser.add_argument(
        "--candidates", default=config.CANDIDATES_ROOT,
        help="Root of the candidate POSCAR tree.",
    )
    parser.add_argument(
        "--device", default=config.DEFAULT_DEVICE,
        help="Compute device for MACE ('cuda' or 'cpu').",
    )
    parser.add_argument(
        "--no-generate", action="store_true",
        help="Skip secondary-phase generation with PyXtal.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Recompute even if a stability_report.json already exists.",
    )
    parser.add_argument(
        "--no-figures", action="store_true",
        help="Skip dashboard figure generation.",
    )
    parser.add_argument(
        "--figure-fmt", default="pdf", choices=["pdf", "png"],
        help="Figure file format.",
    )
    args = parser.parse_args()

    if args.sym_dir == "batch":
        reports = run_stability_pipeline_batch(
            database_root           = args.database,
            candidates_root         = args.candidates,
            operating_T             = args.operating_T,
            device                  = args.device,
            generate_missing_phases = not args.no_generate,
            force_rerun             = args.force,
            save_figures            = not args.no_figures,
            figure_fmt              = args.figure_fmt,
        )
        n_ok   = sum(1 for r in reports if r["overall"] == config.STATUS_OK)
        n_warn = sum(1 for r in reports if r["overall"] == config.STATUS_WARN)
        n_fail = sum(1 for r in reports if r["overall"] == config.STATUS_FAIL)
        print(f"\nBatch complete: {len(reports)} materials assessed.")
        print(f"  OK: {n_ok}  |  WARN: {n_warn}  |  FAIL: {n_fail}")
        if n_fail:
            print("\nFailed materials:")
            for r in reports:
                if r["overall"] == config.STATUS_FAIL:
                    print(f"  {r['material']}/{r['symmetry']}")
                    for flag in r.get("flags", []):
                        print(f"    {flag}")
    else:
        report = run_stability_pipeline(
            sym_dir                 = args.sym_dir,
            operating_T             = args.operating_T,
            candidates_root         = args.candidates,
            database_root           = args.database,
            device                  = args.device,
            generate_missing_phases = not args.no_generate,
            force_rerun             = args.force,
            save_figure             = not args.no_figures,
            figure_fmt              = args.figure_fmt,
        )
        print(f"\nOverall: {report['overall'].upper()}")
        for flag in report.get("flags", []):
            print(f"  {flag}")
        if report.get("stopped_at"):
            print(f"  Pipeline stopped at: {report['stopped_at']}")
        sys.exit(0 if report["overall"] == config.STATUS_OK else 1)


if __name__ == "__main__":
    main()
