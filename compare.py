"""Compare multiple OmniSafe training runs using the built-in Plotter.

Usage:
  # Compare all runs under runs/ directory
  python compare.py --runs runs/

  # Compare specific run directories
  python compare.py \\
      --runs "runs/PPO-{SafetyCarGoal2-v0}" "runs/PPOLag-{SafetyCarGoal2-v0}" \\
      --save-dir experiments/base_line/plots

  # Compare with custom labels
  python compare.py \\
      --runs "runs/PPO-{SafetyCarGoal2-v0}" "runs/PPOLag-{SafetyCarGoal2-v0}" \\
      --legend "PPO (no safety)" "PPOLag (cost≤25)" \\
      --save-dir experiments/base_line/plots
"""

import sys
import argparse
from pathlib import Path

import numpy as np


def find_run_dirs(root: Path) -> list[Path]:
    """Find all OmniSafe run directories (must contain progress.csv)."""
    runs = []
    for p in sorted(root.rglob("progress.csv")):
        runs.append(p.parent)
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare OmniSafe training runs with reward / cost curves"
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        metavar="DIR",
        help="Run directories to compare (can be parent dirs — scanned recursively)",
    )
    parser.add_argument(
        "--legend",
        nargs="+",
        default=None,
        metavar="LABEL",
        help="Display labels for each --runs entry (must match count)",
    )
    parser.add_argument(
        "--cost-limit",
        type=float,
        default=25.0,
        help="Safety cost limit to draw as dashed reference line (default: 25.0)",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=5,
        help="Rolling-average window for smoothing curves (default: 5)",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("experiments/base_line/plots"),
        help="Directory to save generated plots",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "pdf", "svg"],
        help="Image format for saved plots",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots in a window (requires display)",
    )
    args = parser.parse_args()

    args.save_dir.mkdir(parents=True, exist_ok=True)

    # Resolve run directories
    all_logdirs: list[str] = []
    for r in args.runs:
        p = Path(r)
        if not p.exists():
            print(f"WARNING: Directory not found, skipping: {p}")
            continue
        # If it contains progress.csv directly, use it; otherwise scan recursively
        if (p / "progress.csv").exists():
            all_logdirs.append(str(p))
        else:
            found = [str(d) for d in find_run_dirs(p)]
            if not found:
                print(f"WARNING: No progress.csv found under {p}, skipping.")
            else:
                all_logdirs.extend(found)

    if not all_logdirs:
        print("ERROR: No valid run directories found.")
        sys.exit(1)

    # If legend is provided, it must match the number of discovered directories.
    # When not provided or count mismatches, auto-generate from config.json.
    def get_algo_label(logdir: str) -> str:
        import json, os
        cfg_path = os.path.join(logdir, "config.json")
        try:
            with open(cfg_path) as f:
                cfg = json.load(f)
            return cfg.get("algo", Path(logdir).parent.name)
        except Exception:
            return Path(logdir).name

    if args.legend and len(args.legend) == len(all_logdirs):
        legend = args.legend
    else:
        if args.legend:
            print(f"WARNING: --legend count ({len(args.legend)}) != run count "
                  f"({len(all_logdirs)}). Auto-generating labels from config.json.")
        legend = [get_algo_label(d) for d in all_logdirs]

    print("=" * 60)
    print("OmniSafe Training Run Comparison")
    print("=" * 60)
    for d, lbl in zip(all_logdirs, legend):
        print(f"  [{lbl}]  {d}")
    print(f"\nSmoothing:   window={args.smooth}")
    print(f"Cost limit:  {args.cost_limit}")
    print(f"Save dir:    {args.save_dir}")
    print("=" * 60 + "\n")

    from omnisafe.utils.plotter import Plotter

    plotter = Plotter()

    # ── Plot 1: Episode Rewards ──────────────────────────────────────
    print("Generating reward curve...")
    plotter.make_plots(
        all_logdirs=all_logdirs,
        legend=legend,
        xaxis="Steps",
        value="Rewards",
        smooth=args.smooth,
        save_dir=str(args.save_dir),
        save_name="reward_curve",
        save_format=args.format,
        show_image=args.show,
    )
    print(f"  Saved: {args.save_dir}/reward_curve.{args.format}")

    # ── Plot 2: Episode Costs ────────────────────────────────────────
    print("Generating cost curve...")
    # Reset plotter state for second plot
    plotter2 = Plotter()
    plotter2.make_plots(
        all_logdirs=all_logdirs,
        legend=legend,
        xaxis="Steps",
        value="Costs",
        smooth=args.smooth,
        cost_limit=args.cost_limit,
        save_dir=str(args.save_dir),
        save_name="cost_curve",
        save_format=args.format,
        show_image=args.show,
    )
    print(f"  Saved: {args.save_dir}/cost_curve.{args.format}")

    # ── Summary table ─────────────────────────────────────────────────
    print("\nLoading run statistics for summary table...")
    all_datasets = []
    p3 = Plotter()
    for logdir in all_logdirs:
        try:
            ds = p3.get_datasets(logdir)
            all_datasets.extend(ds)
        except Exception as e:
            print(f"  WARNING: Could not load {logdir}: {e}")

    if all_datasets:
        print()
        print(f"{'Algorithm':<20} {'Mean Reward':>14} {'Final Reward':>14} "
              f"{'Mean Cost':>12} {'Final Cost':>12}")
        print("-" * 74)
        for ds in all_datasets:
            label = ds["Condition1"].iloc[0]
            mean_r = ds["Rewards"].mean()
            final_r = ds["Rewards"].iloc[-1]
            mean_c = ds["Costs"].mean()
            final_c = ds["Costs"].iloc[-1]
            print(f"  {label:<18} {mean_r:>14.2f} {final_r:>14.2f} "
                  f"{mean_c:>12.2f} {final_c:>12.2f}")
        print()

    print(f"\nPlots saved to: {args.save_dir.resolve()}")
    print("Done.")


if __name__ == "__main__":
    main()
