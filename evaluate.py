"""Evaluate and record a trained OmniSafe agent on any safety-gymnasium environment."""

import os
import sys
import argparse
from pathlib import Path

import numpy as np


def find_latest_model(torch_save_dir: Path) -> str:
    """Return the filename of the latest epoch-N.pt in torch_save/."""
    pts = sorted(torch_save_dir.glob("epoch-*.pt"),
                 key=lambda p: int(p.stem.split("-")[1]))
    if not pts:
        raise FileNotFoundError(f"No epoch-*.pt found in {torch_save_dir}")
    return pts[-1].name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained OmniSafe agent on any safety-gymnasium environment"
    )
    parser.add_argument(
        "--exp-dir",
        type=Path,
        required=True,
        help="OmniSafe experiment directory (contains config.json and torch_save/)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model filename inside torch_save/ (default: latest epoch-N.pt)",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=10,
        help="Number of episodes to evaluate",
    )
    parser.add_argument(
        "--record-episodes",
        type=int,
        default=3,
        help="How many episodes to record as video (N independent render episodes)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Video width in pixels",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Video height in pixels",
    )
    args = parser.parse_args()

    if not args.exp_dir.exists():
        print(f"ERROR: Experiment directory not found: {args.exp_dir}")
        sys.exit(1)

    torch_save_dir = args.exp_dir / "torch_save"
    if not torch_save_dir.exists():
        print(f"ERROR: torch_save/ not found in {args.exp_dir}")
        sys.exit(1)

    model_name = args.model or find_latest_model(torch_save_dir)
    model_path = torch_save_dir / model_name
    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        sys.exit(1)

    video_dir = args.exp_dir / "videos"
    video_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("OmniSafe Evaluation")
    print("=" * 60)
    print(f"Experiment : {args.exp_dir}")
    print(f"Model      : {model_name}")
    print(f"Episodes   : {args.num_episodes}  (record {args.record_episodes})")
    print(f"Resolution : {args.width}x{args.height}")
    print(f"Video dir  : {video_dir}")
    print("=" * 60)

    import omnisafe

    evaluator = omnisafe.Evaluator()
    evaluator.load_saved(
        save_dir=str(args.exp_dir),
        model_name=model_name,
        render_mode="rgb_array",
        width=args.width,
        height=args.height,
    )

    # --- Evaluate (no video) ---
    print(f"\nRunning {args.num_episodes} evaluation episodes...")
    rewards, costs = evaluator.evaluate(num_episodes=args.num_episodes)

    print("\nReward  — mean: {:7.2f}  std: {:6.2f}  min: {:7.2f}  max: {:7.2f}".format(
        np.mean(rewards), np.std(rewards), np.min(rewards), np.max(rewards)))
    print("Cost    — mean: {:7.4f}  std: {:6.4f}  min: {:7.4f}  max: {:7.4f}".format(
        np.mean(costs), np.std(costs), np.min(costs), np.max(costs)))

    best_ep = int(np.argmax(rewards))
    print(f"\nBest episode: #{best_ep}  reward={rewards[best_ep]:.2f}  cost={costs[best_ep]:.4f}")

    # --- Record video ---
    if args.record_episodes > 0:
        print(f"\nRecording {args.record_episodes} episode(s) to {video_dir} ...")

        # Re-load because evaluate() closed the env
        evaluator2 = omnisafe.Evaluator()
        evaluator2.load_saved(
            save_dir=str(args.exp_dir),
            model_name=model_name,
            render_mode="rgb_array",
            width=args.width,
            height=args.height,
        )
        evaluator2.render(
            num_episodes=args.record_episodes,
            save_replay_path=str(video_dir),
        )
        print(f"\nVideos saved to: {video_dir}")
        for f in sorted(video_dir.glob("*.mp4")):
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.name}  ({size_mb:.1f} MB)")

    print("\nDone.")


if __name__ == "__main__":
    main()
