"""Transfer evaluation: run CarGoal2-trained PPO/PPOLag weights on SafetyCarGoal1-v0.

CarGoal1 and CarGoal2 share the same obs/action dimensions (both 78-dim obs, 2-dim action),
so weights transfer directly without any architectural changes.

Usage:
    python cross_eval.py --ppo-dir   runs/PPO-{SafetyCarGoal2-v0}/seed-042-...
    python cross_eval.py --ppolag-dir runs/PPOLag-{SafetyCarGoal2-v0}/seed-042-...
    python cross_eval.py --ppo-dir runs/... --ppolag-dir runs/...
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np


TARGET_ENV = "SafetyCarGoal1-v0"


def find_latest_model(torch_save_dir: Path) -> str:
    pts = sorted(
        torch_save_dir.glob("epoch-*.pt"),
        key=lambda p: int(p.stem.split("-")[1]),
    )
    if not pts:
        raise FileNotFoundError(f"No epoch-*.pt in {torch_save_dir}")
    return pts[-1].name


def build_transfer_dir(src_dir: Path, target_env: str, output_root: Path) -> Path:
    """Create a patched run directory that points the evaluator to target_env.

    OmniSafe Evaluator reads env_id from config.json in the run directory.
    We create a shadow directory with a patched config and a copy of the model.
    """
    src_dir = src_dir.resolve()
    config_path = src_dir / "config.json"
    torch_save_src = src_dir / "torch_save"

    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {src_dir}")
    if not torch_save_src.exists():
        raise FileNotFoundError(f"torch_save/ not found in {src_dir}")

    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    original_env = config["env_id"]
    algo = config.get("algo", "Unknown")

    # Patch env_id and exp_name in config
    config["env_id"] = target_env
    config["exp_name"] = f"{algo}-{{SafetyCarGoal1-v0}}-transfer"

    # Destination: runs/<algo>-transfer-{CarGoal1}/seed-<orig_seed>/
    seed_part = src_dir.name  # e.g. "seed-042-2026-04-03-..."
    dest = output_root / f"{algo}-transfer-{{{target_env}}}" / seed_part
    dest.mkdir(parents=True, exist_ok=True)

    # Write patched config
    with open(dest / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    # Copy torch_save directory (model weights)
    dest_torch = dest / "torch_save"
    if dest_torch.exists():
        shutil.rmtree(dest_torch)
    shutil.copytree(torch_save_src, dest_torch)

    print(f"  [{algo}] transfer dir: {dest}")
    print(f"  [{algo}] orig env    : {original_env}")
    print(f"  [{algo}] target env  : {target_env}")
    return dest


def run_evaluation(
    transfer_dir: Path,
    model_name: str,
    num_episodes: int,
    record_episodes: int,
    width: int,
    height: int,
) -> dict:
    """Run OmniSafe Evaluator on the patched transfer directory."""
    import omnisafe

    video_dir = transfer_dir / "videos"
    video_dir.mkdir(exist_ok=True)

    print(f"\n  Evaluating {num_episodes} episodes...")
    evaluator = omnisafe.Evaluator()
    evaluator.load_saved(
        save_dir=str(transfer_dir),
        model_name=model_name,
        render_mode="rgb_array",
        width=width,
        height=height,
    )
    rewards, costs = evaluator.evaluate(num_episodes=num_episodes)

    result = {
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "reward_min": float(np.min(rewards)),
        "reward_max": float(np.max(rewards)),
        "cost_mean": float(np.mean(costs)),
        "cost_std": float(np.std(costs)),
        "cost_min": float(np.min(costs)),
        "cost_max": float(np.max(costs)),
        "zero_cost_rate": float(np.mean(np.array(costs) == 0)),
    }

    print(f"  Reward  mean={result['reward_mean']:.2f}  std={result['reward_std']:.2f}"
          f"  min={result['reward_min']:.2f}  max={result['reward_max']:.2f}")
    print(f"  Cost    mean={result['cost_mean']:.4f}  std={result['cost_std']:.4f}"
          f"  zero-cost-rate={result['zero_cost_rate']*100:.1f}%")

    if record_episodes > 0:
        print(f"\n  Recording {record_episodes} episode(s) ...")
        evaluator2 = omnisafe.Evaluator()
        evaluator2.load_saved(
            save_dir=str(transfer_dir),
            model_name=model_name,
            render_mode="rgb_array",
            width=width,
            height=height,
        )
        evaluator2.render(
            num_episodes=record_episodes,
            save_replay_path=str(video_dir),
        )
        videos = sorted(video_dir.glob("*.mp4"))
        for v in videos:
            print(f"    {v.name}  ({v.stat().st_size/1024/1024:.1f} MB)")

    # Save result summary
    result_path = transfer_dir / "videos" / "transfer_result.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

    return result


def main():
    parser = argparse.ArgumentParser(
        description=f"Transfer CarGoal2 weights to {TARGET_ENV} and evaluate"
    )
    parser.add_argument("--ppo-dir", type=Path, default=None,
                        help="PPO experiment dir (CarGoal2 trained)")
    parser.add_argument("--ppolag-dir", type=Path, default=None,
                        help="PPOLag experiment dir (CarGoal2 trained)")
    parser.add_argument("--model", type=str, default=None,
                        help="Model filename (default: latest epoch-N.pt)")
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--record-episodes", type=int, default=3)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"),
                        help="Root output directory for transfer runs")
    args = parser.parse_args()

    if args.ppo_dir is None and args.ppolag_dir is None:
        print("ERROR: provide --ppo-dir and/or --ppolag-dir")
        sys.exit(1)

    sources = []
    if args.ppo_dir:
        sources.append(args.ppo_dir)
    if args.ppolag_dir:
        sources.append(args.ppolag_dir)

    all_results = {}
    print("=" * 60)
    print(f"Transfer Evaluation: CarGoal2 → {TARGET_ENV}")
    print("=" * 60)

    for src in sources:
        if not src.exists():
            print(f"SKIP: {src} does not exist")
            continue

        # Identify algo from config
        with open(src / "config.json", encoding="utf-8") as f:
            cfg = json.load(f)
        algo = cfg.get("algo", src.parent.name.split("-")[0])
        model_name = args.model or find_latest_model(src / "torch_save")
        print(f"\n[{algo}] Source: {src}")
        print(f"[{algo}] Model : {model_name}")

        transfer_dir = build_transfer_dir(src, TARGET_ENV, args.runs_dir)

        result = run_evaluation(
            transfer_dir=transfer_dir,
            model_name=model_name,
            num_episodes=args.num_episodes,
            record_episodes=args.record_episodes,
            width=args.width,
            height=args.height,
        )
        all_results[algo] = result

    # Print comparison table
    if len(all_results) > 1:
        print("\n" + "=" * 60)
        print(f"Comparison on {TARGET_ENV} (transfer from CarGoal2)")
        print("=" * 60)
        print(f"{'Algo':<10} {'Reward Mean':>12} {'Cost Mean':>10} {'Zero-Cost%':>10}")
        print("-" * 46)
        for algo, r in all_results.items():
            print(f"{algo:<10} {r['reward_mean']:>12.2f} {r['cost_mean']:>10.4f} "
                  f"{r['zero_cost_rate']*100:>9.1f}%")

    print("\nDone.")


if __name__ == "__main__":
    main()
