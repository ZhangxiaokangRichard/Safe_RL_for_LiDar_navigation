"""Top-down 2D trajectory visualization for OmniSafe trained agents.

Supports: SafetyCarGoal1-v0, SafetyCarGoal2-v0, SafetyButtonGoal1-v0

Loads a trained OmniSafe model, runs N episodes while collecting
agent (x, y) positions and per-step costs, then renders top-down maps
showing hazard zones, obstacle paths, and the agent trajectory coloured
by safety violations (green=safe, red=cost incurred).

Usage:
    python trajectory_viz.py --exp-dir runs/PPO-{SafetyCarGoal2-v0}/seed-042-...
    python trajectory_viz.py --exp-dir runs/... --env-id SafetyCarGoal1-v0 --num-episodes 5
    python trajectory_viz.py --exp-dir runs/... --best-only --num-episodes 10
"""

from __future__ import annotations   # Python 3.9 compatible union types

import argparse
import copy
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")          # headless-safe; works on Windows/Linux servers
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize


# ──────────────────────────────────────────────────────────────────────────── #
#  Environment metadata registry                                                #
# ──────────────────────────────────────────────────────────────────────────── #
ENV_META: dict = {
    "SafetyCarGoal1-v0": {
        "extent": 1.5, "type": "goal",
        "haz_r": 0.20, "goal_r": 0.30, "vase_r": 0.10,
    },
    "SafetyCarGoal2-v0": {
        "extent": 2.0, "type": "goal",
        "haz_r": 0.20, "goal_r": 0.30, "vase_r": 0.10,
    },
    "SafetyCarButton1-v0": {
        "extent": 1.5, "type": "button",
        "haz_r": 0.20, "goal_r": 0.10, "btn_r": 0.10,
        "grem_travel": 0.35,
    },
}

def get_meta(env_id: str) -> dict:
    for key, val in ENV_META.items():
        if key in env_id:
            return dict(val)
    return {"extent": 2.0, "type": "goal", "haz_r": 0.20, "goal_r": 0.30}


# ──────────────────────────────────────────────────────────────────────────── #
#  Model / environment loading                                                  #
# ──────────────────────────────────────────────────────────────────────────── #

def find_latest_model(torch_save_dir: Path) -> str:
    pts = sorted(torch_save_dir.glob("epoch-*.pt"),
                 key=lambda p: int(p.stem.split("-")[1]))
    if not pts:
        raise FileNotFoundError(f"No epoch-*.pt in {torch_save_dir}")
    return pts[-1].name


def make_work_dir(exp_dir: Path,
                  env_id_override: str | None) -> tuple[Path, str, Path | None]:
    """Return (work_dir, target_env_id, tmp_dir_to_clean).

    If env_id_override differs from the saved config, create a temporary
    directory with a patched config.json pointing to the new environment.
    The caller is responsible for deleting tmp_dir (returned as third value).
    """
    with open(exp_dir / "config.json", encoding="utf-8") as f:
        orig_cfg = json.load(f)

    orig_env = orig_cfg["env_id"]
    target_env = env_id_override or orig_env

    if target_env == orig_env:
        return exp_dir, target_env, None   # no temp dir created

    tmp = Path(tempfile.mkdtemp(prefix="traj_viz_"))
    patched = copy.deepcopy(orig_cfg)
    patched["env_id"] = target_env
    patched["exp_name"] = f"{orig_cfg.get('algo', 'agent')}-traj-{target_env}"
    (tmp / "config.json").write_text(json.dumps(patched, indent=2))
    shutil.copytree(exp_dir / "torch_save", tmp / "torch_save")
    return tmp, target_env, tmp   # caller must clean up tmp


# ──────────────────────────────────────────────────────────────────────────── #
#  Episode runner                                                               #
# ──────────────────────────────────────────────────────────────────────────── #

def _get_sg_task(evaluator_env):
    """Walk the OmniSafe CMDP chain to reach safety-gymnasium's Builder.task.

    Chain: evaluator._env (CMDP wrappers) → SafetyGymnasiumEnv
           → SafetyGymnasiumEnv._env (gymnasium-wrapped gym env)
           → .unwrapped (raw Builder)
           → .task

    The OmniSafe __getattr__ proxy stops before .task because
    SafetyGymnasiumEnv._env is a gymnasium Wrapper that does not
    expose Builder attributes directly — we must call .unwrapped explicitly.
    """
    env = evaluator_env
    while env is not None:
        if type(env).__name__ == "SafetyGymnasiumEnv":
            return env._env.unwrapped.task
        env = getattr(env, "_env", None)
    raise AttributeError(
        "Could not find SafetyGymnasiumEnv in OmniSafe wrapper chain"
    )


def run_episodes(
    work_dir: Path,
    model_name: str,
    target_env: str,
    num_episodes: int,
    boundary_penalty: float = 1.0,
) -> list[dict]:
    """Load model with OmniSafe Evaluator and collect trajectory data.

    Key design:
    - task access : use _get_sg_task(evaluator._env) to walk the CMDP chain
                    to SafetyGymnasiumEnv, then ._env.unwrapped.task reaches
                    the raw safety-gymnasium Builder.task with live MuJoCo data.
    - `evaluator._actor` : Loaded GaussianLearningActor — always present after
                           load_saved() for model-free algorithms.
    - obs and act stay as torch.Tensor throughout; the OmniSafe ActionScale
                           wrapper expects tensors and crashes on numpy arrays.
    """
    import omnisafe

    evaluator = omnisafe.Evaluator()
    evaluator.load_saved(
        save_dir=str(work_dir),
        model_name=model_name,
        render_mode="rgb_array",
    )

    inner_env = evaluator._env   # top-level CMDP wrapper

    # _actor is always set for model-free algos (PPO, PPOLag, …)
    actor = evaluator._actor
    assert actor is not None, (
        "evaluator._actor is None — is this a model-based algorithm? "
        "trajectory_viz only supports model-free actors."
    )

    meta = get_meta(target_env)
    episodes: list[dict] = []

    for ep in range(num_episodes):
        obs, _info = inner_env.reset()
        task = _get_sg_task(inner_env)   # Builder.task via unwrapped chain

        traj: dict = {
            "ep_idx": ep,
            "env_id": target_env,
            "agent_xy": [],
            "costs": [],
            "rewards": [],
            "boundary_flags": [],   # True when agent position exceeds arena bounds
            "goal_xy": [],
            "hazard_xy": None,
            "vase_xy": None,
            "button_xy": None,
            "gremlin_init_xy": None,
        }

        # ── Static obstacle snapshot (after reset) ──────────────────────── #
        for attr, key in [
            ("hazards",  "hazard_xy"),
            ("vases",    "vase_xy"),
            ("buttons",  "button_xy"),
        ]:
            obj = getattr(task, attr, None)
            if obj is not None:
                pos = getattr(obj, "pos", None)
                if pos is not None:
                    arr = np.asarray(pos)
                    if arr.ndim == 2 and arr.shape[0] > 0 and arr.shape[1] >= 2:
                        traj[key] = arr[:, :2].copy()

        grem_obj = getattr(task, "gremlins", None)
        if grem_obj is not None:
            pos = getattr(grem_obj, "pos", None)
            if pos is not None:
                arr = np.asarray(pos)
                if arr.ndim == 2 and arr.shape[0] > 0 and arr.shape[1] >= 2:
                    traj["gremlin_init_xy"] = arr[:, :2].copy()

        # ── Step loop ───────────────────────────────────────────────────── #
        done = False
        while not done:
            # Agent XY — from MuJoCo data via safety-gymnasium task
            agent_pos = np.asarray(task.agent.pos[:2]).copy()
            traj["agent_xy"].append(agent_pos)

            # Boundary violation: agent outside arena extent
            out_of_bounds = bool(np.any(np.abs(agent_pos) > meta["extent"]))
            traj["boundary_flags"].append(out_of_bounds)

            # Goal position (task.goal is the Goal geom)
            goal_obj = getattr(task, "goal", None)
            if goal_obj is not None:
                gp = getattr(goal_obj, "pos", None)
                if gp is not None:
                    traj["goal_xy"].append(np.asarray(gp[:2]).copy())

            # Inference — obs is already a torch.Tensor from OmniSafe CMDP env
            import torch
            with torch.no_grad():
                act = actor.predict(obs, deterministic=True)
                # predict() always returns a Tensor (never a tuple) for
                # GaussianLearningActor — pass directly to env.step()

            # step() expects a torch.Tensor; do NOT convert to numpy
            obs, rew, cost, terminated, truncated, _ = inner_env.step(act)
            # Apply boundary penalty to cost for this step
            augmented_cost = float(cost) + (boundary_penalty if out_of_bounds else 0.0)
            traj["costs"].append(augmented_cost)
            traj["rewards"].append(float(rew))
            done = bool(terminated) or bool(truncated)

        traj["agent_xy"]       = np.array(traj["agent_xy"])          # (T, 2)
        traj["boundary_flags"] = np.array(traj["boundary_flags"])    # (T,) bool
        traj["goal_xy"]  = (np.array(traj["goal_xy"])
                            if traj["goal_xy"] else np.zeros((0, 2)))
        traj["total_reward"]    = float(np.sum(traj["rewards"]))
        traj["total_cost"]      = float(np.sum(traj["costs"]))
        traj["boundary_steps"]  = int(np.sum(traj["boundary_flags"]))
        episodes.append(traj)

        cost_flag = "0-cost" if traj["total_cost"] == 0 else f"cost={traj['total_cost']:.0f}"
        oob_tag   = f"  oob={traj['boundary_steps']}st" if traj["boundary_steps"] > 0 else ""
        print(f"  [ep {ep+1:>2}/{num_episodes}]  "
              f"reward={traj['total_reward']:7.2f}  {cost_flag}{oob_tag}")

    return episodes


# ──────────────────────────────────────────────────────────────────────────── #
#  Plotting                                                                     #
# ──────────────────────────────────────────────────────────────────────────── #

def plot_trajectory(traj: dict, out_path: Path, algo: str = "") -> None:
    meta = get_meta(traj["env_id"])
    ext  = meta["extent"]
    fig, ax = plt.subplots(figsize=(6, 6))

    # ── Arena background ─────────────────────────────────────────────────── #
    arena = mpatches.FancyBboxPatch(
        (-ext, -ext), 2 * ext, 2 * ext,
        boxstyle="square,pad=0",
        edgecolor="#222", facecolor="#f9f9f0", linewidth=2.5, zorder=0)
    ax.add_patch(arena)

    # ── Hazards ──────────────────────────────────────────────────────────── #
    if traj["hazard_xy"] is not None:
        for xy in traj["hazard_xy"]:
            ax.add_patch(plt.Circle(xy, meta["haz_r"],
                                    color="#e63030", alpha=0.18, zorder=2))
            ax.add_patch(plt.Circle(xy, meta["haz_r"], color="#e63030",
                                    fill=False, linewidth=1.4, zorder=3))
            ax.plot(*xy, "+", color="#e63030", ms=5, mew=1.2, zorder=4)

    # ── Vases ─────────────────────────────────────────────────────────────── #
    if traj["vase_xy"] is not None:
        for xy in traj["vase_xy"]:
            ax.add_patch(plt.Circle(xy, meta.get("vase_r", 0.10),
                                    color="#888", alpha=0.40, zorder=2))

    # ── Buttons ───────────────────────────────────────────────────────────── #
    if traj["button_xy"] is not None:
        for i, xy in enumerate(traj["button_xy"]):
            ax.add_patch(plt.Circle(xy, meta.get("btn_r", 0.10),
                                    color="#5544cc", alpha=0.55, zorder=4))
            ax.text(*xy, str(i), ha="center", va="center",
                    fontsize=7, color="white", fontweight="bold", zorder=5)

    # ── Gremlins — initial position + travel ring ─────────────────────────── #
    if traj["gremlin_init_xy"] is not None and meta["type"] == "button":
        travel = meta.get("grem_travel", 0.35)
        for xy in traj["gremlin_init_xy"]:
            ax.add_patch(plt.Circle(xy, travel,
                                    color="#ff8c00", fill=False,
                                    linestyle="--", linewidth=1.1,
                                    alpha=0.70, zorder=3))
            ax.plot(*xy, "D", ms=5, color="#ff8c00", zorder=5)

    # ── Goal positions ────────────────────────────────────────────────────── #
    if len(traj["goal_xy"]) > 1:
        g = traj["goal_xy"]
        diff = np.linalg.norm(np.diff(g, axis=0), axis=1)
        change_idx = np.where(diff > 0.05)[0] + 1
        unique_goals = g[np.concatenate([[0], change_idx])]
        for i, gpos in enumerate(unique_goals):
            ax.plot(*gpos, marker="*", ms=18 if i == 0 else 12,
                    color="#f5c400", markeredgecolor="#806600",
                    markeredgewidth=0.7, zorder=6)

    # ── Agent trajectory (coloured by cost) ──────────────────────────────── #
    xy = traj["agent_xy"]
    costs = np.array(traj["costs"])
    if len(xy) > 1:
        segs  = np.stack([xy[:-1], xy[1:]], axis=1)   # (T-1, 2, 2)
        cvals = (costs[:-1] > 0).astype(float)         # 0=safe, 1=violation
        lc = LineCollection(segs, cmap="RdYlGn_r", norm=Normalize(0, 1),
                             linewidth=1.8, alpha=0.85, zorder=7)
        lc.set_array(cvals)
        ax.add_collection(lc)

    ax.plot(*xy[0],  "go", ms=9, zorder=8,
            markeredgecolor="#005500", markeredgewidth=0.8, label="Start")
    ax.plot(*xy[-1], "r^", ms=9, zorder=8,
            markeredgecolor="#550000", markeredgewidth=0.8, label="End")

    # ── Out-of-bounds markers ─────────────────────────────────────────────── #
    bf = traj.get("boundary_flags")
    if bf is not None and np.any(bf):
        oob_xy = xy[bf]
        ax.scatter(oob_xy[:, 0], oob_xy[:, 1],
                   c="#dd00dd", s=22, marker="x", linewidths=1.4,
                   zorder=11, alpha=0.85)

    # ── Legend ────────────────────────────────────────────────────────────── #
    proxies = [
        mpatches.Patch(color="#e63030", alpha=0.5,  label=f"Hazard r={meta['haz_r']}m"),
        mpatches.Patch(color="#00bb44", alpha=0.8,  label="Safe step"),
        mpatches.Patch(color="#cc2200", alpha=0.8,  label="Cost step"),
    ]
    if meta["type"] == "button":
        proxies += [
            mpatches.Patch(color="#5544cc", alpha=0.5, label="Buttons"),
            mpatches.Patch(color="#ff8c00", alpha=0.5, label="Gremlin path"),
        ]
    elif traj["vase_xy"] is not None:
        proxies.append(mpatches.Patch(color="#888", alpha=0.5, label="Vases"))

    oob_steps = traj.get("boundary_steps", 0)
    if oob_steps > 0:
        proxies.append(mpatches.Patch(color="#dd00dd", alpha=0.8,
                                      label=f"Out-of-bounds ({oob_steps} steps)"))

    ax.legend(handles=proxies, loc="upper right", fontsize=7,
              framealpha=0.85, edgecolor="#ccc")

    # ── Axes decoration ───────────────────────────────────────────────────── #
    pad = 0.2
    ax.set_xlim(-ext - pad, ext + pad)
    ax.set_ylim(-ext - pad, ext + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)", fontsize=9)
    ax.set_ylabel("y (m)", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.tick_params(labelsize=8)

    cost_flag = ("zero-cost" if traj["total_cost"] == 0
                 else f"cost={traj['total_cost']:.0f}")
    ax.set_title(
        f"{algo} | ep #{traj['ep_idx']+1} | "
        f"R={traj['total_reward']:.2f} | {cost_flag}",
        fontsize=9, pad=6)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_multi_trajectories(episodes: list[dict], out_path: Path,
                             algo: str = "") -> None:
    """Overlay all episode trajectories on one map."""
    if not episodes:
        return
    meta = get_meta(episodes[0]["env_id"])
    ext  = meta["extent"]
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.add_patch(mpatches.FancyBboxPatch(
        (-ext, -ext), 2 * ext, 2 * ext, boxstyle="square,pad=0",
        edgecolor="#222", facecolor="#f9f9f0", linewidth=2.5, zorder=0))

    ep0 = episodes[0]
    if ep0["hazard_xy"] is not None:
        for xy in ep0["hazard_xy"]:
            ax.add_patch(plt.Circle(xy, meta["haz_r"],
                                    color="#e63030", alpha=0.15, zorder=2))
            ax.add_patch(plt.Circle(xy, meta["haz_r"], color="#e63030",
                                    fill=False, linewidth=1.2, zorder=3))
    if meta["type"] == "button" and ep0["gremlin_init_xy"] is not None:
        travel = meta.get("grem_travel", 0.35)
        for xy in ep0["gremlin_init_xy"]:
            ax.add_patch(plt.Circle(xy, travel, color="#ff8c00",
                                    fill=False, linestyle="--",
                                    linewidth=1.0, alpha=0.6, zorder=3))

    cmap = plt.get_cmap("tab10")
    for i, ep in enumerate(episodes):
        xy = ep["agent_xy"]
        if len(xy) < 2:
            continue
        col = cmap(i % 10)
        ax.plot(xy[:, 0], xy[:, 1], color=col, linewidth=1.1,
                alpha=0.6, zorder=6,
                label=f"ep{i+1} R={ep['total_reward']:.1f}")
        ax.plot(*xy[0], "o", ms=5, color=col, zorder=7)

    ax.legend(fontsize=6, loc="upper right", framealpha=0.8, ncol=2)
    pad = 0.2
    ax.set_xlim(-ext - pad, ext + pad)
    ax.set_ylim(-ext - pad, ext + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)", fontsize=9)
    ax.set_ylabel("y (m)", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.set_title(
        f"{algo} — trajectory overlay ({len(episodes)} episodes)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────── #
#  CLI                                                                          #
# ──────────────────────────────────────────────────────────────────────────── #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Top-down trajectory visualization for OmniSafe-trained agents"
    )
    parser.add_argument("--exp-dir", type=Path, required=True,
                        help="OmniSafe experiment directory (seed-XXX-timestamp/)")
    parser.add_argument("--env-id", type=str, default=None,
                        help="Override environment ID for cross-env evaluation "
                             "(default: read from config.json)")
    parser.add_argument("--model", type=str, default=None,
                        help="Model filename in torch_save/ (default: latest epoch-N.pt)")
    parser.add_argument("--num-episodes", type=int, default=5,
                        help="Number of episodes to run (default: 5)")
    parser.add_argument("--best-only", action="store_true",
                        help="Save only the episode with highest reward")
    parser.add_argument("--overlay", action="store_true",
                        help="Also save a trajectory overlay of all episodes")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output directory (default: <exp-dir>/trajectories/)")
    parser.add_argument("--boundary-penalty", type=float, default=1.0,
                        help="Extra cost per out-of-bounds step (default: 1.0; 0 = track only)")
    args = parser.parse_args()

    if not args.exp_dir.exists():
        raise SystemExit(f"ERROR: {args.exp_dir} does not exist")

    work_dir, target_env, tmp_to_clean = make_work_dir(args.exp_dir, args.env_id)
    try:
        model_name = args.model or find_latest_model(work_dir / "torch_save")

        out_dir = args.out_dir or (args.exp_dir / "trajectories")
        out_dir.mkdir(parents=True, exist_ok=True)

        with open(work_dir / "config.json", encoding="utf-8") as f:
            cfg = json.load(f)
        algo = cfg.get("algo", "Agent")

        print("=" * 60)
        print(f"Trajectory Visualizer — {algo} on {target_env}")
        print(f"  Model           : {model_name}")
        print(f"  Episodes        : {args.num_episodes}")
        print(f"  Boundary penalty: {args.boundary_penalty} per out-of-bounds step")
        print(f"  Output          : {out_dir}")
        print("=" * 60)

        episodes = run_episodes(work_dir, model_name, target_env, args.num_episodes,
                                boundary_penalty=args.boundary_penalty)

        # ── Per-episode plots ─────────────────────────────────────────────── #
        if args.best_only:
            best = max(episodes, key=lambda e: e["total_reward"])
            plot_trajectory(best, out_dir / "best_trajectory.png", algo=algo)
            print(f"\nBest episode: #{best['ep_idx']+1}  "
                  f"reward={best['total_reward']:.2f}  "
                  f"cost={best['total_cost']:.0f}")
        else:
            for ep in episodes:
                fname = f"trajectory_ep{ep['ep_idx']+1:02d}.png"
                plot_trajectory(ep, out_dir / fname, algo=algo)

        # ── Overlay plot ──────────────────────────────────────────────────── #
        if args.overlay or not args.best_only:
            plot_multi_trajectories(
                episodes, out_dir / "trajectory_overlay.png", algo=algo)

        # ── Stats summary ─────────────────────────────────────────────────── #
        rewards     = [e["total_reward"]   for e in episodes]
        costs       = [e["total_cost"]     for e in episodes]
        oob_steps   = [e.get("boundary_steps", 0) for e in episodes]
        total_steps = sum(len(e["agent_xy"]) for e in episodes)
        print("\n── Summary ──────────────────────────────────────────────────")
        print(f"  Reward  mean={np.mean(rewards):7.2f}  std={np.std(rewards):6.2f}  "
              f"min={np.min(rewards):7.2f}  max={np.max(rewards):7.2f}")
        print(f"  Cost    mean={np.mean(costs):7.2f}  std={np.std(costs):6.2f}  "
              f"zero-cost={100*np.mean(np.array(costs)==0):.0f}%")
        total_oob = sum(oob_steps)
        print(f"  Out-of-bounds: {total_oob} steps total  "
              f"({100*total_oob/max(total_steps,1):.1f}% of all steps)  "
              f"penalty={args.boundary_penalty}/step")
        print(f"\nTrajectory images saved to: {out_dir.resolve()}")

    finally:
        # Clean up temp directory created for cross-env evaluation
        if tmp_to_clean is not None and tmp_to_clean.exists():
            shutil.rmtree(tmp_to_clean, ignore_errors=True)


if __name__ == "__main__":
    main()
