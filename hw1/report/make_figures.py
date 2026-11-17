"""Generate all figures for the HW1 report.

Usage (from hw1/):
    uv run --with matplotlib python report/make_figures.py \
        --mse-dir exp/seed_42_20261116_214703_mse \
        --flow-dir exp/seed_42_20261116_221635_flow
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gym_pusht  # noqa: F401
import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from hw1_imitation.data import Normalizer, download_pusht, load_pusht_zarr

FIG_DIR = Path(__file__).parent / "figures"
COLORS = {"mse": "#2a78c4", "flow": "#d9622b"}
LABELS = {"mse": "MSE policy", "flow": "Flow matching policy"}
CHUNK = 8
FLOW_STEPS = 10


# --------------------------------------------------------------------------- #
# 1. Training curves from log.csv
# --------------------------------------------------------------------------- #
def read_log(path: Path) -> dict[str, np.ndarray]:
    """Return separate (step, value) arrays for loss and reward (rows are sparse)."""
    lines = path.read_text().strip().splitlines()
    header = lines[0].split(",")
    rows = [dict(zip(header, line.split(","))) for line in lines[1:]]

    def series(key: str) -> tuple[np.ndarray, np.ndarray]:
        pts = [(int(r["step"]), float(r[key])) for r in rows if r.get(key)]
        arr = np.array(pts)
        return arr[:, 0], arr[:, 1]

    loss_s, loss = series("train/loss")
    rew_s, rew = series("eval/mean_reward")
    return {"loss_step": loss_s, "loss": loss, "rew_step": rew_s, "rew": rew}


def smooth(x: np.ndarray, w: int = 20) -> np.ndarray:
    if len(x) < w:
        return x
    kernel = np.ones(w) / w
    padded = np.concatenate([np.full(w - 1, x[0]), x])
    return np.convolve(padded, kernel, mode="valid")


def plot_curves(name: str, log: dict, threshold: float) -> None:
    c = COLORS[name]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))

    ax = axes[0]
    ax.plot(log["loss_step"], log["loss"], color=c, alpha=0.25, lw=0.8, label="raw")
    ax.plot(log["loss_step"], smooth(log["loss"]), color=c, lw=2, label="moving avg (20)")
    ax.set_xlabel("training step")
    ax.set_ylabel("training loss")
    ax.set_yscale("log")
    ax.set_title(f"{LABELS[name]}: training loss")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(log["rew_step"], log["rew"], "o-", color=c, lw=2)
    ax.axhline(threshold, color="gray", ls="--", lw=1)
    ax.text(log["rew_step"][0], threshold + 0.015, f"target {threshold}", color="gray")
    ax.set_xlabel("training step")
    ax.set_ylabel("eval mean reward (100 episodes)")
    ax.set_ylim(0, 1)
    ax.set_title(f"{LABELS[name]}: evaluation reward")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}_curves.png", dpi=150)
    plt.close(fig)


def plot_reward_comparison(logs: dict) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for name, log in logs.items():
        ax.plot(log["rew_step"], log["rew"], "o-", color=COLORS[name], lw=2, label=LABELS[name])
    ax.axhline(0.5, color=COLORS["mse"], ls=":", lw=1)
    ax.axhline(0.7, color=COLORS["flow"], ls=":", lw=1)
    ax.set_xlabel("training step")
    ax.set_ylabel("eval mean reward")
    ax.set_ylim(0, 1)
    ax.set_title("Evaluation reward: MSE vs flow matching")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "reward_comparison.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 2. Per-episode evaluation of the final checkpoints
# --------------------------------------------------------------------------- #
def load_final_checkpoint(run_dir: Path, device: torch.device) -> torch.nn.Module:
    ckpts = sorted(
        (run_dir / "wandb" / "files" / "checkpoints").glob("checkpoint_step_*.pkl"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    model = torch.load(ckpts[-1], map_location=device, weights_only=False)
    model.eval()
    print(f"loaded {ckpts[-1]}")
    return model


@torch.no_grad()
def rollout_stats(model, normalizer: Normalizer, device, n_episodes: int = 100) -> dict:
    """Same protocol as evaluation.evaluate_policy, but keeps per-episode numbers."""
    env = gym.make("gym_pusht/PushT-v0", obs_type="state")
    low, high = env.action_space.low, env.action_space.high
    max_rewards, lengths, success = [], [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        done, t, best, idx, chunk = False, 0, 0.0, CHUNK, None
        while not done:
            if chunk is None or idx >= CHUNK:
                s = torch.from_numpy(normalizer.normalize_state(obs)).float().to(device)
                pred = model.sample_actions(s[None], num_steps=FLOW_STEPS).cpu().numpy()[0]
                chunk = np.clip(normalizer.denormalize_action(pred), low, high)
                idx = 0
            obs, r, term, trunc, _ = env.step(chunk[idx].astype(np.float32))
            best = max(best, float(r))
            done = term or trunc
            idx += 1
            t += 1
        max_rewards.append(best)
        lengths.append(t)
        success.append(bool(term))
    env.close()
    return {
        "max_reward": np.array(max_rewards),
        "length": np.array(lengths),
        "success": np.array(success),
    }


def plot_episode_stats(stats: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    bins = np.linspace(0, 1, 21)
    for name, st in stats.items():
        axes[0].hist(
            st["max_reward"], bins=bins, alpha=0.55, color=COLORS[name],
            label=f"{LABELS[name]} (mean {st['max_reward'].mean():.3f})",
        )
    axes[0].set_xlabel("per-episode max reward")
    axes[0].set_ylabel("# episodes (of 100)")
    axes[0].set_title("Distribution of per-episode reward (final checkpoint)")
    axes[0].legend(frameon=False, loc="upper left")
    axes[0].grid(alpha=0.3)

    names = list(stats)
    rates = [stats[n]["success"].mean() * 100 for n in names]
    bars = axes[1].bar([LABELS[n] for n in names], rates, color=[COLORS[n] for n in names])
    for b, r in zip(bars, rates):
        axes[1].text(b.get_x() + b.get_width() / 2, r + 1, f"{r:.0f}%", ha="center")
    axes[1].set_ylabel("success rate (%)")
    axes[1].set_ylim(0, 100)
    axes[1].set_title("Success rate (coverage > 95%)")
    axes[1].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "episode_stats.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 3. Multimodality: many flow samples vs the single MSE prediction
# --------------------------------------------------------------------------- #
@torch.no_grad()
def plot_multimodality(models: dict, normalizer: Normalizer, states, actions,
                       episode_ends, device, n_candidates: int = 2000,
                       n_samples: int = 64, n_show: int = 4, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # valid chunk starts (same rule as the dataset)
    starts = np.concatenate(([0], episode_ends[:-1]))
    valid = np.concatenate([np.arange(s, e - CHUNK + 1) for s, e in zip(starts, episode_ends)])
    cand = rng.choice(valid, size=n_candidates, replace=False)

    s_norm = torch.from_numpy(normalizer.normalize_state(states[cand])).float().to(device)
    # sample n_samples flow chunks per candidate state
    rep = s_norm.repeat_interleave(n_samples, dim=0)
    flow = models["flow"].sample_actions(rep, num_steps=FLOW_STEPS).cpu().numpy()
    flow = normalizer.denormalize_action(flow).reshape(n_candidates, n_samples, CHUNK, 2)
    mse = models["mse"].sample_actions(s_norm).cpu().numpy()
    mse = normalizer.denormalize_action(mse)

    # Bimodality score: spread of flow endpoints, and how far the MSE endpoint
    # sits from the nearest flow sample (large = MSE predicts "between" modes).
    end = flow[:, :, -1, :]                                    # (N, S, 2)
    spread = end.std(axis=1).sum(-1)
    gap = np.linalg.norm(end - mse[:, None, -1, :], axis=-1).min(axis=1)
    score = spread * gap
    order = np.argsort(-score)
    picked: list[int] = []
    for i in order:  # avoid picking near-duplicate frames from the same episode
        if all(abs(cand[i] - cand[j]) > 50 for j in picked):
            picked.append(i)
        if len(picked) == n_show:
            break

    env = gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode="rgb_array")
    env.reset(seed=0)
    fig, axes = plt.subplots(1, n_show, figsize=(4 * n_show, 4.2))
    for ax, i in zip(axes, picked):
        t = cand[i]
        env.reset(options={"reset_to_state": states[t]})
        img = env.render()
        scale = img.shape[0] / 512.0
        ax.imshow(img)
        for k in range(n_samples):
            xy = flow[i, k] * scale
            ax.plot(xy[:, 0], xy[:, 1], "-", color=COLORS["flow"], alpha=0.25, lw=1)
            ax.plot(xy[-1, 0], xy[-1, 1], "o", color=COLORS["flow"], alpha=0.4, ms=3)
        exp_xy = actions[t:t + CHUNK] * scale
        ax.plot(exp_xy[:, 0], exp_xy[:, 1], "-", color="black", lw=2)
        ax.plot(exp_xy[-1, 0], exp_xy[-1, 1], "*", color="black", ms=11)
        m_xy = mse[i] * scale
        ax.plot(m_xy[:, 0], m_xy[:, 1], "-", color=COLORS["mse"], lw=2.5)
        ax.plot(m_xy[-1, 0], m_xy[-1, 1], "s", color=COLORS["mse"], ms=8)
        ax.set_title(f"dataset frame {t}")
        ax.axis("off")
    env.close()

    handles = [
        plt.Line2D([], [], color=COLORS["flow"], lw=2, label=f"flow samples (x{n_samples})"),
        plt.Line2D([], [], color=COLORS["mse"], lw=2.5, marker="s", label="MSE prediction"),
        plt.Line2D([], [], color="black", lw=2, marker="*", label="expert chunk (dataset)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Predicted 8-step action chunks (target positions) from the same state")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(FIG_DIR / "multimodality.png", dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 4. Rollout filmstrips from the logged eval videos
# --------------------------------------------------------------------------- #
def plot_rollout_strip(dirs: dict, episode: int, step: int, every: int = 25,
                       cols: int = 6) -> None:
    import imageio.v2 as imageio

    rows = []  # (label, frames)
    for name, d in dirs.items():
        vid = next((d / "wandb/files/media/videos/eval").glob(f"rollout_ep{episode}_{step}_*.mp4"))
        frames = [f for i, f in enumerate(imageio.get_reader(vid)) if i % every == 0]
        n_total = imageio.get_reader(vid).count_frames()
        for r in range(0, max(len(frames), 1), cols):
            rows.append((name, r * every, frames[r:r + cols], n_total))
    fig, axes = plt.subplots(len(rows), cols, figsize=(cols * 1.25, len(rows) * 1.32))
    for ax_row, (name, t0, frames, n_total) in zip(axes, rows):
        for c, ax in enumerate(ax_row):
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if c < len(frames):
                ax.imshow(frames[c])
                ax.set_title(f"t={t0 + c * every}", fontsize=7, pad=2)
            elif c == len(frames):
                ax.text(0.5, 0.5, f"episode ended\nat t={n_total}", ha="center",
                        va="center", fontsize=7, color="gray", transform=ax.transAxes)
        ax_row[0].set_ylabel(LABELS[name].replace(" policy", ""), fontsize=8,
                             color=COLORS[name], fontweight="bold")
    fig.tight_layout(pad=0.3, h_pad=0.4, w_pad=0.2)
    fig.savefig(FIG_DIR / f"rollout_ep{episode}.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mse-dir", type=Path, required=True)
    ap.add_argument("--flow-dir", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    args = ap.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(0)
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dirs = {"mse": args.mse_dir, "flow": args.flow_dir}
    logs = {n: read_log(d / "log.csv") for n, d in dirs.items()}
    plot_curves("mse", logs["mse"], 0.5)
    plot_curves("flow", logs["flow"], 0.7)
    plot_reward_comparison(logs)
    final_step = int(logs["mse"]["rew_step"][-1])
    for ep in (0, 2):
        plot_rollout_strip(dirs, ep, final_step)

    states, actions, episode_ends = load_pusht_zarr(download_pusht(args.data_dir))
    normalizer = Normalizer.from_data(states, actions)
    models = {n: load_final_checkpoint(d, device) for n, d in dirs.items()}

    stats = {n: rollout_stats(m, normalizer, device) for n, m in models.items()}
    plot_episode_stats(stats)

    plot_multimodality(models, normalizer, states, actions, episode_ends, device)

    summary = {
        n: {
            "mean_max_reward": float(st["max_reward"].mean()),
            "success_rate": float(st["success"].mean()),
            "median_max_reward": float(np.median(st["max_reward"])),
            "frac_reward_below_0.5": float((st["max_reward"] < 0.5).mean()),
            "mean_len_successful": float(st["length"][st["success"]].mean())
            if st["success"].any() else None,
        }
        for n, st in stats.items()
    }
    (FIG_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
