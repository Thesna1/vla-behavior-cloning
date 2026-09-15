"""Generate a comparison plot: baseline BC (collapses/averages) vs CVAE BC
(commits to one side)."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from env import OBSTACLE_CENTER, OBSTACLE_RADIUS, START, GOAL

fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))

# --- Left panel: baseline ---
ax = axes[0]
traj = np.load("outputs/baseline_traj.npy")
ax.plot(traj[:, 0], traj[:, 1], color="crimson", linewidth=2, label="Baseline MLP rollout")
ax.add_patch(Circle(OBSTACLE_CENTER, OBSTACLE_RADIUS, color="gray", alpha=0.4))
ax.scatter(*START, color="black", zorder=5, label="Start")
ax.scatter(*GOAL, color="green", marker="*", s=150, zorder=5, label="Goal")
ax.set_title("Baseline BC (MSE regression)\nAverages left+right demos -> drives through obstacle")
ax.set_xlim(-3, 3); ax.set_ylim(-0.5, 5)
ax.set_aspect("equal")
ax.legend(loc="lower right", fontsize=8)

# --- Right panel: CVAE, several sampled z ---
ax = axes[1]
ax.add_patch(Circle(OBSTACLE_CENTER, OBSTACLE_RADIUS, color="gray", alpha=0.4))
ax.scatter(*START, color="black", zorder=5)
ax.scatter(*GOAL, color="green", marker="*", s=150, zorder=5)

colors = plt.cm.viridis(np.linspace(0, 1, 20))
for i in range(20):
    try:
        traj = np.load(f"outputs/cvae_traj_{i}.npy")
    except FileNotFoundError:
        continue
    side = "left" if traj[:, 0].min() < -0.3 else ("right" if traj[:, 0].max() > 0.3 else "center")
    style = "-" if side in ("left", "right") else "--"
    ax.plot(traj[:, 0], traj[:, 1], style, color=colors[i], alpha=0.7, linewidth=1.3)

ax.set_title("CVAE-conditioned BC\nEach sampled z commits to one coherent side")
ax.set_xlim(-3, 3); ax.set_ylim(-0.5, 5)
ax.set_aspect("equal")

plt.tight_layout()
plt.savefig("outputs/comparison.png", dpi=150)
print("Saved outputs/comparison.png")
