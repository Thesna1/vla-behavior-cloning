"""
Baseline behavior cloning for the PyBullet Panda arm: a single MLP
trained with MSE loss. Expect this to fail on the multimodal
left/right demonstration data, same as toy2d_version/train_baseline_bc.py.
"""

import numpy as np
import torch
import torch.nn as nn

from env import PickPlaceEnv, generate_dataset


class MLPPolicy(nn.Module):
    def __init__(self, state_dim=3, action_dim=3, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, s):
        return self.net(s)


def train(states, actions, epochs=300, lr=1e-3, device="cpu"):
    model = MLPPolicy().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    s = torch.tensor(states, dtype=torch.float32, device=device)
    a = torch.tensor(actions, dtype=torch.float32, device=device)

    for epoch in range(epochs):
        opt.zero_grad()
        pred = model(s)
        loss = nn.functional.mse_loss(pred, a)
        loss.backward()
        opt.step()
        if epoch % 50 == 0:
            print(f"epoch {epoch:4d}  loss={loss.item():.5f}")
    return model


if __name__ == "__main__":
    print("Generating demonstrations (this drives the physics sim, may take a minute)...")
    states, actions, modes = generate_dataset(n_demos_per_mode=15, gui=False)
    print(f"Collected {len(states)} (state, action) pairs")

    model = train(states, actions)
    model.eval()

    env = PickPlaceEnv(gui=False)
    env.reset()
    pos = env.get_ee_position()
    traj, min_clear = [pos.copy()], np.inf
    for _ in range(150):
        with torch.no_grad():
            action = model(torch.tensor(pos, dtype=torch.float32)).numpy()
        pos = env.step(action)
        traj.append(pos.copy())
        min_clear = min(min_clear, env.obstacle_clearance(pos))
        if np.linalg.norm(pos - env.goal_pos) < 0.05:
            break
    env.close()

    traj = np.array(traj)
    print(f"Rollout steps: {len(traj)}")
    print(f"Closest clearance to obstacle: {min_clear:.3f} "
          f"(negative = collision)")
    np.save("outputs/pb_baseline_traj.npy", traj)
