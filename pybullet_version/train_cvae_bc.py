"""
CVAE-conditioned behavior cloning for the PyBullet Panda arm -- the fix
for the multimodality problem demonstrated in train_baseline_bc.py.

Same idea as toy2d_version/cvae.py (encoder q(z|s,a), decoder p(a|s,z)),
implemented in PyTorch here since it's available on a machine with
network access.
"""

import numpy as np
import torch
import torch.nn as nn

from env import PickPlaceEnv, generate_dataset

STATE_DIM, ACTION_DIM, LATENT_DIM = 3, 3, 2


class Encoder(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM + ACTION_DIM, hidden), nn.Tanh(),
        )
        self.mu = nn.Linear(hidden, LATENT_DIM)
        self.logvar = nn.Linear(hidden, LATENT_DIM)

    def forward(self, s, a):
        h = self.net(torch.cat([s, a], dim=-1))
        return self.mu(h), self.logvar(h)


class Decoder(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM + LATENT_DIM, hidden), nn.Tanh(),
            nn.Linear(hidden, ACTION_DIM),
        )

    def forward(self, s, z):
        return self.net(torch.cat([s, z], dim=-1))


class CVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, s, a):
        mu, logvar = self.encoder(s, a)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        a_hat = self.decoder(s, z)
        return a_hat, mu, logvar

    def sample_action(self, s, z):
        with torch.no_grad():
            return self.decoder(s, z)


def loss_fn(a_hat, a, mu, logvar, beta=0.02):
    recon = nn.functional.mse_loss(a_hat, a)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kl, recon, kl


def train(states, actions, epochs=800, lr=1e-3, beta=0.02, device="cpu"):
    model = CVAE().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    s = torch.tensor(states, dtype=torch.float32, device=device)
    a = torch.tensor(actions, dtype=torch.float32, device=device)

    for epoch in range(epochs):
        opt.zero_grad()
        a_hat, mu, logvar = model(s, a)
        loss, recon, kl = loss_fn(a_hat, a, mu, logvar, beta=beta)
        loss.backward()
        opt.step()
        if epoch % 100 == 0:
            print(f"epoch {epoch:4d}  loss={loss.item():.5f}  "
                  f"recon={recon.item():.5f}  kl={kl.item():.5f}")
    return model


if __name__ == "__main__":
    print("Generating demonstrations (this drives the physics sim, may take a minute)...")
    states, actions, modes = generate_dataset(n_demos_per_mode=15, gui=False)
    print(f"Collected {len(states)} (state, action) pairs")

    model = train(states, actions)
    model.eval()

    env = PickPlaceEnv(gui=False)
    results = []
    for i in range(10):
        z = torch.clamp(torch.randn(1, LATENT_DIM), -1.3, 1.3)
        env.reset()
        pos = env.get_ee_position()
        traj, min_clear = [pos.copy()], np.inf
        for _ in range(150):
            s_t = torch.tensor(pos, dtype=torch.float32).unsqueeze(0)
            action = model.sample_action(s_t, z).numpy()[0]
            pos = env.step(action)
            traj.append(pos.copy())
            min_clear = min(min_clear, env.obstacle_clearance(pos))
            if np.linalg.norm(pos - env.goal_pos) < 0.05:
                break
        traj = np.array(traj)
        reached = np.linalg.norm(traj[-1] - env.goal_pos) < 0.05
        print(f"z={z.numpy().round(2).tolist()}  steps={len(traj):3d}  "
              f"clearance={min_clear:.3f}  reached_goal={reached}")
        np.save(f"outputs/pb_cvae_traj_{i}.npy", traj)
        results.append((z, traj, min_clear, reached))
    env.close()

    n_clean = sum(1 for r in results if r[3] and r[2] > 0)
    print(f"\n{n_clean}/10 rollouts reached the goal without colliding.")
