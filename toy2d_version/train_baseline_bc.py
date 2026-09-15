"""
Baseline behavior cloning: a single MLP trained with plain MSE loss to
predict action from state. This is the "naive" approach that most
tutorials start with -- and the one that breaks on multimodal
demonstration data.
"""

import numpy as np
from sklearn.neural_network import MLPRegressor

from env import generate_dataset, rollout_policy, obstacle_distance


def train_baseline(states, actions, seed=0):
    model = MLPRegressor(
        hidden_layer_sizes=(64, 64),
        activation="tanh",
        max_iter=2000,
        random_state=seed,
    )
    model.fit(states, actions)
    return model


def make_policy_fn(model):
    def policy_fn(state):
        return model.predict(state.reshape(1, -1))[0]
    return policy_fn


if __name__ == "__main__":
    states, actions, modes = generate_dataset(n_demos_per_mode=40, seed=0)
    print(f"Training on {len(states)} (state, action) pairs "
          f"({np.sum(modes=='left')} left, {np.sum(modes=='right')} right)")

    model = train_baseline(states, actions)
    policy_fn = make_policy_fn(model)

    traj, collided = rollout_policy(policy_fn)
    min_dist = min(obstacle_distance(p) for p in traj)

    print(f"Rollout length: {len(traj)} steps")
    print(f"Collided with obstacle: {collided}")
    print(f"Closest approach to obstacle center-line clearance: {min_dist:.3f} "
          f"(negative = inside the obstacle)")

    np.save("outputs/baseline_traj.npy", traj)
