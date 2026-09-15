"""
A minimal 2D "reach around an obstacle" environment.

This stands in for a real PyBullet pick-and-place task, but keeps the
same essential structure:
    - a start position and a goal position
    - an obstacle directly between them
    - two equally valid ways to get from start to goal (go left of the
      obstacle, or go right of it)

This is the simplest possible setting in which the imitation-learning
"multimodality problem" appears: a single state can be followed by two
very different but equally correct actions in the demonstration data.
"""

import numpy as np

START = np.array([0.0, 0.0])
GOAL = np.array([0.0, 4.0])
OBSTACLE_CENTER = np.array([0.0, 2.0])
OBSTACLE_RADIUS = 0.8
DT = 0.1  # timestep


def obstacle_distance(pos):
    return np.linalg.norm(pos - OBSTACLE_CENTER) - OBSTACLE_RADIUS


def scripted_trajectory(mode: str, noise_std: float = 0.02, rng=None):
    """
    Generate one scripted demonstration trajectory.

    mode: "left" or "right" -- which side of the obstacle to pass on.
    Returns arrays of states and actions (velocity commands) along the path.
    The final approach to GOAL decelerates smoothly, so the policy also
    learns to slow down and stop instead of flying through the goal.
    """
    if rng is None:
        rng = np.random.default_rng()

    detour_x = -1.4 if mode == "left" else 1.4
    waypoints = [
        START.copy(),
        np.array([detour_x, 2.0]),
        GOAL.copy(),
    ]

    states, actions = [], []
    pos = START.copy()
    for wp_idx, wp in enumerate(waypoints[1:]):
        is_final = (wp_idx == len(waypoints) - 2)
        while np.linalg.norm(wp - pos) > 0.05:
            to_wp = wp - pos
            dist = np.linalg.norm(to_wp)
            direction = to_wp / dist
            if is_final:
                # Decelerate on the final approach so the policy learns to
                # slow down and stop at the goal, not fly through it.
                speed = np.clip(1.2 * (dist / 1.0), 0.05, 1.2)
            else:
                speed = 1.2
            action = direction * speed
            noisy_action = action + rng.normal(0, noise_std, size=2)

            states.append(pos.copy())
            actions.append(noisy_action.copy())

            pos = pos + noisy_action * DT
    return np.array(states), np.array(actions)


def generate_dataset(n_demos_per_mode: int = 25, seed: int = 0):
    """
    Generate a demonstration dataset with roughly half the trajectories
    going left of the obstacle and half going right -- i.e. a genuinely
    multimodal (state -> action) mapping.
    """
    rng = np.random.default_rng(seed)
    all_states, all_actions, all_modes = [], [], []

    for mode in ["left", "right"]:
        for _ in range(n_demos_per_mode):
            s, a = scripted_trajectory(mode, rng=rng)
            all_states.append(s)
            all_actions.append(a)
            all_modes.extend([mode] * len(s))

    states = np.concatenate(all_states, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    modes = np.array(all_modes)
    return states, actions, modes


def rollout_policy(policy_fn, max_steps=200):
    """
    Roll out a policy (a function state -> action) in the environment
    starting from START, and return the resulting trajectory plus whether
    it collided with the obstacle.
    """
    pos = START.copy()
    traj = [pos.copy()]
    collided = False
    for _ in range(max_steps):
        action = policy_fn(pos)
        pos = pos + np.clip(action, -2, 2) * DT
        traj.append(pos.copy())
        if obstacle_distance(pos) < 0:
            collided = True
        if np.linalg.norm(pos - GOAL) < 0.1:
            break
    return np.array(traj), collided
