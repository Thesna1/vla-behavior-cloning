"""
PyBullet pick-and-place environment with an obstacle between the arm's
start pose and the target -- mirrors toy2d_version/env.py but with a real
simulated robot arm (Franka Panda) instead of a point mass.

Run this on a machine with network access:
    pip install pybullet numpy torch

This file is intentionally close in structure to toy2d_version/env.py so
you can compare the two directly: same idea (two valid ways to reach the
goal around an obstacle), same dataset format (states, actions), just
with real arm kinematics and a physics simulator underneath.
"""

import time
import numpy as np
import pybullet as p
import pybullet_data


class PickPlaceEnv:
    def __init__(self, gui=False):
        self.gui = gui
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.reset()

    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.loadURDF("plane.urdf")

        self.robot = p.loadURDF(
            "franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True
        )
        self.ee_link = 11  # Panda end-effector link index
        self.num_joints_controlled = 7  # first 7 revolute joints (arm, not gripper)

        # Obstacle: a fixed box between the arm's natural start and target.
        obstacle_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05, 0.15, 0.15])
        obstacle_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.05, 0.15, 0.15],
                                            rgbaColor=[0.5, 0.5, 0.5, 1])
        self.obstacle_pos = [0.5, 0.0, 0.3]
        self.obstacle_id = p.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=obstacle_col,
            baseVisualShapeIndex=obstacle_vis, basePosition=self.obstacle_pos)

        self.start_pos = np.array([0.3, 0.0, 0.5])
        self.goal_pos = np.array([0.7, 0.0, 0.3])

        self._move_ee_to(self.start_pos)
        for _ in range(20):
            p.stepSimulation()

    def _move_ee_to(self, target_pos, target_orn=None):
        if target_orn is None:
            target_orn = p.getQuaternionFromEuler([np.pi, 0, 0])
        joint_targets = p.calculateInverseKinematics(
            self.robot, self.ee_link, target_pos, target_orn)
        for i in range(self.num_joints_controlled):
            p.setJointMotorControl2(
                self.robot, i, p.POSITION_CONTROL,
                targetPosition=joint_targets[i], force=200)

    def get_ee_position(self):
        state = p.getLinkState(self.robot, self.ee_link)
        return np.array(state[0])

    def step(self, action, substeps=4):
        """action: 3D end-effector velocity command (m/s)."""
        current = self.get_ee_position()
        target = current + np.asarray(action) * (substeps / 240.0)
        self._move_ee_to(target)
        for _ in range(substeps):
            p.stepSimulation()
            if self.gui:
                time.sleep(1 / 240.0)
        return self.get_ee_position()

    def obstacle_clearance(self, pos):
        # Distance from end-effector to the obstacle's nearest surface (approx, box as sphere)
        obstacle_radius_approx = 0.2
        return np.linalg.norm(np.array(pos) - np.array(self.obstacle_pos)) - obstacle_radius_approx

    def close(self):
        p.disconnect(self.client)


def scripted_trajectory(env: PickPlaceEnv, mode: str, noise_std=0.005):
    """
    Scripted demonstration: move from start to goal via a detour waypoint
    that passes the obstacle on the left (+y) or right (-y) side.
    """
    detour_y = 0.25 if mode == "left" else -0.25
    waypoints = [
        env.start_pos.copy(),
        np.array([0.5, detour_y, 0.4]),
        env.goal_pos.copy(),
    ]

    states, actions = [], []
    env.reset()
    pos = env.get_ee_position()
    rng = np.random.default_rng()

    MAX_STEPS_PER_WAYPOINT = 300  # safety cap: ~5 sim-seconds at 4 substeps/step
    for wp_idx, wp in enumerate(waypoints[1:]):
        is_final = wp_idx == len(waypoints) - 2
        step_count = 0
        while np.linalg.norm(wp - pos) > 0.03 and step_count < MAX_STEPS_PER_WAYPOINT:
            to_wp = wp - pos
            dist = np.linalg.norm(to_wp)
            direction = to_wp / (dist + 1e-8)
            speed = np.clip(0.4 * (dist / 0.3), 0.05, 0.4) if is_final else 0.4
            action = direction * speed + rng.normal(0, noise_std, size=3)

            states.append(pos.copy())
            actions.append(action.copy())
            pos = env.step(action)
            step_count += 1

        if step_count >= MAX_STEPS_PER_WAYPOINT:
            print(f"  [warning] waypoint {wp_idx} did not converge "
                  f"(final distance {np.linalg.norm(wp - pos):.3f}m) -- "
                  f"moving on anyway")

    return np.array(states), np.array(actions)


def generate_dataset(n_demos_per_mode=15, gui=False, seed=0):
    env = PickPlaceEnv(gui=gui)
    all_states, all_actions, all_modes = [], [], []
    for mode in ["left", "right"]:
        for _ in range(n_demos_per_mode):
            s, a = scripted_trajectory(env, mode)
            all_states.append(s)
            all_actions.append(a)
            all_modes.extend([mode] * len(s))
    env.close()
    return (np.concatenate(all_states), np.concatenate(all_actions),
            np.array(all_modes))


if __name__ == "__main__":
    states, actions, modes = generate_dataset(n_demos_per_mode=5, gui=True)
    print(f"Collected {len(states)} (state, action) pairs")
    np.save("outputs/pb_states.npy", states)
    np.save("outputs/pb_actions.npy", actions)
    np.save("outputs/pb_modes.npy", modes)
