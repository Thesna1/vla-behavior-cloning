# Behavior Cloning with Latent-Conditioned Policies

the given small project showsvthe **multimodality problem** in imitation
learning, and for the correction i am using conditioning the policy on a latent
variable (CVAE-style), the same idea underlying modern VLA architectures
like ACT and Diffusion Policy.

## The problem

the available data often contains multiple equally valid ways to solve a
task from the same state (for example going left vs. right around an obstacle).
A policy trained with plain MSE regression can't represent "pick one of
several valid actions" -- it can only output a single point estimate, so
it learns something close to the average of the valid actions. That
average is frequently not a valid action at all: in this project's
task, it drives straight through an obstacle that every single
demonstration went around.

## The fix

A Conditional VAE decoder, `p(action | state, z)`, conditioned on a
latent `z`. Sampling `z` once per rollout lets the policy commit to one
coherent mode of behavior (fully left, or fully right) instead of
blending between them at every timestep.

## Two versions in this repo

### `toy2d_version/` -- runs anywhere

A simplified 2D point-mass environment. Same problem structure: start,
goal, an obstacle with two valid detour paths. The CVAE is implemented
from scratch in plain numpy (manual forward/backward pass) so it has
zero external dependencies.

**Tested results** (see `outputs/comparison.png`):
- **Baseline MLP (MSE)**: collides with the obstacle on every single
  rollout. It reliably drives straight down the center -- the average of the left and right
  demonstrations.
- **CVAE-conditioned policy**: 7/20 sampled `z` values produce a clean
  rollout (reaches the goal, never collides). Every clean rollout
  given to one side. This is a minimal from-scratch implementation
  with no validation or early stopping, so the success rate is
  imperfect -- but it never averages through the obstacle center the
  way the baseline does, which is the actual point being demonstrated.

Run it:
```bash
cd toy2d_version
python3 train_baseline_bc.py   # trains baseline, saves outputs/baseline_traj.npy
python3 train_cvae_bc.py       # trains CVAE, saves outputs/cvae_traj_*.npy
python3 visualize.py           # produces outputs/comparison.png
```

### `pybullet_version/` -- the real project, run on your own machine

The actual PyBullet + PyTorch version, using a simulated Franka Panda
arm doing pick-and-place around a physical obstacle. Same structure as
the toy version but with real arm kinematics and physics. 

```bash
cd pybullet_version
pip install pybullet torch numpy
python3 train_baseline_bc.py
python3 train_cvae_bc.py
```

## Conclusion

- Understanding of *why* imitation learning fails on multimodal data,
  not just that it does.
- Implemented the CVAE fix to see, including the practical
  z-clipping trick needed to keep prior-sampled rollouts stable.
- The exact mechanism (latent-conditioned generation to resolve
  multimodal action distributions) that underlies ACT and Diffusion
  Policy in real VLA systems.

## Limitations

- The toy version's CVAE success rate (7/20) reflects a minimal,
  untuned, from-scratch implementation -- not a fundamental limit of
  the method. Production implementations (ACT, Diffusion Policy) use
  more data, better architectures, and validation-based tuning.
