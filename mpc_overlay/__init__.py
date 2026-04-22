"""Client-side MPC overlay for pi 0.5 action chunks.

The VLA is treated as a frozen black box. Per-step, the client sends a vanilla
observation to the policy server and receives an action chunk ``a_vla``. If
user waypoints (drawn or predicted by a foundation model) are available, the
client then refines ``a_vla`` locally with a short CEM optimisation:

    J(a) = lam_p * ||a - a_vla||^2
         + lam_a * masked_L2(project(a), waypoints_px)     # 2D pixel space
         + lam_c * (joint_limits + action_box)
         + lam_s * smoothness

This package is robot-agnostic — every robot-specific quantity (intrinsics,
extrinsic, joint_vel_scale, control_dt, image-flip flag, current q0) is
bundled into a :class:`GuidanceSpec` built by the caller. Real-robot and
simulator clients share this library and only differ in the adapter that
constructs the spec.

Public API:
    franka_fk.fk_ee_position / fk_ee_pose
    trajectory_cost.GuidanceSpec, predict_ee_pixels, trajectory_cost
    mpc.MPCWeights, CEMParams, mpc_overlay, build_mpc_cost, cem_optimize
"""

from mpc_overlay.franka_fk import fk_ee_pose
from mpc_overlay.franka_fk import fk_ee_position
from mpc_overlay.mpc import CEMParams
from mpc_overlay.mpc import MPCWeights
from mpc_overlay.mpc import action_box_penalty
from mpc_overlay.mpc import arrow_penalty
from mpc_overlay.mpc import build_mpc_cost
from mpc_overlay.mpc import cem_optimize
from mpc_overlay.mpc import joint_limit_penalty
from mpc_overlay.mpc import mpc_overlay
from mpc_overlay.mpc import prior_penalty
from mpc_overlay.mpc import smoothness_penalty
from mpc_overlay.trajectory_cost import GuidanceSpec
from mpc_overlay.trajectory_cost import predict_ee_pixels
from mpc_overlay.trajectory_cost import trajectory_cost

__all__ = [
    "CEMParams",
    "GuidanceSpec",
    "MPCWeights",
    "action_box_penalty",
    "arrow_penalty",
    "build_mpc_cost",
    "cem_optimize",
    "fk_ee_pose",
    "fk_ee_position",
    "joint_limit_penalty",
    "mpc_overlay",
    "predict_ee_pixels",
    "prior_penalty",
    "smoothness_penalty",
    "trajectory_cost",
]
