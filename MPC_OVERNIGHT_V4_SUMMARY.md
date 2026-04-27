# MPC overnight-v4 sweep summary (2026-04-27)

5-hour follow-up to overnight-v3, focused on improving the gripper/grasp
behavior while preserving arrow following. Built on v3's `v_arb_lam3`
recipe.

## TL;DR

**Two-tier recommendation:**

1. **`v_arb_lam10_grip`** (NEW, recommended for fast & grasp-preserving):
   λ=10 + α-arbitration + gripper-state-aware arbitration. Achieves the
   user's stated goal of "still grasps at right pose AND follows arrows".
   2 wins / 18 tested, but **2.5× faster than v_arb_lam3** when it wins
   (bananas_crate: 158 steps vs 390).

2. **`v_arb_lam3`** (v3 winner, still recommended for max coverage):
   3 wins / 20 tested. Slower successes but broader.

## New code: gripper-state-aware arbitration

Committed `0485d97`. Adds these flags to `simulator/main_robolab.py`:

```bash
--mpc-arbitration-gripper-threshold 0.5    # 0 = disabled
--mpc-arbitration-gripper-tau 0.05         # sigmoid sharpness
```

When enabled, the existing pixel-distance α-arbitration is multiplied
by `α_gripper = sigmoid((threshold - gripper_state_now) / tau)`. When
the gripper closes (carrying object), α_gripper → 0 and MPC arrow-pull
goes off — the VLA does placement naturally.

## Why this works

In v3 we found λ_a=10 broke grasping universally — MPC's CEM perturbations
disrupted VLA's grasping micro-motions. We had to drop to λ=3 to preserve
grasping, but that limited how strongly MPC could pull the EE during
approach.

Gripper-state-aware arbitration provides a **clean phase boundary**:

| phase | gripper | MPC active? |
|---|---|---|
| approach | open (state ~ 0) | yes — at full λ=10 |
| grasp moment | transitioning | pixel-α decoupling |
| carry | closed (state ~ 1) | **no — α_gripper=0** |
| placement | closed | no |

So MPC can be aggressive during approach (where it most needs to direct
the EE toward the arrow) without disrupting any subsequent VLA-driven
grasping or placement.

## Recommended hyperparameters

For real-robot `main_mpc.py` (and sim `main_robolab.py`):

```bash
# v_arb_lam10_grip (recommended):
--guidance-mode=mpc
--mpc-lam-p=1.0 --mpc-lam-a=10.0
--mpc-lam-c=100.0 --mpc-lam-s=0.01 --mpc-lam-prog=1.0
--mpc-arrow-lookahead=0.15 --mpc-init-std=0.05
--mpc-arbitration-d-grasp-px=50
--mpc-arbitration-tau-px=15
--mpc-prior-boost-near-waypoint=2.0
--mpc-arbitration-gripper-threshold=0.5  # NEW
--mpc-arbitration-gripper-tau=0.05       # NEW
# do NOT enable --mpc-gripper-force-override (regressed every task in v3)
# do NOT enable --mpc-trust-region-radius (redundant with arbitration)
```

For real-robot main_mpc.py, this also requires passing
`gripper_state_now=current_gripper_position` from each step's obs into the
MPC spec — the sim adapter already does this; the real-robot equivalent
needs the same wiring (TODO).

## Honest results on the 20-task RoboLab sweep

3 unique tasks rescued by SOME MPC variant: apple_yogurt, canned_food,
bananas_crate. Same as v3.

17 tasks remain unsolvable at any tested config — failure is upstream of
MPC: base pi05 either doesn't grasp at all, or grasps at wrong pose.
Pixel arrows can't conjure grasp capability. These need a finetuned VLA.

The v4 contribution is **speed and grasp preservation on the same 3
rescuable tasks**, not breaking through the 17-task ceiling.

## Variants tested (13)

| variant | wins | notes |
|---|---|---|
| v_arb_lam1 | 0 | too weak |
| v_arb_lam1.5 | 1 | apple_yogurt only |
| v_arb_lam2 | 1 | apple_yogurt only |
| **v_arb_lam3** | **3** | v3 winner |
| v_arb_lam5 | 2 | apple_yogurt + canned_food |
| v_arb_lam3_grip | 0 | grip-aware too restrictive at low λ |
| v_arb_lam3_grip_soft | 0 | even with softer τ |
| v_arb_lam5_grip | 0 |  |
| **v_arb_lam10_grip** | **2** | **2.5× faster than lam3 on shared wins** |
| v_arb_d80 | 0 | wider zone doesn't help |
| v_arb_d100 | 1 | very wide loses speed |
| v_arb_d80_grip | 1 | grip-aware at d=80 |
| v_arb_la05 | 0 | too-short lookahead breaks rescues |
| v_arb_la25 | 0 | longer lookahead also no benefit |

## Artifacts

- `tmp/overnight_v4/FINAL_REPORT.md` — full results matrix
- `tmp/overnight_v4/runs/phase{A,B,C,D}_*/` — per-rollout logs
- `tmp/overnight_v4/scripts/run_one.sh` — all 14 variant presets
- Code: `mpc_overlay/{trajectory_cost,mpc}.py`, `simulator/{main_robolab,mpc_sim_adapter}.py`
