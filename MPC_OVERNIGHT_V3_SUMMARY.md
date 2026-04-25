# MPC overnight-v3 sweep summary (2026-04-25)

Comprehensive sweep of MPC variants over 20 RoboLab tabletop tasks against
base pi0.5 droid policy, building on the v2 sweep findings.

## TL;DR

**Recommended MPC defaults: `v_arb_lam3`** (λ_a=3 + α-arbitration).

Rescues 2 tasks (apple_yogurt, canned_food) over baseline AND preserves
the one task baseline already solves (bananas_crate). Net: +2 tasks /
0 regressions.

## Variant tested → wins / 20

| variant | wins | preserves baseline (bananas_crate) | one-line description |
|---|---|---|---|
| v0_baseline | 1 | ✓ (by definition) | upstream pi0.5, no MPC |
| v_basic | 1 | ✗ | λ_a=10 MPCC sliding window |
| v_basic_lam3 | 0 | ✗ | λ_a=3 MPCC, no arbitration |
| v_force | 0 | n/a | + gripper force-override |
| v_arb | 1 | ✗ | λ_a=10 + α-arbitration |
| **v_arb_lam3** | **2 + 1 preserved = 3** | **✓** | λ_a=3 + α-arbitration ← **winner** |
| v_combo | 0 | n/a | v_arb + force-override |
| v_combo_lam3 | 0 | ✗ | v_arb_lam3 + force-override |
| v_trust_arb | 0 | n/a | v_arb + trust-region radius |

## Winning hyperparameters

```bash
--guidance-mode=mpc
--mpc-lam-p=1.0 --mpc-lam-a=3.0 --mpc-lam-c=100.0
--mpc-lam-s=0.01 --mpc-lam-prog=1.0
--mpc-arrow-lookahead=0.15 --mpc-init-std=0.05
--mpc-arbitration-d-grasp-px=50
--mpc-arbitration-tau-px=15
--mpc-prior-boost-near-waypoint=2.0
# DO NOT enable --mpc-gripper-force-override (regressed every task in our tests)
# DO NOT enable --mpc-trust-region-radius (redundant with arbitration)
```

## Why λ_a=3 + arbitration?

The strongest signal comes from `bananas_crate`, the one task baseline solves:

| variant | grip_close cycles | result |
|---|---|---|
| baseline (no MPC) | 35 in 145 steps | ✓ success |
| v_basic (λ_a=10, no arb) | 5 in 400 | ✗ fail (regression) |
| v_arb (λ_a=10) | 40 in 400 | ✗ fail |
| **v_arb_lam3 (λ_a=3)** | **65 in 390** | **✓ success** |

At λ_a=10, MPC's CEM perturbations on the joint-velocity chunk dominate
the VLA's grasping micro-motions. λ_a=3 makes MPC influence gentle.
Arbitration further disables MPC's arrow pull when the EE projects within
50 px of an arrow endpoint, fully decoupling MPC from grasp/release timing.

## Tasks tested

3 rescues (= base pi05 fails, v_arb_lam3 succeeds):
- apple_yogurt (`AppleAndYogurtInBowlTask`)
- canned_food (`CannedFoodInBinTask`)

1 baseline-preserved (= base pi05 succeeds, v_arb_lam3 also succeeds):
- bananas_crate (`BananasInCrateTask`)

17 hard tasks, NO MPC variant rescues:
banana_bowl, smartphone_bin, bagels_plate, mug_center, throw_apple,
spoon_mug, small_pumpkin, wood_spatula, fruits_moving, keyboard_out,
put_bowl_shelf, clamp_bin, green_spoons, marker_mug, fruits_onion,
recycle_carton, white_mugs_bin.

For these 17, the failure mode is upstream of MPC: base pi05 either
doesn't try to grasp at all (e.g. fruits_onion, marker_mug have 0 grip
closes in baseline) or grasps at the wrong pose. Pixel-level arrow
guidance can't conjure grasping capability that isn't there. These would
need a finetuned VLA, not better MPC.

## What was NOT tried (next steps if iterating)

- **Gripper-state-aware arbitration**: scale MPC influence by gripper open/closed
  state (closed = carrying = let VLA place). Currently arbitration is purely
  pixel-distance-based.
- **Multi-segment arrows**: split LLM arrow into approach (`home → manip`)
  and place (`manip → target`) segments — currently a single arrow puts
  both endpoints at object positions, never representing the home-pose
  approach phase.
- **Per-task λ tuning**: e.g. λ_a=1.5 for the most VLA-fragile tasks.
- **Higher max_timesteps (600+)**: bananas_crate succeeded at 390/400 steps
  — barely. Some other near-miss tasks might succeed with more headroom.

## Artifacts

Run logs and result.txt files: `tmp/overnight_v3/runs/phase{1..5}_*/<variant>__<task>/<timestamp>/`

Winning rollouts (MP4): `tmp/overnight_v3/winning_videos/v_arb_lam3__{apple_yogurt,canned_food,bananas_crate}.mp4` and `v0_baseline__bananas_crate.mp4`

Full details: `tmp/overnight_v3/FINAL_REPORT.md`
