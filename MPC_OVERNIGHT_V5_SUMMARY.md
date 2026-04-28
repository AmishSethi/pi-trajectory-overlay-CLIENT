# MPC overnight-v5 sweep summary (2026-04-28)

9-hour autonomous broad sweep across 96 / 120 RoboLab tasks. Goal:
**find 20 tasks where MPC beats baseline.** Result: **4 tasks** found.
Honest report.

## TL;DR

MPC (in either of the two best variants from v3/v4) beats baseline on
**4 / 96 tasks tested** — 3 rescues + 1 speedup:

| task | benefit | baseline | best MPC |
|---|---|---|---|
| appleandyogurtinbowl | rescue | ✗ 500 | lam3 ✓ 65 |
| bowlinbin | rescue | ✗ 300 | lam10_grip ✓ 399 |
| onebottleinsquarepail | rescue | ✗ 300 | lam10_grip ✓ 24 |
| toyinbin | 3.7× speedup | ✓ 131 | lam3 ✓ 35 |

PLUS 1 baseline-only success (bananasincrate, 177 steps) that MPC
regresses — keep MPC OFF for that one.

## Why not 20

Base pi05_droid is the fundamental ceiling. **91 / 96 tasks fail at every
config** (baseline + 2 MPC variants). MPC arrow guidance can align EE
pixel with arrow, but cannot fix:
- VLA never emitting a gripper-close (63 / 96 tasks have ≤ 30 baseline
  grip closes — the VLA doesn't try to grasp at all)
- VLA grasping at the wrong pose (30 / 96 tasks have ≥ 30 grip closes
  but still fail — gripper closes but doesn't catch the object)

Pixel-level arrows can't conjure grasping capability. To rescue these
tasks needs a finetuned VLA, not better MPC.

## Per-task variant selection

There's no universal best — different tasks need different variants:
- `apple_yogurt`, `toyinbin` → use **v_arb_lam3** (λ=3 + arbitration)
- `bowlinbin`, `onebottleinsquarepail` → use **v_arb_lam10_grip**
  (λ=10 + arbitration + gripper-state-aware)
- `bananasincrate` → use **baseline** (no MPC). Both MPC variants regressed it.

## Test methodology

5 phases:
- **Phase A**: baseline on all 120 tasks at max_timesteps=300 → 3 successes
- **Phase B**: v_arb_lam10_grip + v_arb_lam3 on top 33 candidates (P1+P2) → 1 + 0 wins
- **Phase C**: clean head-to-head at max_timesteps=500 on 5 candidates → 4 best-variants
- **Phase D**: broad sweep with v_arb_lam10_grip + v_arb_lam3 on 35 tasks at max=500 → 2 NEW rescues
- 24 tasks segfaulted on Isaac Sim shutdown (common, retried partially in Phase A)

## Honest recommendation

Ship MPC as a **per-task opt-in**, not a default. The 4 tasks above are
worth the MPC overhead; the others are not.

For the real-robot main_mpc.py defaults, the v3/v4 recommendation
remains: `v_arb_lam10_grip` (or `v_arb_lam3` as fallback) — but only
turn it on for tasks where it's been verified to help.

## Artifacts

- `tmp/overnight_v5/FINAL_REPORT.md` — full report
- `tmp/overnight_v5/runs/phaseA..D_*/` — 96 task × variant rollouts
- `rollouts/2026_04_28_mpc_sweep_v5/winning_runs/` — the 4 actual wins +
  baseline bananasincrate, with rollout.mp4 each.
- All previous v3/v4 wins (apple_yogurt, canned_food, bananas_crate)
  also reproduced in v5 except canned_food (stochastic — succeeded in
  Phase C only).

