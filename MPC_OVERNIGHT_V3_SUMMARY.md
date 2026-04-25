# MPC overnight-v3 sweep summary (2026-04-25)

Comprehensive 8-hour sweep of MPC variants over 20 RoboLab tabletop tasks
against base pi0.5 droid policy.

## TL;DR

**Recommended MPC defaults: `v_arb_lam3`** (λ_a=3 + α-arbitration).

It is the only variant tested that simultaneously **rescues 2 tasks** the
base pi05 fails (apple_yogurt, canned_food) AND **preserves the 1 task
baseline already solves** (bananas_crate). Net: **+2 / 0 regressions**.

## Variant tested → wins / 20

| variant | wins | preserves baseline (bananas_crate)? |
|---|---|---|
| v0_baseline | 1 (bananas_crate) | ✓ (def.) |
| v_basic (λ_a=10) | 1 (apple_yogurt) | ✗ |
| v_basic_lam3 (λ_a=3, no arb) | 0 | ✗ |
| v_force (force-grip) | 0 | n/a |
| v_force_lam3 | 1 (apple_yogurt) | ✗ |
| v_arb (λ_a=10 + arb) | 2 (apple_yogurt, canned_food) | ✗ |
| **v_arb_lam3 (λ_a=3 + arb)** | **2 + 1 preserved = 3** | **✓** |
| v_combo (arb + force) | 0 | n/a |
| v_combo_lam3 | 0 | ✗ |
| v_trust_arb (arb + trust-region) | 0 | n/a |

## Winning hyperparameters

```bash
--guidance-mode=mpc
--mpc-lam-p=1.0 --mpc-lam-a=3.0      # NOT 10 — that suppresses VLA grasping
--mpc-lam-c=100.0 --mpc-lam-s=0.01
--mpc-lam-prog=1.0
--mpc-arrow-lookahead=0.15
--mpc-init-std=0.05
--mpc-arbitration-d-grasp-px=50
--mpc-arbitration-tau-px=15
--mpc-prior-boost-near-waypoint=2.0
# DO NOT enable: --mpc-gripper-force-override (regressed every task)
# DO NOT enable: --mpc-trust-region-radius   (redundant with arbitration)
```

## Why λ_a=3 + arbitration?

Strongest signal: `bananas_crate`, the one baseline-success task.

| variant | grip_close cycles | result |
|---|---|---|
| baseline (no MPC) | 35 in 145 steps | ✓ |
| v_basic (λ_a=10) | 5 / 400 | ✗ regression |
| v_basic_lam3 (λ_a=3, no arb) | 53 / 400 | ✗ |
| v_arb (λ_a=10 + arb) | 40 / 400 | ✗ |
| **v_arb_lam3 (λ_a=3 + arb)** | **65 / 390** | **✓** |

At λ_a=10, MPC's CEM perturbations dominate the VLA's grasping
micro-motions and suppress closures. λ_a=3 is gentle enough to preserve
them. **Arbitration alone isn't sufficient at λ=10**; **lower λ alone
isn't sufficient without arbitration**. Both are needed.

## The 17 hard tasks

No MPC variant rescues:
banana_bowl, smartphone_bin, bagels_plate, mug_center, throw_apple,
spoon_mug, small_pumpkin, wood_spatula, fruits_moving, keyboard_out,
put_bowl_shelf, clamp_bin, green_spoons, marker_mug, fruits_onion,
recycle_carton, white_mugs_bin.

Failure modes upstream of MPC:
- **VLA never tries to grasp** (0 grip-closes in baseline): fruits_onion,
  put_bowl_shelf, white_mugs_bin, wood_spatula. Arrow can't fix what isn't
  there.
- **VLA grasps at wrong pose** (many closes, none on object): banana_bowl,
  throw_apple, marker_mug.

Both need a finetuned VLA — pixel-level arrows can't conjure grasping
capability the VLA hasn't learned.

## Methodology

- 11 variants × 20 tasks ≈ 220 rollout slots; ~140 completed (segfaults
  during Isaac Sim concurrent boot reduced throughput).
- 5-GPU peak parallelism (3,4,5,6,8); 3-GPU "safe" mode after observing
  segfaults at 5-way concurrency.
- Each rollout: max_timesteps=400, plan-freq=150 (LLM re-plan every 150
  steps via Gemini-ER + GPT-4o-mini), open-loop horizon=8.
- Server: `pi05_droid` base, JAX, port 8001 (sim path).
- Watchdog kills any rollout whose log file is unchanged for >8 min
  (catches Isaac Sim hangs).

## What was NOT tried (next steps)

In priority order if iterating:

1. **Gripper-state-aware arbitration**: scale α by gripper open/closed.
   Currently arbitration is purely pixel-distance based.
2. **Multi-segment arrows**: approach (`home→manip`) + place (`manip→target`).
3. **Per-task λ_a fine sweep**: 1.5/2/5.
4. **Higher max_timesteps**: bananas_crate succeeded at 390/400 in
   v_arb_lam3 (vs 145 baseline) — barely; some near-misses may need more.
5. **Finetuned VLA**: probably the true bottleneck on the 17 hard tasks.

## Artifacts

- `tmp/overnight_v3/FINAL_REPORT.md` — full results matrix + analysis.
- `tmp/overnight_v3/runs/` — per-rollout actions.log, result.txt, etc.
- `tmp/overnight_v3/winning_videos/` — MP4 of the 3 v_arb_lam3 successes
  + the v0_baseline bananas_crate success.
- `tmp/overnight_v3/scripts/` — launchers (`run_one.sh` has all 11 variant presets).

