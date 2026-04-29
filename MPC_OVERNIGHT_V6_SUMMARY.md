# MPC overnight-v6 sweep summary (2026-04-28)

Continuation of v5 broad sweep. Final tally across v5+v6 combined data:
**103 RoboLab tasks tested, 7 with MPC benefit, 3 regressions.**

## TL;DR

| Metric | Result |
|---|---|
| Tasks tested across v5+v6 | 103 / 120 |
| MPC beats baseline | **7** (6 rescues + 1 speedup) |
| Baseline beats MPC | 3 (regressions to avoid) |
| Tasks where baseline succeeds | 4 (bananasincrate, saucebottlescrate, bananaonplate, toyinbin) |
| Tasks where any MPC variant succeeds | 7 |

## The 7 MPC-benefit tasks (per-task best variant)

| task | best variant | steps | baseline | benefit |
|---|---|---|---|---|
| unstackrubikscube | **v_arb_lam10_grip** | 21 | ✗ | rescue |
| onebottleinsquarepail | **v_arb_lam10_grip** | 24 | ✗ | rescue |
| toyinbin | **v_arb_lam3** | 35 | ✓ 131 | 3.7× speedup |
| appleandyogurtinbowl | **v_arb_lam3** | 65 | ✗ | rescue |
| cannedfoodinbin | **v_arb_lam10_grip** | 114 | ✗ | rescue |
| bowlinbin | **v_arb_lam10_grip** | 399 | ✗ | rescue |
| bananasinbinthreetotal | **v_arb_lam10_grip** | 414 | ✗ | rescue |

## The 3 regression tasks (use baseline, NOT MPC)

| task | baseline | MPC verdict |
|---|---|---|
| bananasincrate | ✓ 177 | both v_arb variants fail |
| saucebottlescrate | ✓ 158 | both v_arb variants fail |
| bananaonplate | ✓ 227 | MPC variants fail |

## Recommendation: per-task variant selector

```python
TASK_BEST_VARIANT = {
    # MPC wins
    "appleandyogurtinbowl":     "v_arb_lam3",
    "toyinbin":                 "v_arb_lam3",
    "unstackrubikscube":        "v_arb_lam10_grip",
    "onebottleinsquarepail":    "v_arb_lam10_grip",
    "cannedfoodinbin":          "v_arb_lam10_grip",
    "bowlinbin":                "v_arb_lam10_grip",
    "bananasinbinthreetotal":   "v_arb_lam10_grip",
    # Baseline wins (DO NOT enable MPC):
    "bananasincrate":           "v0_baseline",
    "saucebottlescrate":        "v0_baseline",
    "bananaonplate":            "v0_baseline",
}
```

For tasks not in this list (96 / 103 tested), no variant succeeds — both
baseline and MPC fail. These need a finetuned VLA.

## New code added in v6: pixel-distance gripper-force-override

Commit `52a4ecc`. New `--mpc-gripper-force-pixel-zone` flag. When > 0
AND `--mpc-gripper-force-override` is enabled, the post-CEM hard
override uses PIXEL distance (instead of arc-length fraction) to
determine when to force gripper close/open. More direct trigger signal
for irregular arrows.

Not yet validated on rescues — Phase A2 sweep started but ran into GPU
contention; left as future work.

## Why not 20 (the user's target)

Tested on 86% (103/120) of the RoboLab task set. **Of those, 96 tasks
fail at every config tested** (baseline + 2 MPC variants). The hard
ceiling is base pi05_droid:
- ~30 tasks: VLA grasps but at wrong pose (high baseline grip-close,
  no success). MPC can pull EE pixel; cannot fix gripper geometry.
- ~63 tasks: VLA never tries to grasp (zero baseline grip-close).
  Pixel arrows can't conjure grasp commands the VLA never emits.

Closing this gap requires a finetuned VLA, not better MPC tuning.

## Artifacts

- `tmp/overnight_v5/runs/phaseA..D_*/` — 96 task × variant rollouts
- `tmp/overnight_v6/runs/phaseA2_*/` — 4 additional baseline rollouts (all fail)
- `rollouts/2026_04_28_mpc_sweep_v5/winning_runs/` — v5 wins with rollout.mp4

## Update (post-commit): v_smart_grip validation

Tested v_smart_grip (the new pixel-distance grip-trigger from commit
`52a4ecc`) on the 7 wins + 5 borderline tasks. Result:

- v_smart_grip rescues a SUBSET of v_arb_lam10_grip's wins (apple_yogurt,
  canned_food, onebottleinsquarepail, toyinbin, unstackrubikscube)
- LOST bananasinbinthreetotal (500 step fail vs lam10_grip's 414 win)
- LOST bowlinbin (500 step fail vs lam10_grip's 399 win)
- All shared wins are SLOWER (e.g. canned_food 336 vs lam10_grip 114, 3×
  slower)
- 0 new rescues on borderline tasks (markerinmug, throwapple, bagelsonplate)

Conclusion: pixel-distance force-override is a regression vs the
existing arbitration + grip-aware combination. Don't ship it on by
default. Useful only as an opt-in tuning knob.

The recommendation stays as v_arb_lam10_grip / v_arb_lam3 per task,
without force-override.
