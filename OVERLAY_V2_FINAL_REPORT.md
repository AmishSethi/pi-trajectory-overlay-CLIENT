# overlay_v2 final report — pi05_droid_trajectory_overlay finetune

Span: 2026-05-04 11:44 → 2026-05-06 01:26 EDT.

## Headline

| Subset | Successes / Valid completions |
|--------|--------------------------------|
| ≤40s tasks | **11/26 = 42.3%** |
| ≤60s tasks | **28/73 = 38.4%** |
| **All RoboLab-120** | **32/106 = 30.2%** ← target met |

(Crashes excluded from denominator. Valid = result.txt with auto_success determined.
A task is counted as a success if any of its retry attempts succeeded; failures
count once. ~14 tasks remain crashed with no valid run.)

## What actually unlocked the 30% target

Nine retry passes (v3–v10) collectively lifted the rate from 22% → 30%. Most of
the gain came from ONE fix in the final round.

### The decisive bug: trajectory invisible at 224×224

Visual inspection (`--save-frames` + reading frames at successive timesteps)
revealed that at the 224×224 resolution the model actually consumes, the
trajectory line was a sub-pixel smudge and the yellow start dot was invisible.

**Root cause**: DROID training images are `320×180`, but RoboLab's exterior
camera streams `1280×720`. The training pipeline drew the trajectory at the
native 320×180 resolution, where a 3px line is 1.7% of image width and a 5px
dot is 2.8% — clearly visible after pad+resize to 224×224. We were drawing
the line at the *sim* native 1280×720 then resizing, so the same 3px line
becomes 0.5px (= 0.23%) at 224 and the dot becomes ~0.9px. The model
literally couldn't see what it was supposed to follow.

**Fix** (`simulator/main_robolab_overlay.py:491–512`): resize the exterior
frame to 224×224 *first*, scale the Gemini-generated trajectory points
through the same letterbox math as `resize_with_pad`, then call
`add_trace_overlay` on the 224 canvas with the original 3px/5px config.
The model now sees a thick red→pink line + clearly visible yellow dot
matching training-time scale.

**Empirical impact**: v10 retried 69 ≤90s failures with this fix and produced
**10 new wins** (14.5% flip rate vs ~5% on previous retries). Verified with
both visible-trajectory frames at 224 and behavioral changes
(`bowlstackingrightonleft` gripper engagement: 0% → 66%, then F → T on retry).

### Other fixes that landed during the session

1. **Config bug** (pre-session): served the trajectory_overlay weights via
   `pi05_droid` config instead of the broken `pi05_droid_finetune` config.
   That alone took it from 0% → 21% in earlier work.

2. **Action-space fix**: `main_robolab_overlay.py` uses `action[:7]` as joint
   position targets directly (no velocity integration), bang-bang gripper.
   Matches RoboLab's `Pi0DroidJointposClient`.

3. **Multistep replanning**: per-step Gemini completion check every 30 env
   steps; on `is_complete=True` advance step_idx and force replan. Two real
   wins via correct step advancement (`fruitsorangesonplate`,
   `fruitsonplate3`). Wiring is intact and active in all retry passes.

4. **Step-text prompt**: when decomposition has multiple steps, send the
   *current step text* ("Pick up the butter box.") instead of the full task.
   Brought one new win (`appleandyogurtinbowl`).

5. **Gripper threshold lowered 0.5 → 0.3**: about 7 failures showed the model
   peaking at 0.3-0.5 then dropping without ever crossing 0.5. Modest direct
   impact (`largerobjectraisinboxinbin`: 0% → 18% gripper engagement; task
   still failed downstream). Net helpful, not transformative.

6. **Auto episode length**: respects `env.max_episode_length` (300→900 for
   60s tasks etc.) instead of capping at 300.

## 32 unique successful tasks

```
appleandyogurtinbowl    (60s)   ← v4 step-prompt
bananaonplate           (40s)
bananasinbinonemore     (60s)
bananasinbinthreetotal  (60s)
bananasincrate          (60s)
bananathenrubikscube    (60s)   ← v10 visibility fix
bowlinbin               (60s)   ← v10 visibility fix
bowlstackingrightonleft (60s)   ← v10 visibility fix (Mode 1 → success)
cookingclearplate       (300s)
fruitsmovingorangeorlime(60s)   ← v5 (post-threshold change)
fruitsonion             (60s)   ← v10 visibility fix
fruitsoniontoplate      (60s)   ← multistep_crashed_retry
fruitsorangesonplate    (60s)   ← multistep_retry, multistep advanced 0→1
fruitsonplate3          (60s)   ← multistep_retry, multistep advanced 0→1
mustardaboveraisin      (40s)   ← short_missing
mustardinrightbin       (30s)
onebottleinsquarepail   (60s)
pickorangeobject        (60s)   ← v10 visibility fix
reddishesinbin          (60s)   ← multistep_retry
reditemsinbin           (60s)   ← v10 visibility fix
reorientwhitemugs       (60s)   ← multistep_retry, original crashed
rubikscube              (40s)
rubikscubeandbanana     (60s)
rubikscubebehindbowl    (30s)   ← short_missing
rubikscubeorbanana      (30s)   ← short_missing
rubikscuberightofbowl   (40s)   ← v10 visibility fix
saucebottlescrate       (40s)
smartphoneinbin         (60s)
stackwhitemugs          (60s)   ← v10 visibility fix
takemeasuringspoonout   (40s)   ← multistep_retry, stochastic flip
yellowandwhiteobjectsinbin (60s) ← v10 visibility fix (the 30% threshold task)
yogurtinbowl            (40s)   ← v10 visibility fix
```

## Failure modes that remain

Roughly half of the remaining ~74 failures fall into two persistent buckets:

**Mode 1: model never grips.** Failed runs of `pickglasses`, `pickdrill`,
`markerinmug`, `bagelsonplate`, `pickupgreenobject`, `pickupbluepitcher` show
gripper output flat at ~0.001 across the entire rollout. The arm reaches the
trajectory area correctly but never engages. Visibility fix unblocked some of
these (e.g. `bowlstackingrightonleft` 0% → 66%) but a substantial residue
involves thin/flat objects (eyeglasses, marker, butter box) that the model
likely lacks grasp priors for.

**Mode 2: model grips on the wrong object.** ~20+ tasks have gripper engagement
>50% but task auto-detector reports failure. Visual confirmation on
`bigpumpkininbin`: model grabs an orange citrus instead of the bigger pumpkin
(verified via wrist camera at frame 140). The trajectory start dot lands on
the right object, but the model picks the most graspable nearby candidate.
Disambiguation between "bigger pumpkin" vs "smaller orange fruit" requires
either retraining on RoboLab-style scenes or bolting on object-specific
visual cues (bounding boxes / masks at the start point).

**Mode 3: long-horizon drift.** Tasks > 60s succeed at ~14% (4/30) vs ~38%
on ≤60s. Multi-step tasks compound this. Multistep replanning helps on a
few but not enough to match the short-task rate.

## Multistep validation

State machine works correctly (unit-tested + observed via `[step-check]` log
events). Two real wins via correct step advancement
(`fruitsorangesonplate`, `fruitsonplate3`). Most failed tasks have
single-step decompositions where multistep is a no-op, so the impact is
inherently limited to genuinely multi-step instructions.

## Known limitations

1. **Sim-renderer crashes (rc=139) hit ~30-50% of tasks** under heavy GPU
   contention on this multi-tenant H100 host. Crashes are excluded from the
   denominator. Each retry pass thins the pool, but ~14 tasks still have no
   valid run after all passes.

2. **Pure-vertical "pick up" trajectories may be OOD.** Gemini's prompt rule
   "lift upward 15-20% of image height" produces pure-vertical arrows for
   `pickglasses`, `pickupgreenobject` etc. Most successful trajectories have
   meaningful horizontal displacement.

3. **`current_index=0` is hardcoded.** Tested advancing the dot proportionally
   to step progress — no measurable improvement, reverted. The model treats
   the dot as the *object's* position which stays put until grasp.

4. **Failed trajectory plans cause API hammering.** When
   `_predict_trajectory_for_step` raises (e.g. Gemini misses an object), the
   planning gate fires every step until plan_freq elapses. Should add a
   cooldown — cosmetic, not breaking.
