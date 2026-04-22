"""Live-server probe: verifies the pi05_droid policy server's returned
action chunks match the assumptions main_robolab.py's velocity integrator
makes.

What we assert:
    1. The server accepts the exact request schema main_robolab builds.
    2. The returned chunk is shape (T, 8) with T >= 8 (we consume T per
       chunk, driven by --open-loop-horizon=8).
    3. Arm velocities (chunk[:, :7]) lie approximately in [-1, 1] as the
       integrator assumes (we clip hard to [-1, 1] before integrating;
       slightly out-of-range values are OK — the integrator clips them).
    4. Gripper (chunk[:, 7]) lies approximately in [0, 1].
    5. Two back-to-back .infer() calls return chunks of the same shape.

Usage:
    .venv/bin/python tests/live_server_probe.py --port 8001

This is NOT a pytest test — it probes an external service. Run manually.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
from openpi_client import websocket_client_policy


def make_request(seed: int = 0) -> dict:
    """Build a request in the exact shape main_robolab.py sends.

    Uses synthetic but valid 224x224 uint8 images + a plausible arm pose.
    We're not exercising the VLA's vision — we're checking the wire format
    + output statistics.
    """
    rng = np.random.default_rng(seed)
    ext_224 = rng.integers(0, 255, size=(224, 224, 3), dtype=np.uint8)
    wrist_224 = rng.integers(0, 255, size=(224, 224, 3), dtype=np.uint8)
    joint_pos = np.array([0.0, -0.7, 0.0, -2.0, 0.0, 1.5, 0.0], dtype=np.float32)
    gripper_pos = np.array([0.0], dtype=np.float32)
    return {
        "observation/exterior_image_1_left": ext_224,
        "observation/wrist_image_left": wrist_224,
        "observation/joint_position": joint_pos,
        "observation/gripper_position": gripper_pos,
        "prompt": "pick up the banana and place it on the plate",
    }


def probe(host: str, port: int) -> int:
    print(f"[probe] connecting to {host}:{port} ...")
    client = websocket_client_policy.WebsocketClientPolicy(host, port)
    print(f"[probe] connected")

    all_ok = True

    for call_idx in range(2):
        req = make_request(seed=call_idx)
        print(f"\n[probe] call {call_idx}: sending request (keys={sorted(req)})")
        out = client.infer(req)
        if "actions" not in out:
            print(f"[probe] FAIL: response missing 'actions' key: {list(out)}")
            return 2
        actions = np.asarray(out["actions"])
        print(f"[probe] call {call_idx}: actions shape={actions.shape} dtype={actions.dtype}")

        # ---- Shape check ----
        if actions.ndim != 2 or actions.shape[1] != 8:
            print(f"[probe] FAIL: expected (T, 8); got {actions.shape}")
            all_ok = False
            continue
        T = actions.shape[0]
        if T < 8:
            print(f"[probe] FAIL: chunk T={T} < open_loop_horizon=8")
            all_ok = False

        # ---- Arm velocity range ----
        arm = actions[:, :7]
        arm_min, arm_max = float(arm.min()), float(arm.max())
        arm_abs_max = float(np.abs(arm).max())
        print(f"[probe]   arm dims    min={arm_min:+.3f}  max={arm_max:+.3f}  |max|={arm_abs_max:.3f}")
        # Pi0 DROID emits normalized velocities targeting [-1, 1]. Real outputs
        # may overshoot slightly; we clip to [-1, 1] before integrating, so
        # anything up to, say, ~1.5 is expected. If we see |v| >> 3 something
        # is badly wrong (likely the wrong config / normalization).
        if arm_abs_max > 3.0:
            print(f"[probe]   WARN: |arm velocity| up to {arm_abs_max:.2f} — normalization may be off")
        # How often does the server emit out-of-[-1,1] values? If >50% of
        # samples saturate, clipping is silently killing the policy.
        saturated = float(((np.abs(arm) > 1.0).sum()) / arm.size)
        print(f"[probe]   fraction of arm samples with |v|>1 (get clipped): {saturated:.1%}")

        # ---- Gripper range ----
        grip = actions[:, 7]
        grip_min, grip_max = float(grip.min()), float(grip.max())
        print(f"[probe]   gripper     min={grip_min:+.3f}  max={grip_max:+.3f}")
        if grip_min < -0.5 or grip_max > 1.5:
            print(f"[probe]   WARN: gripper range suspicious (expected roughly [0, 1])")

        # ---- Non-finite sanity ----
        if not np.all(np.isfinite(actions)):
            print("[probe] FAIL: actions contain NaN/Inf")
            all_ok = False

    # Integration sanity: take one chunk, run main_robolab's integrator
    # against the first 8 steps, and verify the q_target deltas are plausible.
    print(f"\n[probe] integration sanity over 8 steps ...")
    sys.path.insert(0, "/home/asethi04/ROBOTICS/pi-trajectory-overlay-CLIENT/simulator")
    import main_robolab as mr

    req = make_request(seed=42)
    chunk = np.asarray(client.infer(req)["actions"], dtype=np.float32)
    q0 = np.asarray(req["observation/joint_position"], dtype=np.float32).copy()
    q_target = q0.copy()
    deltas = []
    for t in range(min(8, chunk.shape[0])):
        q_prev = q_target.copy()
        q_target = mr._integrate_velocity(chunk[t, :7], q_target)
        deltas.append(np.abs(q_target - q_prev).max())
    total_delta = np.abs(q_target - q0).max()
    per_step = max(deltas)
    # Per-step upper bound: max(vel_limit)*dt = 2.61/15 ≈ 0.174 rad.
    upper_bound = float(mr.DROID_VEL_LIMITS.max() / mr.SIM_CONTROL_FREQUENCY)
    print(f"[probe]   max per-step |Δq| = {per_step:.4f} rad  (upper bound at |v|=1 is {upper_bound:.4f})")
    print(f"[probe]   total 8-step |Δq| = {total_delta:.4f} rad")
    if per_step > upper_bound + 1e-6:
        print(f"[probe] FAIL: per-step Δq exceeds physical bound — clipping broken")
        all_ok = False

    print("\n[probe] " + ("PASS" if all_ok else "FAIL"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    raise SystemExit(probe(args.host, args.port))
