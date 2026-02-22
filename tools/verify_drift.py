"""
Drift Verification Script
==========================
Quantitatively measures drift performance under various scenarios.
Compares: raw integration vs ESKF-only vs ESKF+constraints vs ESKF+NeuralZUPT

Usage: python tools/verify_drift.py
"""
import numpy as np
import sys, math, logging
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

from airwriting_imu.fusion.imu_only_fusion import IMUOnlyFusion

logging.basicConfig(level=logging.WARNING)


def make_config(neural=False):
    return {
        "accel_noise_std": 0.5,
        "gyro_noise_std": 0.01,
        "accel_bias_std": 0.0001,
        "gyro_bias_std": 0.00001,
        "initial_covariance": 1.0,
        "zupt": {
            "enabled": True,
            "neural": neural,
            "gyro_threshold": 0.05,
            "accel_variance_threshold": 0.3,
            "noise": 0.001,
            "window_size": 20,
            "adaptive": True,
        },
        "zaru": {"enabled": True, "noise": 0.0001},
        "constraints": {
            "max_velocity": 3.0,
            "max_acceleration": 30.0,
            "velocity_decay": 0.98,
        },
    }

def make_config_no_zupt():
    cfg = make_config(False)
    cfg["zupt"]["enabled"] = False
    cfg["zaru"]["enabled"] = False
    return cfg

# ═══════════════════════════════════════════
# Scenario generators
# ═══════════════════════════════════════════

def scenario_stationary(duration_s=30, rate=100):
    """30 seconds perfectly still — drift should be ~0."""
    noise_a, noise_g = 0.05, 0.005
    for i in range(int(duration_s * rate)):
        accel = np.random.normal(0, noise_a, 3)
        gyro = np.random.normal(0, noise_g, 3)
        yield accel, gyro, np.zeros(3), i * (1_000_000 // rate)

def scenario_stationary_noisy(duration_s=30, rate=100):
    """30 seconds with realistic IMU noise levels."""
    noise_a, noise_g = 0.3, 0.02
    for i in range(int(duration_s * rate)):
        accel = np.random.normal(0, noise_a, 3)
        gyro = np.random.normal(0, noise_g, 3)
        yield accel, gyro, np.zeros(3), i * (1_000_000 // rate)

def scenario_move_and_stop(rate=100):
    """Move for 2s, stop for 3s, repeat 4 times. Should return near origin."""
    noise_a = 0.05
    t = 0
    for cycle in range(4):
        # Move phase (2s)
        for i in range(2 * rate):
            angle = cycle * math.pi / 2  # Different direction each time
            accel = np.array([math.cos(angle), math.sin(angle), 0.0]) * 1.5
            accel += np.random.normal(0, noise_a, 3)
            gyro = np.random.normal(0, 0.005, 3)
            yield accel, gyro, None, t * (1_000_000 // rate)
            t += 1
        # Stop phase (3s)
        for i in range(3 * rate):
            accel = np.random.normal(0, noise_a, 3)
            gyro = np.random.normal(0, 0.002, 3)
            yield accel, gyro, np.zeros(3), t * (1_000_000 // rate)
            t += 1

def scenario_circle_return(duration_s=10, rate=100):
    """Circular motion that returns to start. Final pos should be near origin."""
    r = 0.1
    f = 1.0 / duration_s  # One full revolution
    dt = 1.0 / rate
    prev_vel = np.zeros(3)
    for i in range(int(duration_s * rate)):
        tt = i * dt
        pos = np.array([r * math.cos(2*math.pi*f*tt),
                        r * math.sin(2*math.pi*f*tt), 0.0])
        vel = np.array([-r*2*math.pi*f*math.sin(2*math.pi*f*tt),
                        r*2*math.pi*f*math.cos(2*math.pi*f*tt), 0.0])
        accel = (vel - prev_vel) / dt
        accel += np.random.normal(0, 0.05, 3)
        gyro = np.array([0, 0, 2*math.pi*f]) + np.random.normal(0, 0.005, 3)
        yield accel, gyro, None, i * (1_000_000 // rate)
        prev_vel = vel.copy()

def scenario_long_stationary(duration_s=120, rate=100):
    """2 minutes stationary — tests long-term drift stability."""
    noise_a, noise_g = 0.1, 0.01
    for i in range(int(duration_s * rate)):
        accel = np.random.normal(0, noise_a, 3)
        gyro = np.random.normal(0, noise_g, 3)
        yield accel, gyro, np.zeros(3), i * (1_000_000 // rate)


# ═══════════════════════════════════════════
# Raw integrator (no filter)
# ═══════════════════════════════════════════

class RawIntegrator:
    """Simple double-integration with no corrections — baseline comparison."""
    def __init__(self): self.pos = np.zeros(3); self.vel = np.zeros(3)
    def update(self, accel, dt=0.01):
        self.vel += accel * dt
        self.pos += self.vel * dt


# ═══════════════════════════════════════════
# Run all scenarios
# ═══════════════════════════════════════════

def run_scenario(name, gen, config, expected_final_pos=None):
    fusion = IMUOnlyFusion(config)
    raw = RawIntegrator()
    
    positions = []
    velocities = []
    n_zupt = 0
    
    for accel, gyro, true_pos, ts in gen:
        res = fusion.update(accel, gyro, ts)
        raw.update(accel)
        positions.append(res["position"].copy())
        velocities.append(res["velocity"].copy())
    
    final_pos = fusion.pos
    final_vel = fusion.vel
    pos_drift = np.linalg.norm(final_pos)
    vel_drift = np.linalg.norm(final_vel)
    raw_drift = np.linalg.norm(raw.pos)
    max_pos = max(np.linalg.norm(p) for p in positions)
    
    return {
        "name": name,
        "pos_drift": pos_drift,
        "vel_drift": vel_drift,
        "raw_drift": raw_drift,
        "max_pos": max_pos,
        "n_zupt": fusion.n_zupt,
        "n_zaru": fusion.n_zaru,
        "n_decay": fusion.n_decay,
        "final_pos": final_pos,
        "improvement": raw_drift / pos_drift if pos_drift > 1e-10 else float("inf"),
    }


def main():
    np.random.seed(42)
    
    scenarios = [
        ("30s Stationary (low noise)", scenario_stationary),
        ("30s Stationary (high noise)", scenario_stationary_noisy),
        ("Move & Stop (4 cycles)", scenario_move_and_stop),
        ("Circle Return (10s)", scenario_circle_return),
        ("2min Long Stationary", scenario_long_stationary),
    ]
    
    print("=" * 90)
    print("  DRIFT VERIFICATION REPORT — IMU-Only AirWriting System")
    print("=" * 90)
    
    configs = [
        ("ESKF (no ZUPT)", make_config_no_zupt()),
        ("ESKF + Rule ZUPT", make_config(neural=False)),
    ]
    
    # Check if neural model is available
    try:
        import torch
        model_path = ROOT / "models" / "zupt_net.pt"
        if model_path.exists():
            configs.append(("ESKF + Neural ZUPT", make_config(neural=True)))
    except ImportError:
        pass
    
    all_results = {}
    
    for scenario_name, gen_fn in scenarios:
        print(f"\n{'─' * 90}")
        print(f"  📊 Scenario: {scenario_name}")
        print(f"{'─' * 90}")
        print(f"  {'Configuration':<25} {'Pos Drift (m)':<16} {'Vel Drift (m/s)':<18} "
              f"{'Raw Drift (m)':<16} {'Improvement':<14} {'ZUPT/ZARU/Decay'}")
        print(f"  {'-'*25} {'-'*14} {'-'*16} {'-'*14} {'-'*12} {'-'*20}")
        
        for cfg_name, cfg in configs:
            gen = gen_fn()
            r = run_scenario(scenario_name, gen, cfg)
            
            imp_str = f"{r['improvement']:.1f}×" if r['improvement'] < 1e6 else "∞"
            print(f"  {cfg_name:<25} {r['pos_drift']:<16.6f} {r['vel_drift']:<18.6f} "
                  f"{r['raw_drift']:<16.4f} {imp_str:<14} "
                  f"{r['n_zupt']}/{r['n_zaru']}/{r['n_decay']}")
            
            all_results.setdefault(scenario_name, {})[cfg_name] = r
    
    # ═══ Summary ═══
    print(f"\n{'=' * 90}")
    print("  SUMMARY — Drift Constraint Effectiveness")
    print(f"{'=' * 90}")
    
    best_cfg = "ESKF + Rule ZUPT"  # Primary production config
    all_pass = True
    
    checks = [
        ("30s Stationary (low noise)", 0.01, "< 1cm drift in 30s stationary"),
        ("30s Stationary (high noise)", 0.05, "< 5cm drift with realistic noise"),
        ("Move & Stop (4 cycles)", 1.0, "< 1m after move-and-stop cycles"),
        ("2min Long Stationary", 0.1, "< 10cm drift in 2 minutes"),
    ]
    
    for scenario_name, threshold, desc in checks:
        if scenario_name in all_results and best_cfg in all_results[scenario_name]:
            r = all_results[scenario_name][best_cfg]
            passed = r["pos_drift"] < threshold
            status = "✅ PASS" if passed else "❌ FAIL"
            
            print(f"  {status}  {desc}")
            print(f"         Actual: {r['pos_drift']:.6f}m (threshold: {threshold}m)")
            print(f"         vs Raw: {r['raw_drift']:.4f}m ({r['improvement']:.1f}× improvement)")
            
            if not passed:
                all_pass = False
    
    print(f"\n  {'✅ ALL DRIFT CHECKS PASSED' if all_pass else '❌ SOME CHECKS FAILED'}")
    print(f"{'=' * 90}")
    
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
