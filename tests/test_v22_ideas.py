"""
Tests for v2.2 creative ideas: WritingPlaneDetector + DriftObserver
"""
import pytest
import numpy as np
from airwriting_imu.constraints.writing_plane import WritingPlaneDetector
from airwriting_imu.constraints.drift_observer import DriftObserver


# ════════════════════════════════════════════════
# Writing Plane Detector
# ════════════════════════════════════════════════
class TestWritingPlaneDetector:

    @pytest.fixture
    def plane(self):
        return WritingPlaneDetector(buffer_size=40, min_spread=0.001)

    def test_not_ready_initially(self, plane):
        assert not plane.is_ready()

    def test_detects_xy_plane(self, plane):
        """If all motion is in XY, normal should be ~[0,0,1]."""
        rng = np.random.default_rng(42)
        for i in range(40):
            pos = np.array([
                rng.uniform(-0.1, 0.1),
                rng.uniform(-0.1, 0.1),
                0.0  # Z always 0 → XY plane
            ])
            plane.observe(pos)

        assert plane.is_ready(), "Should detect XY plane"
        normal = plane.get_normal()
        # Normal should be close to [0,0,±1]
        assert abs(abs(normal[2]) - 1.0) < 0.1, \
            f"Normal should point in Z: {normal}"

    def test_detects_xz_plane(self, plane):
        """If motion is in XZ, normal should be ~[0,1,0]."""
        rng = np.random.default_rng(99)
        for i in range(40):
            pos = np.array([
                rng.uniform(-0.1, 0.1),
                0.0,  # Y always 0 → XZ plane
                rng.uniform(-0.1, 0.1),
            ])
            plane.observe(pos)

        assert plane.is_ready()
        normal = plane.get_normal()
        assert abs(abs(normal[1]) - 1.0) < 0.1, \
            f"Normal should point in Y: {normal}"

    def test_constrain_removes_normal_component(self, plane):
        """Constraining should suppress the normal-axis acceleration."""
        # Manually set plane to XY (normal = Z)
        rng = np.random.default_rng(42)
        for i in range(40):
            plane.observe(np.array([
                rng.uniform(-0.1, 0.1),
                rng.uniform(-0.1, 0.1),
                0.0
            ]))

        accel = np.array([1.0, 2.0, 3.0])
        accel_orig = accel.copy()
        plane.constrain(accel)

        # XY components should be mostly unchanged
        assert abs(accel[0] - accel_orig[0]) < 0.5
        assert abs(accel[1] - accel_orig[1]) < 0.5
        # Z component should be suppressed (by ~80%)
        assert abs(accel[2]) < abs(accel_orig[2]) * 0.5

    def test_no_constrain_when_not_ready(self, plane):
        """Should not modify anything when plane isn't detected."""
        accel = np.array([1.0, 2.0, 3.0])
        accel_orig = accel.copy()
        plane.constrain(accel)
        assert np.allclose(accel, accel_orig)

    def test_reset(self, plane):
        rng = np.random.default_rng(42)
        for i in range(40):
            plane.observe(np.array([
                rng.uniform(-0.1, 0.1),
                rng.uniform(-0.1, 0.1),
                0.0
            ]))
        assert plane.is_ready()
        plane.reset()
        assert not plane.is_ready()


# ════════════════════════════════════════════════
# Drift Observer
# ════════════════════════════════════════════════
class TestDriftObserver:

    @pytest.fixture
    def observer(self):
        return DriftObserver(window=10, drift_threshold=0.001)

    def test_no_drift_when_stationary(self, observer):
        """Same position during ZUPT should not trigger drift."""
        pos = np.array([0.1, 0.2, 0.3])
        for _ in range(20):
            result = observer.observe(pos, zupt_active=True)
        assert not result, "Constant position should not trigger drift"

    def test_drift_detected_when_position_varies(self, observer):
        """Varying position during ZUPT should trigger drift detection."""
        rng = np.random.default_rng(42)
        detected = False
        for i in range(20):
            pos = np.array([0.1, 0.2, 0.3]) + rng.uniform(-0.1, 0.1, 3)
            detected = observer.observe(pos, zupt_active=True)
        assert detected, "Varying position during ZUPT should trigger drift"

    def test_no_detection_without_zupt(self, observer):
        """Should never trigger when ZUPT is inactive."""
        rng = np.random.default_rng(42)
        for i in range(20):
            pos = rng.uniform(-1, 1, 3)
            result = observer.observe(pos, zupt_active=False)
            assert not result

    def test_apply_correction(self, observer):
        """Correction should scale bias and zero velocity."""
        ba = np.array([0.1, 0.2, 0.3])
        vel = np.array([1.0, 2.0, 3.0])
        observer.apply_correction(ba, vel)
        assert np.allclose(ba, [0.05, 0.1, 0.15])  # 0.5×
        assert np.allclose(vel, [0.0, 0.0, 0.0])

    def test_counter_increments(self, observer):
        """Drift detection counter should increment."""
        rng = np.random.default_rng(42)
        assert observer.n_drift_detected == 0
        for i in range(20):
            pos = np.array([0.1, 0.2, 0.3]) + rng.uniform(-0.1, 0.1, 3)
            observer.observe(pos, zupt_active=True)
        assert observer.n_drift_detected >= 1

    def test_reset(self, observer):
        rng = np.random.default_rng(42)
        for i in range(20):
            pos = rng.uniform(-0.1, 0.1, 3)
            observer.observe(pos, zupt_active=True)
        observer.reset()
        assert observer.n_drift_detected == 0
        assert observer.n_corrections == 0
