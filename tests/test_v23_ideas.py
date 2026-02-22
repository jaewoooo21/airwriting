"""
Tests for v2.3 creative ideas:
- Adaptive Measurement Noise
- Loop Closure Detection
- Velocity Regression Network
- Multi-Rate Scheduler
- Trajectory Image Encoding
"""
import numpy as np
import pytest

# ═══════════════════════════════════════════════════
# Test 1: Adaptive Measurement Noise (AMN)
# ═══════════════════════════════════════════════════
class TestAdaptiveMeasurementNoise:
    """Verify AMN scales ZUPT/ZARU R matrices by confidence."""

    def setup_method(self):
        from airwriting_imu.fusion.imu_only_fusion import IMUOnlyFusion
        self.fusion = IMUOnlyFusion({"zupt": {"enabled": True}})

    def test_amn_enabled_by_default(self):
        assert self.fusion._amn_enabled is True

    def test_high_confidence_zupt_stronger_correction(self):
        """Higher ZUPT confidence should result in more velocity correction."""
        self.fusion.vel[:] = [0.1, 0.1, 0.1]
        vel_before = self.fusion.vel.copy()

        # Simulate high confidence
        self.fusion.zupt_confidence = 0.9
        self.fusion._apply_zupt()
        correction_high = np.linalg.norm(vel_before - self.fusion.vel)

        # Reset
        self.fusion.vel[:] = [0.1, 0.1, 0.1]

        # Simulate low confidence
        self.fusion.zupt_confidence = 0.1
        self.fusion._apply_zupt()
        correction_low = np.linalg.norm(vel_before - self.fusion.vel)

        # High confidence should correct more
        assert correction_high > correction_low

    def test_zupt_still_works_without_amn(self):
        """ZUPT should still function if AMN is disabled."""
        self.fusion._amn_enabled = False
        self.fusion.vel[:] = [0.1, 0.1, 0.1]
        self.fusion.zupt_confidence = 0.5
        self.fusion._apply_zupt()
        assert np.linalg.norm(self.fusion.vel) < 0.1


# ═══════════════════════════════════════════════════
# Test 2: Loop Closure Detection
# ═══════════════════════════════════════════════════
class TestLoopClosureDetector:
    """Verify loop closure detection on trajectory patterns."""

    def setup_method(self):
        from airwriting_imu.constraints.loop_closure import LoopClosureDetector
        self.lc = LoopClosureDetector(
            min_loop_length=10,
            proximity_m=0.01,
            cooldown=5,
        )

    def test_no_detection_without_stroke(self):
        result = self.lc.detect()
        assert result is None

    def test_circular_loop_detected(self):
        """Drawing a circle should trigger loop closure."""
        self.lc.start_stroke()

        # Draw a circle
        for i in range(30):
            theta = 2 * np.pi * i / 29
            pos = np.array([
                0.05 * np.cos(theta),
                0.05 * np.sin(theta),
                0.0
            ])
            self.lc.track(pos)

        # Last point should be near first
        result = self.lc.detect()
        assert result is not None
        assert result["loop_length"] >= 10
        assert self.lc.n_closures == 1

    def test_line_no_closure(self):
        """A straight line should not trigger closure."""
        self.lc.start_stroke()
        for i in range(30):
            pos = np.array([0.001 * i, 0, 0])
            self.lc.track(pos)

        result = self.lc.detect()
        assert result is None

    def test_cooldown_prevents_rapid_detection(self):
        """After a detection, cooldown should prevent immediate re-detection."""
        self.lc._cooldown_max = 100  # long cooldown
        self.lc.start_stroke()

        # Draw a circle
        for i in range(30):
            theta = 2 * np.pi * i / 29
            pos = np.array([0.05 * np.cos(theta), 0.05 * np.sin(theta), 0])
            self.lc.track(pos)

        first = self.lc.detect()
        assert first is not None

        # Immediately try again
        self.lc.track(np.array([0.05, 0, 0]))
        second = self.lc.detect()
        assert second is None  # cooldown active

    def test_end_stroke_clears(self):
        self.lc.start_stroke()
        self.lc.track(np.array([0, 0, 0]))
        self.lc.end_stroke()
        assert self.lc.detect() is None

    def test_reset(self):
        self.lc.start_stroke()
        for i in range(30):
            self.lc.track(np.array([0.001 * i, 0, 0]))
        self.lc.reset()
        assert self.lc.n_closures == 0


# ═══════════════════════════════════════════════════
# Test 3: Velocity Regression Network
# ═══════════════════════════════════════════════════
class TestVelocityRegressionNet:
    """Test VRN fallback estimator and data generation."""

    def test_fallback_estimator(self):
        from airwriting_imu.ml.velocity_regression import VRNFallback
        fb = VRNFallback(window=5)
        vel = fb.predict(np.array([1.0, 0.0, 0.0]), dt=0.01)
        assert vel.shape == (3,)
        assert vel[0] > 0  # positive x acceleration → positive x velocity

    def test_fallback_stationary(self):
        from airwriting_imu.ml.velocity_regression import VRNFallback
        fb = VRNFallback(window=5)
        for _ in range(10):
            vel = fb.predict(np.zeros(3), dt=0.01)
        assert np.linalg.norm(vel) < 0.01

    def test_data_generator(self):
        from airwriting_imu.ml.velocity_regression import VRNDataGenerator
        data = VRNDataGenerator.generate_synthetic(n_samples=100, window=10)
        assert data["X"].shape == (100, 10, 18)
        assert data["Y"].shape == (100, 3)

    def test_vrn_wrapper_init(self):
        from airwriting_imu.ml.velocity_regression import VelocityRegressionNet
        vrn = VelocityRegressionNet()
        vrn.feed(np.zeros(18))
        # Should not crash even with insufficient data
        result = vrn.predict(np.zeros(3), dt=0.01)
        # May return None if buffer not full enough

    def test_eskf_velocity_measurement(self):
        """Verify ESKF velocity pseudo-measurement works."""
        from airwriting_imu.fusion.imu_only_fusion import IMUOnlyFusion
        fusion = IMUOnlyFusion({"zupt": {"enabled": True}})
        fusion.vel[:] = [0.1, 0.2, 0.3]
        vel_measured = np.array([0.05, 0.15, 0.25])
        fusion.update_velocity_measurement(vel_measured, vel_noise_std=0.05)
        # Velocity should move toward measurement
        assert np.linalg.norm(fusion.vel - vel_measured) < \
               np.linalg.norm(np.array([0.1, 0.2, 0.3]) - vel_measured)


# ═══════════════════════════════════════════════════
# Test 4: Multi-Rate Scheduler
# ═══════════════════════════════════════════════════
class TestMultiRateScheduler:
    """Multi-rate scheduling logic."""

    def setup_method(self):
        from airwriting_imu.fusion.multi_rate import MultiRateScheduler
        self.sched = MultiRateScheduler(
            base_rate_hz=100, fast_multiplier=2, slow_divisor=10
        )

    def test_fast_runs_every_tick(self):
        for _ in range(10):
            assert self.sched.should_fast() is True
            self.sched.tick()

    def test_normal_runs_every_other(self):
        results = []
        for _ in range(10):
            results.append(self.sched.should_normal())
            self.sched.tick()
        # Should alternate: True, False, True, False, ...
        assert results[0] is True
        assert results[1] is False
        assert results[2] is True

    def test_slow_runs_infrequently(self):
        results = []
        for _ in range(30):
            results.append(self.sched.should_slow())
            self.sched.tick()
        # Should be True only every 20 ticks (2*10)
        assert results[0] is True
        assert sum(results) == 2  # tick 0 and tick 20

    def test_gyro_preintegration(self):
        gyro1 = np.array([0.1, 0.2, 0.3])
        gyro2 = np.array([0.2, 0.3, 0.4])
        self.sched.accumulate_gyro(gyro1, 0.005)
        self.sched.accumulate_gyro(gyro2, 0.005)
        mean_g, total_dt = self.sched.get_preintegrated_gyro()
        assert total_dt == pytest.approx(0.01)
        # Weighted mean (equal dt → simple average)
        expected = (gyro1 + gyro2) / 2
        np.testing.assert_allclose(mean_g, expected)

    def test_rates(self):
        rates = self.sched.get_rates()
        assert rates["fast_hz"] == 200
        assert rates["normal_hz"] == 100
        assert rates["slow_hz"] == 10


# ═══════════════════════════════════════════════════
# Test 5: Trajectory Image Encoding
# ═══════════════════════════════════════════════════
class TestTrajectoryEncoder:
    """Trajectory-to-image rendering."""

    def setup_method(self):
        from airwriting_imu.ml.trajectory_recognition import TrajectoryEncoder
        self.enc = TrajectoryEncoder(img_size=64, line_width=1)

    def test_empty_render(self):
        img = self.enc.render()
        assert img.shape == (64, 64)
        assert img.sum() == 0

    def test_single_point_no_render(self):
        self.enc.add_point(np.array([0.0, 0.0]))
        img = self.enc.render()
        assert img.sum() == 0  # needs 2+ points

    def test_line_renders_pixels(self):
        self.enc.add_point(np.array([0.0, 0.0]))
        self.enc.add_point(np.array([0.1, 0.0]))
        img = self.enc.render()
        assert img.sum() > 0  # some pixels should be lit

    def test_circle_renders(self):
        for i in range(50):
            theta = 2 * np.pi * i / 49
            self.enc.add_point(np.array([
                np.cos(theta) * 0.05,
                np.sin(theta) * 0.05,
            ]))
        img = self.enc.render()
        assert img.sum() > 0
        # Circle should have pixels across the image
        assert np.any(img[32, :] > 0)  # middle row

    def test_3d_projection(self):
        p3d = np.array([1.0, 2.0, 3.0])
        normal = np.array([0.0, 0.0, 1.0])
        p2d = self.enc.project_3d(p3d, normal)
        assert p2d.shape == (2,)
        # Z-axis normal → projection is in XY plane
        # The result should capture the X and Y components

    def test_clear(self):
        self.enc.add_point(np.array([0, 0]))
        self.enc.add_point(np.array([1, 1]))
        self.enc.clear()
        assert self.enc.get_point_count() == 0
        assert self.enc.render().sum() == 0

    def test_recognizer_init(self):
        from airwriting_imu.ml.trajectory_recognition import TrajectoryRecognizer
        rec = TrajectoryRecognizer(img_size=64)
        assert rec.encoder is not None
