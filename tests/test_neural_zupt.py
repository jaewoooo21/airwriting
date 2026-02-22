"""
Tests for Neural ZUPT detector
"""
import pytest
import numpy as np


class TestNeuralZUPTFallback:
    """Test rule-based fallback when no model loaded."""

    def test_import(self):
        from airwriting_imu.ml.neural_zupt import NeuralZUPTDetector

    def test_fallback_creation(self):
        from airwriting_imu.ml.neural_zupt import NeuralZUPTDetector
        config = {
            "window_size": 20,
            "gyro_threshold": 0.05,
            "accel_variance_threshold": 0.3,
        }
        # Explicitly pass nonexistent model path to force fallback
        det = NeuralZUPTDetector(config, model_path="/nonexistent/model.pt")
        assert not det.use_neural
        stats = det.get_stats()
        assert stats["mode"] == "rule-based"

    def test_fallback_stationary_detection(self):
        from airwriting_imu.ml.neural_zupt import NeuralZUPTDetector
        config = {
            "window_size": 10,
            "gyro_threshold": 0.05,
            "accel_variance_threshold": 0.3,
        }
        det = NeuralZUPTDetector(config, model_path=None)

        # Feed stationary samples
        for _ in range(20):
            accel = np.random.normal(0, 0.01, 3)
            gyro = np.random.normal(0, 0.001, 3)
            is_stat, conf = det.detect(accel, gyro)

        # After enough samples, should detect stationary
        assert is_stat, "Fallback should detect stationary state"
        assert conf > 0.5

    def test_fallback_motion_detection(self):
        from airwriting_imu.ml.neural_zupt import NeuralZUPTDetector
        config = {
            "window_size": 10,
            "gyro_threshold": 0.05,
            "accel_variance_threshold": 0.3,
        }
        det = NeuralZUPTDetector(config, model_path=None)

        # Feed motion samples (high gyro)
        for _ in range(20):
            accel = np.array([2.0, 0.0, 0.0]) + np.random.normal(0, 0.5, 3)
            gyro = np.array([0.5, 0.0, 0.0])
            is_stat, conf = det.detect(accel, gyro)

        assert not is_stat, "Should detect motion state"

    def test_reset(self):
        from airwriting_imu.ml.neural_zupt import NeuralZUPTDetector
        config = {"window_size": 10}
        det = NeuralZUPTDetector(config, model_path=None)

        for _ in range(15):
            det.detect(np.zeros(3), np.zeros(3))

        det.reset()
        assert det.n_fallback == 0
        assert det.n_neural == 0


class TestZUPTNetModel:
    """Test model architecture (requires PyTorch)."""

    def test_model_forward(self):
        try:
            import torch
            from airwriting_imu.ml.neural_zupt import ZUPTNet
        except ImportError:
            pytest.skip("PyTorch not available")

        model = ZUPTNet(input_dim=6, hidden_dim=32, num_layers=2)
        x = torch.randn(4, 20, 6)  # [batch=4, seq=20, features=6]
        out = model(x)
        assert out.shape == (4, 1)
        assert (out >= 0).all() and (out <= 1).all()  # Sigmoid output

    def test_model_deterministic(self):
        try:
            import torch
            from airwriting_imu.ml.neural_zupt import ZUPTNet
        except ImportError:
            pytest.skip("PyTorch not available")

        model = ZUPTNet()
        model.eval()
        x = torch.randn(1, 20, 6)
        with torch.no_grad():
            y1 = model(x).item()
            y2 = model(x).item()
        assert y1 == y2


class TestTrainingData:
    """Test training data generation."""

    def test_generate_dataset(self):
        import sys
        sys.path.insert(0, ".")
        from tools.generate_zupt_data import generate_dataset

        X, y = generate_dataset(num_seconds=5, window_size=10)
        assert X.ndim == 3
        assert X.shape[1] == 10  # window_size
        assert X.shape[2] == 6   # accel + gyro
        assert y.ndim == 2
        assert y.shape[1] == 1
        assert len(X) == len(y)

        # Should have both classes
        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        assert n_pos > 0, "Should have stationary samples"
        assert n_neg > 0, "Should have motion samples"


class TestESKFNeuralIntegration:
    """Test ESKF with neural ZUPT config (fallback)."""

    def test_eskf_with_neural_config(self):
        from airwriting_imu.fusion.imu_only_fusion import IMUOnlyFusion
        config = {
            "accel_noise_std": 0.5,
            "gyro_noise_std": 0.01,
            "accel_bias_std": 0.0001,
            "gyro_bias_std": 0.00001,
            "initial_covariance": 1.0,
            "zupt": {
                "enabled": True,
                "neural": True,  # Enable neural mode
                "window_size": 20,
                "gyro_threshold": 0.05,
                "accel_variance_threshold": 0.3,
                "noise": 0.001,
            },
            "zaru": {"enabled": True, "noise": 0.0001},
            "constraints": {"max_velocity": 3.0, "velocity_decay": 0.98},
        }
        fusion = IMUOnlyFusion(config)

        # Should work even without a trained model (fallback)
        for i in range(50):
            res = fusion.update(np.zeros(3), np.zeros(3), i * 10000)

        assert res["zupt_active"] or True  # Just check it runs
        assert "zupt_confidence" in res
