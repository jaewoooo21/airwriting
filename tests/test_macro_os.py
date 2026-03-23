import pytest
from airwriting_imu.core.command_bus import CommandBus
from airwriting_imu.core.policy_engine import PolicyEngine

@pytest.fixture
def mock_config():
    return {
        "macro_os": {
            "profiles": {
                "TEST": {
                    "allowed_gestures": ["A", "B"],
                    "confidence_threshold": 0.8,
                    "secure_gestures": ["B"],
                    "mappings": {
                        "A": "CMD_A",
                        "B": "CMD_B"
                    }
                }
            }
        }
    }

def test_policy_engine_validation(mock_config):
    engine = PolicyEngine(mock_config["macro_os"]["profiles"])
    engine.set_profile("TEST")
    
    # Allowed & High Confidence
    res = engine.validate_action("A", 0.9)
    assert res is not None
    assert res["label"] == "A"
    
    # Allowed & Low Confidence
    assert engine.validate_action("A", 0.7) is None
    
    # Not Allowed
    assert engine.validate_action("C", 0.9) is None
    
    # Secure Gesture
    res = engine.validate_action("B", 0.9)
    assert res is not None
    assert res["lane"] == "SECURE"

def test_command_bus_mapping(mock_config):
    engine = PolicyEngine(mock_config["macro_os"]["profiles"])
    engine.set_profile("TEST")
    udp_ports = {"WEB": 12348}
    bus = CommandBus(engine, udp_ports)
    
    assert bus.get_command_token("A") == "CMD_A"
    assert bus.get_command_token("B") == "CMD_B"
    assert bus.get_command_token("C") is None

def test_policy_engine_profile_switch(mock_config):
    engine = PolicyEngine(mock_config["macro_os"]["profiles"])
    engine.set_profile("TEST")
    assert engine.active_profile == "TEST"
    
    engine.set_profile("NON_EXISTENT")
    assert engine.active_profile == "TEST" # Should remain on current

if __name__ == "__main__":
    # Simple manual runner
    config = {
        "macro_os": {
            "profiles": {
                "TEST": {
                    "allowed_gestures": ["A", "B"],
                    "confidence_threshold": 0.8,
                    "secure_gestures": ["B"],
                    "mappings": {
                        "A": "CMD_A",
                        "B": {"keyword": "CMD_B", "intent": "TEST_INTENT"}
                    }
                }
            }
        }
    }
    print("Running manual tests...")
    test_policy_engine_validation(config)
    test_command_bus_mapping(config)
    test_policy_engine_profile_switch(config)
    print("Tests passed!")
