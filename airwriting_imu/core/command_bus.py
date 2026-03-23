import logging
import json
import socket
import time
from typing import Dict, Any, Optional

log = logging.getLogger(__name__)

class CommandBus:
    """
    Centralized command dispatcher for the Macro OS.
    Decouples ML labels from final system actions and intents.
    """
    
    def __init__(self, policy_engine: Any, udp_ports: Dict[str, int]):
        self.policy_engine = policy_engine
        self.udp_ports = udp_ports
        self._tx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        log.info(f"🚌 CommandBus initialized (Context-Aware)")

    def get_command_token(self, label: str) -> Optional[str]:
        """Maps an ML label to a command token based on the active profile."""
        profile_name = self.policy_engine.active_profile
        profile_cfg = self.policy_engine.profiles.get(profile_name, {})
        mappings = profile_cfg.get("mappings", {})
        
        action = mappings.get(label.upper())
        if isinstance(action, dict):
            return action.get("keyword")
        elif isinstance(action, str):
            return action
        return None

    def dispatch(self, label: str, confidence: float, context: Optional[str] = None) -> bool:
        """
        Dispatches a command based on label and active profile.
        """
        profile_name = context or self.policy_engine.active_profile
        profile_cfg = self.policy_engine.profiles.get(profile_name, {})
        mappings = profile_cfg.get("mappings", {})
        
        action = mappings.get(label.upper())
        if not action:
            log.warning(f"⚠️ No action mapped for label: {label} in profile {profile_name}")
            return False

        if isinstance(action, str):
            command_token = action
            intent = ""
        else:
            command_token = action.get("keyword", "UNKNOWN")
            intent = action.get("intent", "")
        
        payload = {
            "type": "command",
            "label": label,
            "token": command_token,
            "intent": intent,
            "confidence": confidence,
            "context": profile_name,
            "timestamp": time.time()
        }
        
        success = True
        try:
            raw = json.dumps(payload).encode('utf-8')
            for target_name, port in self.udp_ports.items():
                try:
                    self._tx_sock.sendto(raw, ("127.0.0.1", port))
                except Exception as e:
                    log.error(f"❌ Failed to send to {target_name}: {e}")
                    success = False
            log.info(f"🚀 Command Executed: {command_token} ('{label}', Profile: {profile_name})")
        except Exception as e:
            log.error(f"❌ Dispatch failed: {e}")
            success = False
            
        return success

    def close(self):
        if hasattr(self, "_tx_sock"):
            self._tx_sock.close()
