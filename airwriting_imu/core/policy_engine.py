import logging
import time
from typing import Dict, List, Any, Optional

log = logging.getLogger(__name__)

class PolicyEngine:
    """
    Core logic for 'Macro OS' decision making.
    Handles profiles, active vocabulary, and execution risk.
    """
    
    def __init__(self, profiles: Dict[str, Any], initial_profile: str = "GLOBAL"):
        self.profiles = profiles
        self.active_profile = initial_profile
        self._last_profile_change = time.time()
        
        log.info(f"🛡️ PolicyEngine initialized with {len(profiles)} profiles. Active: {self.active_profile}")

    def set_profile(self, profile_name: str) -> bool:
        if profile_name in self.profiles or profile_name == "GLOBAL":
            self.active_profile = profile_name
            self._last_profile_change = time.time()
            log.info(f"🔄 Profile switched to: {profile_name}")
            return True
        log.warning(f"⚠️ Attempted to switch to unknown profile: {profile_name}")
        return False

    def validate_action(self, label: str, confidence: float) -> Optional[Dict[str, Any]]:
        """
        Validates if an ML result should be executed given the current profile.
        Implements 'Active Vocabulary' and 'Risk Policy'.
        """
        # 1. Get profile config
        profile_cfg = self.profiles.get(self.active_profile, {})
        active_vocab = profile_cfg.get("allowed_gestures", [])
        
        # 2. Check Active Vocabulary (if defined)
        if active_vocab and label.upper() not in active_vocab:
            log.debug(f"🛑 GESTURE REJECTED: '{label}' is not in active vocabulary for {self.active_profile}")
            return None
            
        # 3. Handle 'REJECT' class
        if label.upper() == "REJECT" or label.upper() == "UNKNOWN":
            return None

        # 4. Check Confidence Thresholds
        threshold = profile_cfg.get("confidence_threshold", 0.6)
        if confidence < threshold:
            log.info(f"⚠️ CONFIDENCE LOW: '{label}' ({confidence:.2f} < {threshold})")
            return None

        # 5. Determine Execution Lane (Reflex vs Secure)
        # For now, we simulate this based on a 'secure_gestures' list
        secure_gestures = profile_cfg.get("secure_gestures", ["CALL", "MESSAGE"])
        
        # We also look at the intent/keyword if possible
        lane = "REFLEX"
        if label.upper() in secure_gestures:
            lane = "SECURE"
            
        log.debug(f"✅ ACTION VALIDATED: '{label}' (Lane: {lane}, Profile: {self.active_profile})")
        
        return {
            "label": label,
            "lane": lane,
            "confidence": confidence,
            "profile": self.active_profile
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "active_profile": self.active_profile,
            "last_change": self._last_profile_change,
            "vocab_size": len(self.profiles.get(self.active_profile, {}).get("allowed_gestures", []))
        }
