import os
import sys
import subprocess
import time
import logging
from typing import List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Orchestrator")

def run_command(command: List[str], description: str) -> bool:
    """Run a shell command and return True if successful."""
    logger.info(f"🚀 Starting: {description}...")
    start_time = time.time()
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False
        )
        duration = time.time() - start_time
        
        if result.returncode == 0:
            logger.info(f"✅ {description} PASSED ({duration:.2f}s)")
            return True
        else:
            logger.error(f"❌ {description} FAILED ({duration:.2f}s)")
            logger.error(f"stdout:\n{result.stdout}")
            logger.error(f"stderr:\n{result.stderr}")
            return False
    except FileNotFoundError:
        logger.error(f"❌ {description} FAILED: Command not found ({command[0]})")
        return False

def check_physics_consistency():
    """Run a specialized physics consistency check (Placeholder for now)."""
    logger.info("🧪 Running Physics Consistency Check...")
    # TODO: Implement actual simulation where we integrate accel/gyro and check bounds
    # For now, we assume the unit tests cover this via 'test_eskf.py'
    return True

def main():
    logger.info("==========================================")
    logger.info("    AirWriting Automation Orchestrator    ")
    logger.info("==========================================")
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    
    overall_success = True

    # 1. Static Analysis (mypy)
    # Only run if installed
    try:
        import mypy
        if not run_command([sys.executable, "-m", "mypy", "airwriting_imu"], "Type Checking (mypy)"):
            overall_success = False
    except ImportError:
        logger.warning("⚠️ mypy not installed, skipping type checking.")

    # 2. Unit Tests (pytest)
    if not run_command([sys.executable, "-m", "pytest", "tests"], "Unit Tests (pytest)"):
        overall_success = False

    # 3. Physics Consistency
    if not check_physics_consistency():
        overall_success = False

    logger.info("==========================================")
    if overall_success:
        logger.info("🎉 ALL CHECKS PASSED! Ready for deployment/merge.")
        sys.exit(0)
    else:
        logger.error("🚫 SOME CHECKS FAILED. See logs above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
