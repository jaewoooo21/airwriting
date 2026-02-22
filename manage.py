import os
import sys
import subprocess
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("Manage")

TOOLS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools")
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def run_script(script_name, args=[]):
    script_path = os.path.join(TOOLS_DIR, script_name)
    if not os.path.exists(script_path):
        logger.error(f"❌ Script not found: {script_path}")
        return
    
    cmd = [sys.executable, script_path] + args
    logger.info(f"🚀 Running {script_name}...")
    try:
        subprocess.run(cmd, check=False)
    except KeyboardInterrupt:
        logger.info("\n⏹️  Interrupted by user.")

def run_main(args=[]):
    main_path = os.path.join(ROOT_DIR, "main.py")
    cmd = [sys.executable, main_path] + args
    logger.info(f"🚀 Running main.py {' '.join(args)}...")
    try:
        subprocess.run(cmd, check=False)
    except KeyboardInterrupt:
        logger.info("\n⏹️  Interrupted by user.")

def show_menu():
    clear_screen()
    print("==========================================")
    print("    AirWriting IMU Project Manager v2.3   ")
    print("==========================================")
    print("── Testing & Diagnostics ──")
    print("1. 🛡️  Run Orchestrator (Tests & Checks)")
    print("2. 📊 Verify Drift (Visual/Stats)")
    print("")
    print("── Simulation ──")
    print("3. 🔌 Start Mock ESP32 IMU Server")
    print("")
    print("── Neural Network ──")
    print("4. 🏋️  Train Neural ZUPT Model")
    print("5. 🎲 Generate Training Data (ZUPT)")
    print("")
    print("── Hardware (Real ESP32) ──")
    print("6. 🏥 Health Check (Test Connection)")
    print("7. 📐 Calibrate & Save Bias")
    print("8. 🚀 Start Real Hardware (Auto-Calibrate)")
    print("9. ⚡ Start Real Hardware (Use Saved Bias)")
    print("------------------------------------------")
    print("q. Quit")
    print("==========================================")

def main():
    if len(sys.argv) > 1:
        # Command line argument mode
        choice = sys.argv[1].strip().lower()
        dispatch(choice)
        return

    # Interactive menu mode
    while True:
        show_menu()
        choice = input("Select an option: ").strip().lower()
        if choice == 'q':
            print("Goodbye! 👋")
            sys.exit(0)
        dispatch(choice)
        input("\nPress Enter to continue...")

def dispatch(choice):
    if choice == '1':
        run_script("orchestrator.py")
    elif choice == '2':
        run_script("verify_drift.py")
    elif choice == '3':
        run_script("mock_esp32_imu.py")
    elif choice == '4':
        run_script("train_zupt.py")
    elif choice == '5':
        run_script("generate_zupt_data.py")
    elif choice == '6':
        run_script("health_check.py")
    elif choice == '7':
        run_script("calibrate.py")
    elif choice == '8':
        run_main([])
    elif choice == '9':
        run_main(["--load-bias"])
    else:
        print(f"Invalid option: {choice}")
        print("Usage: python manage.py [1-9|q]")

if __name__ == "__main__":
    main()
