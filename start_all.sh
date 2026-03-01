#!/bin/bash
echo "🚀 Starting AirWriting Backend Services (Jetson/Linux)..."

# Exit on error
set -e

# Start processes in the background
echo "1. Starting IMU Engine (main.py)..."
python3 main.py &
PID_MAIN=$!

echo "2. Starting WebSocket Relay (tools/web_relay.py)..."
python3 tools/web_relay.py &
PID_RELAY=$!

echo "3. Starting Flask Web App & ML Engine (web_app/app.py)..."
python3 web_app/app.py &
PID_APP=$!

echo "✅ All services running!"
echo "Press [CTRL+C] to stop all services."

# Trap Ctrl+C to kill background jobs
trap "echo 'Stopping services...'; kill $PID_MAIN $PID_RELAY $PID_APP; exit" SIGINT SIGTERM

# Wait indefinitely
wait
