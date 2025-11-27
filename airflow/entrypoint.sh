#!/bin/bash
set -e

echo "=========================================="
echo "Starting Airflow initialization..."
echo "=========================================="

# Clean up any existing PID files and processes
echo "Cleaning up stale processes and PID files..."
pkill -9 -f "airflow webserver" 2>/dev/null || true
pkill -9 -f "airflow scheduler" 2>/dev/null || true
pkill -9 -f "gunicorn" 2>/dev/null || true
rm -f /opt/airflow/*.pid 2>/dev/null || true
sleep 2

# Initialize Airflow database
echo "Initializing Airflow database..."
airflow db migrate || airflow db init

# Create admin user
echo "Creating admin user..."
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin 2>/dev/null || echo "Admin user already exists"

echo "=========================================="
echo "Starting Airflow services..."
echo "=========================================="

# Start webserver in background
echo "Starting webserver..."
airflow webserver --port 8080 &
WEBSERVER_PID=$!

# Give webserver time to initialize
sleep 10

# Start scheduler in foreground (keeps container alive)
echo "Starting scheduler..."
exec airflow scheduler