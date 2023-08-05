#!bin/bash

# Declares the variables required to run the application
export PYTHONPATH="$(pwd)"
export BACKEND_URL="http://172.17.0.1:8000"
export VIDEO_DIR="/workspace/data/video"
export WORKING_DIR="/workspace/temp"

APP_MODULE="main:app"
HOST="0.0.0.0"
PORT="8080"

# The program is run with the following command:
exec uvicorn --reload --host "$HOST" --port "$PORT" "$APP_MODULE"