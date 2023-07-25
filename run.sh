#!bin/bash

# Declares the variables required to run the application
export PYTHONPATH="$(pwd)"
APP_MODULE="main:app"
HOST="0.0.0.0"
PORT="8080"

# The program is run with the following command:
exec uvicorn --reload --host "$HOST" --port "$PORT" "$APP_MODULE"