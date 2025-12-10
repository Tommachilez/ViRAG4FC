#!/bin/bash

# Save Python dependencies into requirements.txt
echo "Saving dependencies to requirements.txt..."

# Check if a virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "No virtual environment detected. Please activate your virtual environment first."
    exit 1
fi

# Export dependencies to requirements.txt
pip freeze > "$PWD/requirements.txt"

if [[ $? -eq 0 ]]; then
    echo "Dependencies saved successfully to $PWD/requirements.txt."
else
    echo "Failed to save dependencies. Please check for errors."
fi