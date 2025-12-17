#!/bin/bash

echo "=========================================="
echo "Setting up Python Environment"
echo "=========================================="

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python version: $PYTHON_VERSION"

# Create virtual environment
echo ""
echo "Creating virtual environment: cms.venv"
python3 -m venv cms.venv

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create virtual environment"
    exit 1
fi

echo "Virtual environment created successfully"

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source cms.venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip