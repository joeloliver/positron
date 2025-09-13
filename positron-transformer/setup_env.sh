#!/bin/bash
# Setup script for Positron Transformer Environment

echo "Setting up Positron Transformer development environment..."

# Create virtual environment
python3 -m venv venv
echo "Virtual environment created."

# Activate virtual environment
source venv/bin/activate
echo "Virtual environment activated."

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
echo "Requirements installed."

# Install project in development mode
pip install -e .
echo "Project installed in development mode."

echo "Setup complete! To activate the environment in the future, run:"
echo "source venv/bin/activate"