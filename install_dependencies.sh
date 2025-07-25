#!/bin/bash

# Clean up any potential conflicts
echo "Cleaning up potential conflicts..."
pip uninstall -y pytest pytest-asyncio imageio scipy

# Install the dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

echo "Installation complete!" 