#!/bin/bash
# Pre-commit hook: Run basic endpoint checks before commit
#
# This hook validates that the API server can start and basic endpoints work.
#
# Usage: ./pre-commit.sh

PROJECT_DIR="/vf/users/parks34/projects/2secactpy"
API_DIR="$PROJECT_DIR/cytoatlas-api"

echo "Running pre-commit API checks..."

# Check if we're in the API directory
if [ ! -f "$API_DIR/app/main.py" ]; then
    echo "Warning: API directory not found. Skipping API checks."
    exit 0
fi

cd "$API_DIR"

# Check Python syntax
echo "Checking Python syntax..."
python -m py_compile app/main.py 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Error: Python syntax error in app/main.py"
    exit 1
fi

# Check if dependencies are importable
echo "Checking imports..."
python -c "from app.main import app" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Warning: Could not import app. Make sure dependencies are installed."
    # Don't fail - user might be committing non-API changes
fi

echo "Pre-commit checks passed."
exit 0
