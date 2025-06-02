#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Create the virtual environment if it doesn't exist
if [ ! -d ~/venvs/rlvr ]; then
    echo "Creating virtual environment at ~/venvs/rlvr"
    python -m venv ~/venvs/rlvr
fi

# Activate the rlvr virtual environment
source ~/venvs/rlvr/bin/activate

# Upgrade pip and install poetry
pip install --upgrade pip
pip install poetry

# Configure poetry to use the existing venv
poetry config virtualenvs.create false

# Update the lock file if necessary
poetry lock

# Install dependencies and the project
poetry install

# Create and install the IPython kernel for the project
python -m ipykernel install --user --name=rlvr --display-name "RLVR"

echo "Jupyter kernel 'rlvr' has been installed."

# Run the tests to verify everything is working
echo "Running tests to verify setup..."
poetry run pytest -v

echo "Project setup complete!"