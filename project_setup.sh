#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Create the virtual environment if it doesn't exist
if [ ! -d ~/venvs/gemma3 ]; then
    echo "Creating virtual environment at ~/venvs/gemma3"
    python -m venv ~/venvs/gemma3
fi

# Activate the gemma3 virtual environment
source ~/venvs/gemma3/bin/activate

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
python -m ipykernel install --user --name=gemma3 --display-name "Gemma3"

echo "Jupyter kernel 'gemma3' has been installed."

# Run the tests to verify everything is working
echo "Running tests to verify setup..."
poetry run pytest -v

echo "Project setup complete!"