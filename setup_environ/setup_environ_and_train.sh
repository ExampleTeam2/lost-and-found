#!/bin/sh
pip3 install poetry
git clone "https://oauth2:$ACCESS_TOKEN@gitlab.com/exampleteam2/dspro2.git"
cd dspro2
poetry install

# Define the path to the notebook
NOTEBOOK_PATH="dspro2/4_modeling/training_resnet_models.ipynb"
OUTPUT_PATH="temp_executed_notebook.ipynb"

# Activate the poetry environment and execute the notebook
poetry run jupyter nbconvert --ExecutePreprocessor.timeout=600 --to notebook --execute "$NOTEBOOK_PATH" --output "$OUTPUT_PATH"
