#!/bin/sh
pip3 install poetry
git clone "https://oauth2:$ACCESS_TOKEN@gitlab.com/exampleteam2/dspro2.git"
cd dspro2
poetry install

# Define the path to the notebook
NOTEBOOK_PATH="dspro2/4_modeling/training_resnet_models.ipynb"
OUTPUT_NAME="temp_executed_notebook"
OUTPUT_PATH="4_modeling/$OUTPUT_NAME.py"

# Activate the poetry environment and convert the notebook to a script
poetry run jupyter nbconvert --ExecutePreprocessor.timeout=600 --to script $NOTEBOOK_PATH --output $OUTPUT_NAME
poetry run python3 $OUTPUT_PATH