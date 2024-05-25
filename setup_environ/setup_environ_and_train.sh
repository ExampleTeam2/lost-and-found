#!/bin/sh
pip3 install poetry --quiet
# Clone the repository withouth the LFS files for now
export GIT_LFS_SKIP_SMUDGE=1 && git clone "https://oauth2:$ACCESS_TOKEN@gitlab.com/exampleteam2/dspro2.git"
cd dspro2
echo "Installing dependencies..."
# Hack to speed up the installation of the dependencies
poetry export -f requirements.txt > requirements.txt
python -m pip install -r requirements.txt --quiet
poetry install --quiet
echo "Dependencies installed"

# Define the path to the notebook
NOTEBOOK_PATH="dspro2/4_modeling/training_resnet_models.ipynb"
OUTPUT_NAME="temp_executed_notebook"
OUTPUT_PATH="dspro2/4_modeling"

# Activate the poetry environment and convert the notebook to a script
poetry run jupyter nbconvert --ExecutePreprocessor.timeout=600 --to script $NOTEBOOK_PATH --output $OUTPUT_NAME
cd $OUTPUT_PATH
# Detect if the notebook is running on colab with a drive mounted
if [ -d /content/drive/MyDrive ]; then
	echo "Running on colab"
	export FILE_LOCATION="/content/drive/MyDrive/.data"
	export JSON_FILE_LOCATION="/content/drive/MyDrive/.data"
	export IMAGE_FILE_LOCATION="/content/drive/MyDrive/.data"
fi
export NESTED="true"
export USE_FILES_LIST="true"
export TMP_DIR_AND_ZIP="true"
export WANDB_TOKEN="$WANDB_TOKEN"
poetry run python3 $OUTPUT_NAME.py
