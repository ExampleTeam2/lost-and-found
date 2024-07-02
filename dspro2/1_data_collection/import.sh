#!/bin/bash

# Run this after putting the files (from kaggle (https://www.kaggle.com/datasets/killusions/street-location-images/), ...) in the .data directory to rename them to our naming convention and move them to the correct directory

# Do not run this when using scraping, as the files are already named correctly

# User settings
original_json_name_prefix="location_"
original_png_name_prefix="location_"
already_enriched=true

# Directory containing the files
directory="./dspro2/1_data_collection/.data/"

new_json_name_prefix="geoguessr_result_singleplayer_"
new_png_name_prefix="geoguessr_location_singleplayer_"

copy_enriched_files_to="./dspro2/3_data_preparation/01_enriching/.data/"

# Rename files ending with .json from $original_json_name_prefix* to $new_json_name_prefix*, optionally copying them to the enriched directory
for file in "${directory}${original_json_name_prefix}"*".json"; do
	# Extract the base name
	base=$(basename "$file" | sed "s/^${original_json_name_prefix}//" | sed "s/\.json$//")
	# Rename the file
	mv "$file" "${directory}${new_json_name_prefix}${base}.json"
	# If already enriched, copy the file to the enriched directory
	if [ "$already_enriched" = true ]; then
		cp "${directory}${new_json_name_prefix}${base}.json" "${move_enriched_files_to}${new_json_name_prefix}${base}.json"
	fi
done

# Rename files ending with .png from $original_png_name_prefix* to $new_png_name_prefix*
for file in "${directory}${original_png_name_prefix}"*".png"; do
	# Extract the base name
	base=$(basename "$file" | sed "s/^${original_png_name_prefix}//" | sed "s/\.png$//")
	# Rename the file
	mv "$file" "${directory}${new_png_name_prefix}${base}.png"
done

# Now run the import.ipynb notebook to use the data_loader while training
echo "Files renamed and moved to the correct directory. Run the import.ipynb notebook now to prepare for training."
