#!/bin/bash

# Directory containing the files
directory="./scraping/data/"

# Temporary file to hold the list of base IDs
temp_ids="/tmp/ids_list.txt"

# Extract the unique base IDs from filenames, including '_3', using awk
ls "${directory}"geoguessr_* | awk -F'[_.]' '{print $(NF-2)"_"$(NF-1)}' | sort -u > "${temp_ids}"

# Initialize counter
delete_count=0

# Iterate over each base ID
while IFS= read -r id; do
    # Define the filenames for the location and result files
    png_file="${directory}geoguessr_location_${id}.png"
    json_file="${directory}geoguessr_result_${id}.json"

    # Check if both files exist
    if [ -f "$png_file" ] && [ ! -f "$json_file" ]; then
        # Delete the PNG file if the JSON file does not exist
        rm "$png_file"
        ((delete_count++))
    elif [ ! -f "$png_file" ] && [ -f "$json_file" ]; then
        # Delete the JSON file if the PNG file does not exist
        rm "$json_file"
        ((delete_count++))
    fi
done < "${temp_ids}"

# Output the number of deleted files
echo "Deleted $delete_count files."

# Cleanup the temporary file
rm "${temp_ids}"
