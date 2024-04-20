#!/bin/bash

# Directory containing the files
directory="./dspro2/1_data_collection/data/"

# Ensure the directory ends with a slash for path correctness
directory="${directory%/}/"

# Temporary directory for the temp file
temp_dir="./dspro2/1_data_collection/tmp/"
# Ensure the temporary directory exists
mkdir -p "${temp_dir}"

# Temporary file to hold the list of base names
temp_names="${temp_dir}names_list.txt"

# File in temp_dir containing game ids, one per line
results_games="${temp_dir}results-games"

# Debug: Show the directory and file pattern being scanned
echo "Scanning directory: $directory for pattern 'geoguessr_result_*_*_*.json'"

# Extract the unique base names from filenames
find "${directory}" -name 'geoguessr_result_*_*_*.json' | awk -F'/' '{n=split($NF,a,"_"); print a[1]"_"a[2]"_"a[3]"_"a[4]}' | sort -u > "${temp_names}"

# Debug: Output the number of unique bases found
echo "$(wc -l < "${temp_names}") unique base names found."

# Initialize counter
delete_count=0

# Iterate over each base name
while IFS= read -r base; do
    # Extract the game ID from the base name
    game_id=$(echo "$base" | awk -F'_' '{print $(NF)}')  # Assuming the game ID is the second-last element

	echo ä

    # Array to check completeness of the sequence
    complete=(0 0 0 0 0)

    # Check each file from 0 to 4
    for i in {0..4}; do
        file="${directory}${base}_${i}.json"
        if [ -f "$file" ]; then
            complete[i]=1
        else
            echo "Missing file: $file"
        fi
    done

    # Check if all elements in the complete array are 1
    if [ "${complete[*]}" != "1 1 1 1 1" ]; then
        echo "Incomplete set for game ID $game_id, deleting files."
        # If not all files are present, delete the existing files
        for i in {0..4}; do
            file="${directory}${base}_${i}.json"
            if [ -f "$file" ]; then
                echo "Deleting: $file"
                rm "$file"
                ((delete_count++))
            fi
        done

		# If results_games does not exist, skip the removal
		if [ ! -f "$results_games" ]; then
			continue
		fi

        # Remove the game_id from results-games if it exists
        if grep -q "^$game_id$" "$results_games"; then
            echo "Removing game ID $game_id from results-games"
            # Use grep to filter out the game_id and overwrite the file
            grep -v "^$game_id$" "$results_games" > "${temp_dir}temp-results-games"
            mv "${temp_dir}temp-results-games" "$results_games"
        fi
    fi
done < "${temp_names}"

# Output the number of deleted files
echo "Deleted $delete_count files."

# Cleanup the temporary file
rm "${temp_names}"
