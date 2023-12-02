#!/bin/bash

# Check if a search string is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <search-string>"
    exit 1
fi

# Assign the search string to a variable
search_string="$1"

# Initialize variables to store the latest commit hash and file paths
latest_commit=""
declare -a file_paths

# Function to search in a specific commit
search_in_commit() {
    local commit=$1
    # Search for the string in .py and .ipynb files in the commit
    local paths=$(git grep -l "$search_string" "$commit" -- '*.py' '*.ipynb')
    echo "$paths"
}

# Loop through each commit that changed .py or .ipynb files
for commit in $(git rev-list --all); do
    file_paths=($(search_in_commit "$commit"))
    if [ "${#file_paths[@]}" -ne 0 ]; then
        latest_commit=$commit
        break # Found the latest commit with the string, exit loop
    fi
done

# Check if a commit was found and display results
if [ ! -z "$latest_commit" ]; then
    echo "Latest commit containing '$search_string': $latest_commit"
    echo "Files containing '$search_string' in this commit:"
    for file_path in "${file_paths[@]}"; do
        echo "$file_path"
    done
else
    echo "No commit found containing '$search_string'"
fi

