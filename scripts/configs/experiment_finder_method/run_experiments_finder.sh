#!/bin/bash

# Define the order of execution
phases=("find" "generate" "evaluate")

for phase in "${phases[@]}"; do
    echo -e "\n========================================"
    echo "PHASE: $phase"
    echo "========================================"

    # Enable nullglob so the loop doesn't run if no files match
    shopt -s nullglob
    # Create an array of files matching the pattern for the current phase
    files=( *_freebase_${phase}_*.yaml )
    shopt -u nullglob

    # Check if any files were found
    if [ ${#files[@]} -eq 0 ]; then
        echo "No files found for phase: $phase"
        continue
    fi

    # Iterate through the found files
    for file in "${files[@]}"; do
        echo "Running config: $file"
        
        # Run the command
        uv run mfdt "$file"
        
        # Optional: Check if the command succeeded
        if [ $? -ne 0 ]; then
            echo "Error: Experiment failed for $file"
            # exit 1 # Uncomment this line to stop the script immediately on error
        fi
    done
done

echo -e "\nAll experiments completed."