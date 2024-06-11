#!/bin/bash
set -x

#!/bin/bash

# Directory where the log files are located
cd /path/to/your/log/files  # Update this path to your actual directory

# Create an array to hold file names, sorted numerically by the part of the filename before .log
files=($(ls *.log | sort -t '_' -k2,2n))

# Counter for renaming files to result_X.txt, where X starts from 0
index=0

# Loop through the sorted list of files
for file in "${files[@]}"; do
    mv "$file" "result_${index}.txt"
    echo "Renamed $file to result_${index}.txt"
    ((index++))
done

echo "All files have been renamed according to sequence."

