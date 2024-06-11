#!/bin/bash
set -x 

#!/bin/bash

# Directory containing the files
DIRECTORY="/work/training4.0/dellemc-mlperf-training-v4.0/Dell/results/XE8640x4H100-SXM-80GB"

# Recursively find all .txt files in the directory and subdirectories
# and replace the placeholders using sed
find $DIRECTORY -type f -name "*.txt" -exec sed -i 's/1xSUBMISSION_PLATFORM_PLACEHOLDER/1xXE8640x4H100-SXM-80GB/g' {} +
find $DIRECTORY -type f -name "*.txt" -exec sed -i 's/SUBMISSION_ORG_PLACEHOLDER/Dell/g' {} +

echo "Recursive replacement in .txt files complete."

