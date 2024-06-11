#!/bin/bash

# Set your IPMI IP address, username, and password
#IPMI_IP="your_ipmi_ip"
#IPMI_USER="your_username"
#IPMI_PASS="your_password"

# Specify the output file
SCRIPT_DIR=$(dirname "$0")
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_FILE="$SCRIPT_DIR/power_readings_$TIMESTAMP.txt"

while true; do
    # Run the ipmitool command and extract the relevant information
    #POWER_OUTPUT=$(ipmitool -I lanplus -H "$IPMI_IP" -U "$IPMI_USER" -P "$IPMI_PASS" dcmi power reading)
    POWER_OUTPUT=$(ipmitool dcmi power reading)
    INSTANTANEOUS_POWER=$(echo "$POWER_OUTPUT" | grep -e "Instantaneous" | cut -d ':' -f2 | awk '{print $1}')
    TIMESTAMP=$(echo "$POWER_OUTPUT" | grep -e "IPMI timestamp" | cut -d ':' -f2-)

    # Print the results
    #echo "Instantaneous power reading: $INSTANTANEOUS_POWER Watts" "IPMI timestamp: $TIMESTAMP"

    # Append the results to the output file
    #echo "Instantaneous power reading: $INSTANTANEOUS_POWER Watts" >> "$OUTPUT_FILE" "IPMI timestamp: $TIMESTAMP" >> "$OUTPUT_FILE"
    echo "Instantaneous power reading:            $INSTANTANEOUS_POWER Watts      IPMI timestamp: $TIMESTAMP" >> "$OUTPUT_FILE"

    # Sleep for a desired interval (e.g., 5 seconds)
    sleep 1
done
