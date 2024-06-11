#!bin/bash

FILE=$1

# collect timestamps
#head_timestamp=$(cat $FILE | head -1 | sed -e 's/.*MLLOG\ \(.*\)/\1/g' | jq -r '.time_ms')
#tail_timestamp=$(cat $FILE | tail -1 | sed -e 's/.*MLLOG\ \(.*\)/\1/g' | jq -r '.time_ms')

# prepend lines
#prepend_lines=':::MLLOG {"namespace": "", "time_ms": __HEAD_TIMESTAMP__, "event_type": "INTERVAL_START", "key": "power_measurement_start", "value": null, "metadata": {"file": "/cm/shared/smc/ah/power/mlperf-logging/mlperf_logging/mllog/examples/power/power_measurement.py", "lineno": 171}}
#prepend_lines=':::MLLOG {"namespace": "", "time_ms": __HEAD_TIMESTAMP__, "event_type": "INTERVAL_START", "key": "power_measurement_start", "value": null, "metadata": ""}\n:::MLLOG {"namespace": "", "time_ms": __HEAD_TIMESTAMP__, "event_type": "POINT_IN_TIME", "key": "conversion_eff", "value": 1.0, "metadata": ""}'
#prepend_lines=$(echo $prepend_lines | sed -e "s/__HEAD_TIMESTAMP__/$head_timestamp/g")
#sed -i "1s/^/$prepend_lines\n/" $FILE 


# append line
#append_line=':::MLLOG {"namespace": "", "time_ms": __TAIL_TIMESTAMP__, "event_type": "INTERVAL_END", "key": "power_measurement_stop", "value": null, "metadata": ""}'
#append_line=$(echo $append_line | sed -e "s/__TAIL_TIMESTAMP__/$tail_timestamp/g")
#echo $append_line >> $FILE

#prepend_1=$(jq --argjson namespace


# get first two lines
first_line_json=$(cat $FILE | head -1 | sed -e 's/.*MLLOG\ \(.*\)/\1/g')
second_line_json=$(cat $FILE | head -2 | tail -1 | sed -e 's/.*MLLOG\ \(.*\)/\1/g')

# modify first two lines
#echo "Orig: $first_line_json"
first_line_json_new=$(echo $first_line_json | jq -c --arg E "INTERVAL_START" --arg K "power_measurement_start" --arg V "null" --arg M "" '.event_type = $E  | .key = $K | .value = $V | .metadata = $M')
#echo "New: $first_line_json_new"
#echo "Orig 2: $first_line_json"
floaty=1.0
second_line_json_new=$(echo $first_line_json | jq -c --arg K "conversion_eff" --argjson V "$floaty" --arg M "" '.key = $K | .value = $V | .metadata = $M')
#echo "New 2: $second_line_json_new"

# modify last line
last_line_json=$(cat $FILE | tail -1 | sed -e 's/.*MLLOG\ \(.*\)/\1/g')
#echo "Orig last: $last_line_json_new"
last_line_json_new=$(echo $first_line_json | jq -c --arg E "INTERVAL_END" --arg K "power_measurement_stop" --arg V "null" --arg M "" '.event_type = $E  | .key = $K | .value = $V | .metadata = $M')
#echo "New last: $last_line_json_new"


# update the file
tmpfile=$(mktemp)
echo ":::MLLOG $first_line_json_new" >> $tmpfile
echo ":::MLLOG $second_line_json_new" >> $tmpfile
tail -n+3 $FILE | head -n -1 >> $tmpfile
echo "::MLLOG $last_line_json_new" >> $tmpfile

#echo "New:"
#cat $tmpfile

cp $tmpfile $FILE

rm $tmpfile

#echo "Check head:"
#head -5 $FILE
#echo "Check tail:"
#tail -5 $FILE
