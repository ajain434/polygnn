#!/bin/bash

# Name of the Python script to run
PYTHON_SCRIPT="benchmark.py"

# Log file to store execution times
LOG_FILE="execution_times_20.log"

# Clear previous log content
echo "Execution times for $PYTHON_SCRIPT:" > $LOG_FILE

# Loop to run the Python script 10 times
for i in {1..10}
do
    echo "Run #$i:" >> $LOG_FILE
    START_TIME=$(date +%s%N) # Start time in nanoseconds
    python3 $PYTHON_SCRIPT
    END_TIME=$(date +%s%N)   # End time in nanoseconds
    ELAPSED_TIME=$((($END_TIME - $START_TIME) / 1000000)) # Convert to milliseconds
    echo "Time taken: ${ELAPSED_TIME} ms" >> $LOG_FILE
done

echo "Execution times logged in $LOG_FILE"
