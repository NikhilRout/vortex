#!/bin/bash

# Test runner script for blackbox testing with different configurations
# Runs all 8 combinations of cores, warps, threads, and apps with rtlsim driver and matrix size 128x128

echo "Starting automated testing for all configurations..."
echo "======================================================="

# Define the configurations
declare -a configs=(
    "1 4 4 sgemmx"
    "1 4 4 dot8"
    "1 16 16 sgemmx"
    "1 16 16 dot8"
    "4 4 4 sgemmx"
    "4 4 4 dot8"
    "4 16 16 sgemmx"
    "4 16 16 dot8"
)

# Counter for tracking progress
counter=1
total=${#configs[@]}

# Log file with timestamp
log_file="test_results_$(date +%Y%m%d_%H%M%S).log"
echo "Results will be logged to: $log_file"
echo ""

# Run each configuration
for config in "${configs[@]}"; do
    # Parse the configuration
    read -r cores warps threads app <<< "$config"
    
    echo "[$counter/$total] Running: cores=$cores, warps=$warps, threads=$threads, app=$app"
    echo "Command: ./ci/blackbox.sh --clusters=1 --cores=$cores --warps=$warps --threads=$threads --driver=rtlsim --app=$app --args=\"-n 128\""
    
    # Log the command being executed
    echo "=== Test $counter/$total - $(date) ===" >> "$log_file"
    echo "Configuration: cores=$cores, warps=$warps, threads=$threads, app=$app" >> "$log_file"
    echo "Command: ./ci/blackbox.sh --clusters=1 --cores=$cores --warps=$warps --threads=$threads --driver=rtlsim --app=$app --args=\"-n 128\"" >> "$log_file"
    echo "" >> "$log_file"
    
    # Execute the command and capture output
    if ./ci/blackbox.sh --clusters=1 --cores="$cores" --warps="$warps" --threads="$threads" --driver=rtlsim --app="$app" --args="-n 128" >> "$log_file" 2>&1; then
        echo "✓ Test $counter completed successfully"
    else
        echo "✗ Test $counter failed (check log for details)"
    fi
    
    echo "" >> "$log_file"
    echo "----------------------------------------" >> "$log_file"
    echo ""
    
    ((counter++))
    
    # Small delay between tests
    sleep 2
done

echo "======================================================="
echo "All tests completed!"
echo "Results saved to: $log_file"
echo "======================================================="