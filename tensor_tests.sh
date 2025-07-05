#!/bin/bash

# Tensor Tests Script with Logging
# Usage: ./tensor_tests.sh

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Log file
LOG_FILE="tensor_tests_$(date +%Y%m%d_%H%M%S).log"

# Function to log and display
log_and_echo() {
    echo -e "$1" | tee -a "$LOG_FILE"
}

# Function to run command with logging
run_cmd() {
    local cmd="$1"
    local desc="$2"
    
    log_and_echo "${BLUE}[$(date '+%H:%M:%S')] Running: $desc${NC}"
    log_and_echo "Command: $cmd"
    
    # Run command and capture output
    if eval "$cmd" >> "$LOG_FILE" 2>&1; then
        log_and_echo "${GREEN}✓ SUCCESS: $desc${NC}"
        return 0
    else
        log_and_echo "${RED}✗ FAILED: $desc${NC}"
        return 1
    fi
}

tensor() {
    log_and_echo "${YELLOW}================================${NC}"
    log_and_echo "${YELLOW}Starting Tensor Tests - $(date)${NC}"
    log_and_echo "${YELLOW}Log file: $LOG_FILE${NC}"
    log_and_echo "${YELLOW}================================${NC}"
    
    local failed_tests=0
    local total_tests=0
    
    # Test 1: SIMX - int8/int32, 2 threads
    ((total_tests++))
    run_cmd "make -C tests/regression/sgemm_tcu clean" "Clean sgemm_tcu (Test 1)" || ((failed_tests++))
    run_cmd 'CONFIGS="-DNUM_THREADS=2 -DITYPE=int8 -DOTYPE=int32" make -C tests/regression/sgemm_tcu' "Build sgemm_tcu (int8/int32, 2 threads)" || ((failed_tests++))
    run_cmd 'CONFIGS="-DNUM_THREADS=2 -DEXT_TCU_ENABLE" ./ci/blackbox.sh --driver=simx --app=sgemm_tcu --debug=3 --log=run_simx.log' "Run SIMX test (int8/int32, 2 threads)" || ((failed_tests++))
    
    # Test 2: SIMX - uint4/int32, 4 threads
    ((total_tests++))
    run_cmd "make -C tests/regression/sgemm_tcu clean" "Clean sgemm_tcu (Test 2)" || ((failed_tests++))
    run_cmd 'CONFIGS="-DNUM_THREADS=4 -DITYPE=uint4 -DOTYPE=int32" make -C tests/regression/sgemm_tcu' "Build sgemm_tcu (uint4/int32, 4 threads)" || ((failed_tests++))
    run_cmd 'CONFIGS="-DNUM_THREADS=4 -DEXT_TCU_ENABLE" ./ci/blackbox.sh --driver=simx --app=sgemm_tcu' "Run SIMX test (uint4/int32, 4 threads)" || ((failed_tests++))
    
    # Test 3: SIMX - fp16/fp32, 8 threads
    ((total_tests++))
    run_cmd "make -C tests/regression/sgemm_tcu clean" "Clean sgemm_tcu (Test 3)" || ((failed_tests++))
    run_cmd 'CONFIGS="-DNUM_THREADS=8 -DITYPE=fp16 -DOTYPE=fp32" make -C tests/regression/sgemm_tcu' "Build sgemm_tcu (fp16/fp32, 8 threads)" || ((failed_tests++))
    run_cmd 'CONFIGS="-DNUM_THREADS=8 -DEXT_TCU_ENABLE" ./ci/blackbox.sh --driver=simx --app=sgemm_tcu' "Run SIMX test (fp16/fp32, 8 threads)" || ((failed_tests++))
    
    # Test 4: SIMX - bf16/bf16, 16 threads
    ((total_tests++))
    run_cmd "make -C tests/regression/sgemm_tcu clean" "Clean sgemm_tcu (Test 4)" || ((failed_tests++))
    run_cmd 'CONFIGS="-DNUM_THREADS=16 -DITYPE=bf16 -DOTYPE=bf16" make -C tests/regression/sgemm_tcu' "Build sgemm_tcu (bf16/bf16, 16 threads)" || ((failed_tests++))
    run_cmd 'CONFIGS="-DNUM_THREADS=16 -DEXT_TCU_ENABLE" ./ci/blackbox.sh --driver=simx --app=sgemm_tcu' "Run SIMX test (bf16/bf16, 16 threads)" || ((failed_tests++))
    
    # Test 5: RTLSIM - int8/int32, 2 threads
    ((total_tests++))
    run_cmd "make -C tests/regression/sgemm_tcu clean" "Clean sgemm_tcu (Test 5)" || ((failed_tests++))
    run_cmd 'CONFIGS="-DNUM_THREADS=2 -DITYPE=int8 -DOTYPE=int32" make -C tests/regression/sgemm_tcu' "Build sgemm_tcu (int8/int32, 2 threads)" || ((failed_tests++))
    run_cmd 'CONFIGS="-DNUM_THREADS=2 -DEXT_TCU_ENABLE" ./ci/blackbox.sh --driver=rtlsim --app=sgemm_tcu --debug=3 --log=run_rtlsim.log' "Run RTLSIM test (int8/int32, 2 threads)" || ((failed_tests++))
    
    # Test 6: RTLSIM - uint4/int32, 4 threads
    ((total_tests++))
    run_cmd "make -C tests/regression/sgemm_tcu clean" "Clean sgemm_tcu (Test 6)" || ((failed_tests++))
    run_cmd 'CONFIGS="-DNUM_THREADS=4 -DITYPE=uint4 -DOTYPE=int32" make -C tests/regression/sgemm_tcu' "Build sgemm_tcu (uint4/int32, 4 threads)" || ((failed_tests++))
    run_cmd 'CONFIGS="-DNUM_THREADS=4 -DEXT_TCU_ENABLE" ./ci/blackbox.sh --driver=rtlsim --app=sgemm_tcu' "Run RTLSIM test (uint4/int32, 4 threads)" || ((failed_tests++))
    
    # Test 7: RTLSIM - fp16/fp32, 8 threads
    ((total_tests++))
    run_cmd "make -C tests/regression/sgemm_tcu clean" "Clean sgemm_tcu (Test 7)" || ((failed_tests++))
    run_cmd 'CONFIGS="-DNUM_THREADS=8 -DITYPE=fp16 -DOTYPE=fp32" make -C tests/regression/sgemm_tcu' "Build sgemm_tcu (fp16/fp32, 8 threads)" || ((failed_tests++))
    run_cmd 'CONFIGS="-DNUM_THREADS=8 -DEXT_TCU_ENABLE" ./ci/blackbox.sh --driver=rtlsim --app=sgemm_tcu' "Run RTLSIM test (fp16/fp32, 8 threads)" || ((failed_tests++))
    
    # Test 8: RTLSIM - fp16/fp32, 8 threads with DSP
    ((total_tests++))
    run_cmd 'CONFIGS="-DNUM_THREADS=8 -DEXT_TCU_ENABLE -DTCU_DSP" ./ci/blackbox.sh --driver=rtlsim --app=sgemm_tcu' "Run RTLSIM test (fp16/fp32, 8 threads, DSP)" || ((failed_tests++))
    
    # Test 9: RTLSIM - bf16/fp32, 16 threads
    ((total_tests++))
    run_cmd "make -C tests/regression/sgemm_tcu clean" "Clean sgemm_tcu (Test 9)" || ((failed_tests++))
    run_cmd 'CONFIGS="-DNUM_THREADS=16 -DITYPE=bf16 -DOTYPE=fp32" make -C tests/regression/sgemm_tcu' "Build sgemm_tcu (bf16/fp32, 16 threads)" || ((failed_tests++))
    run_cmd 'CONFIGS="-DNUM_THREADS=16 -DEXT_TCU_ENABLE" ./ci/blackbox.sh --driver=rtlsim --app=sgemm_tcu' "Run RTLSIM test (bf16/fp32, 16 threads)" || ((failed_tests++))
    
    # Final cleanup
    run_cmd "make -C tests/regression/sgemm_tcu clean" "Final cleanup"
    
    # Summary
    log_and_echo "${YELLOW}================================${NC}"
    log_and_echo "${YELLOW}Tensor Tests Summary - $(date)${NC}"
    log_and_echo "${YELLOW}================================${NC}"
    
    if [ $failed_tests -eq 0 ]; then
        log_and_echo "${GREEN}🎉 ALL TESTS PASSED! ($total_tests/$total_tests)${NC}"
    else
        log_and_echo "${RED}❌ $failed_tests out of $total_tests tests failed${NC}"
    fi
    
    log_and_echo "${BLUE}Full log saved to: $LOG_FILE${NC}"
    log_and_echo "${YELLOW}================================${NC}"
    
    return $failed_tests
}

# Run the tests
tensor