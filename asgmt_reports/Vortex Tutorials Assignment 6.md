# Vortex Tutorials Assignment 6

Extending GPU Microarchitecture to accelerate a kernel in hardware. Implementing a new RISC-V custom instruction for computing integer dot product (VX_DOT8) in RTL

## Step 1 & 2

Completed in assignment 5

## Step 3: Hardware RTL Implementation

### SOLUTION:

```bash
cd vortex/hw/rtl
```

Updating `VX_define.h` to add DOT8 in ALU operation encoding

```verilog
`define INST_ALU_DOT8        4'b0001
```

Update `VX_config.vh` to define latency of dot8 instr as 2

```verilog
`ifndef LATENCY_DOT8
`define LATENCY_DOT8 2
`endif
```

Updating dpi_trace() in `VX_gpu_pkg.sv` to print the new instruction

```verilog
    task trace_ex_op(input int level,
                     input [`EX_BITS-1:0] ex_type,
                     input [`INST_OP_BITS-1:0] op_type,
                     input VX_gpu_pkg::op_args_t op_args
    );
        case (ex_type)
        `EX_ALU: begin
            case (op_args.alu.xtype)
                `ALU_TYPE_ARITH: begin
                    if (op_args.alu.is_w) begin
                        if (op_args.alu.use_imm) begin
			                      ...
                        end else begin
                            case (`INST_ALU_BITS'(op_type))
		                            ...
                                `INST_ALU_DOT8: `TRACE(level, ("DOT8W")) // for RV64
                                default:       `TRACE(level, ("?"))
                            endcase
                        end
                    end else begin
                        if (op_args.alu.use_imm) begin
		                        ...
                        end else begin
                            case (`INST_ALU_BITS'(op_type))
		                            ...
                                `INST_ALU_DOT8:  `TRACE(level, ("DOT8")) // for RV32
                                default:         `TRACE(level, ("?"))
                            endcase
                        end
                    end
                end
            ...
        ...
    endtask
```

Updating `core/VX_decode.sv` to decode the new instruction. Recognizing and routing DOT8 instruction to its appropriate execution unit (ALU)

```verilog
            `INST_EXT1: begin
                case (func7)
										...
                    7'h01: begin
                        case (func3)
                            3'h0: begin // DOT8
                                ex_type = `EX_ALU;
                                op_type = `INST_OP_BITS'(`INST_ALU_DOT8);
                                op_args.alu.xtype = `ALU_TYPE_ARITH;
                                op_args.alu.is_w = 0;
                                op_args.alu.use_PC = 0;
                                op_args.alu.use_imm = 0;
                                use_rd = 1;
                                `USED_IREG (rd);
                                `USED_IREG (rs1);
                                `USED_IREG (rs2);
                            end
                            default:;
                        endcase
                    end                    
                    default:;
                endcase
```

Creating `core/VX_alu_dot8.sv` , module VX_alu_dot8 PEs contain logic for performing the actual dot product computation

```verilog
`include "VX_define.vh"

module VX_alu_dot8 #(
    parameter `STRING INSTANCE_ID = "",
    parameter NUM_LANES = 1
) (
    input wire          clk,
    input wire          reset,

    // Inputs
    VX_execute_if.slave execute_if,

    // Outputs
    VX_commit_if.master commit_if
);
    `UNUSED_SPARAM (INSTANCE_ID)
    localparam PID_BITS = `CLOG2(`NUM_THREADS / NUM_LANES);
    localparam PID_WIDTH = `UP(PID_BITS);
    localparam TAG_WIDTH = `UUID_WIDTH + `NW_WIDTH + NUM_LANES + `PC_BITS + `NR_BITS + 1 + PID_WIDTH + 1 + 1;
    localparam LATENCY_DOT8 = `LATENCY_DOT8;
    localparam PE_RATIO = 2;
    localparam NUM_PES = `UP(NUM_LANES / PE_RATIO);

    `UNUSED_VAR (execute_if.data.op_type)
    `UNUSED_VAR (execute_if.data.tid)
    `UNUSED_VAR (execute_if.data.rs3_data)

    wire [NUM_LANES-1:0][2*`XLEN-1:0] data_in;

    for (genvar i = 0; i < NUM_LANES; ++i) begin :gen_lanes
        assign data_in[i][0 +: `XLEN] = execute_if.data.rs1_data[i];
        assign data_in[i][`XLEN +: `XLEN] = execute_if.data.rs2_data[i];
    end

    wire pe_enable;
    wire [NUM_PES-1:0][2*`XLEN-1:0] pe_data_in;
    wire [NUM_PES-1:0][`XLEN-1:0] pe_data_out;

    // PEs time-multiplexing
    VX_pe_serializer #(
        .NUM_LANES  (NUM_LANES),
        .NUM_PES    (NUM_PES),
        .LATENCY    (LATENCY_DOT8),
        .DATA_IN_WIDTH (2*`XLEN),
        .DATA_OUT_WIDTH (`XLEN),
        .TAG_WIDTH  (TAG_WIDTH),
        .PE_REG     (1)
    ) pe_serializer (
        .clk        (clk),
        .reset      (reset),
        .valid_in   (execute_if.valid),
        .data_in    (data_in),
        .tag_in     ({
            execute_if.data.uuid,
            execute_if.data.wid,
            execute_if.data.tmask,
            execute_if.data.PC,
            execute_if.data.rd,
            execute_if.data.wb,
            execute_if.data.pid,
            execute_if.data.sop,
            execute_if.data.eop
        }),
        .ready_in   (execute_if.ready),
        .pe_enable  (pe_enable),
        .pe_data_in (pe_data_out),
        .pe_data_out(pe_data_in),
        .valid_out  (commit_if.valid),
        .data_out   (commit_if.data.data),
        .tag_out    ({
            commit_if.data.uuid,
            commit_if.data.wid,
            commit_if.data.tmask,
            commit_if.data.PC,
            commit_if.data.rd,
            commit_if.data.wb,
            commit_if.data.pid,
            commit_if.data.sop,
            commit_if.data.eop
        }),
        .ready_out  (commit_if.ready)
    );

    // PEs instancing
    for (genvar i = 0; i < NUM_PES; ++i) begin : gen_pes
        wire [XLEN-1:0] a = pe_data_in[i][0 +: XLEN];
        wire [XLEN-1:0] b = pe_data_in[i][XLEN +: XLEN];

        wire signed [15:0] p0 = $signed(a[7:0]) * $signed(b[7:0]);
        wire signed [15:0] p1 = $signed(a[15:8]) * $signed(b[15:8]);
        wire signed [15:0] p2 = $signed(a[23:16]) * $signed(b[23:16]);
        wire signed [15:0] p3 = $signed(a[31:24]) * $signed(b[31:24]);

        // Tree adder
        wire signed [16:0] sum01 = p0 + p1;
        wire signed [16:0] sum23 = p2 + p3;
        wire signed [17:0] result_18 = sum01 + sum23;
        wire [31:0] c = {{14{result_18[17]}}, result_18};

        reg [31:0] result;
        `BUFFER_EX(result, c, pe_enable, 1, LATENCY_DOT8); // c is the result of the dot product
        assign pe_data_out[i] = `XLEN'(result);
    end

endmodule

```

Updating `core/VX_alu_unit.sv` to add new VX_alu_dot8 module instance as a 3rd sub-unit after VX_alu_muldiv

```verilog
`define EXT_DOT8_ENABLED 1

module VX_alu_unit #(...) (
	...
);
    localparam PE_COUNT     = 1 + `EXT_M_ENABLED + `EXT_DOT8_ENABLED;
    localparam PE_IDX_DOT8  = PE_IDX_MDV + `EXT_DOT8_ENABLED;
    
    for (genvar block_idx = 0; block_idx < BLOCK_SIZE; ++block_idx) begin : g_alus
		    ...
		    always @(*) begin
            pe_select = PE_IDX_INT;
            if (`EXT_M_ENABLED && (per_block_execute_if[block_idx].data.op_args.alu.xtype == `ALU_TYPE_MULDIV))
                pe_select = PE_IDX_MDV;
            if (`EXT_DOT8_ENABLED && (per_block_execute_if[block_idx].data.op_type == `INST_OP_BITS'(`INST_ALU_DOT8)))
                pe_select = PE_IDX_DOT8;
        end
        ...
    `ifdef EXT_DOT8_ENABLED
        VX_alu_dot8 #(
            .INSTANCE_ID (`SFORMATF(("%s-dot8%0d", INSTANCE_ID, block_idx))),
            .NUM_LANES (NUM_LANES)
        ) dot8_unit (
            .clk        (clk),
            .reset      (reset),
            .execute_if (pe_execute_if[PE_IDX_DOT8]),
            .commit_if  (pe_commit_if[PE_IDX_DOT8])
        );
    `endif
    end
endmodule
```

## Step 4: Testing

```bash
cd vortex/build
source ./ci/toolchain_env.sh
make -s

# single test example
./ci/blackbox.sh --clusters=1 --cores=1 --warps=4 --threads=4 --driver=rtlsim--app=dot8--args="-n 128"
```

Creating Bash script to run all configurations one after the other and log reports

```bash
code rtl_tests.sh
chmod +x rtl_tests.sh
./rtl_tests.sh
```

```bash
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
```

![rtl_tests_ss.png](Vortex%20Tutorials%20Assignment%206%202085dbc17d10808c996cd146b78a255d/rtl_tests_ss.png)

### Results Tabulation & Plot

| **Kernel/App** | **Cores** | **Warps** | **Threads** | **IPC** |
| --- | --- | --- | --- | --- |
| sgemmx | 1 | 4 | 4 | 1.376283 |
| dot8 | 1 | 4 | 4 | 1.898650 |
| sgemmx | 4 | 4 | 4 | 3.532519 |
| dot8 | 4 | 4 | 4 | 3.883131 |
| sgemmx | 1 | 16 | 16 | 4.635266 |
| dot8 | 1 | 16 | 16 | 6.459965 |
| sgemmx | 4 | 16 | 16 | 10.334927 |
| dot8 | 4 | 16 | 16 | 9.636362 |

![image.png](Vortex%20Tutorials%20Assignment%206%202085dbc17d10808c996cd146b78a255d/image.png)

### Inference

The RTL implementation finds similar results to the cycle-level simulations

In single-core configurations, increasing warps/threads per core results in higher relative IPC gains (39.3% with 16W-16T ; 38.1% with 4W-4T)

However, in Multi-Core configurations, there is only a marginal improvement at lower warp/thread counts (9.9% with 4W-4T), and in contrast diminishes at higher warp/thread counts (-6.7% with 16W-16T)

This further backs the notion that this behavior is likely due to resource contention (memory bandwidth bottleneck). Using coalesced memory access to read/write groups of 4 values from/to the packed registers used by the dot8 instruction could potentially address this issue
