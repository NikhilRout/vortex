# Vortex Tutorials Assignment 5

## Assignment #5: Dot Product Acceleration (SimX)
https://github.com/vortexgpgpu/vortex_tutorials/blob/main/Exercises/assignment5.md

## Step 1: ISA Extension

VX_DOT8 calculates the dot product of two 4x4 vectors of int8 integers (packed in a 32-bit register)

Dot Product = *(A1B1 + A2B2 + A3B3 + A4B4)*

Instruction format:   *VX_DOT8 rd, rs1, rs2*

rs1 := {A1, A2, A3, A4}
rs2 := {B1, B2, B3, B4}
rd  := destination int32 result

R-Type RISC-V instruction format.

```markdown
| funct7  | rs2    | rs1    | funct3 | rd    | opcode |
|  7 bits | 5 bits | 5 bits | 3 bits | 5 bit | 7 bits |
```

Using custom-0/1 opcode space = 0x0B (0001011) with func7=1 and func3=0; (prevents clash with RISC-V 0x33 R-Type Standard Instructions)

[insn speudo-instruction format reference](https://sourceware.org/binutils/docs/as/RISC_002dV_002dFormats.html) (the .insn directive allows users to explicitly define instructions using a numeric representation, providing more control over instruction encoding)

### SOLUTION:

Modifying `vx_intrinsics.h` (contains inline assembly definitions for all custom Vortex instructions) to add new VX_DOT8 instruction definition

```bash
cd vortex/kernel/include
```

```cpp
// DOT8
inline int vx_dot8(int a, int b) {
  size_t ret; //why isnt this int?
  //size_t guaranteed to be large enough to hold size of largest possible object on the system = 32 bits (unisnged int) --> (immaterial here)
  asm volatile (".insn r %1, 0, 1, %0, %2, %3" : "=r"(ret) : "i"(RISCV_CUSTOM0), "r"(a), "r"(b)); // = --> o/p; r --> gpr; %_ --> positon ref
  //alternatively
  //asm volatile (".insn r 0x0B, 0, 1, %0, %1, %2" : "=r"(ret) : "r"(a), "r"(b));
  return ret;
}
```

## Step 2: Matrix Multiplication Kernel

Implementing a matrix multiplication GPU kernel that uses the new hardware-accelerated vx_dot8 instruction

### SOLUTION:

Cloning sgemmx test directory

```bash
cd vortex/tests/regression
cp -r sgemmx dot8
cd dot8
```

Modifying Test `Makefile`

```makefile
PROJECT := dot8
```

Modifying TYPE definition from float to int8 in `common.h`

```cpp
#define TYPE int8_t
```

Modifying `main.cpp` to operate on int8_t matrices

```cpp
// converting float comparator function to int8
template <>
class Comparator<int8_t> {
public:
static const char* type_str() {
return "int8_t";
}
static int8_t generate() {
return rand() % 128;
}
static bool compare(int8_t a, int8_t b, int index, int errors) {
if (a != b) {
if (errors < 100) {
printf("*** error: [%d] expected=%d, actual=%d\n", index, b, a);
}
return false;
}
return true;
}
};

// updating CPU reference matmul function to operate on int8
static void matmul_cpu(int32_t* out, const int8_t* A, const int8_t* B, uint32_t width, uint32_t height) {
  for (uint32_t row = 0; row < height; ++row) {
    for (uint32_t col = 0; col < width; ++col) {
      int32_t sum = 0;
      for (uint32_t e = 0; e < width; ++e) {
        sum += static_cast<int32_t>(A[row * width + e]) * static_cast<int32_t>(B[e * width + col]);
      }
      out[row * width + col] = sum;
    }
  }
}

int main(int argc, char *argv[]) {
...
	// updating source/destination buffer sizes for comparison and verification
	uint32_t A_buf_size = size_sq * sizeof(int8_t);
  uint32_t B_buf_size = size_sq * sizeof(int8_t);
  uint32_t C_buf_size = size_sq * sizeof(int32_t);
...
	// update all TYPE def expansions to literal int8/int32 accordingly
}
```

Modifying `kernel.cpp` to make use of the new vx_dot8 instruction

```cpp
#include <vx_spawn.h>
#include "common.h"

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
	auto A = reinterpret_cast<int8_t*>(arg->A_addr);
	auto B = reinterpret_cast<int8_t*>(arg->B_addr);
	auto C = reinterpret_cast<int32_t*>(arg->C_addr);
    auto size = arg->size;

    int col = blockIdx.x;
    int row = blockIdx.y;

    int32_t sum(0); //direct (constructor-like) initialization
    //why not just copy initialize 0? user-defined formats support (classes/structs)? 

    for (int e = 0; e < size; e += 4) {
        //packing 4 int8_t from A (row-major)
        uint32_t packedA = 
            ((uint8_t)A[row * size + e]) |
            ((uint8_t)A[row * size + e + 1] << 8) |
            ((uint8_t)A[row * size + e + 2] << 16) |
            ((uint8_t)A[row * size + e + 3] << 24);

        //packing 4 int8_t from B (column-major)
        uint32_t packedB = 
            ((uint8_t)B[e * size + col]) |
            ((uint8_t)B[(e + 1) * size + col] << 8) |
            ((uint8_t)B[(e + 2) * size + col] << 16) |
            ((uint8_t)B[(e + 3) * size + col] << 24);

        sum += vx_dot8(packedA, packedB);
    }

    C[row * size + col] = sum;
}

int main() {
	kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
	return vx_spawn_threads(2, arg->grid_dim, nullptr, (vx_kernel_func_cb)kernel_body, arg);
}
```

## Step 3: Simulation Implementation

Modifying the cycle level simulator  SimX to implement the custom ISA extension

```bash
cd vortex/sim/simx
```

Updating op_string() function in `decode.cpp` to print out new instruction

```cpp
static const char* op_string(const Instr &instr) {
  auto opcode = instr.getOpcode();
...
  switch (opcode) {
  case Opcode::EXT1:
    switch (func7) {
    ...
    case 1:
      switch (func3) {
      case 0:  // DOT8
        return "DOT8";
      default:
        std::abort();
      }
    default:
      std::abort();
    }
...
}
```

Updating Emulator::decode() function to decode new instruction format

```cpp
std::shared_ptr<Instr> Emulator::decode(uint32_t code) const {
...
  switch (iType) {
  case InstType::R:
    switch (op) {
    ...
    case Opcode::EXT1:
      switch (func7) {
      ...
      case 1:
        switch (func3) {
        case 0:  // DOT8
          instr->setDestReg(rd, RegType::Integer);
          instr->addSrcReg(rs1, RegType::Integer);
          instr->addSrcReg(rs2, RegType::Integer);
          break;
        default:
          std::abort();
        }
        break;
      default:
        std::abort();
      }
      break;
    instr->setFunc3(func3);
    instr->setFunc7(func7);
    break;
  ...
  }
 }
```

Updating AluType enum in `types.h` thereby adding new operation emulator can perform

```cpp
enum class AluType {
  ARITH,
  ...
	DOT8
};

inline std::ostream &operator<<(std::ostream &os, const AluType& type) {
  switch (type) {
  ...
  case AluType::DOT8:    os << "DOT8"; break;
  default: assert(false);
  }
  return os;
}
```

Updating Emulator::execute() in `execute.cpp` to implement the actual VX_DOT8 emulation

```cpp
void Emulator::execute(const Instr &instr, uint32_t wid, instr_trace_t *trace) {
  ...
  switch (opcode) {
  ...
  case Opcode::EXT1: {
    switch (func7) {
    ...
    case 1: {
      switch (func3) {
      case 0: { // DOT8
        trace->fu_type = FUType::ALU;
        trace->alu_type = AluType::DOT8;
        trace->src_regs[0] = {RegType::Integer, rsrc0};
        trace->src_regs[1] = {RegType::Integer, rsrc1};
        for (uint32_t t = thread_start; t < num_threads; ++t) {
          if (!warp.tmask.test(t))
            continue;
          // Extract int8 values from 32-bit registers
          int32_t a1 = (rsdata[t][0].i >> 0) & 0xFF;
          int32_t a2 = (rsdata[t][0].i >> 8) & 0xFF;
          int32_t a3 = (rsdata[t][0].i >> 16) & 0xFF;
          int32_t a4 = (rsdata[t][0].i >> 24) & 0xFF;
          
          int32_t b1 = (rsdata[t][1].i >> 0) & 0xFF;
          int32_t b2 = (rsdata[t][1].i >> 8) & 0xFF;
          int32_t b3 = (rsdata[t][1].i >> 16) & 0xFF;
          int32_t b4 = (rsdata[t][1].i >> 24) & 0xFF;
          
          // Calculate dot product
          int32_t result = (a1 * b1) + (a2 * b2) + (a3 * b3) + (a4 * b4);
          
          rddata[t].i = result;
        }
        rd_write = true;
      } break;
      default:
        std::abort();
      }
    } break;
    default:
      std::abort();
    }
  } break;
  ...
}
```

Updating AluUnit::tick() in `func_unit.cpp` to implement the timing of VX_DOT8. Assuming 2 cycles latency for the dot-product execution

```cpp
void AluUnit::tick() {
  for (uint32_t iw = 0; iw < ISSUE_WIDTH; ++iw) {
		...
		int delay = 2;
		switch (trace->alu_type) {
		...
		case AluType::DOT8:
			output.push(trace, 2+delay);
			break;
		default:
			std::abort();
		}
		...
	}
}
```

## Step 4: Testing

```bash
cd vortex/build
source ./ci/toolchain_env.sh
make -s

# single test example
./ci/blackbox.sh --clusters=1 --cores=1 --warps=4 --threads=4 --driver=simx --app=dot8--args="-n 128"
```

Creating Bash script to run all configurations one after the other and log reports

```bash
code run_all_tests.sh
chmod +x run_all_tests.sh
./run_all_tests.sh
```

```bash
#!/bin/bash

# Test runner script for blackbox testing with different configurations
# Runs all 8 combinations of cores, warps, threads, and apps with simx driver and matrix size 128x128

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
    echo "Command: ./ci/blackbox.sh --clusters=1 --cores=$cores --warps=$warps --threads=$threads --driver=simx --app=$app --args=\"-n 128\""
    
    # Log the command being executed
    echo "=== Test $counter/$total - $(date) ===" >> "$log_file"
    echo "Configuration: cores=$cores, warps=$warps, threads=$threads, app=$app" >> "$log_file"
    echo "Command: ./ci/blackbox.sh --clusters=1 --cores=$cores --warps=$warps --threads=$threads --driver=simx --app=$app --args=\"-n 128\"" >> "$log_file"
    echo "" >> "$log_file"
    
    # Execute the command and capture output
    if ./ci/blackbox.sh --clusters=1 --cores="$cores" --warps="$warps" --threads="$threads" --driver=simx --app="$app" --args="-n 128" >> "$log_file" 2>&1; then
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

![run_all_tests_ss.png](Vortex%20Tutorials%20Assignment%205%201e85dbc17d1080deb6ebc4a1a37a4f3a/run_all_tests_ss.png)

### Results Tabulation & Plot

| **Kernel/App** | **Cores** | **Warps** | **Threads** | **IPC** |
| --- | --- | --- | --- | --- |
| sgemmx | 1 | 4 | 4 | 1.222137 |
| dot8 | 1 | 4 | 4 | 1.736676 |
| sgemmx | 4 | 4 | 4 | 3.624350 |
| dot8 | 4 | 4 | 4 | 3.786116 |
| sgemmx | 1 | 16 | 16 | 4.669724 |
| dot8 | 1 | 16 | 16 | 7.599452 |
| sgemmx | 4 | 16 | 16 | 10.276404 |
| dot8 | 4 | 16 | 16 | 9.809991 |

![image.png](Vortex%20Tutorials%20Assignment%205%201e85dbc17d1080deb6ebc4a1a37a4f3a/image.png)

### Inference

The dot8 kernel, utilizing the custom VX_DOT8 instruction, demonstrates a substantial increment in throughput —most notably in single-core configurations— when compared to the sgemmx kernel

Increasing warps/threads per core results in higher relative IPC gains (62.7% with 16W-16T ; 42.1% with 4W-4T)

However, in Multi-Core configurations, there is only a marginal improvement at lower warp/thread counts (4.4% with 4W-4T), and in contrast diminishes at higher warp/thread counts (-4.5% with 16W-16T)

This behavior is likely due to resource contention (memory bandwidth bottleneck). Using coalesced memory access to read/write groups of 4 values from/to the packed registers used by the dot8 instruction could potentially address this issue

___
Note: `git submodule update --init --recursive` to fix toolchain installation errors locally after cloning
