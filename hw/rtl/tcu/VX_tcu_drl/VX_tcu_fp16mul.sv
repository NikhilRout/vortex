// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

`include "VX_define.vh"

module VX_tcu_fp16mul (
    input  wire enable,
    input wire [15:0] a, //fp16 input
    input wire [15:0] b,
    output logic [31:0] y //fp32 output
);

    `UNUSED_VAR(enable);

    //Extract fields from inputs
    wire sign_A = a[15];
    wire sign_B = b[15];
    wire [4:0] exp_A = a[14:10];
    wire [4:0] exp_B = b[14:10];
    wire [9:0] frac_A = a[9:0];
    wire [9:0] frac_B = b[9:0];
    
    //Hidden bits (implied 1)
    wire hidden_A = |exp_A; // 0 if exp_A is all zeros (denormal), 1 otherwise
    wire hidden_B = |exp_B;
    
    //Exception/Special Cases
    wire A_is_zero = (~hidden_A) & (~|frac_A);
    wire B_is_zero = (~hidden_B) & (~|frac_B);
    wire A_is_inf = (&exp_A) & (~|frac_A);
    wire B_is_inf = (&exp_B) & (~|frac_B);
    wire A_is_nan = (&exp_A) & (|frac_A);
    wire B_is_nan = (&exp_B) & (|frac_B);
    
    //Result sign
    wire result_sign = sign_A ^ sign_B;
    
    //Result Mantissa Calculation
    wire [10:0] full_mant_A = {hidden_A, frac_A};
    wire [10:0] full_mant_B = {hidden_B, frac_B};
    wire [21:0] product_mant; // = full_mant_A * full_mant_B;
    VX_tcu_arraymul #(.N(11)) amulfp16(.A(full_mant_A), .B(full_mant_B), .Prod(product_mant));

    //Partial norm for FP32 conversion
    wire normalize_shift = product_mant[21];
    wire [22:0] fp32_mantissa = normalize_shift ? {product_mant[20:0], 2'b00} : {product_mant[19:0], 3'b000};

    //Result Exponent Calculation
    wire [6:0] exp_sum = {2'b0, exp_A} + {2'b0, exp_B} - 7'd15 + {6'd0, normalize_shift};
    wire [7:0] biased_exp = exp_sum + 8'd112;
    wire underflow = exp_sum[6] | (~|exp_sum[5:0]); // Underflow detected --> Negative result or 0
                                                    // Overflow is impossible since inputs are in fp16
    
    //Result selection flags
    wire result_is_nan = A_is_nan | B_is_nan | (A_is_inf & B_is_zero) | (B_is_inf & A_is_zero);
    wire result_is_inf = (A_is_inf & ~B_is_zero) | (B_is_inf & ~A_is_zero);
    wire result_is_zero = underflow | A_is_zero | B_is_zero;
    
    //Final result logic
    logic [7:0] final_exp;
    logic [22:0] final_frac;

    always_comb begin
        case({result_is_nan, result_is_inf, result_is_zero})
            3'b100: begin
                        final_exp = 8'hFF;
                        final_frac = 23'h400000;  //Cannonical NaN
            end
            3'b010: begin
                        final_exp = 8'hFF;
                        final_frac = 23'h000000;  //Infinity
            end
            3'b001: begin
                        final_exp = 8'h00;
                        final_frac = 23'h000000;  //Zero
            end
            default: begin
                        final_exp = biased_exp;
                        final_frac = fp32_mantissa;
            end
        endcase
    end

    assign y = {result_sign, final_exp, final_frac};
endmodule
