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

module VX_tcu_fp32add (
    input  wire enable,
    input wire [31:0] a, //fp32 input
    input wire [31:0] b,
    output logic [31:0] y //fp32 output
);

    `UNUSED_VAR (enable);
    
    //Extract fields from inputs
    wire sign_a = a[31];
    wire sign_b = b[31];
    wire [7:0] exp_a = a[30:23];
    wire [7:0] exp_b = b[30:23];
    wire [22:0] frac_a = a[22:0];
    wire [22:0] frac_b = b[22:0];

    //Hidden bits (implied 1)
    wire hidden_a = |exp_a; // 0 if exp_a is all zeros (denormal), 1 otherwise
    wire hidden_b = |exp_b;

    //Exceptions/Special Cases
    wire a_is_zero = (~hidden_a) & (~|frac_a);
    wire b_is_zero = (~hidden_b) & (~|frac_b);
    wire a_is_inf = (&exp_a) & (~|frac_a);
    wire b_is_inf = (&exp_b) & (~|frac_b);
    wire a_is_nan = (&exp_a) & (|frac_a);
    wire b_is_nan = (&exp_b) & (|frac_b);
    
    //Result selection flags
    wire result_is_nan = a_is_nan | b_is_nan | (a_is_inf & b_is_zero) | (a_is_zero & b_is_inf);
    wire result_is_inf = (a_is_inf & ~b_is_zero) | (b_is_inf & ~a_is_zero);
    wire result_is_zero = a_is_zero & b_is_zero;
    
    //Internal computation variables
    logic sign_s;
    logic [7:0] exp_diff, exp_s, exp_ss;
    logic [22:0] mantissa_ss;
    logic [23:0] mantissa_a, mantissa_b;
    logic [24:0] mantissa_s;

    always_comb begin
        mantissa_a = {hidden_a, frac_a};
        mantissa_b = {hidden_b, frac_b};

        //Align mantissas based on exponents
        if (exp_a >= exp_b) begin
            exp_diff = exp_a - exp_b;
            mantissa_b = mantissa_b >> exp_diff;
            exp_s = exp_a;
        end else begin
            exp_diff = exp_b - exp_a;
            mantissa_a = mantissa_a >> exp_diff;
            exp_s = exp_b;
        end

        //Perform addition or subtraction based on sign
        if (sign_a ~^ sign_b) begin
            mantissa_s = mantissa_a + mantissa_b;
            sign_s = sign_a;
        end else begin
            if (mantissa_a > mantissa_b) begin
                mantissa_s = mantissa_a - mantissa_b;
                sign_s = sign_a;
            end else begin
                mantissa_s = mantissa_b - mantissa_a;
                sign_s = sign_b;
            end            
        end

        //Normalization
        if (mantissa_s[24]) begin
            exp_s = exp_s + 1'b1;
            mantissa_s = mantissa_s >> 1;
        end

        //Final result selection
        case({result_is_nan, result_is_inf, result_is_zero})
            3'b100: begin
                        exp_ss = 8'hFF;
                        mantissa_ss = 23'h400000;  //NaN
            end
            3'b010: begin
                        exp_ss = 8'hFF;
                        mantissa_ss = 23'h000000;  //Infinity
            end
            3'b001: begin
                        exp_ss = 8'h00;
                        mantissa_ss = 23'h000000;  //Zero
            end
            default: begin
                        exp_ss = exp_s;
                        mantissa_ss = mantissa_s[22:0];
            end
        endcase
    end

    assign y = {sign_s, exp_ss, mantissa_ss};
endmodule
