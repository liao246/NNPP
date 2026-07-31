// rtl/requant_relu.sv
//
// Lab 3 skeleton. See docs/architecture_spec.md Section 2 ("FC1 requantization")
// and sw/golden_model.py's fc1_forward_int() -- this module MUST implement the
// exact same 5 steps, in the exact same order, bit-for-bit. This is the piece
// the old version of this project never built at all.
//
// Applied once per FC1 neuron (instantiate as many as you decide in
// digit_top.sv -- see the "requant-bank parallelism" decision in
// docs/lab3_rtl_datapath.md).

module requant_relu #(
    parameter int SHIFT1 = 11   // TODO: replace with YOUR calibrated value from Lab 1 (data/meta.json), not this placeholder
)(
    input  logic clk,
    input  logic rst,
    input  logic en,

    input  logic signed [31:0] acc1_in,   // raw FC1 accumulator (post bias, pre-shift) -- step 1 output

    output logic signed [7:0]  a1_out,    // final FC1 activation -- step 5 output
    output logic                valid_out
);

    // TODO, exactly matching architecture_spec.md Section 2 / golden_model.py:
    //   1. acc1        = acc1_in                                   (already computed upstream by mac_lane/fc_layer)
    //   2. acc1_round  = acc1 + (1 <<< (SHIFT1 - 1))                 // round-half-up, int32
    //   3. acc1_shift  = acc1_round >>> SHIFT1                       // ARITHMETIC right shift (signed)
    //   4. acc1_clamp  = clamp(acc1_shift, -128, 127)                // saturate to int8 range
    //   5. a1          = (acc1_clamp < 0) ? 0 : acc1_clamp           // ReLU
    //
    // Pipeline this across a few registered stages if it helps timing closure
    // (Lab 5) -- but keep the arithmetic identical to golden_model.py either way.
    // valid_out should track en with whatever latency you introduce.

endmodule
