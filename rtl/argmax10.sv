// rtl/argmax10.sv
//
// Lab 3 skeleton. See docs/lab3_rtl_datapath.md's "tie-break bit-exactness"
// note before writing this -- your comparator's tie-break rule MUST match
// Python's max(range(10), key=lambda k: scores[k]) (smallest index wins on a
// tie), or Lab 4's bit-exact comparison will show mismatches on tied inputs
// that have nothing to do with a real datapath bug.

module argmax10 (
    input  logic clk,
    input  logic rst,
    input  logic en,

    input  logic signed [31:0] scores_in [0:9],   // raw FC2 accumulators (acc2) -- NOT requantized, see architecture_spec.md Section 2

    output logic [3:0] pred_digit,
    output logic        valid_out
);

    // TODO: comparator tree (or straightforward sequential/combinational
    // scan -- a 10-input comparison isn't large enough to need a deep
    // pipelined tree, but architecture_spec.md Section 4 budgets ~5 cycles
    // for it, so a small amount of pipelining is reasonable if it helps
    // timing in Lab 5).
    //
    // Tie-break rule: only update the running max/index on a STRICT
    // improvement (>), never on equality -- this keeps the first (smallest
    // index) tied maximum as the winner, matching golden_model.py.

endmodule
