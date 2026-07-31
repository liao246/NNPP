// rtl/digit_top.sv
//
// Lab 3 skeleton. See docs/architecture_spec.md Section 5 for the exact port
// list/board-I/O mapping, and docs/lab3_rtl_datapath.md for how the pieces
// below fit together.
//
// Orchestrates: image ROM -> fc_layer(FC1) -> requant_relu bank -> fc_layer(FC2)
// -> argmax10 -> board-facing outputs.
//
// NOTE: this module does NOT debounce start/rst -- it assumes clean,
// single-cycle inputs. A board-level wrapper adding debounce logic is built
// in Lab 6, not here.

module digit_top #(
    parameter int N_IN     = 196,   // pixels per (pooled) image
    parameter int HIDDEN   = 64,
    parameter int N_CLASS  = 10,
    parameter int NTEST    = 20,    // stored test images
    parameter int LANES    = 16,
    parameter string IMAGE_INIT_FILE = "",   // TODO: 20 x 196 pixel ROM, see architecture_spec.md Section 3
    parameter string LABEL_INIT_FILE = ""    // TODO: 20 x 4-bit true-label ROM (for the `correct` self-check output)
)(
    input  logic clk,
    input  logic rst,
    input  logic start,
    input  logic [$clog2(NTEST)-1:0] img_idx,

    output logic busy,
    output logic done,
    output logic [3:0] pred_digit,
    output logic       correct,
    output logic [15:0] cycle_count,

    // Simulation/ILA-only debug port -- does not need to reach a real board I/O pin
    output logic signed [N_CLASS*32-1:0] score_debug
);

    // TODO: top-level FSM: IDLE -> FC1 -> REQUANT -> FC2 -> ARGMAX -> DONE_S -> IDLE
    // (same shape as the old project's FSM, but now FC1/FC2 are the
    // parallel fc_layer engine, not a serial one-MAC-per-cycle loop, and
    // REQUANT is a real state now instead of being skipped entirely).

    // TODO: image ROM (NTEST x N_IN, uint8, from IMAGE_INIT_FILE) -- feed
    // fc_layer #1's act_addr/act_data pull interface, offset by img_idx*N_IN.

    // TODO: label ROM (NTEST x 4-bit, from LABEL_INIT_FILE) -- compare
    // against pred_digit to drive `correct`.

    // TODO: fc_layer #1 instance, parameterized N_IN=N_IN, N_OUT=HIDDEN, LANES=LANES.
    //   (fc_layer #1 example instantiation -- fill in the weight/bias file
    //   parameters and wire up start/busy/done/act_addr/act_data per its port list)
    //
    // fc_layer #(
    //     .N_IN  (N_IN),
    //     .N_OUT (HIDDEN),
    //     .LANES (LANES)
    //     // TODO: weight/bias init file parameters
    // ) fc1 (
    //     .clk      (clk),
    //     .rst      (rst),
    //     .start    (/* TODO: from top FSM */),
    //     .act_addr (/* TODO */),
    //     .act_data (/* TODO: from image ROM */),
    //     .busy     (/* TODO */),
    //     .done     (/* TODO */),
    //     .acc_out  (/* TODO: 64 raw int32 FC1 accumulators */)
    // );

    // TODO: requant_relu bank -- decide how many instances (see lab3.md's
    // "requant-bank parallelism" decision), feed each fc1's raw acc_out
    // values, capture the resulting 64 int8 a1 values into a register bank
    // that fc_layer #2 will pull from as its activation source.

    // TODO: fc_layer #2 instance, parameterized N_IN=HIDDEN, N_OUT=N_CLASS,
    // LANES=LANES, act_data sourced from the a1 register bank above instead
    // of the image ROM.

    // TODO: argmax10 instance, fed fc_layer #2's raw acc_out (NOT
    // requantized, per architecture_spec.md Section 2) -> pred_digit.

    // TODO: cycle_count -- free-running counter that starts at `start` and
    // latches its value at `done` (see architecture_spec.md Section 4's
    // "practical demo note" -- 9us is too fast to see, this counter is the
    // actual demo/debug readout).

    // TODO: score_debug -- pack fc_layer #2's raw acc_out into this flat bus
    // for simulation/ILA visibility.

endmodule
