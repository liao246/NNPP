// rtl/fc_layer.sv
//
// Lab 3 skeleton. See docs/lab3_rtl_datapath.md before writing any logic here.
//
// Generic, reusable layer engine built from LANES parallel mac_lane
// instances. Reused twice in digit_top.sv: once parameterized for FC1
// (N_IN=196, N_OUT=64) and once for FC2 (N_IN=64, N_OUT=10) -- this reuse is
// the entire reason this module is generic instead of two separate
// hand-written fc1_layer/fc2_layer modules (which is what the old version of
// this project did).
//
// This module does NOT own the activation source memory -- it PULLS one
// value per cycle via act_addr and expects act_data back (with whatever
// latency your memory read takes -- account for it in the batch FSM, see
// lab3.md's BRAM-latency note). The caller (digit_top) is responsible for
// routing act_addr/act_data to the image ROM (FC1) or to the FC1 output
// register bank (FC2).

module fc_layer #(
    parameter int    N_IN   = 196,
    parameter int    N_OUT  = 64,
    parameter int    LANES  = 16,
    parameter string BIAS_INIT_FILE = ""   // TODO: int32 bias ROM for this layer, N_OUT entries
    // TODO: decide how each internal mac_lane gets its own per-lane weight
    // init file (a per-instance file-name parameter array? a generate loop
    // with a naming convention? your call -- see lab3.md's "weight file init
    // is an open decision" note)
)(
    input  logic clk,
    input  logic rst,
    input  logic start,

    // Activation source interface (pull-based; caller supplies act_data for
    // the act_addr presented, with its own read latency -- see lab3.md)
    output logic [$clog2(N_IN)-1:0] act_addr,
    input  logic [8:0]              act_data,

    output logic busy,
    output logic done,

    output logic signed [31:0] acc_out [0:N_OUT-1]
);

    localparam int NUM_BATCHES = (N_OUT + LANES - 1) / LANES;  // ceil(N_OUT/LANES); FC2's last (only) batch is partial: 10 of 16 lanes valid

    // TODO: instantiate LANES x mac_lane, all sharing one broadcast
    // weight_addr and one broadcast act_data (per lab3.md's addressing
    // background) but each with its own WEIGHT_INIT_FILE.

    // TODO: bias ROM (N_OUT x int32), loaded from BIAS_INIT_FILE.

    // TODO: batch FSM:
    //   for batch in 0..NUM_BATCHES-1:
    //     assert acc_clear on all lanes for one cycle, loading each lane's
    //       accumulator with THIS BATCH's neuron's bias (bias_addr = batch*LANES + lane_id,
    //       only valid for lane_id < N_OUT - batch*LANES on a partial final batch)
    //     for i in 0..N_IN-1:
    //       drive act_addr = i, wait for act_data's read latency, drive
    //       weight_addr for this lane's local memory accordingly, assert
    //       acc_en for one MAC cycle
    //     capture each active lane's acc_out into acc_out[batch*LANES + lane_id]
    //   assert done once all batches complete; busy high throughout.
    //
    // Handle the FC2 partial-batch case explicitly: lanes 10-15 during the
    // (only) FC2 batch are still running the same control sequence, but
    // their acc_out values must not be written into the N_OUT=10-sized
    // acc_out array -- gate the capture step by lane validity, not just by
    // batch index.

endmodule
