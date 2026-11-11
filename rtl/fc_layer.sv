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
    // Each lane loads "<FILE_PREFIX>_lane<i>_weights.hex" and
    // "<FILE_PREFIX>_lane<i>_bias.hex". digit_top passes "fc1" / "fc2" so the
    // two instances of this module don't load each other's files.
    parameter string FILE_PREFIX = ""
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

    localparam int BATCH_W = (NUM_BATCHES > 1) ? $clog2(NUM_BATCHES) : 1;

    logic [$clog2(N_IN + 2)-1:0]           weight_count;
    logic [BATCH_W-1:0]                    batch;
    logic [BATCH_W-1:0]                    bias_addr;
    logic [$clog2(NUM_BATCHES * N_IN)-1:0] weight_addr;
    logic weights_done, batches_done;

    // fsm
    typedef enum logic [2:0] {
        IDLE,
        CLEAR,
        FILL0,
        FILL1,
        ACCUMULATE,
        CAPTURE,
        DONE
    } state_t;

    state_t state, next_state;

    logic acc_clear, acc_en, valid_out, batch_clr, count_en;

    // raw per-lane accumulator outputs (transient, LANES wide) vs. the
    // captured result array (persistent, N_OUT wide)
    logic signed [31:0] mac_lane_out [0:LANES-1];
    logic signed [31:0] next_acc_out [0:N_OUT-1];

    assign bias_addr = batch;


    genvar i;

    generate
        for (i = 0; i < LANES; i++) begin : mac_lane_gen
            mac_lane #(
                .LOCAL_MEM_DEPTH(NUM_BATCHES * N_IN),
                .WEIGHT_INIT_FILE($sformatf("%s_lane%0d_weights.hex", FILE_PREFIX, i)),
                .BIAS_INIT_FILE  ($sformatf("%s_lane%0d_bias.hex",    FILE_PREFIX, i)),
                .BIAS_DEPTH(NUM_BATCHES)
            ) mac_lane_inst (
                .clk(clk),
                .rst(rst),
                .weight_addr(weight_addr),
                .act_data(act_data),
                .acc_clear(acc_clear),
                .bias_addr(bias_addr),
                .acc_en(acc_en),
                .acc_out(mac_lane_out[i])
            );
        end
    endgenerate

    always_comb begin
        next_acc_out = acc_out;                          // default: hold all N_OUT
        for (int lane = 0; lane < LANES; lane++) begin
            if (valid_out && (batch*LANES + lane < N_OUT))
                next_acc_out[batch*LANES + lane] = mac_lane_out[lane];
        end
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            for (int k = 0; k < N_OUT; k++) acc_out[k] <= '0;
        end else begin
            acc_out <= next_acc_out;
        end
    end


    always_comb begin
        acc_clear = 0;
        acc_en    = 0;
        valid_out = 0;
        done      = 0;
        busy      = 0;
        batch_clr = 0;
        count_en  = 0;

        case (state)
            IDLE: begin
                if (start) next_state = CLEAR;
                else next_state = IDLE;
            end
            CLEAR: begin
                next_state = FILL0;
            end
            FILL0: begin
                next_state = FILL1;
            end
            FILL1: begin
                next_state = ACCUMULATE;
            end
            ACCUMULATE: begin
                if (weights_done) next_state = CAPTURE;
                else next_state = ACCUMULATE;
            end
            CAPTURE: begin
                if (batches_done) next_state = DONE;
                else next_state = CLEAR;
            end
            DONE: begin
                next_state = IDLE;
            end
            default: next_state = IDLE;
        endcase
        
        case (state)
            IDLE: begin
                acc_clear = 0;
                acc_en = 0;
                valid_out = 0;
                done = 0;
                busy = 0;
                batch_clr = 1;
                count_en = 0;
            end
            CLEAR: begin
                acc_clear = 1;
                busy = 1;
            end
            FILL0: begin
                busy = 1;
                count_en = 1;
            end
            FILL1: begin
                busy = 1;
                count_en = 1;
            end
            ACCUMULATE: begin
                busy = 1;
                acc_en = 1;
                count_en = 1;
            end
            CAPTURE: begin
                busy = 1;
                valid_out = 1;
            end
            DONE: begin
                busy = 0;
                done = 1;
            end
            default: begin
                acc_clear = 0;
                acc_en = 0;
                valid_out = 0;
                done = 0;
                busy = 0;
                batch_clr = 1;
                count_en = 0;
            end
        endcase
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
        end else begin
            state <= next_state;
        end
    end


    // counters:

    counter #(.MAX_COUNT(N_IN + 1)) weights_counter (
    .clk(clk), .rst(rst),
    .clear(acc_clear), .en(count_en),
    .count(weight_count), .done(weights_done)
    );

    counter #(.MAX_COUNT(NUM_BATCHES - 1)) batch_counter (
        .clk(clk), .rst(rst),
        .clear(batch_clr), .en(valid_out),
        .count(batch), .done(batches_done)
    );



    always_comb begin
        if (weight_count > N_IN-1) begin
            act_addr = N_IN-1;
        end else begin
            act_addr = weight_count;
        end
    end

    always_comb begin
        weight_addr = batch * N_IN + act_addr;
    end


endmodule
