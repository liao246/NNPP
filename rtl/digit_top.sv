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
    output logic       correct
);

    // State and signal declarations

    typedef enum logic [3:0] {
        IDLE,
        FC1_START,
        FC1_WAIT,
        REQUANT_CAPTURE,
        FC2_START,
        FC2_WAIT,
        ARGMAX_START,
        ARGMAX_WAIT,
        DONE
    } state_t;

    // FSM control signals
    state_t state, next_state;
    logic fc1_start, fc1_done, fc2_start, fc2_done, requant_en, argmax_en;

    // fc_layer signals
    logic [$clog2(N_IN)-1:0] act_addr1;
    logic [$clog2(HIDDEN)-1:0] act_addr2;
    logic [8:0] act_data, act_data_delay, act_data_delay_r;
    logic [7:0] image_data;
    logic fc1_busy, fc2_busy;
    logic signed [31:0] acc_out [0:HIDDEN-1], acc_out2 [0:N_CLASS-1];

    // requant_relu signals
    logic signed [7:0] requant_out [0:HIDDEN-1];
    logic [HIDDEN*8-1:0] requant_r, next_requant_r;

    logic [$clog2(N_IN*NTEST)-1:0] image_addr;
    logic [$clog2(NTEST)-1:0] img_idx_r, next_img_idx;

    logic [3:0] predicted_digit;
    logic [3:0] next_pred_digit;
    logic argmax_valid_out;
    logic next_correct;
    logic [3:0] label_rom_out;
    logic correct_r;

    // Choose the next FSM state.
    always_comb begin
        case (state)
            IDLE: begin
                if (start) next_state = FC1_START;
                else next_state = IDLE;
            end
            FC1_START: begin
                next_state = FC1_WAIT;
            end
            FC1_WAIT: begin
                if (fc1_done) next_state = REQUANT_CAPTURE;
                else next_state = FC1_WAIT;
            end
            REQUANT_CAPTURE: begin
                next_state = FC2_START;
            end
            FC2_START: begin
                next_state = FC2_WAIT;
            end
            FC2_WAIT: begin
                if (fc2_done) next_state = ARGMAX_START;
                else next_state = FC2_WAIT;
            end
            ARGMAX_START: begin
                next_state = ARGMAX_WAIT;
            end
            ARGMAX_WAIT: begin
                if (argmax_valid_out) next_state = DONE;
                else next_state = ARGMAX_WAIT;
            end
            DONE: begin
                if (start) next_state = FC1_START;
                else next_state = DONE;  // hold done until the next request
            end
            default: next_state = IDLE;
        endcase
    end

    // Store the current FSM state.
    always_ff @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
        end else begin
            state <= next_state;
        end
    end

    // Generate control signals from the current state.
    always_comb begin
        busy = 0;
        done = 0;
        fc1_start = 0;
        fc2_start = 0;
        requant_en = 0;
        argmax_en = 0;

        case (state)
            IDLE: begin
                busy = 0;
                done = 0;
            end
            FC1_START: begin
                busy = 1;
                fc1_start = 1;
            end
            FC1_WAIT: begin
                busy = 1;
            end
            REQUANT_CAPTURE: begin
                busy = 1;
                requant_en = 1;
            end
            FC2_START: begin
                busy = 1;
                fc2_start = 1;
            end
            FC2_WAIT: begin
                busy = 1;
            end
            ARGMAX_START: begin
                busy = 1;
                argmax_en = 1;
            end
            ARGMAX_WAIT: begin
                busy = 1;
            end
            DONE: begin
                busy = 0;
                done = 1;
            end
            default: begin
                busy = 0;
                done = 0;
            end
        endcase
    end

    // Image selection and image ROM
    always_comb begin
        next_img_idx = img_idx_r;
        if (fc1_start) begin
            // With 20 images, indices above 19 select the last image.
            if (img_idx > NTEST-1) begin
                next_img_idx = NTEST-1;
            end else begin
                next_img_idx = img_idx;
            end
        end
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            img_idx_r <= '0;
        end else begin
            img_idx_r <= next_img_idx;
        end
    end

    assign image_addr = img_idx_r * N_IN + act_addr1;
    assign act_data = {1'b0, image_data};

    rom #(
        .DATA_W(8),
        .DEPTH(N_IN * NTEST),
        .INIT_FILE(IMAGE_INIT_FILE)
    ) image_rom (
        .clk(clk),
        .rst(rst),
        .addr(image_addr),
        .data_out(image_data)
    );

    // FC1: image pixels to hidden-layer accumulators
    fc_layer #(
        .N_IN(N_IN),
        .N_OUT(HIDDEN),
        .LANES(LANES),
        .FILE_PREFIX("fc1")
    ) fc_layer1 (
        .clk(clk),
        .rst(rst),
        .start(fc1_start),
        .act_addr(act_addr1),
        .act_data(act_data),
        .busy(fc1_busy),
        .done(fc1_done),
        .acc_out(acc_out)
    );

    // Requantization: convert and store the hidden-layer activations
    genvar i;

    generate
        for (i = 0; i < HIDDEN; i++) begin : requant_relu_gen
            requant_relu #(
                .SHIFT1(11)
            ) requant_relu_inst (
                .acc_out(acc_out[i]),
                .requant_out(requant_out[i])
            );
        end
    endgenerate

    // The requant bank is combinational; capture all activations once.
    always_comb begin
        next_requant_r = requant_r;
        if (requant_en) begin
            for (int i = 0; i < HIDDEN; i++) begin
                next_requant_r[i*8 +: 8] = requant_out[i];
            end
        end
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            requant_r <= '0;
        end else begin
            requant_r <= next_requant_r;
        end
    end

    // FC2 input mux: select one byte, then delay it by one clock.
    // [act_addr2*8 +: 8] selects the eight bits for activation act_addr2.
    assign act_data_delay = {1'b0, requant_r[act_addr2*8 +: 8]};

    always_ff @(posedge clk) begin
        if (rst) begin
            act_data_delay_r <= 9'b0;
        end else begin
            act_data_delay_r <= act_data_delay;
        end
    end

    // FC2: hidden-layer activations to ten class scores
    fc_layer #(
        .N_IN(HIDDEN),
        .N_OUT(N_CLASS),
        .LANES(LANES),
        .FILE_PREFIX("fc2")
    ) fc_layer2 (
        .clk(clk),
        .rst(rst),
        .start(fc2_start),
        .act_addr(act_addr2),
        .act_data(act_data_delay_r),
        .busy(fc2_busy),
        .done(fc2_done),
        .acc_out(acc_out2)
    );

    // Argmax: select the class with the highest score
    argmax10 argmax_inst (
        .clk(clk),
        .rst(rst),
        .en(argmax_en),
        .scores_in(acc_out2),
        .pred_digit(predicted_digit),
        .valid_out(argmax_valid_out)
    );

    // Label ROM and comparison against the predicted class
    rom #(
        .DATA_W(4),
        .DEPTH(NTEST),
        .INIT_FILE(LABEL_INIT_FILE)
    ) label_rom (
        .clk(clk),
        .rst(rst),
        .addr(img_idx_r),
        .data_out(label_rom_out)
    );

    always_comb begin
        if (predicted_digit == label_rom_out) begin
            correct_r = 1'b1;
        end else begin
            correct_r = 1'b0;
        end
    end

    // Output registers: update both results when argmax finishes.
    always_comb begin
        next_pred_digit = pred_digit;  // Hold the previous result.
        if (argmax_valid_out) begin
            next_pred_digit = predicted_digit;
        end
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            pred_digit <= 4'b0;
        end else begin
            pred_digit <= next_pred_digit;
        end
    end

    always_comb begin
        next_correct = correct;  // Hold the previous result.
        if (argmax_valid_out) begin
            next_correct = correct_r;
        end
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            correct <= 1'b0;
        end else begin
            correct <= next_correct;
        end
    end

endmodule
