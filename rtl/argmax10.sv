// rtl/argmax10.sv
//
// Find the class with the highest signed score.
// If multiple classes tie, choose the smallest class index.

module argmax10 (
    input  logic clk,
    input  logic rst,
    input  logic en,

    input  logic signed [31:0] scores_in [0:9],   // raw FC2 accumulators (acc2) -- NOT requantized, see architecture_spec.md Section 2

    output logic [3:0] pred_digit,
    output logic        valid_out
);

    logic signed [31:0] highest_score;
    logic [3:0] winning_digit;
    logic [3:0] next_pred_digit;

    // Compare all ten scores using combinational logic.
    // Start with class 0, not a score of zero: all scores might be negative.
    always_comb begin
        highest_score = scores_in[0];
        winning_digit = 4'd0;

        // This loop builds comparison hardware; it does not take nine clocks.
        for (int i = 1; i < 10; i++) begin
            if (scores_in[i] > highest_score) begin
                highest_score = scores_in[i];
                winning_digit = i[3:0];  // Class indices 0 through 9 fit in 4 bits.
            end
            // Equal scores leave the earlier (smaller) class index unchanged.
        end
    end

    // Keep the previous result unless a new comparison is requested.
    always_comb begin
        next_pred_digit = pred_digit;
        if (en) begin
            next_pred_digit = winning_digit;
        end
    end

    // On an enabled clock edge, register the result and assert valid_out.
    // digit_top pulses en for one cycle, so valid_out is also a one-cycle pulse.
    always_ff @(posedge clk) begin
        if (rst) begin
            pred_digit <= 4'd0;
            valid_out <= 1'b0;
        end else begin
            pred_digit <= next_pred_digit;
            valid_out <= en;
        end
    end

endmodule
