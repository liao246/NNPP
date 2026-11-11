// rtl/counter.sv
//
// Generic up-counter: synchronous clear, enable, and a `done` flag that
// asserts at MAX_COUNT. Holds at MAX_COUNT rather than wrapping, so `done`
// stays high until `clear`.
//
// Instantiated twice in fc_layer -- once for the per-batch input stream
// (MAX_COUNT = N_IN+1) and once for the batch index (MAX_COUNT = NUM_BATCHES-1).

module counter #(
    parameter int MAX_COUNT = 1,
    // MAX_COUNT = 0 (FC2's single batch) would make $clog2(1) = 0, which is
    // an illegal zero-width signal -- hence the guard.
    parameter int WIDTH     = (MAX_COUNT > 0) ? $clog2(MAX_COUNT + 1) : 1
)(
    input  logic clk,
    input  logic rst,
    input  logic clear,
    input  logic en,

    output logic [WIDTH-1:0] count,
    output logic             done
);

    logic [WIDTH-1:0] next_count;

    always_comb begin
        if (clear)            next_count = '0;
        else if (en && !done) next_count = count + 1'b1;
        else                  next_count = count;
    end

    always_ff @(posedge clk) begin
        if (rst) count <= '0;
        else     count <= next_count;
    end

    assign done = (count == MAX_COUNT);

endmodule
