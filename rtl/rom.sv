module rom #(
    parameter int DATA_W = 8,
    parameter int DEPTH  = 3920,
    parameter string INIT_FILE = "",
    parameter bit REGISTER_OUTPUT = 1'b1
)(
    input  logic clk,
    input  logic rst,
    input  logic [((DEPTH > 1) ? $clog2(DEPTH) : 1)-1:0] addr,

    output logic [DATA_W-1:0] data_out
);

    logic [DATA_W-1:0] memory [0:DEPTH-1];
    logic [DATA_W-1:0] next_data_out;

    // Load ROM contents at initialization.
    initial begin
        if (INIT_FILE != "")
            $readmemh(INIT_FILE, memory);
    end

    // Address selects the next output value.
    always_comb begin
        next_data_out = memory[addr];
    end

    generate
        if (REGISTER_OUTPUT) begin : registered_read
            // One-cycle synchronous read.
            always_ff @(posedge clk) begin
                if (rst)
                    data_out <= '0;
                else
                    data_out <= next_data_out;
            end
        end else begin : combinational_read
            // No register or reset behavior, matching a combinational lookup.
            assign data_out = next_data_out;
        end
    endgenerate

endmodule
