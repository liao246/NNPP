module mac_lane #(
    parameter int    LOCAL_MEM_DEPTH   = 1088,  // TODO: confirm sizing against your FC1/FC2 region layout (architecture_spec.md Section 3)
    parameter string WEIGHT_INIT_FILE  = "",     // TODO: this lane's per-lane init file -- see the "weight file init is an open decision" note in lab3
    parameter string BIAS_INIT_FILE = "",
    parameter int BIAS_DEPTH = 4
)(
    input  logic clk,
    input  logic rst,

    // Shared/broadcast address + activation (identical across all 16 lanes each cycle)
    input  logic [$clog2(LOCAL_MEM_DEPTH)-1:0] weight_addr,
    input  logic [8:0]                          act_data,   // zero-extended: uint8 pixel (0..255) or FC1 a1 (0..127) -- see lab3 "non-negative operands" note

    // Accumulator control
    input  logic                acc_clear,   // 1 = synchronously load acc with bias_in instead of accumulating
    input logic [((BIAS_DEPTH > 1) ? $clog2(BIAS_DEPTH) : 1)-1:0] bias_addr,
    input  logic                acc_en,      // 1 = perform one multiply-accumulate this cycle

    output logic signed [31:0]  acc_out
);

    // weight ROM
    logic [$clog2(LOCAL_MEM_DEPTH)-1:0] weight_addr_r;
    always_ff @(posedge clk) begin
        if (rst) begin
            weight_addr_r <= 0;
        end else begin
            weight_addr_r <= weight_addr;
        end
    end

    logic signed [7:0] weight_data;
    rom #(
        .DATA_W(8),
        .DEPTH(LOCAL_MEM_DEPTH),
        .INIT_FILE(WEIGHT_INIT_FILE)
    ) weight_rom (
        .clk(clk),
        .rst(rst),
        .addr(weight_addr_r),
        .data_out(weight_data)
    );

    // Bias ROM -- LUT
    logic signed [31:0] bias_in;
    rom #(
        .DATA_W(32),
        .DEPTH(BIAS_DEPTH),
        .INIT_FILE(BIAS_INIT_FILE),
        .REGISTER_OUTPUT(1'b0)
    ) bias_rom (
        .clk(clk),
        .rst(rst),
        .addr(bias_addr),
        .data_out(bias_in)
    );


    logic signed [31:0] next_acc_out;
    logic signed [8:0] act_data_r;
    always_ff @(posedge clk) begin
        if (rst) begin
            acc_out <= 0;
            act_data_r <= 0;
        end else begin
            acc_out <= next_acc_out;
            act_data_r <= act_data;
        end
    end

    always_comb begin
        if (acc_clear) begin
            next_acc_out = bias_in;
        end else if (acc_en) begin
            next_acc_out = acc_out + (weight_data * act_data_r);
        end else begin
            next_acc_out = acc_out;
        end
    end

endmodule
