module requant_relu #(
    parameter int SHIFT1 = 11   // TODO: replace with YOUR calibrated value from Lab 1 (data/meta.json), not this placeholder
)(
    input logic signed [31:0] acc_out,
    output logic signed [7:0] requant_out
);

    logic signed [31:0] acc1_round, acc1_shift;
    logic signed [7:0] acc1_clamp, a1_comb;
    always_comb begin
        acc1_round = acc_out + (1 <<< (SHIFT1 - 1));
        acc1_shift = acc1_round >>> SHIFT1;
        acc1_clamp = (acc1_shift > 127) ? 127 : (acc1_shift < -128) ? -128 : acc1_shift;
        a1_comb = (acc1_clamp < 0) ? 0 : acc1_clamp;    
    end
    assign requant_out = a1_comb;
endmodule
