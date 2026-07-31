// rtl/mac_lane.sv
//
// Lab 3 skeleton. See docs/lab3_rtl_datapath.md before writing any logic here.
//
// One MAC lane: local weight memory (this lane's slice of a weight matrix,
// FC1 region + FC2 region packed into one address space, see
// docs/architecture_spec.md Section 3) + a DSP48E1-inferring multiply-
// accumulate (docs/lab2_fpga_concepts.md's MACC coding pattern).
//
// weight_addr and act_data are SHARED/BROADCAST signals driven identically
// to every mac_lane instance by fc_layer -- see docs/lab3_rtl_datapath.md's
// "Background: how the batch/addressing scheme actually works" section for
// why that's correct despite each lane computing a different neuron.

module mac_lane #(
    parameter int    LOCAL_MEM_DEPTH   = 1088,  // TODO: confirm sizing against your FC1/FC2 region layout (architecture_spec.md Section 3)
    parameter string WEIGHT_INIT_FILE  = ""     // TODO: this lane's per-lane init file -- see the "weight file init is an open decision" note in lab3
)(
    input  logic clk,
    input  logic rst,

    // Shared/broadcast address + activation (identical across all 16 lanes each cycle)
    input  logic [$clog2(LOCAL_MEM_DEPTH)-1:0] weight_addr,
    input  logic [8:0]                          act_data,   // zero-extended: uint8 pixel (0..255) or FC1 a1 (0..127) -- see lab3 "non-negative operands" note

    // Accumulator control
    input  logic                acc_clear,   // 1 = synchronously load acc with bias_in instead of accumulating
    input  logic signed [31:0]  bias_in,
    input  logic                acc_en,      // 1 = perform one multiply-accumulate this cycle

    output logic signed [31:0]  acc_out
);

    // TODO: declare the local weight memory array and preload it with
    // WEIGHT_INIT_FILE via $readmemh in an initial block. Register the read
    // address (weight_addr) so Vivado infers a real BRAM primitive rather
    // than fabric LUTs -- see docs/lab2_fpga_concepts.md's BRAM inference
    // section. Remember: a registered-address read returns data ONE cycle
    // later, not the same cycle -- fc_layer's FSM must be built assuming
    // that latency (see lab3.md).

    // TODO: implement the DSP48E1 MACC-style accumulate:
    //   - register weight_addr's resulting weight value and act_data (the
    //     DSP48E1's registered A/B inputs)
    //   - on acc_clear, load acc_out <= bias_in instead of accumulating
    //   - on acc_en (and not acc_clear), acc_out <= acc_out + weight * act_data
    //   - weight is signed (int8, from local memory); act_data must be
    //     ZERO-extended to match the multiplier width, not sign-extended
    //     (see the "non-negative operands" note in docs/lab3_rtl_datapath.md)

endmodule
