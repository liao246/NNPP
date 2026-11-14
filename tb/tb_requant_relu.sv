`timescale 1ns/1ps

module tb_requant_relu;

    // Match the shift used by the design.
    localparam int SHIFT1 = 11;
    localparam int SCALE = 2 ** SHIFT1;

    // Signals connected to the module being tested.
    logic signed [31:0] acc_out;
    logic signed [7:0] requant_out;
 
    // DUT means "design under test."
    requant_relu #(
        .SHIFT1(SHIFT1)
    ) dut (
        .acc_out(acc_out),
        .requant_out(requant_out)
    );



    task automatic check_case(input logic signed [31:0] input_value);
        longint signed expected;
        longint signed scale;
        longint signed remainder;

        begin
            acc_out = input_value;

            scale = 2 ** SHIFT1;

            // Calculate the expected output based on the requantization formula.
            if (input_value < 0) begin
                expected = 0;
            end else begin
                expected = input_value / scale;
                remainder = input_value % scale;

                if (remainder >= (scale / 2)) begin
                    expected = expected + 1;
                end

                if (expected > 127) expected = 127;
            end

            #1;  // Allow the output to settle.

            if (requant_out !== expected) begin
                $error("Test failed: input=%0d, expected=%0d, actual=%0d", acc_out, expected, requant_out);
            end else begin
                $display("Test passed: input=%0d, output=%0d", acc_out, requant_out);
            end
        end
    endtask

    // No clock or reset is needed: requant_relu is combinational.
    initial begin
        acc_out = 32'sd0;
        #1;  // Allow the output to settle before reading it.

        $display("Input = %0d, output = %0d", acc_out, requant_out);


        check_case(32'sd0);          // Test zero input
        check_case(32'sd2048); // testing exact scale value (2048 / 2048 = 1)
        check_case(-32'sd4096); // Test negative input
        check_case(32'sd1023); // Just below the halfway threshold: rounds to 0
        check_case(32'sd1024); // Exactly halfway between 0 and 1: rounds to 1
        check_case(32'sd1025); // Just above the halfway threshold: rounds to 1
        check_case(32'sd4096); // Exact scaled integer: 4096 / 2048 = 2
        check_case(32'sd259071); // Just below 126.5 after scaling: rounds to 126
        check_case(32'sd259072); // Exactly 126.5 after scaling: rounds to 127
        check_case(32'sd261120); // Test a large positive input (261120 / 2048 = 127.5, rounds to 128, but should clamp to 127)

        $finish;
    end

endmodule
