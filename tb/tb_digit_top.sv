// tb/tb_digit_top.sv
//
// Lab 4 skeleton. See docs/lab4_verification.md before writing any logic here.
//
// Clock/reset generation and reference-vector file reading are filled in.
// The bit-exact comparison against golden_model.py's dumped reference
// vectors, and the functional coverage, are left as TODO.

`timescale 1ns/1ps

module tb_digit_top;

    // -----------------------------------------------------------------
    // Parameters (must match rtl/digit_top.sv and data/meta.json)
    // -----------------------------------------------------------------
    localparam int N_IN    = 196;
    localparam int HIDDEN  = 64;
    localparam int N_CLASS = 10;
    localparam int NTEST   = 20;

    // -----------------------------------------------------------------
    // DUT I/O
    // -----------------------------------------------------------------
    logic clk;
    logic rst;
    logic start;
    logic [$clog2(NTEST)-1:0] img_idx;

    logic busy;
    logic done;
    logic [3:0] pred_digit;
    logic correct;
    logic [15:0] cycle_count;
    logic signed [N_CLASS*32-1:0] score_debug;

    digit_top #(
        .N_IN    (N_IN),
        .HIDDEN  (HIDDEN),
        .N_CLASS (N_CLASS),
        .NTEST   (NTEST)
        // TODO: weight/image/label/bias init file parameters, matching
        // whatever you decided in Lab 3
    ) dut (
        .clk         (clk),
        .rst         (rst),
        .start       (start),
        .img_idx     (img_idx),
        .busy        (busy),
        .done        (done),
        .pred_digit  (pred_digit),
        .correct     (correct),
        .cycle_count (cycle_count),
        .score_debug (score_debug)
    );

    // -----------------------------------------------------------------
    // Clock: 100MHz (10ns period), matching the real board target
    // -----------------------------------------------------------------
    initial clk = 0;
    always #5 clk = ~clk;

    // -----------------------------------------------------------------
    // Waveform dump
    // -----------------------------------------------------------------
    initial begin
        $dumpfile("wave_digit_top.vcd");
        $dumpvars(0, tb_digit_top);
    end

    // -----------------------------------------------------------------
    // Reference vectors, dumped by sw/golden_model.py (Lab 1, step 8).
    // Format here assumes one file per image with plain decimal integers,
    // in the order: acc1[0..63], a1[0..63], acc2[0..9], pred (1 value).
    // If you chose a different format in golden_model.py, update this
    // reader to match it -- the two must agree exactly.
    // -----------------------------------------------------------------
    logic signed [31:0] ref_acc1  [0:NTEST-1][0:HIDDEN-1];
    logic signed [7:0]  ref_a1    [0:NTEST-1][0:HIDDEN-1];
    logic signed [31:0] ref_acc2  [0:NTEST-1][0:N_CLASS-1];
    logic [3:0]          ref_pred [0:NTEST-1];

    task automatic load_reference_vectors();
        integer fd, r, img, i;
        integer tmp;
        string  fname;
        begin
            for (img = 0; img < NTEST; img++) begin
                fname = $sformatf("data/ref_img%0d.txt", img);
                fd = $fopen(fname, "r");
                if (fd == 0) begin
                    $display("ERROR: could not open %s -- did you run sw/golden_model.py?", fname);
                    $fatal(1);
                end

                for (i = 0; i < HIDDEN; i++) begin
                    r = $fscanf(fd, "%d\n", tmp);
                    if (r != 1) $fatal(1, "ERROR reading acc1[%0d] for image %0d", i, img);
                    ref_acc1[img][i] = tmp;
                end
                for (i = 0; i < HIDDEN; i++) begin
                    r = $fscanf(fd, "%d\n", tmp);
                    if (r != 1) $fatal(1, "ERROR reading a1[%0d] for image %0d", i, img);
                    ref_a1[img][i] = tmp;
                end
                for (i = 0; i < N_CLASS; i++) begin
                    r = $fscanf(fd, "%d\n", tmp);
                    if (r != 1) $fatal(1, "ERROR reading acc2[%0d] for image %0d", i, img);
                    ref_acc2[img][i] = tmp;
                end
                r = $fscanf(fd, "%d\n", tmp);
                if (r != 1) $fatal(1, "ERROR reading pred for image %0d", img);
                ref_pred[img] = tmp[3:0];

                $fclose(fd);
            end
            $display("Loaded golden reference vectors for %0d images.", NTEST);
        end
    endtask

    // -----------------------------------------------------------------
    // TODO: covergroup(s) for the 5 target coverpoints from
    // docs/lab4_verification.md -- weight/activation extremes, shift-
    // saturation hit/not-hit, ReLU clip/pass, argmax tie-break, all 20
    // images + start-while-busy. Sample on the appropriate DUT-internal
    // signals (you may need to add a hierarchical reference into dut's
    // internals, or expose additional debug ports from digit_top -- your
    // call, document which you chose).
    // -----------------------------------------------------------------

    // -----------------------------------------------------------------
    // Task: run one image through the DUT and wait for done, with a
    // timeout safety net.
    // -----------------------------------------------------------------
    task automatic run_one_image(input int img);
        int timeout;
        begin
            img_idx = img[$clog2(NTEST)-1:0];

            @(negedge clk);
            start = 1;
            @(negedge clk);
            start = 0;

            timeout = 0;
            while (done !== 1'b1) begin
                @(posedge clk);
                timeout++;
                if (timeout > 100000) $fatal(1, "TIMEOUT img=%0d: done never asserted", img);
            end
            @(posedge clk);  // settle
        end
    endtask

    // -----------------------------------------------------------------
    // Main test: for every stored image, run it, then bit-exact compare
    // acc1/a1/acc2/pred_digit against the golden reference. TODO.
    // -----------------------------------------------------------------
    int errors;

    initial begin
        start   = 0;
        img_idx = 0;
        rst     = 1;
        repeat (5) @(posedge clk);
        rst = 0;
        repeat (2) @(posedge clk);

        load_reference_vectors();
        errors = 0;

        for (int img = 0; img < NTEST; img++) begin
            run_one_image(img);

            // TODO: compare against ref_acc1[img], ref_a1[img],
            // ref_acc2[img], ref_pred[img]. You'll need a way to observe
            // acc1/a1 from the testbench -- either expose them as
            // additional digit_top debug ports (simplest) or reference
            // dut's internals hierarchically (dut.<instance>.<signal>).
            // Document which you chose. score_debug already exposes acc2
            // per architecture_spec.md Section 5.
            //
            // On any mismatch: increment errors, and $display enough detail
            // (image index, which value, expected vs. actual) to debug
            // without immediately reaching for a waveform viewer.

            // TODO: also compare pred_digit against ref_pred[img] AND
            // against `correct`/the true label, as a cross-check.
        end

        $display("======================================");
        $display("RESULTS: %0d/%0d images bit-exact, %0d errors", NTEST - errors, NTEST, errors);
        $display("======================================");

        // TODO: report coverage summary ($display or rely on the
        // simulator's coverage report output, per your toolchain).

        $finish;
    end

endmodule
