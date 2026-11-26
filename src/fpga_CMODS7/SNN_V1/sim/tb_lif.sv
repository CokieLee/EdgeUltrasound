`timescale 1ns/1ps
module tb_lif;
    reg signed [31:0] previous;
    reg signed [15:0] current;
    wire signed [31:0] membrane;
    wire spike;
    lif_step dut(.*);

    task check;
        input signed [31:0] old_mem, expected_mem;
        input signed [15:0] injection;
        input expected_spike;
        begin
            previous=old_mem; current=injection;
            #1;
            if (membrane !== expected_mem || spike !== expected_spike)
                $fatal(1, "old=%0d current=%0d: got %0d/%b, expected %0d/%b",
                       previous,current,membrane,spike,expected_mem,expected_spike);
        end
    endtask
    initial begin
        check(0, 1024, 1024, 0);
        check(0, 1025, 1025, 1);
        check(1024, 1024, 103, 0);
        check(1025, -102, 0, 0);
        check(-1, -1, 0, 0);
        check(0, 600, 600, 0);
        check(600, 1139, 600, 1);
        check(1139, 601, 600, 0);
        check(601, 1140, 600, 1);
        $display("PASS LIF threshold, signed leak and delayed reset");
        $finish;
    end
endmodule
