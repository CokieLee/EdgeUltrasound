`timescale 1ns/1ps
module tb_uart_link;
    localparam integer CPB=104;
    reg clk=0;
    always #5 clk=~clk;
    reg rst=1;
    reg rx=1;
    wire tx;
    wire [7:0] byte_data;
    wire byte_valid;
    wire [31:0] received_bytes;
    wire overflow_error, framing_error;
    reg [7:0] expected [0:2047];
    integer queued=0, seen=0, echoed=0;
    integer i;
    uart_byte_link dut (
        .clk(clk), .rst(rst), .uart_rx_pin(rx), .uart_tx_pin(tx),
        .byte_data(byte_data), .byte_valid(byte_valid), .received_bytes(received_bytes),
        .overflow_error(overflow_error), .framing_error(framing_error)
    );

    always @(posedge clk) begin
        if (rst) seen <= 0;
        else if (byte_valid) begin
            if (seen >= queued || byte_data !== expected[seen])
                $fatal(1, "RX mismatch at %0d: got %h", seen, byte_data);
            seen <= seen+1;
        end
    end

    // Independent bit-level decoder of FPGA TX (does not reuse uart_rx).
    initial begin : monitor_echo
        reg [7:0] value;
        integer bit_num;
        forever begin
            @(negedge tx);
            repeat (CPB+CPB/2) @(posedge clk);
            for (bit_num=0; bit_num<8; bit_num=bit_num+1) begin
                value[bit_num]=tx;
                repeat (CPB) @(posedge clk);
            end
            if (tx !== 1'b1) $fatal(1, "TX stop bit missing");
            if (echoed >= queued || value !== expected[echoed])
                $fatal(1, "Echo mismatch at %0d: got %h", echoed, value);
            echoed=echoed+1;
        end
    end

    task send_byte(input [7:0] value, input integer bit_clocks, input bit good_stop);
        integer b;
        begin
            @(negedge clk); rx=0;
            repeat(bit_clocks) @(negedge clk);
            for (b=0; b<8; b=b+1) begin
                rx=value[b];
                repeat(bit_clocks) @(negedge clk);
            end
            rx=good_stop;
            repeat(bit_clocks) @(negedge clk);
            rx=1;
        end
    endtask

    initial begin
        repeat(10) @(negedge clk); rst=0;
        repeat(10) @(negedge clk);
        for (i=0; i<1024; i=i+1) begin
            expected[queued]=(i*73+19)&255;
            queued=queued+1;
            // Alternate nominal and slightly slower host baud periods.
            send_byte(expected[queued-1], CPB+(i%2), 1);
        end
        wait(echoed==queued);
        repeat(CPB*3) @(negedge clk);
        if (seen!=queued || received_bytes!=queued || overflow_error || framing_error)
            $fatal(1, "Normal transfer count/error failure");

        // Invalid stop bit: must not emit a payload byte, and error must latch.
        send_byte(8'h55, CPB, 0);
        repeat(CPB*3) @(negedge clk);
        if (!framing_error || received_bytes!=queued || seen!=queued || echoed!=queued)
            $fatal(1, "Bad stop bit handling failed");

        rst=1;
        repeat(10) @(negedge clk);
        queued=0; echoed=0;
        rst=0;
        repeat(10) @(negedge clk);
        for (i=0; i<256; i=i+1) begin
            expected[queued]=i;
            queued=queued+1;
            send_byte(expected[queued-1], CPB, 1);
        end
        wait(echoed==queued);
        repeat(CPB*3) @(negedge clk);
        if (seen!=256 || received_bytes!=256 || framing_error || overflow_error)
            $fatal(1, "Reset/restart failed");
        $display("PASS: UART receive, independent echo decode, baud variation, framing error, FIFO wrap, reset");
        $finish;
    end
    initial begin
        repeat(2_000_000) @(posedge clk);
        $fatal(1, "Simulation timeout");
    end
endmodule
