`timescale 1ns/1ps
module tb_uart_stream;
    localparam CPB=16;
    localparam EXPECTED_RX_BYTES=155148;
    reg clk=0; always #5 clk=~clk;
    reg rst_button=1,host_tx=1;
    wire fpga_tx;
    wire [3:0] led;
    top_stream #(.CLOCK_HZ(1600),.BAUD(100)) dut(
        .clk(clk),.rst_button(rst_button),.uart_rx(host_tx),.uart_tx(fpga_tx),.led(led));
    wire [7:0] captured_byte;
    wire captured_valid,captured_error;
    uart_rx #(.CLOCK_HZ(1600),.BAUD(100)) monitor(
        .clk(clk),.rst(rst_button),.rx(fpga_tx),.data(captured_byte),
        .valid(captured_valid),.framing_error(captured_error));
    reg [7:0] stimulus[0:40205];
    integer capture_file,received=0,i;

    task send_byte;
        input [7:0] value;
        integer b;
        begin
            @(negedge clk); host_tx=0;
            repeat(CPB) @(negedge clk);
            for(b=0;b<8;b=b+1) begin
                host_tx=value[b]; repeat(CPB) @(negedge clk);
            end
            host_tx=1; repeat(CPB) @(negedge clk);
        end
    endtask

    always @(posedge clk) begin
        if(!rst_button && captured_error) $fatal(1,"FPGA TX framing error");
        if(!rst_button && led[2]) $fatal(1,"FPGA protocol error LED");
        if(captured_valid) begin
            $fwrite(capture_file,"%c",captured_byte);
            received=received+1;
        end
    end

    initial begin
        $readmemh("build/vectors/uart_stimulus.mem",stimulus);
        capture_file=$fopen("build/uart_capture.bin","wb");
        if(!capture_file) $fatal(1,"could not open UART capture");
        repeat(8) @(negedge clk); rst_button=0;
        repeat(8) @(negedge clk);
        send_byte(8'h48);
        wait(received==2); // B5 03 byte-split diagnostic identity.
        repeat(CPB*2) @(negedge clk);
        for(i=0;i<30221;i=i+1) send_byte(stimulus[i]);
        wait(received==3); // Parameter acknowledgment.
        repeat(CPB*2) @(negedge clk);
        for(i=30221;i<40206;i=i+1) send_byte(stimulus[i]);
        wait(received==EXPECTED_RX_BYTES);
        repeat(CPB*2) @(negedge clk);
        $fclose(capture_file);
        $display("PASS UART stream transfer: %0d bytes; host must validate tensors",received);
        $finish;
    end
    initial begin
        repeat(90_000_000) @(posedge clk);
        $fatal(1,"UART trace timeout; received %0d bytes",received);
    end
endmodule
