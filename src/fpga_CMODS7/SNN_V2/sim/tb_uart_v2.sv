`timescale 1ns/1ps
module tb_uart_v2;
    localparam CPB=16;
    reg clk=0; always #5 clk=~clk;
    reg rst_button=0, host_tx=1;
    wire fpga_tx;
    wire [3:0] led;
    top_snn_v2 #(.CLOCK_HZ(1600),.BAUD(100)) dut(
        .clk(clk),.rst_button(rst_button),.uart_rx(host_tx),.uart_tx(fpga_tx),.led(led));
    wire [7:0] captured_byte;
    wire captured_valid,captured_error;
    uart_rx #(.CLOCK_HZ(1600),.BAUD(100)) monitor(
        .clk(clk),.rst(rst_button),.rx(fpga_tx),.data(captured_byte),
        .valid(captured_valid),.framing_error(captured_error));

    reg [7:0] expected [0:2219];
    reg [7:0] stimulus [0:6143];
    integer received=0, frame, step, i, target=0;
    reg with_trace;

    task send_byte(input [7:0] value);
        integer b;
        begin
            @(negedge clk); host_tx=0;
            repeat(CPB) @(negedge clk);
            for (b=0;b<8;b=b+1) begin
                host_tx=value[b]; repeat(CPB) @(negedge clk);
            end
            host_tx=1; repeat(CPB) @(negedge clk);
        end
    endtask

    always @(negedge clk) begin
        if (!rst_button && (captured_error || led[2])) $fatal(1,"UART framing or protocol error");
        if (captured_valid) begin
            if (received>=2220 || captured_byte !== expected[received])
                $fatal(1,"UART output byte %0d: got %h expected %h",received,captured_byte,expected[received]);
            received=received+1;
        end
    end

    initial begin
        $readmemh("build/vectors/input.mem",stimulus);
        $readmemh("build/vectors/output.mem",expected);
        #1 rst_button=1;
        repeat(8) @(negedge clk); rst_button=0;
        repeat(8) @(negedge clk);
        send_byte("H");
        target=10;
        wait(received==target);
        repeat(CPB*2) @(negedge clk);

        for (frame=0;frame<3;frame=frame+1) begin
            with_trace=(frame!=1);
            send_byte(with_trace ? "T" : "I");
            for (step=0;step<16;step=step+1) begin
                for (i=0;i<128;i=i+1) send_byte(stimulus[frame*2048+step*128+i]);
                target=target+(with_trace ? 67 : 0)+(step==15 ? 7 : 1);
                wait(received==target);
                repeat(CPB*2) @(negedge clk);
            end
            $display("PASS frame %0d, trace %0d",frame,with_trace);
        end
        if (received!=2220) $fatal(1,"Incomplete UART capture");
        $display("PASS identity, exact spikes and voltages, scores, CRC, and frame reset");
        $finish;
    end

    initial begin
        #20_000_000;
        $fatal(1,"UART timeout, %0d bytes received",received);
    end
endmodule
