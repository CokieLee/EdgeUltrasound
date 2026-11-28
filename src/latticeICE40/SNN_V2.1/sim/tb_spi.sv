`timescale 1ns/1ps
module tb_spi;
    reg clk=0;
    always #41.667 clk=~clk;
    reg sck=0, mosi=0, flash_cs_n=1;
    wire miso;
    top_upduino_spi dut(.clk(clk),.spi_sck(sck),.spi_mosi(mosi),
                        .flash_cs_n(flash_cs_n),.spi_miso(miso));

    reg [7:0] expected [0:2219];
    reg [7:0] stimulus [0:6143];
    reg [15:0] reply;
    integer received=0, frame, step, i, target=0;
    reg with_trace;

    // Same mode-0 edges as MPSSE, at 250 kHz.
    task slot(input [15:0] payload);
        integer bit_no;
        begin
            for (bit_no=15;bit_no>=0;bit_no=bit_no-1) begin
                mosi=payload[bit_no];
                #2000;
                sck=1;
                reply[bit_no]=miso;
                #2000;
                sck=0;
            end
            case (reply[15:8])
                8'h00: begin end
                8'h01: begin
                    if (received>=2220 || reply[7:0] !== expected[received])
                        $fatal(1,"SPI byte %0d: got %h expected %h",received,reply[7:0],expected[received]);
                    received=received+1;
                end
                default: $fatal(1,"Invalid reply slot %h",reply);
            endcase
        end
    endtask

    task expect_bytes(input integer goal);
        begin
            while (received<goal) slot(16'h0000);
            if (received!=goal) $fatal(1,"Unexpected response length");
        end
    endtask

    initial begin
        $readmemh("build/vectors/input.mem",stimulus);
        $readmemh("build/vectors/output.mem",expected);
        #2000 flash_cs_n=0;
        #1 if (miso !== 1'bz) $fatal(1,"MISO should be released while flash is selected");
        #3900 flash_cs_n=1;
        #2000;
        slot(16'h0148); // H
        target=10;
        expect_bytes(target);
        for (frame=0;frame<3;frame=frame+1) begin
            with_trace=(frame!=1);
            slot(with_trace ? 16'h0154 : 16'h0149); // T or I
            for (step=0;step<16;step=step+1) begin
                for (i=0;i<128;i=i+1) slot({8'h01,stimulus[frame*2048+step*128+i]});
                target=target+(with_trace ? 67 : 0)+(step==15 ? 7 : 1);
                expect_bytes(target);
            end
            $display("PASS SPI frame %0d",frame);
        end
        repeat(3) slot(16'h0000);
        if (received!=2220) $fatal(1,"Unexpected trailing response");
        $display("PASS exact identity, spikes, voltages, scores, CRC and flash-bus release");
        $finish;
    end

    initial begin
        #650_000_000;
        $fatal(1,"SPI timeout, %0d bytes received",received);
    end
endmodule
