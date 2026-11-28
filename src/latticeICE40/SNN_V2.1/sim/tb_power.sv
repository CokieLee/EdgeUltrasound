`timescale 1ns/1ps
module tb_power;
    localparam [31:0] TEST_CYCLES=45000;
    reg clk=0;
    always #5 clk=~clk;
    reg rst=0;
    reg [7:0] rx_data=0;
    reg rx_valid=0, tx_ready=0;
    wire [7:0] tx_data;
    wire tx_valid;
    snn_v2_stream #(.SETTLE_CYCLES(20),.BENCH_CYCLES(TEST_CYCLES)) dut(
        .clk(clk),.rst(rst),.rx_data(rx_data),.rx_valid(rx_valid),.rx_error(1'b0),
        .tx_data(tx_data),.tx_valid(tx_valid),.tx_ready(tx_ready));

    reg [7:0] input_data [0:6143];
    reg [7:0] expected [0:2219];
    reg [7:0] value;
    reg [119:0] result;
    integer i, token;
    integer launch_count=0;
    always @(posedge clk) if (dut.start_token) launch_count<=launch_count+1;

    task send(input [7:0] data);
        begin
            rx_data=data; rx_valid=1;
            @(negedge clk);
            rx_valid=0;
            @(negedge clk);
        end
    endtask

    task read_byte(output [7:0] data);
        begin
            while (!tx_valid) @(negedge clk);
            data=tx_data;
            tx_ready=1;
            @(negedge clk);
            tx_ready=0;
        end
    endtask

    task measure(input active);
        begin
            launch_count=0;
            send(active ? "B" : "J");
            for (token=0;token<16;token=token+1) begin
                for (i=0;i<128;i=i+1) send(input_data[token*128+i]);
                read_byte(value);
                if (value !== 8'ha1) $fatal(1,"Missing upload credit");
            end
            read_byte(value);
            if (value !== 8'hd2) $fatal(1,"Incorrect benchmark tag");
            for (i=0;i<15;i=i+1) begin
                read_byte(value);
                result[i*8+:8]=value;
            end
            if (result[119:112] !== 0 || result[111:96] !== {expected[1103],expected[1102]})
                $fatal(1,"Benchmark flags or CRC differ: %h",result);
            if (active) begin
                if (launch_count!=48 || result[31:0]!==32'd3 ||
                    result[63:32]<TEST_CYCLES || result[63:32]>TEST_CYCLES+22000 ||
                    $signed(result[95:64])!==-32'sd1281222)
                    $fatal(1,"Active benchmark count, cycles or score differ: %h",result);
            end else begin
                if (launch_count!=0 || result[31:0]!==0 || result[63:32]!==TEST_CYCLES ||
                    result[95:64]!==0) $fatal(1,"Idle benchmark activated compute or timed incorrectly: %h",result);
            end
        end
    endtask

    initial begin
        $readmemh("build/vectors/input.mem",input_data);
        $readmemh("build/vectors/output.mem",expected);
        #1 rst=1;
        repeat (4) @(negedge clk);
        rst=0;
        measure(1);
        measure(0);
        measure(1);
        $display("PASS active, idle, active: exact score, CRC, repeat count and timer");
        $finish;
    end
    initial begin
        #3_000_000;
        $fatal(1,"Benchmark simulation timed out");
    end
endmodule
