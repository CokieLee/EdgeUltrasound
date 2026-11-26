`timescale 1ns/1ps
module tb_snn_engine;
    reg clk=0; always #5 clk=~clk;
    reg rst=0, start=0, trace_enable=0, trace_ready=0, weight_we=0, input_we=0;
    reg [13:0] weight_addr=0;
    reg [12:0] input_addr=0, replay_addr=0;
    reg signed [15:0] weight_data=0, input_data=0;
    reg [39:0] right_shifts, bias_left_shifts;
    wire busy, done, trace_valid;
    wire [2:0] trace_stage;
    wire signed [15:0] replay_data, score, trace_data;
    snn_engine dut(.*);

    reg [15:0] weights [0:15104];
    reg [15:0] frame [0:4991];
    reg [7:0] config_bytes [0:9];
    reg [15:0] expected_trace [0:72479];
    reg [15:0] expected_score [0:0];
    integer i, stage;

    task launch;
        input streaming;
        begin
            trace_enable=streaming; start=1;
            @(negedge clk); start=0;
        end
    endtask

    task check_finish;
        begin
            while (!done) @(negedge clk);
            if (score !== expected_score[0])
                $fatal(1,"Wrong score: got %0d",score);
            for (i=0; i<4992; i=i+1) begin
                replay_addr=i;
                @(negedge clk);
                if (replay_data !== frame[i]) $fatal(1,"Input replay index %0d mismatch",i);
            end
        end
    endtask

    initial begin
        $readmemh("build/vectors/weights.mem",weights);
        $readmemh("build/vectors/input.mem",frame);
        $readmemh("build/vectors/config.mem",config_bytes);
        $readmemh("build/vectors/trace.mem",expected_trace);
        $readmemh("build/vectors/score.mem",expected_score);
        #1 rst=1;
        repeat (3) @(negedge clk);
        rst=0;
        for (i=0; i<5; i=i+1) begin
            right_shifts[i*8+:8]=config_bytes[2*i];
            bias_left_shifts[i*8+:8]=config_bytes[2*i+1];
        end
        weight_we=1;
        for (i=0; i<15105; i=i+1) begin
            weight_addr=i; weight_data=weights[i];
            @(negedge clk);
        end
        weight_we=0;
        input_we=1;
        for (i=0; i<4992; i=i+1) begin
            input_addr=i; input_data=frame[i];
            @(negedge clk);
        end
        input_we=0;
        launch(1);
        for (i=0; i<72480; i=i+1) begin
            while (!trace_valid) @(negedge clk);
            stage=i<39936 ? 0 : i<59904 ? 1 : i<69888 ? 2 : i<72448 ? 3 : 4;
            if (i%257==0) repeat (3) @(negedge clk);
            if (!trace_valid || trace_stage !== stage[2:0] || trace_data !== expected_trace[i])
                $fatal(1,"Trace word %0d stage %0d: got %0d, expected %0d",
                       i,stage,trace_data,$signed(expected_trace[i]));
            trace_ready=1;
            @(negedge clk); trace_ready=0;
        end
        check_finish;
        $display("PASS exact streamed stem, spikes, counts, signed score and input replay");
        launch(0);
        check_finish;
        if (trace_valid) $fatal(1,"Unexpected trace in continuous mode");
        $display("PASS second frame, LIF reset and continuous score");
        $finish;
    end

    initial begin
        #900_000_000;
        $fatal(1,"Engine simulation timed out");
    end
endmodule
