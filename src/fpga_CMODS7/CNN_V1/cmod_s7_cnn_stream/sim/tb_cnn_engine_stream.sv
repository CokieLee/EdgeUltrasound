`timescale 1ns/1ps
module tb_cnn_engine_stream;
    reg clk=0; always #5 clk=~clk;
    reg rst=1,start=0,weight_we=0,input_we=0;
    reg [13:0] weight_addr=0;
    reg [12:0] input_addr=0,replay_addr=0;
    reg signed [15:0] weight_data=0,input_data=0;
    wire signed [15:0] replay_data,score;
    wire busy,done;
    reg [39:0] right_shifts=40'h0002020200;
    reg [39:0] bias_left_shifts=0;
    reg trace_enable=1,trace_ready=0;
    wire trace_valid;
    wire [2:0] trace_stage;
    wire signed [15:0] trace_data;
    reg signed [15:0] w[0:15104],x[0:4991],expected_score[0:0];
    reg signed [15:0] stem[0:39935],conv1[0:19967],conv2[0:9983];
    reg signed [15:0] conv3[0:2559],pool[0:31];
    reg signed [15:0] expected,held;
    integer i,stage_number,count,total=0;
    reg continuous_phase=0;
    wire reference_busy,reference_done;
    wire signed [15:0] reference_score,reference_replay_data;
    cnn_engine_stream dut(.*);
    cnn_engine reference_engine(.busy(reference_busy),.done(reference_done),
        .score(reference_score),.replay_data(reference_replay_data),.*);

    always @(negedge clk) if(continuous_phase && !rst) begin
        if(busy !== reference_busy || done !== reference_done)
            $fatal(1,"continuous cycle sequence differs from original engine");
        if(done && score !== reference_score) $fatal(1,"continuous score differs from original");
        if(trace_valid) $fatal(1,"unexpected trace word in continuous mode");
    end

    initial begin
        $readmemh("build/vectors/weights.mem",w);
        $readmemh("build/vectors/input.mem",x);
        $readmemh("build/vectors/score.mem",expected_score);
        $readmemh("build/vectors/stem.mem",stem);
        $readmemh("build/vectors/conv1.mem",conv1);
        $readmemh("build/vectors/conv2.mem",conv2);
        $readmemh("build/vectors/conv3.mem",conv3);
        $readmemh("build/vectors/pool.mem",pool);
        repeat(5) @(negedge clk); rst=0;
        for(i=0;i<15105;i=i+1) begin
            @(negedge clk); weight_we=1; weight_addr=i; weight_data=w[i];
        end
        @(negedge clk); weight_we=0;
        for(i=0;i<4992;i=i+1) begin
            @(negedge clk); input_we=1; input_addr=i; input_data=x[i];
        end
        @(negedge clk); input_we=0;
        @(negedge clk); start=1;
        @(negedge clk); start=0;
        for(stage_number=0;stage_number<5;stage_number=stage_number+1) begin
            case(stage_number)
                0: count=39936;
                1: count=19968;
                2: count=9984;
                3: count=2560;
                4: count=32;
            endcase
            for(i=0;i<count;i=i+1) begin
                wait(trace_valid); @(negedge clk);
                if(trace_stage !== stage_number) $fatal(1,"stage order mismatch");
                case(stage_number)
                    0: expected=stem[i];
                    1: expected=conv1[i];
                    2: expected=conv2[i];
                    3: expected=conv3[i];
                    4: expected=pool[i];
                endcase
                if(trace_data !== expected)
                    $fatal(1,"stage %0d index %0d expected %0d got %0d",stage_number,i,expected,trace_data);
                held=trace_data;
                repeat(1+(i%3)) @(negedge clk);
                if(!trace_valid || trace_data !== held || !busy)
                    $fatal(1,"produced word changed under backpressure");
                trace_ready=1; @(negedge clk); trace_ready=0;
                if(trace_valid) $fatal(1,"accepted word did not clear valid");
                total=total+1;
            end
            $display("PASS streamed stage %0d: %0d exact words",stage_number,count);
        end
        wait(done); @(negedge clk);
        if(score !== expected_score[0] || total!=72480) $fatal(1,"streamed final result mismatch");
        for(i=0;i<4992;i=i+1) begin
            replay_addr=i; repeat(2) @(negedge clk);
            if(replay_data !== x[i]) $fatal(1,"original input replay mismatch %0d",i);
        end
        $display("PASS stream engine: all outputs, backpressure, preserved input, score %0d",score);
        trace_enable=0; continuous_phase=1;
        start=1; @(negedge clk); start=0;
        wait(done); @(negedge clk);
        if(score !== expected_score[0]) $fatal(1,"continuous score mismatch");
        $display("PASS continuous engine: same cycle sequence and score as original core");
        $finish;
    end
    initial begin
        repeat(45_000_000) @(posedge clk);
        $fatal(1,"stream engine timeout");
    end
endmodule
