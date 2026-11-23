`timescale 1ns/1ps
module tb_cnn_engine;
    reg clk=0; always #5 clk=~clk;
    reg rst=1,start=0,weight_we=0,input_we=0;
    reg [13:0] weight_addr=0;
    reg [12:0] input_addr=0,replay_addr=0;
    reg signed [15:0] weight_data=0,input_data=0;
    wire signed [15:0] replay_data,score;
    wire busy,done;
    // Little-endian byte lanes: shifts stem, conv1, conv2, conv3, head.
    reg [39:0] right_shifts=40'h0002020200;
    reg [39:0] bias_left_shifts=0;
    reg signed [15:0] w[0:15104], x[0:4991], expected_score[0:0];
    reg signed [15:0] exp_stem[0:39935], exp_conv1[0:19967];
    reg signed [15:0] exp_conv2[0:9983], exp_conv3[0:2559],exp_pool[0:31];
    integer i,j;
    integer cycles=0;
    reg checked_stem=0,checked_conv1=0,checked_conv2=0,checked_conv3=0;
    cnn_engine dut(.*);

    always @(posedge clk) begin
        if(busy) cycles<=cycles+1;
        if(!rst && dut.state==1 && dut.layer==1 && !checked_stem) begin
            for(j=0;j<39936;j=j+1)
                if(dut.buffer_a[j] !== exp_stem[j]) $fatal(1,"stem mismatch %0d",j);
            checked_stem=1; $display("PASS stem");
        end
        if(!rst && dut.state==1 && dut.layer==2 && !checked_conv1) begin
            for(j=0;j<19968;j=j+1)
                if(dut.buffer_b[j] !== exp_conv1[j]) $fatal(1,"conv1 mismatch %0d",j);
            checked_conv1=1; $display("PASS conv1 + ReLU");
        end
        if(!rst && dut.state==1 && dut.layer==3 && !checked_conv2) begin
            for(j=0;j<9984;j=j+1)
                if(dut.buffer_a[j] !== exp_conv2[j]) $fatal(1,"conv2 mismatch %0d",j);
            checked_conv2=1; $display("PASS conv2 + ReLU");
        end
        if(!rst && dut.state==9 && !checked_conv3) begin
            for(j=0;j<2560;j=j+1)
                if(dut.buffer_b[j] !== exp_conv3[j]) $fatal(1,"conv3 mismatch %0d",j);
            checked_conv3=1; $display("PASS conv3 + ReLU");
        end
    end
    initial begin
        $readmemh("build/vectors/weights.mem",w);
        $readmemh("build/vectors/input.mem",x);
        $readmemh("build/vectors/score.mem",expected_score);
        $readmemh("build/vectors/stem.mem",exp_stem);
        $readmemh("build/vectors/conv1.mem",exp_conv1);
        $readmemh("build/vectors/conv2.mem",exp_conv2);
        $readmemh("build/vectors/conv3.mem",exp_conv3);
        $readmemh("build/vectors/pool.mem",exp_pool);
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
        wait(done); @(negedge clk);
        if(score !== expected_score[0])
            $fatal(1,"score expected %0d got %0d",expected_score[0],score);
        if(!(checked_stem&&checked_conv1&&checked_conv2&&checked_conv3))
            $fatal(1,"not every layer was checked");
        for(i=0;i<32;i=i+1)
            if(dut.pooled[i] !== exp_pool[i]) $fatal(1,"pool mismatch %0d",i);
        for(i=0;i<4992;i=i+1) begin
            replay_addr=i;
            repeat(2) @(negedge clk);
            if(replay_data !== x[i]) $fatal(1,"original input replay mismatch %0d",i);
        end
        $display("PASS: all CNN layers, pooling, score=%0d, preserved input; %0d compute clocks",score,cycles);
        $finish;
    end
    initial begin
        repeat(20_000_000) @(posedge clk);
        $fatal(1,"timeout");
    end
endmodule
