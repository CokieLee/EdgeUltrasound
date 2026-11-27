// SNN V2 byte protocol, independent of its physical transport.
module snn_v2_stream #(
    parameter [31:0] SETTLE_CYCLES=32'd24000000, BENCH_CYCLES=32'd360000000
) (
    input wire clk, input wire rst,
    input wire [7:0] rx_data,
    input wire rx_valid, input wire rx_error,
    output reg [7:0] tx_data,
    output reg tx_valid,
    input wire tx_ready
);
    `include "weights/model_id.svh"
    localparam [3:0] CMD=0, RX_LO=1, RX_HI=2, LAUNCH=3, WAIT_TOKEN=4,
        TRACE=5, CREDIT=6, RESULT=7, IDENTITY=8, ERROR=9, LOAD_CREDIT=10,
        SETTLE=11, RESTART=12, REPEAT_LAUNCH=13, REPEAT_WAIT=14, BENCH_RESULT=15;
    reg [3:0] state;
    reg [6:0] tx_index;
    reg [5:0] word_index;
    reg [3:0] step;
    reg trace_enabled, new_frame, start_token, token_we;
    reg [15:0] token_data;
    reg [9:0] token_waddr;
    reg [7:0] low_byte;
    reg [15:0] checksum;
    reg benchmark, bench_active, score_changed;
    reg [31:0] elapsed, completed;
    reg signed [31:0] first_score;
    wire [3:0] bench_index = tx_index[3:0]-1'b1;
    wire [6:0] trace_offset = tx_index-7'd3;
    wire [2:0] trace_neuron = trace_offset[4:2];
    wire signed [31:0] trace_v1, trace_v2, score;
    wire [31:0] bench_score = bench_active ? score : 32'd0;
    wire [119:0] bench_payload = {7'b0,score_changed,checksum,bench_score,elapsed,completed};
    wire [31:0] trace_word = trace_offset[5] ? trace_v2 : trace_v1;
    wire [7:0] spikes1, spikes2;
    wire busy, token_done, frame_done;

    snn_v2_core core(
        .clk(clk),.rst(rst),.new_frame(new_frame),.start_token(start_token),
        .token_we(token_we),.token_waddr(token_waddr),.token_data(token_data),
        .trace_neuron(trace_neuron),.spikes1(spikes1),.spikes2(spikes2),
        .trace_v1(trace_v1),.trace_v2(trace_v2),.busy(busy),.token_done(token_done),
        .frame_done(frame_done),.score(score));

    function automatic [15:0] crc_byte(input [15:0] old, input [7:0] value);
        reg [15:0] remainder;
        integer bit_index;
        begin
            remainder=old ^ {value,8'b0};
            for (bit_index=0; bit_index<8; bit_index=bit_index+1)
                remainder=(remainder << 1) ^ (remainder[15] ? 16'h1021 : 16'h0000);
            crc_byte=remainder;
        end
    endfunction

    always @* begin
        tx_data=0; tx_valid=0;
        case (state)
            IDENTITY: begin
                tx_valid=1;
                if (tx_index==0) tx_data=8'hb7;
                else if (tx_index==1) tx_data=8'h01;
                else tx_data=MODEL_ID[(7'd9-tx_index)*8+:8];
            end
            TRACE: begin
                tx_valid=1;
                if (tx_index==0) tx_data={4'hc,step};
                else if (tx_index==1) tx_data=spikes1;
                else if (tx_index==2) tx_data=spikes2;
                else tx_data=trace_word[trace_offset[1:0]*8+:8];
            end
            CREDIT, LOAD_CREDIT: begin tx_valid=1; tx_data=8'ha1; end
            BENCH_RESULT: begin
                tx_valid=1;
                tx_data=tx_index==0 ? 8'hd2 : bench_payload[{bench_index,3'b0}+:8];
            end
            RESULT: begin
                tx_valid=1;
                case (tx_index)
                    0: tx_data=8'hd1;
                    1: tx_data=score[7:0];
                    2: tx_data=score[15:8];
                    3: tx_data=score[23:16];
                    4: tx_data=score[31:24];
                    5: tx_data=checksum[7:0];
                    default: tx_data=checksum[15:8];
                endcase
            end
            ERROR: begin tx_valid=1; tx_data=8'hee; end
            default: begin end
        endcase
    end

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state<=CMD; tx_index<=0; word_index<=0; step<=0;
            low_byte<=0; token_data<=0; token_waddr<=0; token_we<=0;
            trace_enabled<=0; new_frame<=0; start_token<=0;
            checksum<=16'hffff;
            benchmark<=0; bench_active<=0; score_changed<=0;
            elapsed<=0; completed<=0; first_score<=0;
        end else begin
            token_we<=0; new_frame<=0; start_token<=0;
            if (rx_error || (rx_valid && state!=CMD && state!=RX_LO && state!=RX_HI)) begin
                state<=ERROR;
            end else case (state)
                CMD: if (rx_valid) begin
                    tx_index<=0;
                    case (rx_data)
                        "H": state<=IDENTITY;
                        "I", "T", "B", "J": begin
                            trace_enabled<=(rx_data=="T");
                            benchmark<=(rx_data=="B" || rx_data=="J");
                            bench_active<=(rx_data=="B");
                            checksum<=16'hffff; new_frame<=1; word_index<=0; step<=0;
                            state<=RX_LO;
                        end
                        default: begin state<=ERROR; end
                    endcase
                end
                RX_LO: if (rx_valid) begin
                    checksum<=crc_byte(checksum,rx_data);
                    low_byte<=rx_data; state<=RX_HI;
                end
                RX_HI: if (rx_valid) begin
                    if ({rx_data,low_byte}>16'd20480) begin state<=ERROR; end
                    else begin
                        checksum<=crc_byte(checksum,rx_data);
                        token_we<=1; token_data<={rx_data,low_byte}; token_waddr<={step,word_index};
                        if (word_index==63) state<=benchmark ? LOAD_CREDIT : LAUNCH;
                        else begin word_index<=word_index+1'b1; state<=RX_LO; end
                    end
                end
                LOAD_CREDIT: if (tx_ready) begin
                    word_index<=0;
                    if (step==15) begin
                        elapsed<=0; completed<=0; score_changed<=0; first_score<=0;
                        state<=SETTLE;
                    end else begin
                        step<=step+1'b1; state<=RX_LO;
                    end
                end
                SETTLE: begin
                    if (elapsed==SETTLE_CYCLES-1) begin
                        elapsed<=0; tx_index<=0;
                        state<=bench_active ? RESTART : REPEAT_WAIT;
                    end else elapsed<=elapsed+1'b1;
                end
                RESTART: begin
                    elapsed<=elapsed+1'b1;
                    new_frame<=1; state<=REPEAT_LAUNCH;
                end
                REPEAT_LAUNCH: begin
                    elapsed<=elapsed+1'b1;
                    start_token<=1; state<=REPEAT_WAIT;
                end
                REPEAT_WAIT: begin
                    elapsed<=elapsed+1'b1;
                    if (!bench_active) begin
                        if (elapsed==BENCH_CYCLES-1) state<=BENCH_RESULT;
                    end else if (frame_done) begin
                        completed<=completed+1'b1;
                        if (completed==0) first_score<=score;
                        else if (score!=first_score) score_changed<=1;
                        state<=elapsed>=BENCH_CYCLES-1 ? BENCH_RESULT : RESTART;
                    end else if (token_done) state<=REPEAT_LAUNCH;
                end
                BENCH_RESULT: if (tx_ready) begin
                    if (tx_index==15) state<=CMD;
                    else tx_index<=tx_index+1'b1;
                end
                LAUNCH: begin start_token<=1; state<=WAIT_TOKEN; end
                WAIT_TOKEN: if (token_done) begin
                    tx_index<=0;
                    if (trace_enabled) state<=TRACE;
                    else state<=frame_done ? RESULT : CREDIT;
                end
                TRACE: if (tx_ready) begin
                    if (tx_index==66) begin tx_index<=0; state<=step==15 ? RESULT : CREDIT; end
                    else tx_index<=tx_index+1'b1;
                end
                CREDIT: if (tx_ready) begin step<=step+1'b1; word_index<=0; state<=RX_LO; end
                RESULT: if (tx_ready) begin
                    if (tx_index==6) state<=CMD;
                    else tx_index<=tx_index+1'b1;
                end
                IDENTITY: if (tx_ready) begin
                    if (tx_index==9) state<=CMD;
                    else tx_index<=tx_index+1'b1;
                end
                ERROR: if (tx_ready) state<=CMD;
                default: begin state<=ERROR; end
            endcase
        end
    end
endmodule
