module top_snn_v2 #(
    parameter integer CLOCK_HZ=12_000_000,
    parameter integer BAUD=115_200
) (
    input wire clk, input wire rst_button, input wire uart_rx,
    output wire uart_tx, output wire [3:0] led
);
    `include "weights/model_id.svh"
    (* ASYNC_REG="TRUE" *) reg [1:0] reset_sync;
    always @(posedge clk or posedge rst_button)
        if (rst_button) reset_sync<=2'b11;
        else reset_sync<={reset_sync[0],1'b0};
    wire rst=reset_sync[1];
    wire [7:0] rx_data;
    wire rx_valid, rx_error, tx_ready;
    reg [7:0] tx_data;
    reg tx_valid;
    uart_rx #(.CLOCK_HZ(CLOCK_HZ),.BAUD(BAUD)) receiver(
        .clk(clk),.rst(rst),.rx(uart_rx),.data(rx_data),.valid(rx_valid),.framing_error(rx_error));
    uart_tx #(.CLOCK_HZ(CLOCK_HZ),.BAUD(BAUD)) transmitter(
        .clk(clk),.rst(rst),.data(tx_data),.valid(tx_valid),.ready(tx_ready),.tx(uart_tx));

    localparam [3:0] CMD=0, RX_LO=1, RX_HI=2, LAUNCH=3, WAIT_TOKEN=4,
        TRACE=5, CREDIT=6, RESULT=7, IDENTITY=8, ERROR=9;
    reg [3:0] state;
    reg [6:0] tx_index;
    reg [5:0] word_index;
    reg [3:0] step;
    reg trace_enabled, error, new_frame, start_token, token_we;
    reg [15:0] token_data;
    reg [5:0] token_waddr;
    reg [7:0] low_byte;
    reg [23:0] heartbeat;
    reg [15:0] checksum;
    wire [6:0] trace_offset = tx_index-7'd3;
    wire [2:0] trace_neuron = trace_offset[4:2];
    wire signed [31:0] trace_v1, trace_v2, score;
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
            CREDIT: begin tx_valid=1; tx_data=8'ha1; end
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
            trace_enabled<=0; error<=0; new_frame<=0; start_token<=0;
            checksum<=16'hffff; heartbeat<=0;
        end else begin
            heartbeat<=heartbeat+1'b1;
            token_we<=0; new_frame<=0; start_token<=0;
            if (rx_error || (rx_valid && state!=CMD && state!=RX_LO && state!=RX_HI)) begin
                error<=1; state<=ERROR;
            end else case (state)
                CMD: if (rx_valid) begin
                    error<=0; tx_index<=0;
                    case (rx_data)
                        "H": state<=IDENTITY;
                        "I", "T": begin
                            trace_enabled<=(rx_data=="T");
                            checksum<=16'hffff; new_frame<=1; word_index<=0; step<=0;
                            state<=RX_LO;
                        end
                        default: begin error<=1; state<=ERROR; end
                    endcase
                end
                RX_LO: if (rx_valid) begin
                    checksum<=crc_byte(checksum,rx_data);
                    low_byte<=rx_data; state<=RX_HI;
                end
                RX_HI: if (rx_valid) begin
                    if ({rx_data,low_byte}>16'd20480) begin error<=1; state<=ERROR; end
                    else begin
                        checksum<=crc_byte(checksum,rx_data);
                        token_we<=1; token_data<={rx_data,low_byte}; token_waddr<=word_index;
                        if (word_index==63) state<=LAUNCH;
                        else begin word_index<=word_index+1'b1; state<=RX_LO; end
                    end
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
                default: begin error<=1; state<=ERROR; end
            endcase
        end
    end
    assign led[0]=heartbeat[23];
    assign led[1]=state==RX_LO || state==RX_HI || state==LAUNCH ||
                  state==WAIT_TOKEN || state==TRACE || state==CREDIT || state==RESULT;
    assign led[2]=error;
    assign led[3]=busy;
endmodule
