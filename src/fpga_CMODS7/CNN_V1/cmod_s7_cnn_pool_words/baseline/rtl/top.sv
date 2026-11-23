// W: 10 config bytes + 15105 little-endian int16 parameters -> A1 ack.
// I: 4992 little-endian int16 input values -> infer -> D1, score16,
//    original input bytes (9984), D2. No immediate input echo.
module top (
    input wire clk, input wire rst_button, input wire uart_rx,
    output wire uart_tx, output wire [3:0] led
);
    (* ASYNC_REG="TRUE" *) reg [1:0] reset_sync;
    always @(posedge clk or posedge rst_button)
        if (rst_button) reset_sync<=2'b11;
        else reset_sync<={reset_sync[0],1'b0};
    wire rst=reset_sync[1];
    wire [7:0] rx_data;
    wire rx_valid, rx_error, tx_ready;
    reg [7:0] tx_data;
    reg tx_valid;
    uart_rx rx_inst(.clk(clk),.rst(rst),.rx(uart_rx),.data(rx_data),
        .valid(rx_valid),.framing_error(rx_error));
    uart_tx tx_inst(.clk(clk),.rst(rst),.data(tx_data),.valid(tx_valid),
        .ready(tx_ready),.tx(uart_tx));
    reg [39:0] right_shifts, bias_shifts;
    reg weight_we, input_we, start;
    reg [13:0] weight_addr;
    reg [12:0] input_addr, replay_addr;
    reg signed [15:0] weight_data, input_data;
    wire signed [15:0] replay_data, score;
    wire busy, done;
    cnn_engine engine(.clk(clk),.rst(rst),.start(start),
        .weight_we(weight_we),.weight_addr(weight_addr),.weight_data(weight_data),
        .input_we(input_we),.input_addr(input_addr),.input_data(input_data),
        .right_shifts(right_shifts),.bias_left_shifts(bias_shifts),
        .replay_addr(replay_addr),.replay_data(replay_data),
        .busy(busy),.done(done),.score(score));
    localparam CMD=0, CFG=1, WLO=2, WHI=3, ACK=4,
        ILO=5, IHI=6, LAUNCH=7, RUN=8, TAG=9, SCORELO=10,
        SCOREHI=11, RWAIT1=12, RWAIT2=13, RLO=14, RHI=15,
        END_TAG=16, ERROR=17;
    reg [4:0] state;
    reg [3:0] cfg_index;
    reg [13:0] index;
    reg [7:0] low_byte;
    reg weights_loaded, error_latched;
    reg [23:0] heartbeat;
    always @* begin
        tx_data=0; tx_valid=0;
        case(state)
            ACK: begin tx_data=8'ha1; tx_valid=1; end
            TAG: begin tx_data=8'hd1; tx_valid=1; end
            SCORELO: begin tx_data=score[7:0]; tx_valid=1; end
            SCOREHI: begin tx_data=score[15:8]; tx_valid=1; end
            RLO: begin tx_data=replay_data[7:0]; tx_valid=1; end
            RHI: begin tx_data=replay_data[15:8]; tx_valid=1; end
            END_TAG: begin tx_data=8'hd2; tx_valid=1; end
            default: begin end
        endcase
    end
    always @(posedge clk or posedge rst) begin
        if(rst) begin
            state<=CMD; cfg_index<=0; index<=0; low_byte<=0;
            right_shifts<=0; bias_shifts<=0; weight_we<=0; input_we<=0; start<=0;
            weight_addr<=0; input_addr<=0; replay_addr<=0;
            weight_data<=0; input_data<=0;
            weights_loaded<=0; error_latched<=0; heartbeat<=0;
        end else begin
            heartbeat<=heartbeat+1'b1;
            weight_we<=0; input_we<=0; start<=0;
            if (rx_error || (rx_valid && !(state==CMD || state==CFG ||
                    state==WLO || state==WHI || state==ILO || state==IHI))) begin
                error_latched<=1; state<=ERROR;
            end else case(state)
                CMD: if(rx_valid) begin
                    index<=0;
                    if(rx_data==8'h57) begin cfg_index<=0; weights_loaded<=0; state<=CFG; end
                    else if(rx_data==8'h49 && weights_loaded) state<=ILO;
                    else begin error_latched<=1; state<=ERROR; end
                end
                CFG: if(rx_valid) begin
                    if(rx_data>31) begin error_latched<=1; state<=ERROR; end
                    else begin
                        if(!cfg_index[0]) right_shifts[(cfg_index>>1)*8 +: 8]<=rx_data;
                        else bias_shifts[(cfg_index>>1)*8 +: 8]<=rx_data;
                        if(cfg_index==9) state<=WLO;
                        else cfg_index<=cfg_index+1'b1;
                    end
                end
                WLO: if(rx_valid) begin low_byte<=rx_data; state<=WHI; end
                WHI: if(rx_valid) begin
                    weight_addr<=index; weight_data<={rx_data,low_byte}; weight_we<=1;
                    if(index==15104) begin weights_loaded<=1; state<=ACK; end
                    else begin index<=index+1'b1; state<=WLO; end
                end
                ACK: if(tx_ready) state<=CMD;
                ILO: if(rx_valid) begin low_byte<=rx_data; state<=IHI; end
                IHI: if(rx_valid) begin
                    input_addr<=index[12:0]; input_data<={rx_data,low_byte}; input_we<=1;
                    if(index==4991) state<=LAUNCH;
                    else begin index<=index+1'b1; state<=ILO; end
                end
                LAUNCH: begin start<=1; state<=RUN; end
                RUN: if(done) begin replay_addr<=0; state<=TAG; end
                TAG: if(tx_ready) state<=SCORELO;
                SCORELO: if(tx_ready) state<=SCOREHI;
                SCOREHI: if(tx_ready) state<=RWAIT1;
                RWAIT1: state<=RWAIT2;
                RWAIT2: state<=RLO;
                RLO: if(tx_ready) state<=RHI;
                RHI: if(tx_ready) begin
                    if(replay_addr==4991) state<=END_TAG;
                    else begin replay_addr<=replay_addr+1'b1; state<=RWAIT1; end
                end
                END_TAG: if(tx_ready) state<=CMD;
                ERROR: begin end // Reset and resend weights after protocol errors.
                default: begin error_latched<=1; state<=ERROR; end
            endcase
        end
    end
    assign led[0]=heartbeat[23];
    assign led[1]=weights_loaded;
    assign led[2]=error_latched;
    assign led[3]=busy;
endmodule
