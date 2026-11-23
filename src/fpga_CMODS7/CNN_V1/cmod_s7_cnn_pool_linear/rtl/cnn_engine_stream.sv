// Fixed network: 1x32x156 -> 8x32x156 -> 16x16x78 -> 32x8x39
// -> 32x4x20 -> global average -> linear(32,1).
// All tensor/parameter storage is signed int16. Accumulation is signed 48-bit.
// See README for the mandatory quantization/export contract.
// Minimal instrumentation: tap produced words; original RAM ports are untouched.
module cnn_engine_stream (
    input wire trace_enable, input wire trace_ready,
    output reg trace_valid, output reg [2:0] trace_stage,
    output reg signed [15:0] trace_data,
    input wire clk, input wire rst, input wire start,
    input wire weight_we, input wire [13:0] weight_addr,
    input wire signed [15:0] weight_data,
    input wire input_we, input wire [12:0] input_addr,
    input wire signed [15:0] input_data,
    input wire [39:0] right_shifts, input wire [39:0] bias_left_shifts,
    input wire [12:0] replay_addr, output wire signed [15:0] replay_data,
    output reg busy, output reg done, output reg signed [15:0] score
);
    reg signed [15:0] weights [0:15104];
    reg signed [15:0] original [0:4991];
    reg signed [15:0] buffer_a [0:39935];
    reg signed [15:0] buffer_b [0:19967];
    reg signed [15:0] pooled [0:31];
    reg signed [15:0] w_q, input_q, a_q, b_q;
    assign replay_data = input_q;
    reg [15:0] src_addr;
    reg [13:0] w_addr;
    reg signed [47:0] acc;
    integer layer, in_c, in_h, in_w, out_c, out_h, out_w, kernel, stride, pad;
    integer weight_base, bias_base;
    integer oc, oy, ox, ic, ky, kx, pool_c, pool_p;
    integer sy, sx, out_addr;
    reg padded;
    wire signed [15:0] operand = layer==0 ? input_q :
        layer==1 ? a_q : layer==2 ? b_q : layer==3 ? a_q : pooled[ic];
    wire signed [31:0] product = operand * w_q;
    wire [7:0] shift_amt = right_shifts >> (layer*8);
    wire [7:0] bias_shift = bias_left_shifts >> (layer*8);
    localparam IDLE=0, CONFIG=1, BREQ=2, BWAIT=3, BUSE=4,
        TREQ=5, TWAIT=6, MAC=7, STORE=8,
        PREQ=9, PWAIT=10, PADD=11, PDONE=12,
        WAIT_TRACE=13, NEXT_OUTPUT=14, NEXT_POOL=15;
    reg [3:0] state;
    reg trace_enabled;

    function automatic signed [15:0] requantize;
        input signed [47:0] value;
        input [7:0] amount;
        input apply_relu;
        reg signed [47:0] shifted;
        begin
            shifted = value >>> amount;
            if (apply_relu && shifted < 0) requantize=0;
            else if (shifted > 32767) requantize=16'sh7fff;
            else if (shifted < -32768) requantize=16'sh8000;
            else requantize=shifted[15:0];
        end
    endfunction

    // Synchronous memory reads; no memory-array resets (BRAM inference).
    // The explicit REQ/WAIT/USE states accommodate their read latency.
    always @(posedge clk) begin
        w_q <= weights[w_addr];
        input_q <= original[busy ? src_addr[12:0] : replay_addr];
        a_q <= buffer_a[src_addr];
        b_q <= buffer_b[src_addr];
        if (weight_we && !busy) weights[weight_addr] <= weight_data;
        if (input_we && !busy) original[input_addr] <= input_data;
        if (state==STORE && layer<4) begin
            if (layer==0 || layer==2)
                buffer_a[(oc*out_h+oy)*out_w+ox] <= requantize(acc,shift_amt,layer!=0);
            else buffer_b[(oc*out_h+oy)*out_w+ox] <= requantize(acc,shift_amt,1'b1);
        end
    end

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state<=IDLE; busy<=0; done<=0; score<=0; layer<=0;
            trace_enabled<=0; trace_valid<=0; trace_data<=0; trace_stage<=0;
            in_c<=1; in_h<=32; in_w<=156; out_c<=8; out_h<=32; out_w<=156;
            kernel<=1; stride<=1; pad<=0; weight_base<=0; bias_base<=8;
            oc<=0; oy<=0; ox<=0; ic<=0; ky<=0; kx<=0;
            pool_c<=0; pool_p<=0; src_addr<=0; w_addr<=0; acc<=0; padded<=0;
        end else begin
            done<=0;
            case (state)
                IDLE: if (start) begin
                    busy<=1; layer<=0; state<=CONFIG;
                    trace_enabled<=trace_enable; trace_valid<=0;
                end
                CONFIG: begin
                    case(layer)
                        0: begin
                            in_c<=1; in_h<=32; in_w<=156; out_c<=8; out_h<=32; out_w<=156;
                            kernel<=1; stride<=1; pad<=0; weight_base<=0; bias_base<=8;
                        end
                        1: begin
                            in_c<=8; in_h<=32; in_w<=156; out_c<=16; out_h<=16; out_w<=78;
                            kernel<=3; stride<=2; pad<=1; weight_base<=16; bias_base<=1168;
                        end
                        2: begin
                            in_c<=16; in_h<=16; in_w<=78; out_c<=32; out_h<=8; out_w<=39;
                            kernel<=3; stride<=2; pad<=1; weight_base<=1184; bias_base<=5792;
                        end
                        3: begin
                            in_c<=32; in_h<=8; in_w<=39; out_c<=32; out_h<=4; out_w<=20;
                            kernel<=3; stride<=2; pad<=1; weight_base<=5824; bias_base<=15040;
                        end
                        default: begin
                            in_c<=32; in_h<=1; in_w<=1; out_c<=1; out_h<=1; out_w<=1;
                            kernel<=1; stride<=1; pad<=0; weight_base<=15072; bias_base<=15104;
                        end
                    endcase
                    oc<=0; oy<=0; ox<=0; state<=BREQ;
                end
                BREQ: begin w_addr<=bias_base+oc; state<=BWAIT; end
                BWAIT: state<=BUSE;
                BUSE: begin
                    acc <= {{32{w_q[15]}},w_q} <<< bias_shift;
                    ic<=0; ky<=0; kx<=0; state<=TREQ;
                end
                TREQ: begin
                    sy=oy*stride+ky-pad;
                    sx=ox*stride+kx-pad;
                    padded <= (sy<0 || sy>=in_h || sx<0 || sx>=in_w);
                    if (sy<0 || sy>=in_h || sx<0 || sx>=in_w) src_addr<=0;
                    else src_addr<=(ic*in_h+sy)*in_w+sx;
                    w_addr<=weight_base+((oc*in_c+ic)*kernel+ky)*kernel+kx;
                    state<=TWAIT;
                end
                TWAIT: state<=MAC;
                MAC: begin
                    if (!padded) acc<=acc+{{16{product[31]}},product};
                    if (kx+1<kernel) begin kx<=kx+1; state<=TREQ; end
                    else if (ky+1<kernel) begin kx<=0; ky<=ky+1; state<=TREQ; end
                    else if (ic+1<in_c) begin kx<=0; ky<=0; ic<=ic+1; state<=TREQ; end
                    else state<=STORE;
                end
                STORE: begin
                    if (layer==4) begin
                        score<=requantize(acc,shift_amt,1'b0);
                        busy<=0; done<=1; state<=IDLE;
                    end else if (trace_enabled) begin
                        // Same value written by the untouched memory block above.
                        trace_data<=requantize(acc,shift_amt,layer!=0);
                        trace_stage<=layer; trace_valid<=1; state<=WAIT_TRACE;
                    end else if (ox+1<out_w) begin ox<=ox+1; state<=BREQ; end
                    else if (oy+1<out_h) begin ox<=0; oy<=oy+1; state<=BREQ; end
                    else if (oc+1<out_c) begin ox<=0; oy<=0; oc<=oc+1; state<=BREQ; end
                    else if (layer==3) begin
                        pool_c<=0; pool_p<=0; src_addr<=0; acc<=0; state<=PREQ;
                    end else begin layer<=layer+1; state<=CONFIG; end
                end
                // Conv3 is stored in channel-major order. Walk all 2560 words
                // with the existing read address register; keep the read wait.
                PREQ: state<=PWAIT;
                PWAIT: state<=PADD;
                PADD: begin
                    acc<=acc+{{32{b_q[15]}},b_q};
                    src_addr<={4'b0, (src_addr[11:0]+12'd1)};
                    if (pool_p==79) state<=PDONE;
                    else begin pool_p<=pool_p+1; state<=PREQ; end
                end
                PDONE: begin
                    pooled[pool_c]<=acc/48'sd80; // positive ReLU values: floor average
                    if (trace_enabled) begin
                        trace_data<=acc/48'sd80; trace_stage<=4;
                        trace_valid<=1; state<=WAIT_TRACE;
                    end else if (pool_c==31) begin layer<=4; state<=CONFIG; end
                    else begin pool_c<=pool_c+1; pool_p<=0; acc<=0; state<=PREQ; end
                end
                WAIT_TRACE: if(trace_ready) begin
                    trace_valid<=0;
                    state<=(trace_stage==4) ? NEXT_POOL : NEXT_OUTPUT;
                end
                // Advance only after the consumer has captured both output bytes.
                // These are the original STORE/PDONE advancement operations.
                NEXT_OUTPUT: begin
                    if (ox+1<out_w) begin ox<=ox+1; state<=BREQ; end
                    else if (oy+1<out_h) begin ox<=0; oy<=oy+1; state<=BREQ; end
                    else if (oc+1<out_c) begin ox<=0; oy<=0; oc<=oc+1; state<=BREQ; end
                    else if (layer==3) begin
                        pool_c<=0; pool_p<=0; src_addr<=0; acc<=0; state<=PREQ;
                    end else begin layer<=layer+1; state<=CONFIG; end
                end
                NEXT_POOL: begin
                    if (pool_c==31) begin layer<=4; state<=CONFIG; end
                    else begin pool_c<=pool_c+1; pool_p<=0; acc<=0; state<=PREQ; end
                end
                default: begin state<=IDLE; busy<=0; trace_valid<=0; end
            endcase
        end
    end
endmodule
