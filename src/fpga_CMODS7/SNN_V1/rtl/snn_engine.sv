// Four-step, layer-major SNN for one 32x156 RF channel.
// Pack all four spikes for one neuron into a nibble.
module snn_engine (
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
    (* ram_style = "block" *) reg [7:0] weights_lo [0:15104], weights_hi [0:15104];
    (* ram_style = "block" *) reg [7:0] original_lo [0:4991], original_hi [0:4991];
    (* ram_style = "block" *) reg [7:0] stem_lo [0:39935], stem_hi [0:39935];
    (* ram_style = "block" *) reg [3:0] spikes1 [0:19967], spikes2 [0:9983];
    reg [8:0] counts [0:31]; // Up to 80 locations * 4 steps per channel.

    reg signed [15:0] w_q, input_q, stem_q;
    reg [3:0] s1_q, s2_q;
    assign replay_data = input_q;

    reg [15:0] src_addr, dst_addr;
    reg [13:0] w_addr;
    wire [12:0] original_addr = busy ? src_addr[12:0] : replay_addr;
    reg signed [47:0] acc;
    reg [2:0] layer;
    reg [1:0] step;
    reg signed [31:0] membrane;
    reg [3:0] spike_word;
    reg [8:0] channel_count;
    reg [5:0] in_c, in_h, out_c, out_h;
    reg [7:0] in_w, out_w;
    reg [1:0] kernel, stride;
    reg pad;
    reg [13:0] weight_base, bias_base;
    reg [4:0] oc, oy, ic;
    reg [7:0] ox;
    reg [1:0] ky, kx;
    wire signed [7:0] sy = $signed({3'b0,oy}) * $signed({6'b0,stride})
                         + $signed({6'b0,ky}) - $signed({7'b0,pad});
    wire signed [9:0] sx = $signed({2'b0,ox}) * $signed({8'b0,stride})
                         + $signed({8'b0,kx}) - $signed({9'b0,pad});
    wire outside = sy < 0 || sy >= $signed({2'b0,in_h})
                || sx < 0 || sx >= $signed({2'b0,in_w});
    wire [10:0] source_row = ic*in_h + sy[4:0];
    reg padded, trace_enabled;
    reg [4:0] state, resume_state;
    localparam IDLE=0, CONFIG=1, BREQ=2, BWAIT=3, BUSE=4,
        TREQ=5, TWAIT=6, MAC=7, STORE=8, LIF=9, SAVE_SPIKES=10,
        ADVANCE=11, WAIT_TRACE=12, COUNT_OUT=13, DIV_INIT=14,
        DIV_STEP=15, FINISH=16;

    wire [7:0] shift_amt = right_shifts >> (layer*8);
    wire [7:0] bias_shift = bias_left_shifts >> (layer*8);
    wire signed [47:0] extended_w = {{32{w_q[15]}},w_q};
    wire signed [15:0] dense_operand =
        layer==0 ? input_q : layer==1 ? stem_q : $signed({7'b0,counts[ic[4:0]]});
    wire signed [31:0] dense_product = dense_operand * w_q;
    wire selected_spike = layer==2 ? s1_q[step] : s2_q[step];
    wire signed [47:0] mac_term = (layer==2 || layer==3)
        ? (selected_spike ? extended_w : 48'sd0)
        : {{16{dense_product[31]}},dense_product};

    function automatic signed [15:0] saturate;
        input signed [47:0] value;
        begin
            if (value > 32767) saturate=16'sh7fff;
            else if (value < -32768) saturate=16'sh8000;
            else saturate=value[15:0];
        end
    endfunction

    wire signed [15:0] current_word = saturate(acc >>> shift_amt);
    reg signed [15:0] lif_current;
    wire signed [31:0] membrane_next;
    wire spike;
    lif_step lif(.previous(membrane),.current(lif_current),
                 .membrane(membrane_next),.spike(spike));

    wire [2:0] neuron_spike_count = {2'b0,spike_word[0]} + {2'b0,spike_word[1]}
                                  + {2'b0,spike_word[2]} + {2'b0,spike_word[3]};
    wire [8:0] count_next = channel_count + {6'b0,neuron_spike_count};
    wire last_in_channel = (ox+1==out_w && oy+1==out_h);

    // Restoring unsigned divide by 320, one bit per cycle. For negative n,
    // divide (abs(n)+319) and negate to obtain floor(n/320).
    reg [31:0] div_bits;
    reg [8:0] div_rem;
    reg [5:0] div_count;
    reg div_negative;
    wire [9:0] div_trial = {div_rem,div_bits[31]};
    wire div_take = div_trial >= 10'd320;
    wire [9:0] div_remainder_next = div_take ? div_trial-10'd320 : div_trial;
    wire signed [47:0] div_value = div_negative
        ? -$signed({16'b0,div_bits}) : $signed({16'b0,div_bits});

    always @(posedge clk) begin
        w_q <= {weights_hi[w_addr],weights_lo[w_addr]};
        input_q <= {original_hi[original_addr],original_lo[original_addr]};
        stem_q <= {stem_hi[src_addr],stem_lo[src_addr]};
        s1_q <= spikes1[src_addr];
        s2_q <= spikes2[src_addr];
        if (weight_we && !busy) begin
            weights_lo[weight_addr] <= weight_data[7:0];
            weights_hi[weight_addr] <= weight_data[15:8];
        end
        if (input_we && !busy) begin
            original_lo[input_addr] <= input_data[7:0];
            original_hi[input_addr] <= input_data[15:8];
        end
        if (state==STORE && layer==0) begin
            stem_lo[dst_addr] <= current_word[7:0];
            stem_hi[dst_addr] <= current_word[15:8];
        end
        if (state==SAVE_SPIKES) begin
            if (layer==1) spikes1[dst_addr] <= spike_word;
            if (layer==2) spikes2[dst_addr] <= spike_word;
        end
    end

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state<=IDLE; resume_state<=IDLE; busy<=0; done<=0; score<=0;
            trace_enabled<=0; trace_valid<=0; trace_stage<=0; trace_data<=0;
            layer<=0; step<=0; membrane<=0; lif_current<=0; spike_word<=0;
            channel_count<=0; src_addr<=0; dst_addr<=0; w_addr<=0; acc<=0;
            in_c<=1; in_h<=32; in_w<=156; out_c<=8; out_h<=32; out_w<=156;
            kernel<=1; stride<=1; pad<=0; weight_base<=0; bias_base<=8;
            oc<=0; oy<=0; ox<=0; ic<=0; ky<=0; kx<=0; padded<=0;
            div_bits<=0; div_rem<=0; div_count<=0; div_negative<=0;
        end else begin
            done<=0;
            case (state)
                IDLE: if (start) begin
                    busy<=1; layer<=0; state<=CONFIG;
                    trace_enabled<=trace_enable; trace_valid<=0;
                end
                CONFIG: begin
                    case (layer)
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
                    oc<=0; oy<=0; ox<=0; step<=0; membrane<=0; spike_word<=0;
                    dst_addr<=0; channel_count<=0; state<=BREQ;
                end
                BREQ: begin w_addr<=bias_base+oc; state<=BWAIT; end
                BWAIT: state<=BUSE;
                BUSE: begin
                    // Apply the head bias to the undivided spatiotemporal sum.
                    acc <= layer==4 ? ((extended_w <<< 8)+(extended_w <<< 6))
                                    : (extended_w <<< bias_shift);
                    ic<=0; ky<=0; kx<=0; state<=TREQ;
                end
                TREQ: begin
                    padded<=outside;
                    src_addr<=outside ? 16'b0 : source_row*in_w+sx[7:0];
                    w_addr<=weight_base+((oc*in_c+ic)*kernel+ky)*kernel+kx;
                    state<=TWAIT;
                end
                TWAIT: state<=MAC;
                MAC: begin
                    if (!padded) acc<=acc+mac_term;
                    if (kx+1<kernel) begin kx<=kx+1; state<=TREQ; end
                    else if (ky+1<kernel) begin kx<=0; ky<=ky+1; state<=TREQ; end
                    else if (ic+1<in_c) begin kx<=0; ky<=0; ic<=ic+1; state<=TREQ; end
                    else state<=STORE;
                end
                STORE: begin
                    if (layer==4) state<=DIV_INIT;
                    else if (layer==0) begin
                        if (trace_enabled) begin
                            trace_data<=current_word; trace_stage<=0; trace_valid<=1;
                            resume_state<=ADVANCE; state<=WAIT_TRACE;
                        end else state<=ADVANCE;
                    end else begin
                        lif_current<=current_word; state<=LIF;
                    end
                end
                LIF: begin
                    membrane<=membrane_next;
                    spike_word[step]<=spike;
                    if (step==3) state<=SAVE_SPIKES;
                    else begin
                        step<=step+1'b1;
                        if (layer!=1) state<=BREQ;
                    end
                end
                SAVE_SPIKES: begin
                    if (layer==3) begin
                        channel_count<=last_in_channel ? 9'b0 : count_next;
                        if (last_in_channel) counts[oc[4:0]]<=count_next;
                    end
                    if (trace_enabled) begin
                        trace_data<={12'b0,spike_word}; trace_stage<=layer; trace_valid<=1;
                        resume_state<=ADVANCE; state<=WAIT_TRACE;
                    end else state<=ADVANCE;
                end
                ADVANCE: begin
                    step<=0; membrane<=0; spike_word<=0; dst_addr<=dst_addr+1'b1;
                    state<=BREQ;
                    if (ox+1<out_w) ox<=ox+1;
                    else if (oy+1<out_h) begin ox<=0; oy<=oy+1; end
                    else if (oc+1<out_c) begin ox<=0; oy<=0; oc<=oc+1; end
                    else if (layer==3 && trace_enabled) begin
                        oc<=0; state<=COUNT_OUT;
                    end else begin layer<=layer+1'b1; state<=CONFIG; end
                end
                COUNT_OUT: begin
                    trace_data<={7'b0,counts[oc[4:0]]};
                    trace_stage<=4; trace_valid<=1; state<=WAIT_TRACE;
                    if (oc==31) begin
                        layer<=4; resume_state<=CONFIG;
                    end else begin oc<=oc+1; resume_state<=COUNT_OUT; end
                end
                WAIT_TRACE: if (trace_ready) begin trace_valid<=0; state<=resume_state; end
                DIV_INIT: begin
                    div_negative<=acc[31]; div_rem<=0; div_count<=0;
                    div_bits<=acc[31] ? (~acc[31:0]+32'd320) : acc[31:0];
                    state<=DIV_STEP;
                end
                DIV_STEP: begin
                    div_bits<={div_bits[30:0],div_take};
                    div_rem<=div_remainder_next[8:0];
                    if (div_count==31) state<=FINISH;
                    else div_count<=div_count+1'b1;
                end
                FINISH: begin
                    score<=saturate(div_value >>> shift_amt);
                    busy<=0; done<=1; state<=IDLE;
                end
                default: begin state<=IDLE; busy<=0; trace_valid<=0; end
            endcase
        end
    end
endmodule
