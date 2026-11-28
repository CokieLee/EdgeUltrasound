// Fixed checkpoint, one token buffer, and one shared dense multiply-accumulate datapath.
module snn_v2_core (
    input wire clk, input wire rst, input wire new_frame, input wire start_token,
    input wire token_we, input wire [5:0] token_waddr,
    input wire [15:0] token_data,
    input wire [2:0] trace_neuron,
    output reg [7:0] spikes1, output reg [7:0] spikes2,
    output wire signed [31:0] trace_v1, output wire signed [31:0] trace_v2,
    output reg busy, output reg token_done, output reg frame_done,
    output reg signed [31:0] score
);
    localparam signed [31:0] THRESHOLD = 32'sd131072; // 0.5 in Q18
    localparam [1:0] FC1=0, FC2=1, SKIP=2, HEAD=3;
    localparam [2:0] IDLE=0, BIAS_READ=1, BIAS_USE=2, TERM_READ=3,
                     TERM_USE=4, UPDATE=5;
    reg [2:0] state;
    reg [1:0] phase;
    reg [3:0] token_index;
    reg [2:0] neuron;
    reg [5:0] col;

    // Separate byte lanes avoid treating a model data bit as BRAM parity.
    (* ram_style = "block" *) reg [7:0] weight_lo[0:664];
    (* ram_style = "block" *) reg [7:0] weight_hi[0:664];
    initial begin
        $readmemh("weights/lo.mem", weight_lo);
        $readmemh("weights/hi.mem", weight_hi);
    end
    reg [15:0] token[0:63];
    reg signed [31:0] m1[0:7], m2[0:7], voltage_total[0:7];
    reg signed [55:0] acc, skip_total;
    reg signed [15:0] weight_q;
    reg [15:0] input_q;
    reg [9:0] weight_addr;
    integer i;

    always @* begin
        case (phase)
            FC1: weight_addr = state==BIAS_READ ? 10'd512+{7'b0,neuron}
                                              : {1'b0,neuron,col};
            FC2: weight_addr = state==BIAS_READ ? 10'd584+{7'b0,neuron}
                                              : 10'd520+{4'b0,neuron,col[2:0]};
            SKIP: weight_addr = 10'd601+{4'b0,col};
            default: weight_addr = state==BIAS_READ ? 10'd600
                                                  : 10'd592+{7'b0,col[2:0]};
        endcase
    end

    always @(posedge clk) begin
        weight_q <= {weight_hi[weight_addr],weight_lo[weight_addr]};
        input_q <= token[col];
        if (token_we) token[token_waddr] <= token_data;
    end

    wire signed [55:0] weight_extended = {{40{weight_q[15]}},weight_q};
    wire signed [31:0] operand =
        phase==HEAD ? voltage_total[col[2:0]] : $signed({16'b0,input_q});
    wire signed [47:0] product = operand * weight_q;
    wire signed [55:0] term = phase==FC2
        ? (spikes1[col[2:0]] ? weight_extended : 56'sd0)
        : {{8{product[47]}},product};
    wire signed [55:0] current1_wide = (acc+56'sd256) >>> 9; // Q27 -> Q18
    wire signed [55:0] current2_wide = acc <<< 3;           // Q15 -> Q18
    wire signed [31:0] current = phase==FC1 ? current1_wide[31:0] : current2_wide[31:0];

    wire signed [31:0] previous = phase==FC1 ? m1[neuron] : m2[neuron];
    wire signed [34:0] previous_wide = {{3{previous[31]}},previous};
    wire signed [34:0] leaked_wide = ((previous_wide <<< 3) - previous_wide) >>> 3;
    wire signed [31:0] voltage = $signed(leaked_wide[31:0]) + current;
    wire spike = voltage >= THRESHOLD;
    wire signed [31:0] membrane = voltage - (spike ? THRESHOLD : 32'sd0);
    assign trace_v1 = m1[trace_neuron] + (spikes1[trace_neuron] ? THRESHOLD : 32'sd0);
    assign trace_v2 = m2[trace_neuron] + (spikes2[trace_neuron] ? THRESHOLD : 32'sd0);
    // Head accumulates Q33 over 16 tokens. Remove 4 average and 13 fractional bits.
    wire signed [55:0] rounded_score = (acc+56'sd65536) >>> 17;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state<=IDLE; phase<=FC1; token_index<=0; neuron<=0; col<=0;
            busy<=0; token_done<=0; frame_done<=0; score<=0;
            spikes1<=0; spikes2<=0; skip_total<=0; acc<=0;
            for (i=0; i<8; i=i+1) begin
                m1[i]<=0; m2[i]<=0; voltage_total[i]<=0;
            end
        end else begin
            token_done<=0; frame_done<=0;
            case (state)
                IDLE: if (new_frame) begin
                    token_index<=0; skip_total<=0;
                    spikes1<=0; spikes2<=0;
                    for (i=0; i<8; i=i+1) begin
                        m1[i]<=0; m2[i]<=0; voltage_total[i]<=0;
                    end
                end else if (start_token) begin
                    busy<=1; phase<=FC1; neuron<=0; state<=BIAS_READ;
                end
                BIAS_READ: state<=BIAS_USE;
                BIAS_USE: begin
                    case (phase)
                        FC1: acc<=weight_extended <<< 12;
                        FC2: acc<=weight_extended;
                        default: acc<=(weight_extended <<< 22)+(skip_total <<< 6);
                    endcase
                    col<=0; state<=TERM_READ;
                end
                TERM_READ: state<=TERM_USE;
                TERM_USE: begin
                    acc<=acc+term;
                    if (((phase==FC1 || phase==SKIP) && col==63)
                        || ((phase==FC2 || phase==HEAD) && col==7)) state<=UPDATE;
                    else begin col<=col+1'b1; state<=TERM_READ; end
                end
                UPDATE: case (phase)
                    FC1: begin
                        m1[neuron]<=membrane; spikes1[neuron]<=spike;
                        state<=BIAS_READ;
                        if (neuron==7) begin phase<=FC2; neuron<=0; end
                        else neuron<=neuron+1'b1;
                    end
                    FC2: begin
                        m2[neuron]<=membrane; spikes2[neuron]<=spike;
                        voltage_total[neuron]<=voltage_total[neuron]+voltage;
                        if (neuron==7) begin
                            phase<=SKIP; col<=0; acc<=0; state<=TERM_READ;
                        end else begin neuron<=neuron+1'b1; state<=BIAS_READ; end
                    end
                    SKIP: begin
                        skip_total<=skip_total+acc;
                        if (token_index==15) begin phase<=HEAD; state<=BIAS_READ; end
                        else begin
                            token_index<=token_index+1'b1;
                            busy<=0; token_done<=1; state<=IDLE;
                        end
                    end
                    HEAD: begin
                        score<=rounded_score[31:0]; frame_done<=1; token_done<=1;
                        busy<=0; state<=IDLE;
                    end
                endcase
                default: begin state<=IDLE; busy<=0; end
            endcase
        end
    end
endmodule
