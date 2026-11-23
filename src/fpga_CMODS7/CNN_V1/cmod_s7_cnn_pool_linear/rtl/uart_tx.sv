// UART 8N1 transmitter. Accept data on a clock with valid && ready.
module uart_tx #(
    parameter integer CLOCK_HZ = 12_000_000,
    parameter integer BAUD = 115_200
) (
    input wire clk,
    input wire rst,
    input wire [7:0] data,
    input wire valid,
    output wire ready,
    output reg tx
);
    localparam integer CPB = (CLOCK_HZ + BAUD/2) / BAUD;
    localparam [1:0] IDLE=0, START=1, DATA=2, STOP=3;
    reg [1:0] state;
    reg [15:0] ticks;
    reg [2:0] bit_index;
    reg [7:0] shift_reg;
    assign ready = (state == IDLE);

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= IDLE;
            ticks <= 0;
            bit_index <= 0;
            shift_reg <= 0;
            tx <= 1;
        end else begin
            case (state)
                IDLE: begin
                    tx <= 1;
                    if (valid) begin
                        shift_reg <= data;
                        ticks <= CPB-1;
                        tx <= 0;
                        state <= START;
                    end
                end
                START: if (ticks != 0) ticks <= ticks-1'b1;
                    else begin
                        ticks <= CPB-1;
                        bit_index <= 0;
                        tx <= shift_reg[0];
                        state <= DATA;
                    end
                DATA: if (ticks != 0) ticks <= ticks-1'b1;
                    else begin
                        ticks <= CPB-1;
                        if (bit_index == 7) begin
                            tx <= 1;
                            state <= STOP;
                        end else begin
                            bit_index <= bit_index+1'b1;
                            tx <= shift_reg[bit_index+1'b1];
                        end
                    end
                STOP: if (ticks != 0) ticks <= ticks-1'b1;
                    else state <= IDLE;
                default: begin state <= IDLE; tx <= 1; end
            endcase
        end
    end
endmodule
