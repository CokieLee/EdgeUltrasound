// UART 8N1 receiver. Integer baud divider; idle line is high.
module uart_rx #(
    parameter integer CLOCK_HZ = 12_000_000,
    parameter integer BAUD = 115_200
) (
    input  wire clk,
    input  wire rst,
    input  wire rx,
    output reg [7:0] data,
    output reg valid,
    output reg framing_error
);
    localparam integer CPB = (CLOCK_HZ + BAUD/2) / BAUD;
    localparam [1:0] IDLE=0, START=1, DATA=2, STOP=3;
    (* ASYNC_REG = "TRUE" *) reg rx_meta, rx_sync;
    reg [1:0] state;
    reg [15:0] ticks;
    reg [2:0] bit_index;
    reg [7:0] shift_reg;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            rx_meta <= 1'b1;
            rx_sync <= 1'b1;
        end else begin
            rx_meta <= rx;
            rx_sync <= rx_meta;
        end
    end

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= IDLE;
            ticks <= 0;
            bit_index <= 0;
            shift_reg <= 0;
            data <= 0;
            valid <= 0;
            framing_error <= 0;
        end else begin
            valid <= 0;
            framing_error <= 0;
            case (state)
                IDLE: if (!rx_sync) begin
                    ticks <= CPB/2-1;
                    state <= START;
                end
                START: if (ticks != 0) ticks <= ticks-1'b1;
                    else if (rx_sync) state <= IDLE; // false start
                    else begin
                        ticks <= CPB-1;
                        bit_index <= 0;
                        state <= DATA;
                    end
                DATA: if (ticks != 0) ticks <= ticks-1'b1;
                    else begin
                        shift_reg[bit_index] <= rx_sync;
                        ticks <= CPB-1;
                        if (bit_index == 7) state <= STOP;
                        else bit_index <= bit_index+1'b1;
                    end
                STOP: if (ticks != 0) ticks <= ticks-1'b1;
                    else begin
                        if (rx_sync) begin
                            data <= shift_reg;
                            valid <= 1;
                        end else framing_error <= 1;
                        state <= IDLE;
                    end
                default: state <= IDLE;
            endcase
        end
    end
endmodule
