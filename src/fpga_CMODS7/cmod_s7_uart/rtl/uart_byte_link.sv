// Receive bytes, expose them to an always-ready consumer, and echo them to PC.
// The 256-byte FIFO buffers the echo ONLY, not an entire tensor.
module uart_byte_link #(
    parameter integer CLOCK_HZ = 12_000_000,
    parameter integer BAUD = 115_200
) (
    input wire clk,
    input wire rst,
    input wire uart_rx_pin,
    output wire uart_tx_pin,
    output wire [7:0] byte_data,
    output wire byte_valid,
    output reg [31:0] received_bytes,
    output reg overflow_error,
    output reg framing_error
);
    wire rx_error;
    wire tx_ready;
    reg [7:0] echo_fifo [0:255];
    reg [7:0] write_ptr, read_ptr;
    reg [8:0] count;
    wire pop = (count != 0) && tx_ready;
    wire push = byte_valid && ((count < 256) || pop);

    uart_rx #(.CLOCK_HZ(CLOCK_HZ), .BAUD(BAUD)) rx_inst (
        .clk(clk), .rst(rst), .rx(uart_rx_pin),
        .data(byte_data), .valid(byte_valid), .framing_error(rx_error)
    );
    uart_tx #(.CLOCK_HZ(CLOCK_HZ), .BAUD(BAUD)) tx_inst (
        .clk(clk), .rst(rst), .data(echo_fifo[read_ptr]),
        .valid(count != 0), .ready(tx_ready), .tx(uart_tx_pin)
    );

    // No reset of the memory array: only reset its pointers and occupancy.
    always @(posedge clk) begin
        if (!rst && push) echo_fifo[write_ptr] <= byte_data;
    end
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            write_ptr <= 0;
            read_ptr <= 0;
            count <= 0;
            received_bytes <= 0;
            overflow_error <= 0;
            framing_error <= 0;
        end else begin
            if (byte_valid) received_bytes <= received_bytes+1'b1;
            if (rx_error) framing_error <= 1;
            if (byte_valid && !push) overflow_error <= 1;
            if (push) write_ptr <= write_ptr+1'b1;
            if (pop) read_ptr <= read_ptr+1'b1;
            case ({push, pop})
                2'b10: count <= count+1'b1;
                2'b01: count <= count-1'b1;
                default: count <= count;
            endcase
        end
    end
endmodule
