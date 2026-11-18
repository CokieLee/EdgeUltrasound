module top #(
    parameter integer CLOCK_HZ = 12_000_000,
    parameter integer BAUD = 115_200
) (
    input wire clk,
    input wire rst_button,
    input wire uart_rx,
    output wire uart_tx,
    output wire [3:0] led
);
    // Button 0: asynchronous assertion, synchronous release of reset.
    (* ASYNC_REG = "TRUE" *) reg [1:0] reset_sync;
    always @(posedge clk or posedge rst_button) begin
        if (rst_button) reset_sync <= 2'b11;
        else reset_sync <= {reset_sync[0], 1'b0};
    end
    wire rst = reset_sync[1];
    wire [7:0] byte_data;
    wire byte_valid;
    wire [31:0] received_bytes;
    wire overflow_error, framing_error;
    reg [23:0] heartbeat;
    reg seen_byte;
    always @(posedge clk or posedge rst) begin
        if (rst) begin heartbeat <= 0; seen_byte <= 0; end
        else begin
            heartbeat <= heartbeat+1'b1;
            if (byte_valid) seen_byte <= 1;
        end
    end

    uart_byte_link #(.CLOCK_HZ(CLOCK_HZ), .BAUD(BAUD)) link_inst (
        .clk(clk), .rst(rst), .uart_rx_pin(uart_rx), .uart_tx_pin(uart_tx),
        .byte_data(byte_data), .byte_valid(byte_valid),
        .received_bytes(received_bytes), .overflow_error(overflow_error),
        .framing_error(framing_error)
    );
    // Future input decoder connects to byte_data/byte_valid here.
    // byte_valid is one clock per received byte; there is no consumer backpressure.
    assign led[0] = heartbeat[23];
    assign led[1] = seen_byte;
    assign led[2] = overflow_error | framing_error;
    assign led[3] = received_bytes[8];
endmodule
