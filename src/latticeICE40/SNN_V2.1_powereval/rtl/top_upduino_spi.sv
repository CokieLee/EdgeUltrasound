module top_upduino_spi #(
    parameter [31:0] SETTLE_CYCLES=32'd24000000, BENCH_CYCLES=32'd360000000
) (
    input wire clk,
    input wire spi_sck, input wire spi_mosi, input wire flash_cs_n,
    output wire spi_miso
);
    reg [7:0] startup = 8'b0;
    always @(posedge clk) startup <= {startup[6:0],1'b1};
    reg [1:0] flash_deselected = 2'b00;
    always @(posedge clk) flash_deselected <= {flash_deselected[0],flash_cs_n};
    wire rst = !startup[7] || !flash_deselected[1];
    wire miso_data;
    // Release immediately on flash selection, separate from the synchronized reset.
    assign spi_miso = flash_cs_n && !rst ? miso_data : 1'bz;

    wire [7:0] rx_data, tx_data;
    wire rx_valid, rx_error, tx_valid, tx_ready;
    spi_byte_link link(
        .clk(clk),.rst(rst),.spi_sck(spi_sck),.spi_mosi(spi_mosi),
        .flash_cs_n(flash_cs_n),.spi_miso(miso_data),
        .rx_data(rx_data),.rx_valid(rx_valid),.rx_error(rx_error),
        .tx_data(tx_data),.tx_valid(tx_valid),.tx_ready(tx_ready));
    snn_v2_stream #(.SETTLE_CYCLES(SETTLE_CYCLES), .BENCH_CYCLES(BENCH_CYCLES)) stream(
        .clk(clk),.rst(rst),.rx_data(rx_data),.rx_valid(rx_valid),.rx_error(rx_error),
        .tx_data(tx_data),.tx_valid(tx_valid),.tx_ready(tx_ready));
endmodule
