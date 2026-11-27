module probe (
    input wire clk,
    input wire spi_mosi,
    input wire flash_cs_n,
    output wire spi_miso
);
    // The echo needs no clock. After 1024 input clock edges, invert it.
    reg [10:0] edges_seen = 11'd0;
    always @(posedge clk)
        if (!edges_seen[10]) edges_seen <= edges_seen + 1'b1;
    assign spi_miso = flash_cs_n ? spi_mosi ^ edges_seen[10] : 1'bz;
endmodule
