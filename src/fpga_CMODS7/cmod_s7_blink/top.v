module top (
    input  wire clk,
    input  wire rst,
    output reg  led
);
    reg [23:0] count;
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            count <= 24'd0;
            led   <= 1'b0;
        end else begin
            count <= count + 1'b1;
            led   <= count[23];
        end
    end
endmodule