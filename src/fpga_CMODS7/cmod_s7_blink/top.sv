module top (
    input  logic clk,
    input  logic rst,
    output logic led
);
    logic [23:0] count;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            count <= 24'd0;
            led   <= 1'b0;
        end else begin
            count <= count + 24'd1;
            led   <= count[23];
        end
    end
endmodule