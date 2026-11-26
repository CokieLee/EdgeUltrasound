module lif_step (
    input  wire signed [31:0] previous,
    input  wire signed [15:0] current,
    output wire signed [31:0] membrane,
    output wire spike
);
    localparam signed [31:0] THRESHOLD = 32'sd1024;
    // beta = round(0.9 * 32768) / 32768. Arithmetic shift rounds down.
    wire signed [47:0] beta_product = previous * 16'sd29491;
    wire signed [31:0] leak = beta_product >>> 15;
    wire signed [31:0] injection = {{16{current[15]}}, current};
    assign membrane = leak + injection - (previous > THRESHOLD ? THRESHOLD : 32'sd0);
    assign spike = membrane > THRESHOLD;
endmodule
