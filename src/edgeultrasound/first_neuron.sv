module lif_neuron #(
    parameter int WIDTH = 16,
    parameter int NUM_INPUTS = 8,                 // Number of incoming connections (fan-in)
    parameter logic signed [WIDTH-1:0] DEFAULT_THRESHOLD = 16'sd100,
    parameter int LEAK_SHIFT = 4
)(
    input  logic                          clk,
    input  logic                          rst_n, // reset signal
    input  logic [NUM_INPUTS-1:0]         spike_in,       // Vector of incoming spikes
    input  logic signed [WIDTH-1:0]       weight [NUM_INPUTS], // Unpacked array of weights for each input
    output logic                          spike_out,
    output logic signed [WIDTH-1:0]       potential_out // for debugging
);
    // internally defined signals
    logic signed [WIDTH-1:0] membrane_potential;
    logic signed [WIDTH-1:0] leak_term;
    logic signed [WIDTH-1:0] total_incoming_current;

    // Combinational logic: Sum up the weights of all inputs that are currently spiking
    always_comb begin
        total_incoming_current = '0;
        for (int i = 0; i < NUM_INPUTS; i++) begin
            if (spike_in[i]) begin
                total_incoming_current += weight[i];
            end
        end
    end

    // Sequential logic: Update membrane potential and check for firing
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            membrane_potential <= '0;
            spike_out          <= 1'b0;
        end else begin
            // Calculate leak
            leak_term = membrane_potential >>> LEAK_SHIFT;

            // Integrate: old potential - leak + sum of active incoming weights
            membrane_potential <= (membrane_potential - leak_term) + total_incoming_current;

            // Threshold check
            if (membrane_potential >= DEFAULT_THRESHOLD) begin
                spike_out          <= 1'b1;
                membrane_potential <= '0; // Reset after firing
            end else begin
                spike_out          <= 1'b0;
            end
        end
    end

    assign potential_out = membrane_potential;

endmodule