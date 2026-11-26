#!/usr/bin/env bash
set -euo pipefail

mkdir -p build-upduino-blink
cd build-upduino-blink

cat > blink.sv <<'SV'
module blink(output wire led_green);
    wire clk;
    reg [23:0] counter = 24'd0;

    // Divide the nominal 48 MHz internal oscillator by four.
    SB_HFOSC #(.CLKHF_DIV("0b10")) oscillator (
        .CLKHFPU(1'b1), .CLKHFEN(1'b1), .CLKHF(clk)
    );

    always @(posedge clk)
        counter <= counter + 1'b1;

    SB_RGBA_DRV #(.RGB0_CURRENT("0b000001")) led_driver (
        .CURREN(1'b1), .RGBLEDEN(1'b1),
        .RGB0PWM(counter[23]), .RGB1PWM(1'b0), .RGB2PWM(1'b0),
        .RGB0(led_green), .RGB1(), .RGB2()
    );
endmodule
SV

cat > upduino.pcf <<'PCF'
set_io led_green 39
PCF

yosys -p 'read_verilog -sv blink.sv; synth_ice40 -top blink -json blink.json'
nextpnr-ice40 --up5k --package sg48 --freq 12 --pcf upduino.pcf --json blink.json --asc blink.asc
icepack blink.asc blink.bin
