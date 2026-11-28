// SPI mode 0, MSB first. Each 16-bit slot is a flag byte and a data byte.
// Flag 0 means no byte, flag 1 means the data byte is valid.
module spi_byte_link (
    input wire clk, input wire rst,
    input wire spi_sck, input wire spi_mosi, input wire flash_cs_n,
    output wire spi_miso,
    output reg [7:0] rx_data,
    output reg rx_valid, output reg rx_error,
    input wire [7:0] tx_data,
    input wire tx_valid,
    output reg tx_ready
);
    (* ASYNC_REG = "TRUE" *) reg [2:0] sck;
    (* ASYNC_REG = "TRUE" *) reg [1:0] mosi, cs_n;
    reg [3:0] index;
    reg [14:0] incoming;
    reg [7:0] outgoing;
    reg outgoing_valid;

    wire rising = sck[1] && !sck[2];
    wire falling = !sck[1] && sck[2];
    wire data_bit = index[3] ? outgoing[~index[2:0]] : (index==7 && outgoing_valid);
    assign spi_miso = data_bit;

    always @(posedge clk) begin
        if (rst) begin
            sck<=0; mosi<=0; cs_n<=0;
        end else begin
            sck<={sck[1:0],spi_sck};
            mosi<={mosi[0],spi_mosi};
            cs_n<={cs_n[0],flash_cs_n};
        end
    end

    always @(posedge clk) begin
        rx_valid<=0; rx_error<=0; tx_ready<=0;
        if (rst || !cs_n[1]) begin
            index<=0; incoming<=0; outgoing<=0; outgoing_valid<=0; rx_data<=0;
        end else begin
            if (rising) begin
                incoming<={incoming[13:0],mosi[1]};
                if (index==0) begin
                    outgoing_valid<=tx_valid;
                    outgoing<=tx_valid ? tx_data : 8'b0;
                end
                if (index==15) begin
                    if (incoming[14:7]==8'h01) begin
                        rx_valid<=1;
                        rx_data<={incoming[6:0],mosi[1]};
                    end else if (incoming[14:7]!=8'h00) begin
                        rx_error<=1;
                    end
                    tx_ready<=outgoing_valid;
                end
            end
            if (falling) index<=index+1'b1;
        end
    end
endmodule
