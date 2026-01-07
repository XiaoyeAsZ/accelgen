module timer #(
    parameter int WIDTH = 10,
    parameter int UP_BOUND = 10'b1111111111
) (
    input logic clk,
    input logic resetn,
    output logic [WIDTH-1:0] cnt
);

  always_ff @(posedge clk) begin
    if (!resetn) cnt <= 0;
    else cnt <= cnt + 1;
  end

endmodule

