module test (
    input logic clock,
    input logic resetn,
    input logic [15:0] din,
    output logic [15:0] dout
);
  always_ff @(posedge clock) begin
    if (!resetn) dout <= 0;
    else dout <= din;
  end
endmodule
