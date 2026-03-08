`include "bf16_buf_if.svh"

module mulf_bf16_bf16 (
    input clk,
    input resetn,
    bf16_bus_if.consumer a_bus,
    bf16_bus_if.consumer b_bus,
    bf16_bus_if.producer out_bus
);

  logic a_zero, b_zero;
  logic a_inf, b_inf;
  logic a_nan, b_nan;

  assign a_zero = (a_bus.exponent == 0) && (a_bus.mantissa == 0);
  assign b_zero = (b_bus.exponent == 0) && (b_bus.mantissa == 0);

  assign a_inf  = (a_bus.exponent == 8'hFF) && (a_bus.mantissa == 0);
  assign b_inf  = (b_bus.exponent == 8'hFF) && (b_bus.mantissa == 0);

  assign a_nan  = (a_bus.exponent == 8'hFF) && (a_bus.mantissa != 0);
  assign b_nan  = (b_bus.exponent == 8'hFF) && (b_bus.mantissa != 0);



  logic [15:0] mantissa_mul;
  assign mantissa_mul = {1'b1, a_bus.mantissa} * {1'b1, b_bus.mantissa};

  logic [8:0] exponent_add;
  assign exponent_add = {1'b0, a_bus.exponent} + {1'b0, b_bus.exponent} - 8'd127;


  /** Normalization stage **/
  logic need_norm;
  assign need_norm = mantissa_mul[15];

  logic [15:0] mantissa_mul_norm;
  assign mantissa_mul_norm = need_norm ? mantissa_mul >> 1'b1 : mantissa_mul;

  logic [8:0] exponent_add_norm;
  assign exponent_add_norm = need_norm ? exponent_add + 1'b1 : exponent_add;

  /** Rounding Stage **/
  logic [8:0] mantissa_norm;
  assign mantissa_norm = mantissa_mul_norm[15:7];

  logic guard, round_bit, sticky, lsb;
  assign lsb = mantissa_mul_norm[7];
  assign guard = mantissa_mul_norm[6];
  assign round_bit = mantissa_mul_norm[5];
  assign sticky = |mantissa_mul_norm[4:0];

  logic need_round;
  assign need_round = guard && (round_bit || sticky || lsb);

  logic [8:0] mantissa_norm_round;
  assign mantissa_norm_round = need_round ? mantissa_norm + 1'b1 : mantissa_norm;

  logic mantissa_norm_round_overflow;
  assign mantissa_norm_overflow = mantissa_norm_round[8];

  logic [8:0] mantissa_round_final;
  assign mantissa_round_final = mantissa_norm_round_overflow ? mantissa_norm_round >> 1'b1 : mantissa_norm_round;

  logic [8:0] exponent_round_final;
  assign exponent_round_final = mantissa_norm_round_overflow ? exponent_add_norm + 1'b1 :exponent_add_norm;


  /** Final Stage **/
  logic sign;
  logic [7:0] exponent;
  logic [6:0] mantissa;

  assign sign = a_bus.sign ^ b_bus.sign;

  always_comb begin
    if (a_nan || b_nan || a_inf || b_inf  // a or b is nan or inf
        || exponent_round_final > 9'd254  // Rounding overflow
        )
      exponent = 8'hFF;
    else exponent = exponent_round_final[7:0];
  end

  always_comb begin
    if (a_nan || b_nan  // a or b is nan -> nan
        || (a_inf && b_zero) || (b_inf && a_zero)  // one is inf the other is 0 -> nan
        )
      mantissa = 7'h40;
    else if (a_inf || b_inf) mantissa = 7'h00;
    else mantissa = mantissa_round_final[6:0];
  end


  /** Producer modport **/
  always_ff @(posedge clk) begin
    if (~resetn) out_bus.valid <= 1'b0;
    else out_bus.valid <= a_bus.valid & b_bus.valid;
  end

  always_ff @(posedge clk) begin
    out_bus.sign <= sign;
    out_bus.exponent <= exponent;
    out_bus.mantissa <= mantissa;
  end

  /** Consumer modport **/
  assign a_bus.ready = 1'b1;
  assign b_bus.ready = 1'b1;

endmodule


