interface bf16_bus_if;
  logic       sign;
  logic [7:0] exponent;
  logic [6:0] mantissa;
  logic       valid;
  logic       ready;

  modport producer(input ready, output sign, mantissa, exponent, valid);
  modport consumer(input sign, mantissa, exponent, valid, output ready);

endinterface
