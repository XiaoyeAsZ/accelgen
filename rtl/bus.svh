interface data_bus;
    logic ready;
    logic valid;
    logic [31:0] data;

    modport up_stream (
        input ready,
        output valid, data
    );

    modport down_stream (
        output ready,
        input valid, data
    );
endinterface

module broadcast #(
    parameter integer N_FANOUT = 7
) (
    data_bus.up_stream ibus,
    data_bus.down_stream [N_FANOUT-1:0] obus
);

    genvar i;
    generate
    for (i = 0; i < N_FANOUT; i++) begin
        assign obus[i].valid = ibus.valid;
        assign obus[i].data  = ibus.data;
    end
    endgenerate

    generate
    for (i = 0; i < N_FANOUT; i++) begin
        assign ibus.ready &= obus[i].ready;
    end
    endgenerate

endmodule



module decompose_dataflow #(
    parameter integer N_FANOUT = 7,
    parameter integer WIDTH_OUTPUT = 16
) (
    data_bus.up_stream ibus,
    data_bus.down_stream [N_FANOUT-1:0] obus
);

    genvar i;
    generate
    for (i = 0; i < N_FANOUT; i++) begin
        assign obus[i].valid = ibus.valid;
        assign obus[i].data  = ibus.data[(i+1)*WIDTH_OUTPUT-1:i*WIDTH_OUTPUT];
    end
    endgenerate

    generate
    for (i = 0; i < N_FANOUT; i++) begin
        assign ibus.ready &= obus[i].ready;
    end
    endgenerate

endmodule



module mac(
    data_bus.down_stream a,
    data_bus.down_stream b,
    data_bus.down_stream c,
    data_bus.up_stream out
);

    assign out.data = a.data * b.data + c.data;
    assign out.valid = a.valid & b.valid & c.valid;
    assign a.ready = 1;
    assign b.ready = 1;
    assign c.ready = 1;
    
endmodule

