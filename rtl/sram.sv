// Single SRAM bank with 1w and 1r port, behaviour simulation only !!!
// module sram #(
//     parameter DATA_WIDTH = 32,
//     parameter DEPTH      = 1024,
//     parameter ADDR_WIDTH = $clog2(DEPTH)
// ) (
//     input  logic                  clk,
//     input  logic                  en,     // chip enable
//     input  logic                  we,     // write enable
//     input  logic [ADDR_WIDTH-1:0] addr,
//     input  logic [DATA_WIDTH-1:0] wdata,
//     output logic [DATA_WIDTH-1:0] rdata
// );

//   // Memory array
//   logic [DATA_WIDTH-1:0] mem[0:DEPTH-1];

//   always_ff @(posedge clk) begin
//     if (en) begin
//       if (we) begin
//         mem[addr] <= wdata;
//       end
//       rdata <= mem[addr];  // synchronous read
//     end
//   end

// endmodule
