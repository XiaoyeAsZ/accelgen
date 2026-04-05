#include "accelgen/Dialect/Dap/DapOps.h"

#include "accelgen/Dialect/Dap/DapDialect.h"

#define GET_OP_CLASSES
#include "accelgen/Dialect/Dap/DapOps.cpp.inc"

mlir::Type mlir::dap::DapDialect::parseType(
    ::mlir::DialectAsmParser& parser) const {
  parser.emitError(parser.getNameLoc(), "unknown type in DapDialect");
  return mlir::Type();
}

void mlir::dap::DapDialect::printType(::mlir::Type type,
                                      ::mlir::DialectAsmPrinter& os) const {
  llvm_unreachable("DapDialect has no types to print");
}

void mlir::dap::DapDialect::registerTypes() {
  // If you plan to register types later, call addTypes(...)
}

// void mlir::dap::DataNodeOp::build(::mlir::OpBuilder& odsBuilder,
//                                   ::mlir::OperationState& odsState,
//                                   mlir::Value din, mlir::Value dout) {
//   odsState.addTypes({din.getType(), dout.getType()});

//   odsState.addOperands({din, dout});
// }

void mlir::dap::MuxOp::build(::mlir::OpBuilder& odsBuilder,
                             ::mlir::OperationState& odsState, int64_t fanin) {
  auto i32 = odsBuilder.getI32Type();

  odsState.addAttribute("fanin", odsBuilder.getI32IntegerAttr(fanin));

  odsState.addTypes(i32);

  for (int i = 0; i < fanin; ++i) odsState.addTypes(i32);
}

void mlir::dap::DemuxOp::build(::mlir::OpBuilder& odsBuilder,
                               ::mlir::OperationState& odsState,
                               int64_t fanout) {
  auto i32 = odsBuilder.getI32Type();

  odsState.addAttribute("fanout", odsBuilder.getI32IntegerAttr(fanout));

  odsState.addTypes(i32);

  for (int i = 0; i < fanout; ++i) odsState.addTypes(i32);
}

void mlir::dap::DecomposeOp::build(::mlir::OpBuilder& odsBuilder,
                                   ::mlir::OperationState& odsState,
                                   int64_t fanout) {
  auto i32 = odsBuilder.getI32Type();

  odsState.addAttribute("fanout", odsBuilder.getI32IntegerAttr(fanout));

  odsState.addTypes(i32);

  for (int i = 0; i < fanout; ++i) odsState.addTypes(i32);
}

void mlir::dap::ComposeOp::build(::mlir::OpBuilder& odsBuilder,
                                 ::mlir::OperationState& odsState,
                                 int64_t fanin) {
  auto i32 = odsBuilder.getI32Type();

  odsState.addAttribute("fanin", odsBuilder.getI32IntegerAttr(fanin));

  for (int i = 0; i < fanin; ++i) odsState.addTypes(i32);

  odsState.addTypes(i32);
}

void mlir::dap::BroadcastOp::build(::mlir::OpBuilder& odsBuilder,
                                   ::mlir::OperationState& odsState,
                                   int64_t fanout) {
  auto i32 = odsBuilder.getI32Type();

  odsState.addAttribute("fanout", odsBuilder.getI32IntegerAttr(fanout));

  odsState.addTypes(i32);

  for (int i = 0; i < fanout; ++i) odsState.addTypes(i32);
}