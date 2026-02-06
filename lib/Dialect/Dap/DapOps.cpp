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