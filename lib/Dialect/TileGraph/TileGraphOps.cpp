#include "accelgen/Dialect/TileGraph/TileGraphOps.h"

#include "accelgen/Dialect/TileGraph/TileGraphDialect.h"

#define GET_OP_CLASSES
#include "accelgen/Dialect/TileGraph/TileGraphOps.cpp.inc"

mlir::Type mlir::tile_graph::TileGraphDialect::parseType(
    ::mlir::DialectAsmParser& parser) const {
  parser.emitError(parser.getNameLoc(), "unknown type in SpeDialect");
  return mlir::Type();
}

void mlir::tile_graph::TileGraphDialect::printType(
    ::mlir::Type type, ::mlir::DialectAsmPrinter& os) const {
  llvm_unreachable("SpeDialect has no types to print");
}

void mlir::tile_graph::TileGraphDialect::registerTypes() {
  // If you plan to register types later, call addTypes(...)
}