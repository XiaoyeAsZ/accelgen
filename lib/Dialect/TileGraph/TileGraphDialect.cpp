#include "mlir/IR/DialectImplementation.h"

#include "accelgen/Dialect/TileGraph/TileGraphDialect.h"
#include "accelgen/Dialect/TileGraph/TileGraphOps.h"

using namespace mlir;
using namespace mlir::tile_graph;

#include "accelgen/Dialect/TileGraph/TileGraphOpsDialect.cpp.inc"

void TileGraphDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "accelgen/Dialect/TileGraph/TileGraphOps.cpp.inc"
      >();
}
