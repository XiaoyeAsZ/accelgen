#ifndef TILEGRAPH_DIALECT_H
#define TILEGRAPH_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"

#include "accelgen/Dialect/TileGraph/TileGraphOpsDialect.h.inc"

#endif  // TILEGRAPH_DIALECT_H