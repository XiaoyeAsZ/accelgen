#ifndef CONSTRUCT_TILE_GRAPH_PASS_H
#define CONSTRUCT_TILE_GRAPH_PASS_H

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
namespace accelgen {

#define GEN_PASS_DECL
#include "accelgen/Passes/ConstructTileGraphPass.h.inc"

}  // namespace accelgen

}  // namespace mlir

#endif  // CONSTRUCT_TILE_GRAPH_PASS_H