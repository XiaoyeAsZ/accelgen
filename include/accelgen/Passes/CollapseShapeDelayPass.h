#ifndef COLLAPSE_SHAPE_DELAY_PASS_H
#define COLLAPSE_SHAPE_DELAY_PASS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace accelgen {

#define GEN_PASS_DECL
#include "accelgen/Passes/CollapseShapeDelayPass.h.inc"

}  // namespace accelgen

}  // namespace mlir

#endif  // COLLAPSE_SHAPE_DELAY_PASS_H