//===- LinalgLoopReorder.h - Linalg Loop Reorder Pass ----------*- C++ -*-===//
//
// This file declares the Linalg Loop Reorder pass.
//
//===----------------------------------------------------------------------===//

#ifndef LINALG_LOOP_REORDER_H
#define LINALG_LOOP_REORDER_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace accelgen {

#define GEN_PASS_DECL
#include "accelgen/Passes/LinalgLoopReorderPass.h.inc"

// #define GEN_PASS_REGISTRATION
// #include "accelgen/Passes/LinalgLoopReorderPass.h.inc"
}  // namespace accelgen

}  // namespace mlir

#endif  // LINALG_LOOP_REORDER_H