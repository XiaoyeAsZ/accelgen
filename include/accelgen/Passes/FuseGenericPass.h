#ifndef FUSE_GENERIC_PASS_H
#define FUSE_GENERIC_PASS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace accelgen {

#define GEN_PASS_DECL
#include "accelgen/Passes/FuseGenericPass.h.inc"

}  // namespace accelgen

}  // namespace mlir

#endif  // FUSE_GENERIC_PASS_H