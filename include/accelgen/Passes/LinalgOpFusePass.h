#ifndef LINALG_OP_FUSE_PASS_H
#define LINALG_OP_FUSE_PASS_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace accelgen {

#define GEN_PASS_DECL
#include "accelgen/Passes/LinalgOpFusePass.h.inc"

// #define GEN_PASS_REGISTRATION
// #include "accelgen/Passes/LinalgOpFusePass.h.inc"
}  // namespace accelgen

}  // namespace mlir

#endif  // LINALG_OP_FUSE_PASS_H