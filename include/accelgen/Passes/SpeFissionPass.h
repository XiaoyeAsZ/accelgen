#ifndef SPE_FISSION_PASS_H
#define SPE_FISSION_PASS_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace accelgen {

#define GEN_PASS_DECL
#include "accelgen/Passes/SpeFissionPass.h.inc"

}  // namespace accelgen

}  // namespace mlir

#endif  // SPE_FISSION_PASS_H