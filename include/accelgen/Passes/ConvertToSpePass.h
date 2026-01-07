#ifndef CONVERT_TO_SPE_PASS_H
#define CONVERT_TO_SPE_PASS_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace accelgen {

#define GEN_PASS_DECL
#include "accelgen/Passes/ConvertToSpePass.h.inc"

}  // namespace accelgen

}  // namespace mlir

#endif  // CONVERT_TO_SPE_PASS_H