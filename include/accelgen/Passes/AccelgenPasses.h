#ifndef PASSES_H
#define PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir::accelgen {

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "accelgen/Passes/AccelgenPasses.h.inc"

}  // namespace mlir::accelgen

#endif