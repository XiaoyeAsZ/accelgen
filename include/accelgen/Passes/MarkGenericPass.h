#ifndef MARK_GENERIC_PASS_H
#define MARK_GENERIC_PASS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace accelgen {

#define GEN_PASS_DECL
#include "accelgen/Passes/MarkGenericPass.h.inc"

} // namespace accelgen

} // namespace mlir

#endif // MARK_GENERIC_PASS_H