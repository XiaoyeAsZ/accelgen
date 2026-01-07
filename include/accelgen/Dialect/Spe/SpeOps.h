#ifndef SPE_OPS_H
#define SPE_OPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

#define GET_OP_CLASSES
#include "accelgen/Dialect/Spe/SpeOps.h.inc"

#endif  // SPE_OPS_H