#ifndef DAP_OPS_H
#define DAP_OPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

#define GET_OP_CLASSES
#include "accelgen/Dialect/Dap/DapOps.h.inc"

#endif  // DAP_OPS_H