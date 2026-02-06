#include "mlir/IR/DialectImplementation.h"

#include "accelgen/Dialect/Dap/DapDialect.h"
#include "accelgen/Dialect/Dap/DapOps.h"

using namespace mlir;
using namespace mlir::dap;

#include "accelgen/Dialect/Dap/DapOpsDialect.cpp.inc"

void DapDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "accelgen/Dialect/Dap/DapOps.cpp.inc"
      >();
}
