#include "mlir/IR/DialectImplementation.h"

#include "accelgen/Dialect/Spe/SpeDialect.h"
#include "accelgen/Dialect/Spe/SpeOps.h"

using namespace mlir;
using namespace mlir::spe;

#include "accelgen/Dialect/Spe/SpeOpsDialect.cpp.inc"

void SpeDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "accelgen/Dialect/Spe/SpeOps.cpp.inc"
      >();
}
