#ifndef OPERATION_UTILS
#define OPERATION_UTILS

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include <vector>

namespace mlir {
namespace accelgen {
std::vector<mlir::Operation*> getTopoOrder(std::vector<mlir::Operation*> ops);

std::vector<linalg::GenericOp> getProducerGeneric(linalg::GenericOp op);
}  // namespace accelgen
}  // namespace mlir

#endif