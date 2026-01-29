#ifndef OPERATION_UTILS
#define OPERATION_UTILS

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Pass/Pass.h"
#include <vector>

namespace mlir {
namespace accelgen {
std::vector<mlir::Operation*> getTopoOrder(
    const std::vector<mlir::Operation*>& ops);

std::vector<linalg::GenericOp> getProducerGeneric(mlir::Value operand);

llvm::ArrayRef<int64_t> getOperandShape(const mlir::Value& operand);
}  // namespace accelgen
}  // namespace mlir

#endif