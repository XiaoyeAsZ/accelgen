#ifndef AFFINE_MAP_UTILS
#define AFFINE_MAP_UTILS

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

namespace mlir {
namespace accelgen {
llvm::SmallVector<int64_t> getAffineMapAccessDims(mlir::AffineMap& affineMap);
}
}  // namespace mlir

#endif