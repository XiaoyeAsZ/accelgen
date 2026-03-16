#ifndef AFFINE_MAP_UTILS
#define AFFINE_MAP_UTILS

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

namespace mlir {
namespace accelgen {
llvm::SmallVector<int64_t> getAffineMapAccessDims(
    const mlir::AffineMap& affineMap);
}

template <typename DimFn, typename ConstFn>
void traverseAffineMapResults(mlir::AffineMap map, DimFn&& dimHandler,
                              ConstFn&& constHandler) {
  for (auto [idxExpr, expr] : llvm::enumerate(map.getResults())) {
    if (auto dim = llvm::dyn_cast<mlir::AffineDimExpr>(expr)) {
      dimHandler(idxExpr, dim);
      continue;
    }
    if (auto cst = llvm::dyn_cast<mlir::AffineConstantExpr>(expr)) {
      constHandler(idxExpr, cst);
      continue;
    }
  }
}

}  // namespace mlir

#endif