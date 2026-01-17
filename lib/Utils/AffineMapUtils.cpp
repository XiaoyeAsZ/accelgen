#include "accelgen/Utils/AffineMapUtils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

llvm::SmallVector<int64_t> mlir::accelgen::getAffineMapAccessDims(
    mlir::AffineMap& affineMap) {
  llvm::SmallVector<int64_t> accessDims;
  for (auto expr : affineMap.getResults()) {
    expr.walk([&](AffineExpr e) {
      if (auto dim = mlir::dyn_cast<AffineDimExpr>(e)) {
        accessDims.push_back(dim.getPosition());
      }
    });
  }
  return accessDims;
}