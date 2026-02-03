#include "accelgen/Utils/AffineMapUtils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

namespace mlir::accelgen {
llvm::SmallVector<int64_t> getAffineMapAccessDims(
    const mlir::AffineMap& affineMap) {
  llvm::SmallVector<int64_t> accessDims;
  for (const auto& expr : affineMap.getResults()) {
    expr.walk([&](AffineExpr e) {
      if (auto dim = mlir::dyn_cast<AffineDimExpr>(e)) {
        accessDims.push_back(dim.getPosition());
      }
    });
  }
  return accessDims;
}

std::vector<int64_t> getAccessOrder(const llvm::SmallVector<int64_t>& order,
                                    const mlir::AffineMap& affineMap) {
  auto accessDims = getAffineMapAccessDims(affineMap);
}

}  // namespace mlir::accelgen
