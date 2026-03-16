#include "accelgen/Utils/AffineMapUtils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

namespace mlir::accelgen {
llvm::SmallVector<int64_t> getAffineMapAccessDims(
    const mlir::AffineMap& affineMap) {
  llvm::SmallVector<int64_t> accessDims;
  for (const auto& [index, expr] : llvm::enumerate(affineMap.getResults())) {
    // expr.walk([&](AffineExpr e) {
    //   if (auto dim = mlir::dyn_cast<AffineDimExpr>(e)) {
    //     accessDims.push_back(dim.getPosition());
    //   } else {
    //     affineMap.dump();
    //     assert(0);
    //   }
    // });
    if (auto dimExpr = mlir::dyn_cast<AffineDimExpr>(expr)) {
      accessDims.push_back(dimExpr.getPosition());
    } else if (auto constExpr = mlir::dyn_cast<AffineConstantExpr>(expr);
               constExpr && constExpr.getValue() == 0) {
      accessDims.push_back(index);
    } else
      assert(0);
  }
  return accessDims;
}

std::vector<int64_t> getAccessOrder(const llvm::SmallVector<int64_t>& order,
                                    const mlir::AffineMap& affineMap) {
  auto accessDims = getAffineMapAccessDims(affineMap);
}

}  // namespace mlir::accelgen
