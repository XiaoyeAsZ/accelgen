

#include "accelgen/Passes/AccelgenPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include <assert.h>
#include <queue>
#include <vector>

#include "accelgen/Utils/AffineMapUtils.h"
#include "accelgen/Utils/DebugUtils.h"
#include "accelgen/Utils/OperationUtils.h"

namespace mlir::accelgen {
#define GEN_PASS_DEF_FOLDTENSOROPPASS
#include "accelgen/Passes/AccelgenPasses.h.inc"

namespace {

class CollapseArgumentPattern
    : public mlir::OpRewritePattern<tensor::CollapseShapeOp> {
  using mlir::OpRewritePattern<tensor::CollapseShapeOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      tensor::CollapseShapeOp collapseOp,
      mlir::PatternRewriter& rewriter) const override {
    mlir::MLIRContext* ctx = rewriter.getContext();

    if (collapseOp.getSrc().getDefiningOp() != nullptr) return mlir::failure();

    return mlir::success();
  }
};

class CollapseExpandPattern
    : public mlir::OpRewritePattern<tensor::ExpandShapeOp> {
  using mlir::OpRewritePattern<tensor::ExpandShapeOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      tensor::ExpandShapeOp expandOp,
      mlir::PatternRewriter& rewriter) const override {
    mlir::MLIRContext* ctx = rewriter.getContext();

    auto collapseOp =
        expandOp.getSrc().getDefiningOp<tensor::CollapseShapeOp>();
    if (!collapseOp) return mlir::failure();

    auto interValue = expandOp.getSrc();
    auto interShape = interValue.getType().getShape();

    auto collapseMap = collapseOp.getReassociationMaps();
    auto expandMap = expandOp.getReassociationMaps();
    assert(collapseMap.size() == expandMap.size());

    bool isMatch = false;

    for (auto [idxShape, itemShape] : llvm::enumerate(interShape)) {
      auto collapseDims = getAffineMapAccessDims(collapseMap[idxShape]);
      auto expandDims = getAffineMapAccessDims(expandMap[idxShape]);

      if (collapseDims.size() > 1 && expandDims.size() > 1 &&
          collapseDims.size() <= expandDims.size()) {
        isMatch = true;
      } else if (collapseDims.size() == 1 && expandDims.size() == 1) {
        continue;
      } else
        return mlir::failure();
    }

    if (isMatch) {
      auto inputShape = collapseOp.getSrc().getType().getShape();
      auto outputShape = expandOp.getResult().getType().getShape();

      int64_t srcP = 0, destP = 0;
      int64_t indexDim = 0;
      llvm::SmallVector<ReassociationIndices> expandReassMap;
      while (srcP < inputShape.size()) {
        ReassociationIndices idx;
        if (inputShape[srcP] == outputShape[destP]) {
          srcP++;
          destP++;
          idx.push_back(indexDim++);
        } else {
          int64_t tmp = 1;
          while (tmp != inputShape[srcP]) {
            idx.push_back(indexDim++);
            tmp *= outputShape[destP];
            destP++;
          }
          srcP++;
        }
        expandReassMap.push_back(idx);
      }

      rewriter.setInsertionPointAfter(expandOp);
      auto updateExpandOp = rewriter.create<tensor::ExpandShapeOp>(
          expandOp.getLoc(), expandOp.getResult().getType(),
          collapseOp.getSrc(), expandReassMap);
      rewriter.replaceOp(expandOp, updateExpandOp);

      return mlir::success();
    } else
      return mlir::failure();
  }
};

class FoldTensorOpPass : public impl::FoldTensorOpPassBase<FoldTensorOpPass> {
 public:
  using impl::FoldTensorOpPassBase<FoldTensorOpPass>::FoldTensorOpPassBase;

  void runOnOperation() final {
    mlir::MLIRContext& ctx = getContext();
    mlir::RewritePatternSet patterns(&ctx);

    patterns.add<CollapseExpandPattern>(&ctx);
    if (mlir::failed(applyPatternsAndFoldGreedily(getOperation(),
                                                  std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace
}  // namespace mlir::accelgen
