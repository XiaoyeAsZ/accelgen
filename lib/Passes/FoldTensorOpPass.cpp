

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

    mlir::Value input = collapseOp.getSrc();
    auto blockArg = mlir::dyn_cast<BlockArgument>(input);
    if (!blockArg) return mlir::failure();
    if (!input.hasOneUse()) return mlir::failure();

    auto func = collapseOp->getParentOfType<func::FuncOp>();
    if (!func) return mlir::failure();

    auto outType =
        mlir::dyn_cast<RankedTensorType>(collapseOp.getResult().getType());

    rewriter.modifyOpInPlace(func, [&]() {
      blockArg.setType(outType);
      SmallVector<Type> newArgTypes(func.getArgumentTypes());
      newArgTypes[blockArg.getArgNumber()] = outType;
      func.setType(
          rewriter.getFunctionType(newArgTypes, func.getResultTypes()));
    });

    rewriter.replaceOp(collapseOp, blockArg);

    return mlir::success();
  }
};

class ExpandArgumentPattern
    : public mlir::OpRewritePattern<tensor::ExpandShapeOp> {
  using mlir::OpRewritePattern<tensor::ExpandShapeOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      tensor::ExpandShapeOp expandOp,
      mlir::PatternRewriter& rewriter) const override {
    mlir::MLIRContext* ctx = rewriter.getContext();

    if (expandOp.getSrc().getDefiningOp() != nullptr) return mlir::failure();

    mlir::Value input = expandOp.getSrc();
    auto blockArg = mlir::dyn_cast<BlockArgument>(input);
    if (!blockArg) return mlir::failure();
    if (!input.hasOneUse()) return mlir::failure();

    auto func = expandOp->getParentOfType<func::FuncOp>();
    if (!func) return mlir::failure();

    auto outType =
        mlir::dyn_cast<RankedTensorType>(expandOp.getResult().getType());

    rewriter.modifyOpInPlace(func, [&]() {
      blockArg.setType(outType);
      SmallVector<Type> newArgTypes(func.getArgumentTypes());
      newArgTypes[blockArg.getArgNumber()] = outType;
      func.setType(
          rewriter.getFunctionType(newArgTypes, func.getResultTypes()));
    });

    rewriter.replaceOp(expandOp, blockArg);

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

      if (collapseDims.size() > 1 && expandDims.size() > 1) {
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
      llvm::SmallVector<ReassociationIndices> reassMap;

      if (inputShape.size() == outputShape.size()) {
        return mlir::failure();
      }
      // Expand
      else if (inputShape.size() < outputShape.size()) {
        while (srcP < inputShape.size() && destP < outputShape.size()) {
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
          reassMap.push_back(idx);
        }

        rewriter.setInsertionPointAfter(expandOp);
        auto updateExpandOp = rewriter.create<tensor::ExpandShapeOp>(
            expandOp.getLoc(), expandOp.getResult().getType(),
            collapseOp.getSrc(), reassMap);
        rewriter.replaceOp(expandOp, updateExpandOp);
      }
      // Collapse
      else {
        while (srcP < inputShape.size() && destP < outputShape.size()) {
          ReassociationIndices idx;
          if (inputShape[srcP] == outputShape[destP]) {
            srcP++;
            destP++;
            idx.push_back(indexDim++);
          } else {
            int64_t tmp = 1;
            while (tmp != outputShape[destP]) {
              idx.push_back(indexDim++);
              tmp *= inputShape[srcP];
              srcP++;
            }
            destP++;
          }
          reassMap.push_back(idx);
        }

        rewriter.setInsertionPointAfter(expandOp);
        auto updateCollapseOp = rewriter.create<tensor::CollapseShapeOp>(
            expandOp.getLoc(), expandOp.getResult().getType(),
            collapseOp.getSrc(), reassMap);
        rewriter.replaceOp(expandOp, updateCollapseOp);
      }

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
    patterns.add<CollapseArgumentPattern>(&ctx);
    patterns.add<ExpandArgumentPattern>(&ctx);
    if (mlir::failed(applyPatternsAndFoldGreedily(getOperation(),
                                                  std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace
}  // namespace mlir::accelgen
