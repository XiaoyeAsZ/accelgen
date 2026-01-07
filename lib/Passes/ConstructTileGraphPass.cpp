#include "accelgen/Passes/LinalgOpFusePass.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "accelgen/Dialect/TileGraph/TileGraphDialect.h"
#include "accelgen/Dialect/TileGraph/TileGraphOps.h"

namespace mlir::accelgen {
#define GEN_PASS_DEF_CONSTRUCTTILEGRAPH
#include "accelgen/Passes/ConstructTileGraphPass.h.inc"

namespace {

class ConstructComputeNodePattern
    : public mlir::OpRewritePattern<mlir::linalg::GenericOp> {
  using mlir::OpRewritePattern<mlir::linalg::GenericOp>::OpRewritePattern;

 public:
  mlir::LogicalResult matchAndRewrite(
      mlir::linalg::GenericOp genericOp,
      mlir::PatternRewriter& rewriter) const override {
    mlir::MLIRContext* ctx = rewriter.getContext();
    // Check how many dims are in affine map
    AffineMapAttr affineMap =
        mlir::cast<AffineMapAttr>(genericOp.getIndexingMaps()[0]);
    unsigned int nDim = affineMap.getAffineMap().getNumDims();

    SmallVector<int64_t> initVec(nDim, 1);
    rewriter.getI64ArrayAttr(initVec);
    ArrayAttr tilingVecAttr = rewriter.getI64ArrayAttr(initVec);
    ArrayAttr unrollVecAttr = rewriter.getI64ArrayAttr(initVec);

    ArrayAttr l2ReorderingMat = generateDigMatrix(nDim, rewriter);
    ArrayAttr l1ReorderingMat = generateDigMatrix(nDim, rewriter);

    // llvm::errs() << "[ComputeNodePattern] Processing op: "
    //              << genericOp.getOperation()->getName() << "\n";

    // genericOp->dump();  // prints full op IR

    rewriter.setInsertionPoint(genericOp);
    tile_graph::ComputeNodeOp computeNodeOp =
        rewriter.create<tile_graph::ComputeNodeOp>(
            genericOp.getLoc(), genericOp.getResultTypes()[0],
            genericOp.getInputs(), genericOp.getIndexingMapsAttr(),
            genericOp.getIteratorTypesAttr(), tilingVecAttr, unrollVecAttr,
            l2ReorderingMat, l1ReorderingMat);

    rewriter.replaceOp(genericOp, computeNodeOp.getResults());

    return mlir::success();
  }

 private:
  ArrayAttr generateDigMatrix(int64_t n,
                              mlir::PatternRewriter& rewriter) const {
    SmallVector<int64_t> initMat(n * n, 0);
    for (int64_t i = 0; i < n; i++) {
      for (int64_t j = 0; j < n; j++) {
        if (i == j) initMat[i * n + j] = 1;
      }
    }
    return rewriter.getI64ArrayAttr(initMat);
  }
};

class ConstructDataPathPattern
    : public mlir::OpRewritePattern<tile_graph::ComputeNodeOp> {
  using mlir::OpRewritePattern<tile_graph::ComputeNodeOp>::OpRewritePattern;

 public:
  mlir::LogicalResult matchAndRewrite(
      tile_graph::ComputeNodeOp computeNodeOp,
      mlir::PatternRewriter& rewriter) const override {
    mlir::MLIRContext* ctx = rewriter.getContext();

    auto operands = computeNodeOp.getOperands();

    if (operands[0].getDefiningOp<tile_graph::DataPathOp>()) return failure();

    rewriter.setInsertionPoint(computeNodeOp);
    for (unsigned int i = 0; i < operands.size(); i++) {
      auto dataPathOp = rewriter.create<tile_graph::DataPathOp>(
          computeNodeOp.getLoc(), operands[i].getType(), operands[i],
          rewriter.getStringAttr("ddr"));
      computeNodeOp.setOperand(i, dataPathOp.getResult());
    }

    return mlir::success();
  }
};

class ConstructTileGraph
    : public impl::ConstructTileGraphBase<ConstructTileGraph> {
 public:
  using impl::ConstructTileGraphBase<
      ConstructTileGraph>::ConstructTileGraphBase;

  void runOnOperation() final {
    mlir::MLIRContext* ctx = &getContext();
    {
      mlir::RewritePatternSet patterns(ctx);
      patterns.add<ConstructComputeNodePattern>(ctx);

      if (mlir::failed(applyPatternsAndFoldGreedily(getOperation(),
                                                    std::move(patterns)))) {
        signalPassFailure();
      }
    }
    {
      mlir::RewritePatternSet patterns(ctx);
      patterns.add<ConstructDataPathPattern>(ctx);

      if (mlir::failed(applyPatternsAndFoldGreedily(getOperation(),
                                                    std::move(patterns)))) {
        signalPassFailure();
      }
    }
  }
};

}  // namespace
}  // namespace mlir::accelgen
