#include "accelgen/Passes/LinalgOpFusePass.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::accelgen {
#define GEN_PASS_DEF_LINALGOPFUSE
#include "accelgen/Passes/LinalgOpFusePass.h.inc"

namespace {

class FuseTransposeExpandBatchMatmulPattern
    : public mlir::OpRewritePattern<linalg::BatchMatmulOp> {
  using mlir::OpRewritePattern<linalg::BatchMatmulOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      linalg::BatchMatmulOp matmulOp,
      mlir::PatternRewriter &rewriter) const override {
    mlir::MLIRContext *ctx = rewriter.getContext();

    mlir::tensor::ExpandShapeOp expandShapeOpOrErr =
        mlir::dyn_cast_or_null<mlir::tensor::ExpandShapeOp>(
            matmulOp.getOperand(1).getDefiningOp());

    if (!(expandShapeOpOrErr)) return mlir::failure();
    auto shapeExpandOpInput = mlir::dyn_cast<mlir::RankedTensorType>(
                                  expandShapeOpOrErr.getSrc().getType())
                                  .getShape();
    auto shapeExpandOpOutput = mlir::dyn_cast<mlir::RankedTensorType>(
                                   expandShapeOpOrErr.getResult().getType())
                                   .getShape();

    if (!(shapeExpandOpInput.size() == 2 && shapeExpandOpOutput.size() == 3 &&
          shapeExpandOpInput[0] == shapeExpandOpOutput[1] &&
          shapeExpandOpInput[1] == shapeExpandOpOutput[2]))
      return mlir::failure();

    auto transposeOpOrErr = mlir::dyn_cast_or_null<mlir::linalg::TransposeOp>(
        expandShapeOpOrErr.getOperand(0).getDefiningOp());

    if (!(transposeOpOrErr)) return mlir::failure();

    auto shapeTransposeOpInput = mlir::dyn_cast<mlir::RankedTensorType>(
                                     transposeOpOrErr.getOperand(0).getType())
                                     .getShape();
    auto shapeTransposeOpOutput =
        mlir::dyn_cast<mlir::RankedTensorType>(
            transposeOpOrErr.getResults()[0].getType())
            .getShape();

    if (!(shapeTransposeOpInput.size() == 2 &&
          shapeTransposeOpOutput.size() == 2 &&
          shapeTransposeOpInput[0] == shapeTransposeOpOutput[1] &&
          shapeTransposeOpInput[1] == shapeTransposeOpOutput[0]))
      return mlir::failure();

    auto fusedRhs = transposeOpOrErr.getOperand(0);

    auto xshape =
        mlir::dyn_cast<mlir::RankedTensorType>(fusedRhs.getType()).getShape();

    mlir::ArrayAttr affineMap = matmulOp.getIndexingMapsAttr();
    mlir::AffineMap affineMapLhs =
        mlir::dyn_cast<mlir::AffineMapAttr>(affineMap[0]).getValue();
    mlir::AffineMap affineMapRhs = mlir::AffineMap::get(
        4, 0, {mlir::getAffineDimExpr(2, ctx), mlir::getAffineDimExpr(3, ctx)},
        ctx);
    mlir::AffineMap affineMapResult =
        mlir::dyn_cast<mlir::AffineMapAttr>(affineMap[2]).getValue();

    auto indexingMaps = rewriter.getAffineMapArrayAttr(
        {affineMapLhs, affineMapRhs, affineMapResult});

    auto iteratorTypes = rewriter.getArrayAttr(
        {mlir::linalg::IteratorTypeAttr::get(
             rewriter.getContext(), mlir::utils::IteratorType::parallel),
         mlir::linalg::IteratorTypeAttr::get(
             rewriter.getContext(), mlir::utils::IteratorType::parallel),
         mlir::linalg::IteratorTypeAttr::get(
             rewriter.getContext(), mlir::utils::IteratorType::parallel),
         mlir::linalg::IteratorTypeAttr::get(
             rewriter.getContext(), mlir::utils::IteratorType::reduction)});

    // mlir::NamedAttribute namedIndexingMaps(
    //     rewriter.getStringAttr("indexing_maps"), indexingMaps);

    rewriter.setInsertionPoint(matmulOp);
    // mlir::linalg::BatchMatmulOp fusedOp =
    //     rewriter.create<mlir::linalg::BatchMatmulOp>(
    //         matmulOp.getLoc(),
    //         mlir::ValueRange{matmulOp.getOperand(0), fusedRhs},
    //         mlir::ValueRange(matmulOp.getResults()));
    // fusedOp.setIndexingMapsAttr(indexingMaps);

    auto fuseOpOutputType =
        mlir::dyn_cast<RankedTensorType>(matmulOp.getResult(0).getType());
    auto emptyTensorOp = rewriter.create<mlir::tensor::EmptyOp>(
        matmulOp.getLoc(), fuseOpOutputType.getShape(),
        fuseOpOutputType.getElementType());

    auto zeroOp = rewriter.create<mlir::arith::ConstantOp>(
        matmulOp.getLoc(),
        rewriter.getZeroAttr(fuseOpOutputType.getElementType()));

    auto fillTensorOp = rewriter.create<mlir::linalg::FillOp>(
        matmulOp.getLoc(), ValueRange{zeroOp.getODSResults(0)},
        ValueRange{emptyTensorOp.getODSResults(0)});

    auto fusedOp = rewriter.create<mlir::linalg::GenericOp>(
        matmulOp.getLoc(), matmulOp.getResultTypes(),
        ValueRange{matmulOp.getOperand(0), fusedRhs},
        ValueRange{fillTensorOp.getResult(0)}, indexingMaps, iteratorTypes,
        nullptr, nullptr, [&](OpBuilder &b, Location loc, ValueRange args) {
          Value a = args[0];
          Value bVal = args[1];
          Value c = args[2];
          auto mul = b.create<mlir::arith::MulFOp>(loc, a, bVal);
          auto add = b.create<mlir::arith::AddFOp>(loc, c, mul);
          b.create<mlir::linalg::YieldOp>(loc, add.getResult());
        });

    rewriter.replaceOp(matmulOp, fusedOp.getResults());
    rewriter.eraseOp(expandShapeOpOrErr);
    rewriter.eraseOp(transposeOpOrErr);

    return mlir::success();
  }
};

class LinalgOpFuse : public impl::LinalgOpFuseBase<LinalgOpFuse> {
 public:
  using impl::LinalgOpFuseBase<LinalgOpFuse>::LinalgOpFuseBase;

  void runOnOperation() final {
    mlir::MLIRContext *ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);

    // Register your rewrite pattern
    patterns.add<FuseTransposeExpandBatchMatmulPattern>(ctx);

    // Apply to the whole operation (your pass probably runs on FuncOp or
    // ModuleOp)
    if (mlir::failed(applyPatternsAndFoldGreedily(getOperation(),
                                                  std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
}  // namespace
}  // namespace mlir::accelgen
