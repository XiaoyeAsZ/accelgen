#include "accelgen/Passes/AccelgenPasses.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::accelgen {
#define GEN_PASS_DEF_RESOLVEMIXEDPRECISIONPASS
#include "accelgen/Passes/AccelgenPasses.h.inc"

namespace {

class MixedPrecisionBatchMatmulPattern
    : public mlir::OpRewritePattern<linalg::BatchMatmulOp> {
  using mlir::OpRewritePattern<linalg::BatchMatmulOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      linalg::BatchMatmulOp matmulOp,
      mlir::PatternRewriter& rewriter) const override {
    auto inputs = matmulOp.getInputs();
    auto outputs = matmulOp.getOutputs();
    if (!matmulOp.getResult(0).hasOneUse()) return mlir::failure();

    auto user = *matmulOp.getResult(0).getUsers().begin();

    auto truncGeneric = mlir::dyn_cast<mlir::linalg::GenericOp>(user);
    if (!truncGeneric) return mlir::failure();

    auto& body = truncGeneric.getRegion().front();
    if (!llvm::hasSingleElement(body.without_terminator()) ||
        !mlir::isa<mlir::arith::TruncFOp>(body.front()))
      return mlir::failure();

    rewriter.setInsertionPoint(truncGeneric);
    auto outputType = mlir::cast<mlir::RankedTensorType>(
        truncGeneric.getOutputs().front().getType());
    auto emptyOp = rewriter.create<mlir::tensor::EmptyOp>(
        matmulOp.getLoc(), outputType.getShape(), outputType.getElementType());
    auto bf16MatmulOp = rewriter.create<mlir::linalg::BatchMatmulOp>(
        matmulOp.getLoc(), matmulOp.getInputs(),
        mlir::ValueRange{emptyOp.getResult()});

    rewriter.replaceOp(truncGeneric, bf16MatmulOp.getResults());
    rewriter.eraseOp(matmulOp);
    return mlir::success();
  }
};

class ResolveMixedPrecision
    : public impl::ResolveMixedPrecisionPassBase<ResolveMixedPrecision> {
 public:
  using impl::ResolveMixedPrecisionPassBase<
      ResolveMixedPrecision>::ResolveMixedPrecisionPassBase;

  void runOnOperation() final {
    mlir::MLIRContext* ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);

    patterns.add<MixedPrecisionBatchMatmulPattern>(ctx);

    if (mlir::failed(
            applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace
}  // namespace mlir::accelgen
