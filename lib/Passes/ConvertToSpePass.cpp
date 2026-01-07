#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "accelgen/Passes/ConvertToSpePass.h"
#include "accelgen/Dialect/Spe/SpeDialect.h"
#include "accelgen/Dialect/Spe/SpeOps.h"

namespace mlir::accelgen {
#define GEN_PASS_DEF_CONVERTTOSPE
#include "accelgen/Passes/ConvertToSpePass.h.inc"

namespace {

class MacPattern : public mlir::OpRewritePattern<mlir::arith::AddFOp> {
  using mlir::OpRewritePattern<mlir::arith::AddFOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::arith::AddFOp addfOp,
      mlir::PatternRewriter &rewriter) const override {
    auto lhsAddfOp = addfOp.getLhs();
    auto rhsAddfOp = addfOp.getRhs();
    auto lhsDefiningOp =
        mlir::dyn_cast_or_null<mlir::arith::MulFOp>(lhsAddfOp.getDefiningOp());
    auto rhsDefiningOp =
        mlir::dyn_cast_or_null<mlir::arith::MulFOp>(rhsAddfOp.getDefiningOp());

    mlir::Value a, b, c;
    mlir::arith::MulFOp opToErase;

    if (lhsDefiningOp &&
        std::distance(lhsAddfOp.use_begin(), lhsAddfOp.use_end()) == 1) {
      a = lhsDefiningOp.getLhs();
      b = lhsDefiningOp.getRhs();
      c = rhsAddfOp;
      opToErase = lhsDefiningOp;
    } else if (rhsDefiningOp &&
               std::distance(rhsAddfOp.use_begin(), rhsAddfOp.use_end()) == 1) {
      a = rhsDefiningOp.getLhs();
      b = rhsDefiningOp.getRhs();
      c = lhsAddfOp;
      opToErase = rhsDefiningOp;
    } else
      return mlir::failure();

    rewriter.setInsertionPoint(addfOp);
    auto macOp =
        rewriter.create<mlir::spe::MacOp>(addfOp.getLoc(), ValueRange{a, b, c});

    rewriter.replaceOp(addfOp, macOp.getResult());
    rewriter.eraseOp(opToErase);

    return mlir::success();
  }
};

class ExpPattern : public mlir::OpRewritePattern<mlir::math::ExpOp> {
  using mlir::OpRewritePattern<mlir::math::ExpOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::math::ExpOp mathExpOp,
      mlir::PatternRewriter &rewriter) const override {
    rewriter.setInsertionPoint(mathExpOp);

    auto speExpOp = rewriter.create<mlir::spe::ExpOp>(
        mathExpOp.getLoc(), ValueRange{mathExpOp.getOperand()});

    rewriter.replaceOp(mathExpOp, speExpOp.getResult());

    return mlir::success();
  }
};

class NegPattern : public mlir::OpRewritePattern<mlir::arith::NegFOp> {
  using mlir::OpRewritePattern<mlir::arith::NegFOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::arith::NegFOp arithNegOp,
      mlir::PatternRewriter &rewriter) const override {
    rewriter.setInsertionPoint(arithNegOp);

    auto speNegOp = rewriter.create<mlir::spe::NegOp>(
        arithNegOp.getLoc(), ValueRange{arithNegOp.getOperand()});

    rewriter.replaceOp(arithNegOp, speNegOp.getResult());

    return mlir::success();
  }
};

class DivPattern : public mlir::OpRewritePattern<mlir::arith::DivFOp> {
  using mlir::OpRewritePattern<mlir::arith::DivFOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::arith::DivFOp arithDivOp,
      mlir::PatternRewriter &rewriter) const override {
    auto lhsDefiningOp = mlir::dyn_cast_or_null<mlir::arith::ConstantOp>(
        arithDivOp.getLhs().getDefiningOp());
    auto constValue =
        mlir::dyn_cast_or_null<mlir::FloatAttr>(lhsDefiningOp.getValue());
    assert(constValue && "Not a float attr");

    rewriter.setInsertionPoint(arithDivOp);

    auto speInvOp = rewriter.create<mlir::spe::InvOp>(
        arithDivOp.getLoc(), ValueRange{arithDivOp.getOperand(1)});

    if (constValue.getValueAsDouble() == 1.0) {
      rewriter.replaceOp(arithDivOp, speInvOp.getResult());
      return mlir::success();
    }

    auto speMulOp = rewriter.create<mlir::spe::MulOp>(
        arithDivOp.getLoc(),
        ValueRange{arithDivOp.getOperand(0), speInvOp.getResult()});

    rewriter.replaceOp(arithDivOp, speMulOp.getResult());

    return mlir::success();
  }
};

class AddPattern : public mlir::OpRewritePattern<mlir::arith::AddFOp> {
  using mlir::OpRewritePattern<mlir::arith::AddFOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::arith::AddFOp arithAddfOp,
      mlir::PatternRewriter &rewriter) const override {
    rewriter.setInsertionPoint(arithAddfOp);
    auto speAddOp = rewriter.create<mlir::spe::AddOp>(
        arithAddfOp.getLoc(),
        ValueRange{arithAddfOp.getLhs(), arithAddfOp.getRhs()});

    rewriter.replaceOp(arithAddfOp, speAddOp.getResult());

    return mlir::success();
  }
};

class MulPattern : public mlir::OpRewritePattern<mlir::arith::MulFOp> {
  using mlir::OpRewritePattern<mlir::arith::MulFOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::arith::MulFOp arithMulfOp,
      mlir::PatternRewriter &rewriter) const override {
    rewriter.setInsertionPoint(arithMulfOp);
    auto speMulOp = rewriter.create<mlir::spe::MulOp>(
        arithMulfOp.getLoc(),
        ValueRange{arithMulfOp.getLhs(), arithMulfOp.getRhs()});

    rewriter.replaceOp(arithMulfOp, speMulOp.getResult());

    return mlir::success();
  }
};

class ConvertToSpe : public impl::ConvertToSpeBase<ConvertToSpe> {
 public:
  using impl::ConvertToSpeBase<ConvertToSpe>::ConvertToSpeBase;

  void runOnOperation() final {
    mlir::MLIRContext *ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);

    patterns.add<MacPattern>(ctx);
    patterns.add<ExpPattern>(ctx);
    patterns.add<DivPattern>(ctx);
    patterns.add<NegPattern>(ctx);
    patterns.add<AddPattern>(ctx);
    patterns.add<MulPattern>(ctx);

    if (mlir::failed(applyPatternsAndFoldGreedily(getOperation(),
                                                  std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace
}  // namespace mlir::accelgen