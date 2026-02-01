

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

#include "accelgen/Utils/OperationUtils.h"

#include "accelgen/Passes/AccelgenPasses.h"

// #include "accelgen/Passes/MarkGenericPass.h"

namespace mlir::accelgen {
#define GEN_PASS_DEF_MARKGENERICPASS
// #include "accelgen/Passes/MarkGenericPass.h.inc"
#include "accelgen/Passes/AccelgenPasses.h.inc"

namespace {

class ConstFillGenericPattern
    : public mlir::OpRewritePattern<linalg::GenericOp> {
  using mlir::OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      linalg::GenericOp genericOp,
      mlir::PatternRewriter& rewriter) const override {
    if (!genericOp->hasAttr("accelgen.memory_transformation"))
      return mlir::failure();
    if (genericOp.getInputs().size() == 1 &&
        genericOp.getInputs()[0].getDefiningOp<arith::ConstantOp>() &&
        (!genericOp->hasAttr("accelgen.const_broadcast"))) {
      genericOp->setAttr("accelgen.const_broadcast",
                         BoolAttr::get(rewriter.getContext(), true));
      return mlir::success();
    } else
      return mlir::failure();
  }
};

class ExpandGenericPattern : public mlir::OpRewritePattern<linalg::GenericOp> {
  using mlir::OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      linalg::GenericOp genericOp,
      mlir::PatternRewriter& rewriter) const override {
    if (!genericOp->hasAttr("accelgen.memory_transformation"))
      return mlir::failure();
    if (genericOp->hasAttr("accelgen.expand")) return mlir::failure();
    if (!mlir::dyn_cast<TensorType>(genericOp.getInputs()[0].getType()))
      return mlir::failure();
    auto shapeIns = getOperandShape(genericOp.getInputs()[0]);
    auto shapeOuts = getOperandShape(genericOp.getOutputs()[0]);
    int64_t rIns = 1;
    for (auto s : shapeIns) rIns *= s;
    int64_t rOuts = 1;
    for (auto s : shapeOuts) rOuts *= s;
    if (rOuts > rIns) {
      genericOp->setAttr("accelgen.expand",
                         BoolAttr::get(rewriter.getContext(), true));
      return mlir::success();
    } else
      return mlir::failure();
  }
};

class TransposeGenericPattern
    : public mlir::OpRewritePattern<linalg::GenericOp> {
  using mlir::OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      linalg::GenericOp genericOp,
      mlir::PatternRewriter& rewriter) const override {
    if (!genericOp->hasAttr("accelgen.memory_transformation"))
      return mlir::failure();
    if (genericOp->hasAttr("accelgen.transpose")) return mlir::failure();
    if (!mlir::dyn_cast<TensorType>(genericOp.getInputs()[0].getType()))
      return mlir::failure();
    auto shapeIns = getOperandShape(genericOp.getInputs()[0]);
    auto shapeOuts = getOperandShape(genericOp.getOutputs()[0]);
    int64_t rIns = 1;
    for (auto s : shapeIns) rIns *= s;
    int64_t rOuts = 1;
    for (auto s : shapeOuts) rOuts *= s;
    if (shapeIns.size() == shapeOuts.size() && rOuts == rIns) {
      genericOp->setAttr("accelgen.transpose",
                         BoolAttr::get(rewriter.getContext(), true));
      return mlir::success();
    } else
      return mlir::failure();
  }
};

class MemoryTransformationPattern
    : public mlir::OpRewritePattern<linalg::GenericOp> {
  using mlir::OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      linalg::GenericOp genericOp,
      mlir::PatternRewriter& rewriter) const override {
    if (genericOp->hasAttr("accelgen.memory_transformation"))
      return mlir::failure();
    auto ins = genericOp.getInputs();
    auto outs = genericOp.getOutputs();
    if (ins.size() != 1 || outs.size() != 1) return mlir::failure();
    auto& ops = genericOp.getBody()->getOperations();
    if (ops.size() != 1) return mlir::failure();
    for (auto& op : ops) {
      if (!mlir::isa<linalg::YieldOp>(op)) return mlir::failure();
    }
    genericOp->setAttr("accelgen.memory_transformation",
                       BoolAttr::get(rewriter.getContext(), true));
    return mlir::success();
  }
};

class MarkGenericPass : public impl::MarkGenericPassBase<MarkGenericPass> {
 public:
  using impl::MarkGenericPassBase<MarkGenericPass>::MarkGenericPassBase;

  void runOnOperation() final {
    mlir::MLIRContext& ctx = getContext();
    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<MemoryTransformationPattern>(&ctx);
    patterns.add<ExpandGenericPattern>(&ctx);
    patterns.add<ConstFillGenericPattern>(&ctx);
    patterns.add<TransposeGenericPattern>(&ctx);
    if (mlir::failed(applyPatternsAndFoldGreedily(getOperation(),
                                                  std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace
}  // namespace mlir::accelgen
