

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

#include "accelgen/Passes/FuseGenericPass.h"
#include "accelgen/Utils/AffineMapUtils.h"
#include "accelgen/Utils/OperationUtils.h"

namespace mlir::accelgen {
#define GEN_PASS_DEF_FUSEGENERICPASS
#include "accelgen/Passes/FuseGenericPass.h.inc"

namespace {

class ArgumentTransposePattern
    : public mlir::OpRewritePattern<linalg::GenericOp> {
  using mlir::OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(linalg::GenericOp genericOp,
                  mlir::PatternRewriter &rewriter) const override {
    if (!genericOp->hasAttr("accelgen.transpose"))
      return mlir::failure();
    mlir::Value input = genericOp.getInputs()[0];
    auto blockArg = mlir::dyn_cast<BlockArgument>(input);
    if (!blockArg)
      return mlir::failure();
    if (!input.hasOneUse())
      return mlir::failure();

    auto func = genericOp->getParentOfType<func::FuncOp>();
    if (!func)
      return mlir::failure();

    auto outType =
        mlir::dyn_cast<RankedTensorType>(genericOp.getResults()[0].getType());

    rewriter.modifyOpInPlace(func, [&]() {
      blockArg.setType(outType);
      SmallVector<Type> newArgTypes(func.getArgumentTypes());
      newArgTypes[blockArg.getArgNumber()] = outType;
      func.setType(
          rewriter.getFunctionType(newArgTypes, func.getResultTypes()));
    });

    rewriter.replaceOp(genericOp, blockArg);

    return mlir::success();
  }
};

class ExpandGenericPattern : public mlir::OpRewritePattern<linalg::GenericOp> {
  using mlir::OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(linalg::GenericOp genericOp,
                  mlir::PatternRewriter &rewriter) const override {
    if (!genericOp->hasAttr("accelgen.expand"))
      return mlir::failure();

    auto results = genericOp.getResults();
    assert(results.size() == 1);
    if (!results[0].hasOneUse())
      return mlir::failure();
    linalg::GenericOp consumerGeneric;
    for (auto use : results[0].getUsers()) {
      if (mlir::dyn_cast<linalg::GenericOp>(use))
        consumerGeneric = mlir::dyn_cast<linalg::GenericOp>(use);
      else
        return mlir::failure();
    }

    genericOp.dump();

    auto ctx = rewriter.getContext();

    llvm::SmallVector<mlir::Value> operands = consumerGeneric.getOperands();
    auto it = std::find(operands.begin(), operands.end(), results[0]);
    assert(it != operands.end());
    auto valueIdx = std::distance(operands.begin(), it);
    operands[valueIdx] = genericOp.getInputs()[0];
    auto nIns = consumerGeneric.getInputs().size();
    llvm::SmallVector<mlir::Value> inputs(operands.data(),
                                          operands.data() + nIns);
    llvm::SmallVector<mlir::Value> outputs(operands.data() + nIns,
                                           operands.data() + operands.size());

    auto indexingMaps = consumerGeneric.getIndexingMapsArray();
    auto affineMapExpand = genericOp.getIndexingMapsArray()[0];
    auto affineMapConsumer = consumerGeneric.getIndexingMapsArray()[valueIdx];
    auto realDims = getAffineMapAccessDims(affineMapExpand);
    llvm::SmallVector<AffineExpr> expr;
    for (auto [i, e] : llvm::enumerate(affineMapConsumer.getResults())) {
      llvm::errs() << i << "\n";
      e.dump();
      if (std::find(realDims.begin(), realDims.end(), i) == realDims.end()) {
        // expr[i] = getAffineConstantExpr(0, ctx);
        continue;
      } else {
        // expr[i] = e;
        expr.push_back(e);
      }
    }
    auto affineMap =
        AffineMap::get(affineMapConsumer.getNumDims(),
                       affineMapConsumer.getNumSymbols(), expr, ctx);
    indexingMaps[valueIdx] = affineMap;
    auto iteratorTypes = consumerGeneric.getIteratorTypesArray();

    for (auto x : indexingMaps)
      x.dump();

    rewriter.setInsertionPointAfter(consumerGeneric);
    auto fusedGeneric = rewriter.create<linalg::GenericOp>(
        genericOp.getLoc(), consumerGeneric.getResultTypes(), inputs, outputs,
        indexingMaps, iteratorTypes);

    auto oldBlock = consumerGeneric.getBody();
    Block &newBlock = fusedGeneric.getRegion().emplaceBlock();
    newBlock.addArguments(oldBlock->getArgumentTypes(),
                          SmallVector<Location>(oldBlock->getNumArguments(),
                                                fusedGeneric.getLoc()));

    mlir::IRMapping mapping;
    for (auto [oldArg, newArg] :
         llvm::zip(oldBlock->getArguments(), newBlock.getArguments())) {
      mapping.map(oldArg, newArg);
    }

    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToEnd(&newBlock);

    for (mlir::Operation &op : oldBlock->without_terminator()) {
      rewriter.clone(op, mapping);
    }
    rewriter.clone(*(oldBlock->getTerminator()), mapping);

    rewriter.replaceOp(consumerGeneric, fusedGeneric);
    rewriter.eraseOp(genericOp);
  }
};

class FuseGenericPass : public impl::FuseGenericPassBase<FuseGenericPass> {
public:
  using impl::FuseGenericPassBase<FuseGenericPass>::FuseGenericPassBase;

  void runOnOperation() final {
    mlir::MLIRContext &ctx = getContext();
    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ArgumentTransposePattern>(&ctx);
    patterns.add<ExpandGenericPattern>(&ctx);
    if (mlir::failed(applyPatternsAndFoldGreedily(getOperation(),
                                                  std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::accelgen
