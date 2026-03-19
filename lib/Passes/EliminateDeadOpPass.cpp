

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
#define GEN_PASS_DEF_ELIMINATEDEADOPPASS
#include "accelgen/Passes/AccelgenPasses.h.inc"

namespace {

class DeadGenericResultPattern
    : public mlir::OpRewritePattern<linalg::GenericOp> {
  using mlir::OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      linalg::GenericOp genericOp,
      mlir::PatternRewriter& rewriter) const override {
    llvm::SmallVector<int64_t> aliveResults;
    for (auto [idxResult, itemResult] :
         llvm::enumerate(genericOp.getResults())) {
      if (!itemResult.use_empty()) {
        aliveResults.push_back(idxResult);
      }
    }

    if (aliveResults.size() == genericOp.getResults().size())
      return mlir::failure();

    auto nIns = genericOp.getResults().size();

    llvm::SmallVector<mlir::Type> resultTypes;
    llvm::SmallVector<mlir::Value> updateOutpus;

    for (auto [idxAliveResult, itemAliveResult] :
         llvm::enumerate(aliveResults)) {
      resultTypes.push_back(genericOp.getResults()[itemAliveResult].getType());
      updateOutpus.push_back(genericOp.getOutputs()[itemAliveResult]);
    }

    llvm::SmallVector<AffineMap> updateIndexingMaps;
    for (auto [idxIns, itemIns] : llvm::enumerate(genericOp.getInputs())) {
      updateIndexingMaps.push_back(genericOp.getIndexingMapsArray()[idxIns]);
    }
    for (auto [idxAliveResult, itemAliveResult] :
         llvm::enumerate(aliveResults)) {
      updateIndexingMaps.push_back(
          genericOp.getIndexingMapsArray()[nIns + itemAliveResult]);
    }

    rewriter.setInsertionPoint(genericOp);
    auto updateGeneric = rewriter.create<linalg::GenericOp>(
        genericOp.getLoc(), resultTypes, genericOp.getInputs(), updateOutpus,
        updateIndexingMaps, genericOp.getIteratorTypesArray());

    Block* body = new Block();
    updateGeneric.getRegion().push_back(body);

    for (auto [idxOperand, itemOperand] :
         llvm::enumerate(updateGeneric.getOperands()))
      body->addArguments(getElementTypeOrSelf(itemOperand.getType()),
                         itemOperand.getLoc());

    rewriter.setInsertionPointToStart(updateGeneric.getBody());

    IRMapping mapper;
    mapper.map(genericOp.getRegion().front().getArguments(),
               updateGeneric.getRegion().front().getArguments());

    for (auto& op : genericOp.getRegion().front().without_terminator()) {
      rewriter.clone(op, mapper);
    }

    auto yieldOp =
        llvm::cast<linalg::YieldOp>(genericOp.getBody()->getTerminator());

    llvm::SmallVector<mlir::Value> updateYeildValues;
    for (auto [idxAliveResult, itemAliveResult] :
         llvm::enumerate(aliveResults)) {
      updateYeildValues.push_back(
          mapper.lookup(yieldOp.getOperands()[itemAliveResult]));
    }
    rewriter.create<linalg::YieldOp>(yieldOp.getLoc(), updateYeildValues);

    for (auto [idxAliveResult, itemAliveResult] :
         llvm::enumerate(aliveResults)) {
      rewriter.replaceAllUsesWith(genericOp.getResults()[itemAliveResult],
                                  updateGeneric.getResults()[idxAliveResult]);
    }
    rewriter.eraseOp(genericOp);

    return mlir::success();
  }
};

class EliminateDeadOpPass
    : public impl::EliminateDeadOpPassBase<EliminateDeadOpPass> {
 public:
  using impl::EliminateDeadOpPassBase<
      EliminateDeadOpPass>::EliminateDeadOpPassBase;

  void runOnOperation() final {
    mlir::MLIRContext& ctx = getContext();
    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<DeadGenericResultPattern>(&ctx);
    if (mlir::failed(applyPatternsAndFoldGreedily(getOperation(),
                                                  std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace
}  // namespace mlir::accelgen
