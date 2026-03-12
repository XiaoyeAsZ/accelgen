#include "accelgen/Passes/AccelgenPasses.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
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
#define GEN_PASS_DEF_COLLAPSESHAPEDELAYPASS
// #include "accelgen/Passes/FuseGenericPass.h.inc"
#include "accelgen/Passes/AccelgenPasses.h.inc"

namespace {

class CollapseGenericPattern
    : public mlir::OpRewritePattern<linalg::GenericOp> {
  using mlir::OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(linalg::GenericOp genericOp,
                  mlir::PatternRewriter &rewriter) const override {
    auto context = rewriter.getContext();

    mlir::Type elementType =
        mlir::dyn_cast<mlir::TensorType>(genericOp.getResult(0).getType())
            .getElementType();

    bool hasCollapseFront = false;
    llvm::SmallVector<mlir::Value> frontValues;
    llvm::SmallVector<tensor::CollapseShapeOp> collapseOpToErase;
    llvm::SmallVector<llvm::SmallVector<int64_t>> dimsShapeMapping(
        genericOp.getNumLoops());
    for (auto [idx, d] : llvm::enumerate(dimsShapeMapping)) {
      d = {genericOp.getStaticLoopRanges()[idx]};
    }

    llvm::SmallVector<llvm::SmallVector<int64_t>> dimsDimMapping(
        genericOp.getNumLoops());
    // for (auto [idx, d] : llvm::enumerate(dimsDimMapping)) d = {int64_t(idx)};

    for (auto [idx, operand] : llvm::enumerate(genericOp.getOperands())) {
      if (auto collapse = operand.getDefiningOp<tensor::CollapseShapeOp>()) {
        collapse.dump();
        hasCollapseFront = true;
        // collapse.getSrc().dump();
        frontValues.push_back(collapse.getSrc());
        collapseOpToErase.push_back(collapse);

        auto reassociationMaps = collapse.getReassociationMaps();
        auto accessDims =
            getAffineMapAccessDims(genericOp.getIndexingMapsArray()[idx]);

        if (reassociationMaps.size() != accessDims.size()) {
          genericOp.dump();
          for (auto xx : reassociationMaps)
            xx.dump();
          ECHO_LIST(accessDims, ",")
        }
        assert(reassociationMaps.size() == accessDims.size());

        for (auto [i, dimAffineMap] : llvm::enumerate(reassociationMaps)) {
          auto dims = getAffineMapAccessDims(dimAffineMap);

          llvm::SmallVector<int64_t> dimShapes;
          for (auto dim : dims) {
            auto shape = collapse.getSrc().getType().getShape()[dim];
            dimShapes.push_back(shape);
          }

          if (dimShapes.size() > dimsShapeMapping[accessDims[i]].size()) {
            dimsShapeMapping[accessDims[i]] = dimShapes;
          }
          //   ECHO_LIST(dimsShapeMapping[accessDims[i]], ",")
        }
      } else {
        // operand.dump();
        frontValues.push_back(operand);
      }
    }
    if (!hasCollapseFront)
      return mlir::failure();

    for (auto xx : dimsShapeMapping) {
      ECHO_LIST(xx, ",")
    }

    // genericOp.dump();

    int64_t nDimsUpdateGenericOp = 0;
    for (auto d : dimsShapeMapping)
      nDimsUpdateGenericOp += d.size();

    int64_t tmpDim = 0;
    for (auto [idx, d] : llvm::enumerate(dimsShapeMapping)) {
      for (auto ud : d) {
        dimsDimMapping[idx].push_back(tmpDim);
        tmpDim++;
      }
      //   ECHO_LIST(dimsDimMapping[idx], ",")
    }

    llvm::SmallVector<utils::IteratorType> iteratorTypesUpdate;
    for (auto [idx, d] : llvm::enumerate(dimsShapeMapping)) {
      for (auto ud : d) {
        iteratorTypesUpdate.push_back(genericOp.getIteratorTypesArray()[idx]);
      }
    }

    llvm::SmallVector<bool> needInsertExpand;
    llvm::SmallVector<mlir::Value> updateValues(genericOp.getOperands().size());
    llvm::SmallVector<llvm::SmallVector<int64_t>> updateShapes(
        genericOp.getOperands().size());
    llvm::SmallVector<mlir::Value> updateInputValues(
        genericOp.getOperands().size());
    llvm::SmallVector<SmallVector<ReassociationIndices>> updateReassMaps(
        genericOp.getOperands().size());

    // Create expand op each input operand
    genericOp.dump();

    for (auto [idx, v] : llvm::enumerate(frontValues)) {
      auto srcType =
          mlir::dyn_cast<mlir::TensorType>(frontValues[idx].getType());
      assert(srcType);
      auto srcShape = srcType.getShape();

      //   auto originAccessDims =
      //       getAffineMapAccessDims(genericOp.getIndexingMapsArray()[idx]);

      //   for (auto [i, d] : llvm::enumerate(originAccessDims)) {
      //     expandShape.append(dimsShapeMapping[d].begin(),
      //                        dimsShapeMapping[d].end());
      //   }

      llvm::SmallVector<int64_t> expandShape;
      // int64_t tmpParDim = 0;
      for (auto [idxExper, expr] : llvm::enumerate(
               genericOp.getIndexingMapsArray()[idx].getResults())) {
        if (auto constExpr = mlir::dyn_cast<AffineConstantExpr>(expr)) {
          if (constExpr.getValue() == 0) {
            // while (genericOp.getIteratorTypesArray()[tmpParDim] !=
            //        mlir::utils::IteratorType::parallel)
            //   tmpParDim++;
            for (auto [i, s] : llvm::enumerate(dimsShapeMapping[idxExper]))
              expandShape.push_back(1);
            // tmpParDim++;
          } else
            assert(0);
        } else if (auto dimExpr = mlir::dyn_cast<AffineDimExpr>(expr)) {
          expandShape.append(dimsShapeMapping[dimExpr.getPosition()].begin(),
                             dimsShapeMapping[dimExpr.getPosition()].end());
        } else
          assert(0);
      }

      if (srcShape.equals(expandShape)) {
        updateValues[idx] = v;
        needInsertExpand.push_back(false);
        ECHO("operand check", "\n")
        v.dump();
        continue;
      }
      needInsertExpand.push_back(true);

      ECHO("need expand", "\n")
      ECHO_LIST(expandShape, ",")

      int64_t srcP = 0, destP = 0;
      llvm::SmallVector<AffineMap> affineMaps;

      SmallVector<ReassociationIndices> reassociationMaps;
      ECHO_LIST(srcShape, ",")
      ECHO_LIST(expandShape, ",")
      while (srcP != srcShape.size()) {
        // genericOp.dump();
        // ECHO(srcP, " ")
        // ECHO(destP, " ")
        // ECHO(srcShape.size(), " ")
        // ECHO(expandShape.size(), "\n")
        // ECHO_LIST(srcShape, ",")
        // ECHO_LIST(expandShape, ",")
        ReassociationIndices reassociationIdx;
        if (srcShape[srcP] == expandShape[destP]) {
          //   affineMaps.push_back(
          //       AffineMap::get(dimRangeUpdate.size(), 0,
          //                      {getAffineConstantExpr(destP, context)}));
          reassociationIdx.append({destP});
          destP++;
        } else {
          int64_t tmp = 1;
          llvm::SmallVector<int64_t> expandDims;
          llvm::SmallVector<mlir::AffineExpr> affineExprs;
          while (tmp != srcShape[srcP] && destP < expandShape.size()) {
            tmp *= expandShape[destP];
            // affineExprs.push_back(getAffineConstantExpr(destP, context));
            expandDims.push_back(destP);
            destP++;
          }

          assert(tmp == srcShape[srcP]);
          //   affineMaps.push_back(
          //       AffineMap::get(dimRangeUpdate.size(), 0, affineExprs,
          //       context));
          reassociationIdx.append(expandDims.begin(), expandDims.end());
        }
        srcP++;

        reassociationMaps.push_back(reassociationIdx);
      }

      //   auto expandOp = rewriter.create<tensor::ExpandShapeOp>(
      //       genericOp.getLoc(),
      //       mlir::RankedTensorType::get(expandShape, elementType), v,
      //       reassociationMaps);
      updateShapes[idx] = expandShape;
      updateInputValues[idx] = v;
      updateReassMaps[idx] = reassociationMaps;
      //   updateValues.push_back(expandOp.getResult());
    }

    // Create expand op
    rewriter.setInsertionPoint(genericOp);
    for (auto [idx, ins] : llvm::enumerate(updateValues)) {
      if (needInsertExpand[idx]) {
        auto expandOp = rewriter.create<tensor::ExpandShapeOp>(
            genericOp.getLoc(),
            mlir::RankedTensorType::get(
                updateShapes[idx],
                mlir::getElementTypeOrSelf(updateInputValues[idx])),
            updateInputValues[idx], updateReassMaps[idx]);
        ECHO("here push", "")
        expandOp.dump();
        expandOp.getResult().dump();
        updateValues[idx] = expandOp.getResult();
        expandOp.dump();
      }
    }

    // Update indexing maps
    llvm::SmallVector<mlir::AffineMap> indexingMapsUpdate;
    for (auto [idx, operand] : llvm::enumerate(genericOp.getOperands())) {
      llvm::SmallVector<mlir::AffineExpr> affineExprs;
      for (auto [idxExpr, expr] : llvm::enumerate(
               genericOp.getIndexingMapsArray()[idx].getResults())) {
        if (auto dimExpr = mlir::dyn_cast<mlir::AffineDimExpr>(expr)) {
          for (auto ud : dimsDimMapping[dimExpr.getPosition()]) {
            affineExprs.push_back(getAffineDimExpr(ud, context));
          }
        } else if (auto constExpr =
                       mlir::dyn_cast<mlir::AffineConstantExpr>(expr)) {
          assert(constExpr.getValue() == 0);
          for (auto [idxUd, ud] : llvm::enumerate(dimsShapeMapping[idxExpr])) {
            affineExprs.push_back(getAffineConstantExpr(0, context));
          }
        } else
          assert(0);
      }
      // auto oldAccessDims =
      //     getAffineMapAccessDims(genericOp.getIndexingMapsArray()[idx]);
      // for (auto od : oldAccessDims) {
      //   for (auto ud : dimsDimMapping[od]) {
      //     affineExprs.push_back(getAffineDimExpr(ud, context));
      //   }
      // }
      indexingMapsUpdate.push_back(
          AffineMap::get(nDimsUpdateGenericOp, 0, affineExprs, context));
    }

    // Rewrite generic op
    llvm::SmallVector<Type> resultTypes;
    for (auto [idx, result] : llvm::enumerate(genericOp.getOutputs())) {
      llvm::SmallVector<int64_t> shape;
      for (auto [idxExpr, expr] : llvm::enumerate(
               genericOp
                   .getIndexingMapsArray()[genericOp.getInputs().size() + idx]
                   .getResults())) {
        if (auto dimExpr = mlir::dyn_cast<mlir::AffineDimExpr>(expr)) {
          shape.append(dimsShapeMapping[dimExpr.getPosition()].begin(),
                       dimsShapeMapping[dimExpr.getPosition()].end());
        } else if (auto constExpr =
                       mlir::dyn_cast<mlir::AffineConstantExpr>(expr)) {
          assert(constExpr.getValue() == 0);
          for (auto [idxUd, ud] : llvm::enumerate(dimsShapeMapping[idxExpr])) {
            shape.push_back(1);
          }
        } else
          assert(0);
      }
      resultTypes.push_back(mlir::RankedTensorType::get(
          shape, mlir::getElementTypeOrSelf(result)));
    }

    // for (auto xx : updateValues) xx.dump();

    llvm::SmallVector<Value> inputsUpdate(updateValues.begin(),
                                          updateValues.begin() +
                                              genericOp.getInputs().size());
    llvm::SmallVector<Value> outputsUpdate(updateValues.begin() +
                                               genericOp.getInputs().size(),
                                           updateValues.end());

    for (auto xx : inputsUpdate)
      xx.dump();
    for (auto xx : outputsUpdate)
      xx.dump();

    auto genericOpUpdate = rewriter.create<linalg::GenericOp>(
        genericOp.getLoc(), resultTypes, inputsUpdate, outputsUpdate,
        indexingMapsUpdate, iteratorTypesUpdate);

    // TODO : print detail here

    genericOpUpdate.dump();

    // for (auto amap : indexingMapsUpdate) amap.dump();

    rewriter.cloneRegionBefore(genericOp.getRegion(),
                               genericOpUpdate.getRegion(),
                               genericOpUpdate.getRegion().end());

    Block &oldBlock = genericOp.getRegion().front();
    Block &newBlock = genericOpUpdate.getRegion().front();
    // for (auto& arg : oldBlock.getArguments())
    //   newBlock->addArgument(arg.getType(), arg.getLoc());

    // for (auto operand : genericOpUpdate.getOperands()) {
    //   newBlock.addArgument(mlir::getElementTypeOrSelf(operand),
    //                        operand.getLoc());
    // }
    // mlir::IRMapping mapping;
    // for (auto [oldArg, newArg] :
    //      llvm::zip(oldBlock.getArguments(), newBlock.getArguments()))
    //   mapping.map(oldArg, newArg);
    // for (auto &op : oldBlock) {
    //   Operation *clonedOp = op.clone(mapping);
    //   newBlock.push_back(clonedOp);
    // }
    // genericOpUpdate.getRegion().push_back(&newBlock);

    genericOpUpdate.dump();
    for (auto xx : inputsUpdate)
      xx.dump();
    for (auto xx : outputsUpdate)
      xx.dump();

    // Create collapse op
    rewriter.setInsertionPointAfter(genericOp);
    for (auto [idx, operand] : llvm::enumerate(genericOp.getOutputs())) {
      llvm::SmallVector<ReassociationIndices> reassociationMaps;
      int64_t tmp = 0;
      for (auto [idxExpr, expr] : llvm::enumerate(
               genericOp
                   .getIndexingMapsArray()[genericOp.getInputs().size() + idx]
                   .getResults())) {
        ReassociationIndices reassociationIdx;
        if (auto dimExpr = mlir::dyn_cast<mlir::AffineDimExpr>(expr)) {
          for (auto ud : dimsDimMapping[dimExpr.getPosition()]) {
            reassociationIdx.push_back(tmp);
            tmp++;
          }
        } else if (auto constExpr =
                       mlir::dyn_cast<mlir::AffineConstantExpr>(expr)) {
          assert(constExpr.getValue() == 0);
          for (auto ud : dimsDimMapping[idxExpr]) {
            reassociationIdx.push_back(tmp);
            tmp++;
          }
        } else
          assert(0);
        reassociationMaps.push_back(reassociationIdx);
      }

      auto collapseOp = rewriter.create<tensor::CollapseShapeOp>(
          genericOp.getLoc(), operand.getType(), genericOpUpdate.getResult(idx),
          reassociationMaps);

      ECHO("check!!!", "\n")
      collapseOp.dump();

      genericOp.getResult(idx).replaceAllUsesWith(collapseOp.getResult());
    }

    // Erase old op
    rewriter.eraseOp(genericOp);
    for (auto collapse : collapseOpToErase) {
      bool flag = true;
      for (auto use : collapse->getUsers()) {
        flag = false;
        break;
      }
      if (flag)
        rewriter.eraseOp(collapse);
    }

    return mlir::success();
  }
};

class CollapseShapeDelayPass
    : public impl::CollapseShapeDelayPassBase<CollapseShapeDelayPass> {
public:
  using impl::CollapseShapeDelayPassBase<
      CollapseShapeDelayPass>::CollapseShapeDelayPassBase;

  void runOnOperation() final {
    mlir::MLIRContext &ctx = getContext();
    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<CollapseGenericPattern>(&ctx);

    GreedyRewriteConfig config;
    config.setUseTopDownTraversal();

    if (mlir::failed(applyPatternsAndFoldGreedily(
            getOperation(), std::move(patterns), config))) {
      ECHO("err pass", "\n")
      getOperation()->walk([&](mlir::Operation *nestedOp) {
        if (mlir::failed(mlir::verify(nestedOp))) {
          llvm::errs() << "Verification failed for op:\n";
          nestedOp->dump();
        }
      });

      signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::accelgen
