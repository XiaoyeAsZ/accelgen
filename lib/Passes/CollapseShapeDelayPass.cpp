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
#include "mlir/Analysis/TopologicalSortUtils.h"

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

  mlir::LogicalResult matchAndRewrite(
      linalg::GenericOp genericOp,
      mlir::PatternRewriter& rewriter) const override {
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
        // collapse.dump();
        hasCollapseFront = true;
        frontValues.push_back(collapse.getSrc());
        collapseOpToErase.push_back(collapse);

        auto reassociationMaps = collapse.getReassociationMaps();
        auto accessDims =
            getAffineMapAccessDims(genericOp.getIndexingMapsArray()[idx]);

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
        frontValues.push_back(operand);
      }
    }
    if (!hasCollapseFront) return mlir::failure();

    genericOp.dump();

    int64_t nDimsUpdateGenericOp = 0;
    for (auto d : dimsShapeMapping) nDimsUpdateGenericOp += d.size();

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

    llvm::SmallVector<mlir::Value> updateValues;
    llvm::SmallVector<llvm::SmallVector<int64_t>> updateShapes;
    llvm::SmallVector<mlir::Value> updateInputValues;
    llvm::SmallVector<SmallVector<ReassociationIndices>> updateReassMaps;

    // Create expand op each input operand

    for (auto [idx, v] : llvm::enumerate(frontValues)) {
      auto srcType =
          mlir::dyn_cast<mlir::TensorType>(frontValues[idx].getType());
      assert(srcType);
      auto srcShape = srcType.getShape();

      auto originAccessDims =
          getAffineMapAccessDims(genericOp.getIndexingMapsArray()[idx]);

      llvm::SmallVector<int64_t> expandShape;
      for (auto [i, d] : llvm::enumerate(originAccessDims)) {
        expandShape.append(dimsShapeMapping[d].begin(),
                           dimsShapeMapping[d].end());
      }
      if (srcShape.equals(expandShape)) {
        updateValues.push_back(v);
        continue;
      }

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
      updateShapes.push_back(expandShape);
      updateInputValues.push_back(v);
      updateReassMaps.push_back(reassociationMaps);
      //   updateValues.push_back(expandOp.getResult());
    }

    // Create expand op
    rewriter.setInsertionPoint(genericOp);
    for (auto [idx, ins] : llvm::enumerate(updateInputValues)) {
      auto expandOp = rewriter.create<tensor::ExpandShapeOp>(
          genericOp.getLoc(),
          mlir::RankedTensorType::get(updateShapes[idx], elementType),
          updateInputValues[idx], updateReassMaps[idx]);
      updateValues.push_back(expandOp.getResult());
    }

    // Update indexing maps
    llvm::SmallVector<mlir::AffineMap> indexingMapsUpdate;
    for (auto [idx, operand] : llvm::enumerate(genericOp.getOperands())) {
      auto oldAccessDims =
          getAffineMapAccessDims(genericOp.getIndexingMapsArray()[idx]);
      llvm::SmallVector<mlir::AffineExpr> affineExprs;
      for (auto od : oldAccessDims) {
        for (auto ud : dimsDimMapping[od]) {
          affineExprs.push_back(getAffineDimExpr(ud, context));
        }
      }
      indexingMapsUpdate.push_back(
          AffineMap::get(nDimsUpdateGenericOp, 0, affineExprs, context));
    }

    // Rewrite generic op
    llvm::SmallVector<Type> resultTypes;
    for (auto [idx, result] : llvm::enumerate(genericOp.getOutputs())) {
      auto accessDims =
          getAffineMapAccessDims(genericOp.getIndexingMapsArray()[idx]);
      llvm::SmallVector<int64_t> shape;
      for (auto d : accessDims) {
        shape.append(dimsShapeMapping[d].begin(), dimsShapeMapping[d].end());
      }
      resultTypes.push_back(mlir::RankedTensorType::get(shape, elementType));
    }

    auto genericOpUpdate = rewriter.create<linalg::GenericOp>(
        genericOp.getLoc(), resultTypes,
        llvm::ArrayRef(updateValues.begin(),
                       updateValues.begin() + genericOp.getInputs().size()),
        llvm::ArrayRef(updateValues.begin() + genericOp.getInputs().size(),
                       updateValues.end()),
        indexingMapsUpdate, iteratorTypesUpdate);

    // TODO : print detail here

    // for (auto amap : indexingMapsUpdate) amap.dump();

    Block& oldBlock = genericOp.getRegion().front();
    Block* newBlock = new Block();
    for (auto& arg : oldBlock.getArguments())
      newBlock->addArgument(arg.getType(), arg.getLoc());
    mlir::IRMapping mapping;
    for (auto [oldArg, newArg] :
         llvm::zip(oldBlock.getArguments(), newBlock->getArguments()))
      mapping.map(oldArg, newArg);
    for (auto& op : oldBlock) {
      Operation* clonedOp = op.clone(mapping);
      newBlock->push_back(clonedOp);
    }
    genericOpUpdate.getRegion().push_back(newBlock);
    // genericOpUpdate.dump();

    // Create collapse op
    rewriter.setInsertionPointAfter(genericOp);
    for (auto [idx, operand] : llvm::enumerate(genericOp.getOutputs())) {
      auto oldAccessDims = getAffineMapAccessDims(
          genericOp.getIndexingMapsArray()[genericOp.getInputs().size() + idx]);

      llvm::SmallVector<ReassociationIndices> reassociationMaps;
      int64_t tmp = 0;
      for (auto [i, d] : llvm::enumerate(oldAccessDims)) {
        ReassociationIndices reassociationIdx;
        for (auto dd : dimsDimMapping[d]) {
          reassociationIdx.push_back(tmp);
          tmp++;
        }
        reassociationMaps.push_back(reassociationIdx);
      }

      auto collapseOp = rewriter.create<tensor::CollapseShapeOp>(
          genericOp.getLoc(), operand.getType(), genericOpUpdate.getResult(idx),
          reassociationMaps);

      genericOp.getResult(idx).replaceAllUsesWith(collapseOp.getResult());
    }

    // Erase old op
    rewriter.eraseOp(genericOp);
    for (auto collapse : collapseOpToErase) rewriter.eraseOp(collapse);

    return mlir::success();
  }
};

class CollapseShapeDelayPass
    : public impl::CollapseShapeDelayPassBase<CollapseShapeDelayPass> {
 public:
  using impl::CollapseShapeDelayPassBase<
      CollapseShapeDelayPass>::CollapseShapeDelayPassBase;

  void runOnOperation() final {
    mlir::MLIRContext& ctx = getContext();
    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<CollapseGenericPattern>(&ctx);
    if (mlir::failed(applyPatternsAndFoldGreedily(getOperation(),
                                                  std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace
}  // namespace mlir::accelgen
