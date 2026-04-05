

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

// #include "accelgen/Passes/Passes.h"
// #include "accelgen/Passes/FuseGenericPass.h"
#include "accelgen/Utils/AffineMapUtils.h"
#include "accelgen/Utils/DebugUtils.h"
#include "accelgen/Utils/OperationUtils.h"

namespace mlir::accelgen {
#define GEN_PASS_DEF_FUSEGENERICPASS
// #include "accelgen/Passes/FuseGenericPass.h.inc"
#include "accelgen/Passes/AccelgenPasses.h.inc"

namespace {

class ArgumentTransposePattern
    : public mlir::OpRewritePattern<linalg::GenericOp> {
  using mlir::OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      linalg::GenericOp genericOp,
      mlir::PatternRewriter& rewriter) const override {
    if (!genericOp->hasAttr("accelgen.transpose")) return mlir::failure();
    mlir::Value input = genericOp.getInputs()[0];
    auto blockArg = mlir::dyn_cast<BlockArgument>(input);
    if (!blockArg) return mlir::failure();
    if (!input.hasOneUse()) return mlir::failure();

    auto func = genericOp->getParentOfType<func::FuncOp>();
    if (!func) return mlir::failure();

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

class TransposeGenericPattern
    : public mlir::OpRewritePattern<linalg::GenericOp> {
  using mlir::OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      linalg::GenericOp genericOp,
      mlir::PatternRewriter& rewriter) const override {
    if (!genericOp->hasAttr("accelgen.transpose")) return mlir::failure();

    auto ctx = rewriter.getContext();

    llvm::SmallVector<mlir::Operation*> users;
    for (auto use : genericOp->getUsers()) {
      if (!use->hasAttr("accelgen.memory_transformation")) users.push_back(use);
    }

    // genericOp.dump();
    // ECHO(users.size(), "\n")
    // users[0]->dump();

    bool hasGeneric = false;
    for (auto use : users) {
      auto generic = mlir::dyn_cast<linalg::GenericOp>(use);
      if (!generic) continue;
      hasGeneric = true;

      mlir::Value updateInput = genericOp.getInputs()[0];

      auto index = std::distance(
          generic.getInputs().begin(),
          std::find(generic.getInputs().begin(), generic.getInputs().end(),
                    genericOp.getResults()[0]));

      // ECHO("index", "\n")
      // ECHO(index, "\n")
      // updateInput.dump();
      llvm::SmallVector<mlir::Value> originInputs = generic.getInputs();
      originInputs[index] = updateInput;
      llvm::SmallVector<mlir::Value> originOutputs = generic.getOutputs();

      auto originIndexingMaps = generic.getIndexingMapsArray();
      auto accDims =
          getAffineMapAccessDims(genericOp.getIndexingMapsArray()[0]);
      auto transAccDims =
          getAffineMapAccessDims(genericOp.getIndexingMapsArray()[1]);
      llvm::DenseMap<int64_t, int64_t> dimMap;
      for (auto [i, s] : llvm::enumerate(accDims)) dimMap[s] = i;
      llvm::SmallVector<int64_t> permuIndex;
      for (auto s : transAccDims) permuIndex.push_back(dimMap[s]);

      auto dimsToAdjust = getAffineMapAccessDims(originIndexingMaps[index]);
      llvm::SmallVector<AffineExpr> dimsAfterAdjust;
      assert(dimsToAdjust.size() == permuIndex.size());
      for (auto i : permuIndex)
        dimsAfterAdjust.push_back(getAffineDimExpr(dimsToAdjust[i], ctx));
      auto updateAffineMap = AffineMap::get(
          originIndexingMaps[index].getNumDims(), 0, dimsAfterAdjust, ctx);

      originIndexingMaps[index] = updateAffineMap;

      rewriter.setInsertionPointAfter(generic);
      auto updateGeneric = rewriter.create<linalg::GenericOp>(
          generic.getLoc(), generic.getResults().getTypes(), originInputs,
          originOutputs, originIndexingMaps, generic.getIteratorTypesArray());

      rewriter.cloneRegionBefore(generic.getRegion(), updateGeneric.getRegion(),
                                 updateGeneric.getRegion().begin());

      // updateGeneric.dump();

      rewriter.replaceOp(generic, updateGeneric.getResults()[0]);
      // rewriter.eraseOp(generic);
    }

    if (hasGeneric) return mlir::success();

    return mlir::failure();
  }
};

class ExpandGenericPattern : public mlir::OpRewritePattern<linalg::GenericOp> {
  using mlir::OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      linalg::GenericOp genericOp,
      mlir::PatternRewriter& rewriter) const override {
    if (!genericOp->hasAttr("accelgen.expand")) return mlir::failure();

    auto results = genericOp.getResults();
    assert(results.size() == 1);
    if (!results[0].hasOneUse()) return mlir::failure();
    linalg::GenericOp consumerGeneric;
    for (auto use : results[0].getUsers()) {
      if (mlir::dyn_cast<linalg::GenericOp>(use))
        consumerGeneric = mlir::dyn_cast<linalg::GenericOp>(use);
      else
        return mlir::failure();
    }

    // genericOp.dump();

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
      // llvm::errs() << i << "\n";
      // e.dump();
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

    // for (auto x : indexingMaps)
    //   x.dump();

    rewriter.setInsertionPointAfter(consumerGeneric);
    auto fusedGeneric = rewriter.create<linalg::GenericOp>(
        genericOp.getLoc(), consumerGeneric.getResultTypes(), inputs, outputs,
        indexingMaps, iteratorTypes);

    auto oldBlock = consumerGeneric.getBody();
    Block& newBlock = fusedGeneric.getRegion().emplaceBlock();
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

    for (mlir::Operation& op : oldBlock->without_terminator()) {
      rewriter.clone(op, mapping);
    }
    rewriter.clone(*(oldBlock->getTerminator()), mapping);

    rewriter.replaceOp(consumerGeneric, fusedGeneric);
    rewriter.eraseOp(genericOp);
  }
};

class ExpandCollapseGenericPattern
    : public mlir::OpRewritePattern<linalg::GenericOp> {
  using mlir::OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      linalg::GenericOp genericOp,
      mlir::PatternRewriter& rewriter) const override {
    if (genericOp->hasAttr("accelgen.memory_transformation"))
      return mlir::failure();
    // genericOp.dump();
    auto context = rewriter.getContext();
    bool hasExpandFront = false;

    assert(genericOp.getNumLoops() == genericOp.getStaticLoopRanges().size());
    llvm::SmallVector<llvm::SmallVector<int64_t>> dimShapeReassMap(
        genericOp.getNumLoops());
    for (auto [idxReassIdx, reassIdx] : llvm::enumerate(dimShapeReassMap))
      dimShapeReassMap[idxReassIdx] = {
          genericOp.getStaticLoopRanges()[idxReassIdx]};

    for (auto [idxOperand, operand] :
         llvm::enumerate(genericOp.getOperands())) {
      auto preGeneric = operand.getDefiningOp<linalg::GenericOp>();
      auto preCollapse = operand.getDefiningOp<tensor::CollapseShapeOp>();
      auto prePreGeneric =
          preCollapse ? preCollapse.getSrc().getDefiningOp<linalg::GenericOp>()
                      : nullptr;

      if (preCollapse) {
        // Update shape of generic op
        auto collapseReassMaps = preCollapse.getReassociationMaps();
        auto indexingMap = genericOp.getIndexingMapsArray()[idxOperand];
        assert(indexingMap.getResults().size() == collapseReassMaps.size());
        traverseAffineMapResults(
            indexingMap,
            [&](uint64_t idxExpr, AffineDimExpr dimExpr) {
              auto nowDimShape = dimShapeReassMap[dimExpr.getPosition()];
              llvm::SmallVector<int64_t> updateDimShape;
              for (auto [idxDim, dim] : llvm::enumerate(
                       getAffineMapAccessDims(collapseReassMaps[idxExpr]))) {
                updateDimShape.push_back(
                    preCollapse.getSrc().getType().getShape()[dim]);
              }

              if (updateDimShape.size() > nowDimShape.size())
                dimShapeReassMap[dimExpr.getPosition()] = updateDimShape;
            },
            [&](uint64_t idxExpr, AffineConstantExpr constExpr) {
              assert(constExpr.getValue() == 0);
            });
      }

      if (preGeneric) {
        if (!preGeneric->hasAttr("accelgen.expand")) continue;
        hasExpandFront = true;
      } else if (preCollapse) {
        if (prePreGeneric && !prePreGeneric->hasAttr("accelgen.expand"))
          continue;
        hasExpandFront = true;
      } else
        continue;
    }
    if (!hasExpandFront) return mlir::failure();

    llvm::SmallVector<ReassociationIndices> dimDimReassMap(
        genericOp.getNumLoops());
    int64_t tmp = 0;
    for (auto [idxReassIdx, reassIdx] : llvm::enumerate(dimDimReassMap)) {
      for (auto [idxDimShapeReass, dimShapeReass] :
           llvm::enumerate(dimShapeReassMap[idxReassIdx])) {
        dimDimReassMap[idxReassIdx].push_back(tmp++);
      }
    }

    int64_t updateNumDims = 0;
    for (auto [idxDimShapeReassMap, itemDimShapeReassMap] :
         llvm::enumerate(dimShapeReassMap)) {
      for (auto [idxShape, itemShape] : llvm::enumerate(itemDimShapeReassMap))
        updateNumDims++;
    }

    // Rewrite stage

    // For create expand op
    llvm::SmallVector<bool> needInsertExpand(genericOp.getNumLoops(), false);
    llvm::SmallVector<mlir::Value> updateExpandInputs(genericOp.getNumLoops());
    llvm::SmallVector<llvm::SmallVector<int64_t>> updateExpandShapes(
        genericOp.getNumLoops());
    llvm::SmallVector<llvm::SmallVector<ReassociationIndices>>
        updateExpandReassMaps(genericOp.getNumLoops());

    // For create generic op

    llvm::SmallVector<mlir::Value> updateInputs(genericOp.getOperands().size());
    for (auto [idxOperand, itemOperand] :
         llvm::enumerate(genericOp.getOperands())) {
      auto preGeneric = itemOperand.getDefiningOp<linalg::GenericOp>();
      auto preCollapse = itemOperand.getDefiningOp<tensor::CollapseShapeOp>();
      auto prePreGeneric =
          preCollapse ? preCollapse.getSrc().getDefiningOp<linalg::GenericOp>()
                      : nullptr;
      if (preGeneric && preGeneric->hasAttr("accelgen.expand")) {
        updateInputs[idxOperand] = preGeneric.getInputs()[0];
      } else if (preCollapse && prePreGeneric &&
                 prePreGeneric->hasAttr("accelgen.expand")) {
        updateInputs[idxOperand] = prePreGeneric.getInputs()[0];
      } else {
        updateInputs[idxOperand] = itemOperand;
      }
    }

    llvm::SmallVector<mlir::utils::IteratorType> updateIteratorTypes;
    for (auto [idxDimShapeReassMap, itemDimShapeReassMap] :
         llvm::enumerate(dimShapeReassMap)) {
      for (auto [idxShape, itemShape] : llvm::enumerate(itemDimShapeReassMap))
        updateIteratorTypes.push_back(
            genericOp.getIteratorTypesArray()[idxDimShapeReassMap]);
    }

    // For create expand op
    for (auto [idxOperand, operand] :
         llvm::enumerate(genericOp.getOperands())) {
      auto preGeneric = operand.getDefiningOp<linalg::GenericOp>();
      auto preCollapse = operand.getDefiningOp<tensor::CollapseShapeOp>();
      auto prePreGeneric =
          preCollapse ? preCollapse.getSrc().getDefiningOp<linalg::GenericOp>()
                      : nullptr;

      llvm::SmallVector<int64_t> shapeInUpdateGeneric;
      int64_t tmp = 0;
      traverseAffineMapResults(
          genericOp.getIndexingMapsArray()[idxOperand],
          [&](uint64_t idxExpr, AffineDimExpr dimExpr) {
            shapeInUpdateGeneric.append(
                dimShapeReassMap[dimExpr.getPosition()].begin(),
                dimShapeReassMap[dimExpr.getPosition()].end());
          },
          [&](uint64_t idxExpr, AffineConstantExpr constExpr) {
            assert(constExpr.getValue() == 0);
            shapeInUpdateGeneric.append(dimShapeReassMap[idxExpr].begin(),
                                        dimShapeReassMap[idxExpr].end());
          });

      llvm::SmallVector<int64_t> shapeBeforeExpand;
      llvm::SmallVector<int64_t> shapeAfterExpand;
      mlir::ShapedType shapedAfter;
      mlir::Value updateInput;
      if (preGeneric && preGeneric->hasAttr("accelgen.expand")) {
        updateInput = preGeneric.getInputs()[0];
        shapedAfter = mlir::dyn_cast<mlir::ShapedType>(
            preGeneric.getOutputs()[0].getType());
        shapeBeforeExpand =
            llvm::SmallVector<int64_t>(mlir::dyn_cast<mlir::ShapedType>(
                                           preGeneric.getInputs()[0].getType())
                                           .getShape());

      } else if (preCollapse && prePreGeneric &&
                 prePreGeneric->hasAttr("accelgen.expand")) {
        updateInput = prePreGeneric.getInputs()[0];
        shapedAfter = mlir::dyn_cast<mlir::ShapedType>(
            prePreGeneric.getOutputs()[0].getType());
        shapeBeforeExpand = llvm::SmallVector<int64_t>(
            mlir::dyn_cast<mlir::ShapedType>(
                prePreGeneric.getInputs()[0].getType())
                .getShape());

      } else {
        updateInput = operand;
        shapedAfter = mlir::dyn_cast<mlir::ShapedType>(operand.getType());
      }
      shapeAfterExpand.append(shapedAfter.getShape().begin(),
                              shapedAfter.getShape().end());

      // Generate reassociation map for expand
      int64_t srcP = 0, updateP = 0;
      llvm::SmallVector<ReassociationIndices> updateReassMaps;
      while (srcP < shapeAfterExpand.size()) {
        ReassociationIndices tmpReassIdx;
        // ECHO(shapeAfterExpand.size(), "\n")
        // ECHO(shapeInUpdateGeneric.size(), "\n")
        // ECHO(srcP, "\n")
        // ECHO(updateP, "\n")
        if (shapeAfterExpand[srcP] == shapeInUpdateGeneric[updateP]) {
          tmpReassIdx.push_back(updateP);
          srcP++;
          updateP++;
        } else {
          int64_t tmp = 1;
          while (tmp != shapeAfterExpand[srcP]) {
            tmp *= shapeInUpdateGeneric[updateP];
            tmpReassIdx.push_back(updateP);
            updateP++;
          }
          srcP++;
        }
        updateReassMaps.push_back(tmpReassIdx);
      }

      // Adjust reassociation map and update shape by eliminating those expand
      // dims
      llvm::SmallVector<int64_t> shapeExpand;
      llvm::SmallVector<ReassociationIndices> reassMapsExpand;
      if (shapeBeforeExpand.size() != 0) {
        llvm::SmallVector<int64_t> expandDims;
        int64_t srcP = 0, updateP = 0;
        while (srcP < shapeBeforeExpand.size()) {
          if (shapeBeforeExpand[srcP] == shapeAfterExpand[updateP]) {
            srcP++;
            updateP++;
          } else {
            expandDims.push_back(updateP);
            updateP++;
          }
        }
        // Adjust update shape
        for (auto [idxShape, itemShape] : llvm::enumerate(shapeAfterExpand)) {
          if (!llvm::is_contained(expandDims, idxShape)) {
            for (auto [idxDim, itemDim] :
                 llvm::enumerate(updateReassMaps[idxShape])) {
              shapeExpand.push_back(shapeInUpdateGeneric[itemDim]);
            }
          }
        }

        // Adjust reassociation map
        int64_t tmp = 0;
        for (auto [idxReassIdx, itemReassIdx] :
             llvm::enumerate(updateReassMaps)) {
          if (!llvm::is_contained(expandDims, idxReassIdx)) {
            ReassociationIndices tmpReassIdx;
            for (auto [idxReassDim, itemReassDim] :
                 llvm::enumerate(itemReassIdx)) {
              tmpReassIdx.push_back(tmp++);
            }
            reassMapsExpand.push_back(tmpReassIdx);
          }
        }

      } else {
        shapeExpand = shapeInUpdateGeneric;
        reassMapsExpand = updateReassMaps;
      }

      if (shapeAfterExpand != shapeInUpdateGeneric) {
        needInsertExpand[idxOperand] = true;
        updateExpandInputs[idxOperand] = updateInput;
        updateExpandShapes[idxOperand] = shapeExpand;
        updateExpandReassMaps[idxOperand] = reassMapsExpand;
      }
    }

    // Create expand op for those needed
    rewriter.setInsertionPoint(genericOp);
    for (auto [idxNeedExpand, needExpand] : llvm::enumerate(needInsertExpand)) {
      if (needExpand) {
        auto expandOp = rewriter.create<tensor::ExpandShapeOp>(
            genericOp.getLoc(),
            RankedTensorType::get(
                updateExpandShapes[idxNeedExpand],
                getElementTypeOrSelf(updateExpandInputs[idxNeedExpand])),
            updateExpandInputs[idxNeedExpand],
            updateExpandReassMaps[idxNeedExpand]);
        updateInputs[idxNeedExpand] = expandOp.getResult();
      }
    }

    // Update indexing maps
    llvm::SmallVector<AffineMap> updateIndexingMaps(
        genericOp.getOperands().size());
    for (auto [idxOperand, itemOperand] :
         llvm::enumerate(genericOp.getOperands())) {
      auto preGeneric = itemOperand.getDefiningOp<linalg::GenericOp>();
      auto preCollapse = itemOperand.getDefiningOp<tensor::CollapseShapeOp>();
      auto prePreGeneric =
          preCollapse ? preCollapse.getSrc().getDefiningOp<linalg::GenericOp>()
                      : nullptr;
      if (preGeneric && preGeneric->hasAttr("accelgen.expand")) {
        auto shaped = mlir::dyn_cast<mlir::ShapedType>(
            updateInputs[idxOperand].getType());
        llvm::SmallVector<int64_t> srcShape =
            shaped ? llvm::to_vector(shaped.getShape())
                   : llvm::SmallVector<int64_t>{};

        llvm::SmallVector<std::pair<int64_t, int64_t>> updateShapeMap;
        traverseAffineMapResults(
            genericOp.getIndexingMapsArray()[idxOperand],
            [&](uint64_t idxExpr, AffineDimExpr dimExpr) {
              for (auto [dim, shape] :
                   llvm::zip(dimDimReassMap[dimExpr.getPosition()],
                             dimShapeReassMap[dimExpr.getPosition()])) {
                updateShapeMap.push_back({dim, shape});
              }
            },
            [&](uint64_t idxExpr, AffineConstantExpr constExpr) {
              assert(constExpr.getValue() == 0);
              for (auto [dim, shape] : llvm::zip(dimDimReassMap[idxExpr],
                                                 dimShapeReassMap[idxExpr])) {
                updateShapeMap.push_back({dim, shape});
              }
            });

        llvm::SmallVector<AffineExpr> dimExprs;
        // ECHO("here!!!!!!", "\n")
        // ECHO(needInsertExpand[idxOperand], "\n")
        // ECHO_LIST(srcShape, ",")
        // for (auto [xx, yy] : updateShapeMap) {
        //   ECHO(xx, "\n")
        //   ECHO(yy, "\n")
        // }
        int64_t srcP = 0, updateP = 0;
        while (srcP < srcShape.size()) {
          // ECHO(srcShape.size(), "\n")
          // ECHO(updateShapeMap.size(), "\n")
          // ECHO(srcP, "\n")
          // ECHO(updateP, "\n")
          if (srcShape[srcP] == updateShapeMap[updateP].second) {
            dimExprs.push_back(
                getAffineDimExpr(updateShapeMap[updateP].first, getContext()));
            srcP++;
            updateP++;
          } else if (srcShape[srcP] == 1) {
            dimExprs.push_back(getAffineConstantExpr(0, getContext()));
            srcP++;
            updateP++;
          } else {
            // assert(0);
            updateP++;
          }
        }
        updateIndexingMaps[idxOperand] =
            AffineMap::get(updateNumDims, 0, dimExprs, context);

        // ECHO("affine map check 1", "\n")
        // updateIndexingMaps[idxOperand].dump();
      } else if (preCollapse && prePreGeneric &&
                 prePreGeneric->hasAttr("accelgen.expand")) {
        auto shaped = mlir::dyn_cast<mlir::ShapedType>(
            updateInputs[idxOperand].getType());
        llvm::SmallVector<int64_t> srcShape =
            shaped ? llvm::to_vector(shaped.getShape())
                   : llvm::SmallVector<int64_t>{};

        llvm::SmallVector<std::pair<int64_t, int64_t>> updateShapeMap;
        traverseAffineMapResults(
            genericOp.getIndexingMapsArray()[idxOperand],
            [&](uint64_t idxExpr, AffineDimExpr dimExpr) {
              for (auto [dim, shape] :
                   llvm::zip(dimDimReassMap[dimExpr.getPosition()],
                             dimShapeReassMap[dimExpr.getPosition()])) {
                updateShapeMap.push_back({dim, shape});
              }
            },
            [&](uint64_t idxExpr, AffineConstantExpr constExpr) {
              assert(constExpr.getValue() == 0);
              for (auto [dim, shape] : llvm::zip(dimDimReassMap[idxExpr],
                                                 dimShapeReassMap[idxExpr])) {
                updateShapeMap.push_back({dim, shape});
              }
            });
        // shaped.dump();
        // ECHO("check", "\n")
        // for (auto xx : dimShapeReassMap) {
        //   ECHO_LIST(xx, ",")
        // }
        // for (auto xx : dimDimReassMap) {
        //   ECHO_LIST(xx, ",")
        // }
        // genericOp.getIndexingMapsArray()[idxOperand].dump();
        // for (auto [xx, yy] : updateShapeMap) {
        //   ECHO(xx, "\n")
        //   ECHO(yy, "\n")
        // }
        llvm::SmallVector<AffineExpr> dimExprs;
        int64_t srcP = 0, updateP = 0;
        while (srcP < srcShape.size()) {
          if (srcShape[srcP] == updateShapeMap[updateP].second) {
            dimExprs.push_back(
                getAffineDimExpr(updateShapeMap[updateP].first, getContext()));
            srcP++;
            updateP++;
          } else if (srcShape[srcP] == 1) {
            dimExprs.push_back(getAffineConstantExpr(0, getContext()));
            srcP++;
            updateP++;
          } else {
            // assert(0);
            updateP++;
          }
        }
        updateIndexingMaps[idxOperand] =
            AffineMap::get(updateNumDims, 0, dimExprs, context);

        // ECHO("affine map check 2", "\n")
        // updateIndexingMaps[idxOperand].dump();

      } else {
        llvm::SmallVector<AffineExpr> dimExprs;
        traverseAffineMapResults(
            genericOp.getIndexingMapsArray()[idxOperand],
            [&](uint64_t idxExpr, AffineDimExpr dimExpr) {
              for (auto [idxDim, itemDim] :
                   llvm::enumerate(dimDimReassMap[dimExpr.getPosition()])) {
                dimExprs.push_back(getAffineDimExpr(itemDim, getContext()));
              }
            },
            [&](uint64_t idxExpr, AffineConstantExpr constExpr) {
              assert(constExpr.getValue() == 0);
              for (auto [idxDim, itemDim] :
                   llvm::enumerate(dimDimReassMap[idxExpr])) {
                dimExprs.push_back(getAffineConstantExpr(0, getContext()));
              }
            });
        // for (auto xx : dimExprs) xx.dump();
        // ECHO(updateNumDims, "\n")
        updateIndexingMaps[idxOperand] =
            AffineMap::get(updateNumDims, 0, dimExprs, context);
        // ECHO("affine map check 3", "\n")
        // updateIndexingMaps[idxOperand].dump();
      }
    }

    // Create update generic op, fold expand generic op and collapse shape op
    llvm::SmallVector<Type> resultTypes;
    for (auto [idxOutputs, itemOutputs] :
         llvm::enumerate(genericOp.getOutputs())) {
      llvm::SmallVector<int64_t> shape;
      traverseAffineMapResults(
          genericOp.getIndexingMapsArray()[idxOutputs +
                                           genericOp.getInputs().size()],
          [&](uint64_t idxExpr, AffineDimExpr dimExpr) {
            shape.append(dimShapeReassMap[dimExpr.getPosition()].begin(),
                         dimShapeReassMap[dimExpr.getPosition()].end());
          },
          [&](uint64_t idxExpr, AffineConstantExpr constExpr) {
            shape.append(dimShapeReassMap[idxExpr].begin(),
                         dimShapeReassMap[idxExpr].end());
          });
      resultTypes.push_back(mlir::RankedTensorType::get(
          shape, mlir::getElementTypeOrSelf(itemOutputs)));
    }

    llvm::SmallVector<Value> inputsUpdate(
        updateInputs.begin(),
        updateInputs.begin() + genericOp.getInputs().size());
    llvm::SmallVector<Value> outputsUpdate(
        updateInputs.begin() + genericOp.getInputs().size(),
        updateInputs.end());

    auto genericOpUpdate = rewriter.create<linalg::GenericOp>(
        genericOp.getLoc(), resultTypes, inputsUpdate, outputsUpdate,
        updateIndexingMaps, updateIteratorTypes);
    rewriter.cloneRegionBefore(genericOp.getRegion(),
                               genericOpUpdate.getRegion(),
                               genericOpUpdate.getRegion().end());

    // genericOpUpdate.dump();

    llvm::SmallVector<mlir::Value> updateCollapseOpResults;
    for (auto [idxOutputs, itemOutputs] :
         llvm::enumerate(genericOp.getOutputs())) {
      llvm::SmallVector<ReassociationIndices> updateCollapseReassMap(
          genericOp
              .getIndexingMapsArray()[idxOutputs + genericOp.getInputs().size()]
              .getResults()
              .size());
      // ECHO(updateCollapseReassMap.size(), ",")
      // ECHO(
      //     genericOp
      //         .getIndexingMapsArray()[idxOutputs +
      //         genericOp.getInputs().size()] .getResults() .size(),
      //     "\n")
      int64_t tmp = 0;
      traverseAffineMapResults(
          genericOp.getIndexingMapsArray()[idxOutputs +
                                           genericOp.getInputs().size()],
          [&](uint64_t idxExpr, AffineDimExpr dimExpr) {
            for (auto i :
                 llvm::seq(dimDimReassMap[dimExpr.getPosition()].size()))
              updateCollapseReassMap[idxExpr].push_back(tmp++);
          },
          [&](uint64_t idxExpr, AffineConstantExpr constExpr) {
            for (auto i : llvm::seq(dimDimReassMap[idxExpr].size()))
              updateCollapseReassMap[idxExpr].push_back(tmp++);
          });
      auto collapseOp = rewriter.create<tensor::CollapseShapeOp>(
          genericOp.getLoc(), itemOutputs.getType(),
          genericOpUpdate.getResult(idxOutputs), updateCollapseReassMap);
      updateCollapseOpResults.push_back(collapseOp.getResult());
    }

    rewriter.replaceAllUsesWith(genericOp.getResults(),
                                updateCollapseOpResults);

    return mlir::success();
  }
};

class FuseGenericPass : public impl::FuseGenericPassBase<FuseGenericPass> {
 public:
  using impl::FuseGenericPassBase<FuseGenericPass>::FuseGenericPassBase;

  void runOnOperation() final {
    mlir::MLIRContext& ctx = getContext();
    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ArgumentTransposePattern>(&ctx);
    patterns.add<TransposeGenericPattern>(&ctx);
    patterns.add<ExpandGenericPattern>(&ctx);
    // patterns.add<ExpandCollapseGenericPattern>(&ctx);
    if (mlir::failed(applyPatternsAndFoldGreedily(getOperation(),
                                                  std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace
}  // namespace mlir::accelgen
