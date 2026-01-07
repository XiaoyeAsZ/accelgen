#include "accelgen/Passes/SpeFissionPass.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "accelgen/Dialect/Spe/SpeDialect.h"
#include "accelgen/Dialect/Spe/SpeOps.h"

namespace mlir::accelgen {
#define GEN_PASS_DEF_SPEFISSION
#include "accelgen/Passes/SpeFissionPass.h.inc"

namespace {

class ComposedSpeAffinePattern
    : public mlir::OpRewritePattern<affine::AffineForOp> {
  using mlir::OpRewritePattern<affine::AffineForOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      affine::AffineForOp affineForOp,
      mlir::PatternRewriter& rewriter) const override {
    auto& innermost = affineForOp;
    while (isa<affine::AffineForOp>(
        innermost.getBody()->getOperations().front())) {
      innermost = mlir::dyn_cast<affine::AffineForOp>(
          innermost.getBody()->getOperations().front());
    }

    auto& opLilst = innermost.getBody()->getOperations();
    opLilst.for (unsigned int opIter = 0; opIter < opLilst.size(); opIter++) {
      if (opIter != opLilst.size() - 1 &&
          (opLilst[opIter]).getDialect()->getNamespace() == "spe" &&
          (opLilst[opIter + 1]).getDialect()->getNamespace() == "spe") {
        auto storeOpRef = mlir::dyn_cast<affine::AffineStoreOp>(opLilst.back());
        auto loadOpRef = mlir::dyn_cast<affine::AffineLoadOp>(
            (*opIter).getOperand(0).getDefiningOp());
        assert(loadOpRef);
        auto allocOpRef = mlir::dyn_cast<memref::AllocOp>(
            loadOpRef.getMemRef().getDefiningOp());
        assert(allocOpRef);

        rewriter.setInsertionPointAfter(affineForOp);
        auto allocOp = rewriter.create<memref::AllocOp>(
            affineForOp.getLoc(), allocOpRef.getOperandTypes());
        rewriter.setInsertionPointAfter(&(*opIter));
        rewriter.create<affine::AffineStoreOp>((*opIter).getLoc(),
                                               (*opIter).getResult(0));
        return mlir::success();
      }
    };
    return mlir::failure();
  }
};

class SpeFission : public impl::SpeFissionBase<SpeFission> {
 public:
  using impl::SpeFissionBase<SpeFission>::SpeFissionBase;

  void runOnOperation() final {
    mlir::MLIRContext* ctx = &getContext();
    auto module = getOperation();

    module.walk([&](affine::AffineForOp affineForOp) {
      if (affineForOp.getBody()->front().hasTrait<OpTrait::AffineScope>())
        return;

      SmallVector<mlir::Operation*> affineLoadOps, affineStoreOps, speOps;
    });
  }
};

}  // namespace
}  // namespace mlir::accelgen
