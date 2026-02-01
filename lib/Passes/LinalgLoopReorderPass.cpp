// #include "accelgen/Passes/LinalgLoopReorderPass.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "accelgen/Passes/AccelgenPasses.h"
namespace mlir::accelgen {
#define GEN_PASS_DEF_LINALGLOOPREORDER
// #include "accelgen/Passes/LinalgLoopReorderPass.h.inc"
#include "accelgen/Passes/AccelgenPasses.h.inc"

namespace {
class LinalgLoopReorder
    : public impl::LinalgLoopReorderBase<LinalgLoopReorder> {
 public:
  using impl::LinalgLoopReorderBase<LinalgLoopReorder>::LinalgLoopReorderBase;
  void runOnOperation() final {
    mlir::ModuleOp moduleOp = getOperation();
    auto* ctx = &getContext();
    mlir::OpBuilder builder(ctx);

    moduleOp.walk([&](mlir::linalg::LinalgOp linalgOp) {
      mlir::TypeSwitch<mlir::Operation*>(linalgOp)
          .Case<mlir::linalg::BatchMatmulOp>([&](mlir::linalg::BatchMatmulOp
                                                     batchMatmul) {
            mlir::OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPoint(batchMatmul);

            // Extract operands
            auto lhs = batchMatmul.getDpsInputOperand(0)->get();
            auto rhs = batchMatmul.getDpsInputOperand(1)->get();
            auto output = batchMatmul.getDpsInitOperand(0)->get();

            // Create indexing maps
            auto loc = batchMatmul.getLoc();
            auto lhsMap = mlir::AffineMap::get(
                /*dimCount=*/4, /*symbolCount=*/0,
                {mlir::getAffineDimExpr(0, ctx), mlir::getAffineDimExpr(2, ctx),
                 mlir::getAffineDimExpr(1, ctx)},
                ctx);
            auto rhsMap = mlir::AffineMap::get(
                /*dimCount=*/4, /*symbolCount=*/0,
                {mlir::getAffineDimExpr(0, ctx), mlir::getAffineDimExpr(1, ctx),
                 mlir::getAffineDimExpr(3, ctx)},
                ctx);
            auto outMap = mlir::AffineMap::get(
                /*dimCount=*/4, /*symbolCount=*/0,
                {mlir::getAffineDimExpr(0, ctx), mlir::getAffineDimExpr(2, ctx),
                 mlir::getAffineDimExpr(3, ctx)},
                ctx);

            llvm::SmallVector<mlir::AffineMap, 3> indexingMaps = {
                lhsMap, rhsMap, outMap};

            llvm::SmallVector<mlir::Attribute, 4> iteratorTypes = {
                mlir::linalg::IteratorTypeAttr::get(
                    builder.getContext(), mlir::utils::IteratorType::parallel),
                mlir::linalg::IteratorTypeAttr::get(
                    builder.getContext(), mlir::utils::IteratorType::reduction),
                mlir::linalg::IteratorTypeAttr::get(
                    builder.getContext(), mlir::utils::IteratorType::parallel),
                mlir::linalg::IteratorTypeAttr::get(
                    builder.getContext(), mlir::utils::IteratorType::parallel)};

            // Create generic op with reduction on k (dim 1)
            auto genericOp = builder.create<mlir::linalg::GenericOp>(
                loc,
                /*resultTypes=*/batchMatmul.getResultTypes(),
                /*inputs=*/ValueRange{lhs, rhs},
                /*outputs=*/ValueRange{output},
                builder.getAffineMapArrayAttr(indexingMaps),
                /*iteratorTypes=*/builder.getArrayAttr(iteratorTypes),
                /*docString=*/nullptr,
                /*libraryCall=*/nullptr,
                [&](OpBuilder& b, Location loc, ValueRange args) {
                  Value a = args[0];
                  Value bVal = args[1];
                  Value c = args[2];
                  auto mul = b.create<mlir::arith::MulFOp>(loc, a, bVal);
                  auto add = b.create<mlir::arith::AddFOp>(loc, c, mul);
                  b.create<mlir::linalg::YieldOp>(loc, add.getResult());
                });

            // Replace original batchMatmul
            batchMatmul.replaceAllUsesWith(genericOp.getResults());
            batchMatmul.erase();
          })
          .Default([](mlir::Operation* other) {
            // ignore or log
          });
    });
  }
};
}  // namespace
}  // namespace mlir::accelgen
