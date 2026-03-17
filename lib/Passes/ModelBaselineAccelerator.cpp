

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
#define GEN_PASS_DEF_MODELBASELINEACCELERATOR
#include "accelgen/Passes/AccelgenPasses.h.inc"

namespace {

class ModelBaselineAccelerator
    : public impl::ModelBaselineAcceleratorBase<ModelBaselineAccelerator> {
 public:
  using impl::ModelBaselineAcceleratorBase<
      ModelBaselineAccelerator>::ModelBaselineAcceleratorBase;

  void runOnOperation() final {
    mlir::MLIRContext& ctx = getContext();
    mlir::RewritePatternSet patterns(&ctx);

    // Some important parameters :
    // 1. acceleratorName
    // 2. configPath

    // Step : construct arch.yaml according parameter

    auto module = getOperation();
    module.walk([&](mlir::Operation* op) {
      llvm::TypeSwitch<mlir::Operation*>(op)
          .Case<linalg::BatchMatmulOp>(
              [&](linalg::BatchMatmulOp batchMatmulOp) {
                // Step : Check dimension of batch matmul
                // Step : Construct problem of timeloop according to dimension
                // Step : Call timeloop
                // Step : Read results (Maybe you need some data structure to
                // save results)
              })
          // You need add other op here. Typically we assume all generic ops are
          // element-wise
          .Default([](mlir::Operation* unRecogOp) {});
    });
  }
};

}  // namespace
}  // namespace mlir::accelgen
