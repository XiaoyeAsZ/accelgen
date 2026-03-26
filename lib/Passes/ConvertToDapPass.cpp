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

#include "accelgen/Dialect/Dap/DapOps.h"
#include "accelgen/Utils/OperationUtils.h"
#include "accelgen/Passes/AccelgenPasses.h"
#include "accelgen/Utils/GenericGraph.h"
#include "accelgen/Utils/DebugUtils.h"

namespace mlir::accelgen {
#define GEN_PASS_DEF_CONVERTTODAP
// #include "accelgen/Passes/MarkGenericPass.h.inc"
#include "accelgen/Passes/AccelgenPasses.h.inc"

namespace {

class ConvertToDap : public impl::ConvertToDapBase<ConvertToDap> {
 public:
  using impl::ConvertToDapBase<ConvertToDap>::ConvertToDapBase;

  void runOnOperation() final {
    mlir::MLIRContext* ctx = &getContext();
    auto module = getOperation();
    std::vector<mlir::func::FuncOp> funcs;
    module.walk([&](mlir::func::FuncOp funcOp) { funcs.push_back(funcOp); });

    mlir::OpBuilder builder(ctx);

    for (auto [indexFuncOp, itemFuncOp] : llvm::enumerate(funcs)) {
      // Create func op
      builder.setInsertionPoint(itemFuncOp);
      auto dapFunc = builder.create<func::FuncOp>(
          itemFuncOp.getLoc(), itemFuncOp.getName().str() + "_dap",
          FunctionType::get(ctx, {}, {}));
      auto entryBlock = dapFunc.addEntryBlock();
      builder.setInsertionPointToStart(entryBlock);
      builder.create<func::ReturnOp>(UnknownLoc::get(ctx));
      builder.setInsertionPointToStart(entryBlock);

      // Build dap graph
      auto graph = GenericGraph(itemFuncOp);

      llvm::DenseMap<mlir::OpOperand*, dap::SramOp> valueSramMap;

      // Build SRAM for all values

      // For concat op, sram is shared for 2- input and 1- output, special
      // process
      for (auto tensorOp : graph.getTensorOps()) {
        if (auto concatOp = mlir::dyn_cast<tensor::ConcatOp>(tensorOp)) {
          auto sramOp =
              builder.create<dap::SramOp>(concatOp.getLoc(), 512, 128, 1, 2);
          for (auto& opOperand : concatOp->getOpOperands()) {
            auto prevOpOperand = graph.getPreviousOpOperand(opOperand);
            assert(!valueSramMap.contains(prevOpOperand));
            valueSramMap[prevOpOperand] = sramOp;
          }
          assert(concatOp.getResult().getNumUses() == 1);
          for (auto& use : concatOp->getUses()) {
            assert(mlir::dyn_cast<linalg::GenericOp>(use.getOwner()));
            valueSramMap[&use] = sramOp;
          }
        }
      }

      // For other op, standard process
      for (auto genericOp : graph.getGenericOpsInTopoOrder()) {
        // For inputs, check whether is associated with an sram, yes -> direct
        // mapping, no -> create sram (block arguments)
        for (auto [indexIns, itemIns] :
             llvm::enumerate(genericOp.getInputs())) {
          auto& opOperand = genericOp->getOpOperand(indexIns);
          auto prevOpOperand = graph.getPreviousOpOperand(opOperand);
          // Sram has been created by concat op
          if (valueSramMap.contains(&opOperand)) {
            continue;
          }
          // Sram is not created, but has producer
          else if (valueSramMap.contains(prevOpOperand)) {
            valueSramMap[&opOperand] = valueSramMap[prevOpOperand];
          } else {
            auto sramOp = builder.create<dap::SramOp>(opOperand.get().getLoc(),
                                                      512, 128, 1, 2);
            valueSramMap[&opOperand] = sramOp;
          }
        }
        // For results, build sram for each use, except for those have been
        // mapped to a sram
        for (auto& use : genericOp->getUses()) {
          // Sram has been created by concat op
          if (valueSramMap.contains(&use)) {
            continue;
          } else {
            auto sramOp =
                builder.create<dap::SramOp>(use.get().getLoc(), 512, 128, 1, 2);
            valueSramMap[&use] = sramOp;
          }
        }
      }

      // Build routing node and arith node
      for (auto genericOp : graph.getGenericOpsInTopoOrder()) {
      }
    }

    for (auto funcOp : funcs) funcOp.erase();
  }
};

}  // namespace
}  // namespace mlir::accelgen
