

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
#include <span>
#include <variant>
#include <vector>

#include "accelgen/DSE/KernelScheduleDSE.h"
#include "accelgen/Passes/AccelgenPasses.h"
// #include "accelgen/Passes/KernelSchedulePass.h"
#include "accelgen/Utils/AffineMapUtils.h"
#include "accelgen/Utils/OperationUtils.h"

namespace mlir::accelgen {
#define GEN_PASS_DEF_KERNELSCHEDULE
// #include "accelgen/Passes/KernelSchedulePass.h.inc"
#include "accelgen/Passes/AccelgenPasses.h.inc"

namespace {

class KernelSchedule : public impl::KernelScheduleBase<KernelSchedule> {
 public:
  using impl::KernelScheduleBase<KernelSchedule>::KernelScheduleBase;

  void runOnOperation() final {
    mlir::MLIRContext& ctx = getContext();
    mlir::func::FuncOp func = getOperation();
    mlir::ModuleOp module = func->getParentOfType<ModuleOp>();

    PerfModel model = PerfModel();
    ArchConfig archCfg;
    archCfg.bandwidth = 128;
    archCfg.sramCapacity = 128 * 1024;

    // GenericOpClusterDAG clusterDAG;
    ScheduledGenericOpCluster scheduledCluster("pruning_brute_force");
    // func.walk([&](mlir::linalg::GenericOp genericOp) {
    //   scheduledCluster.insertGenericOp(genericOp);
    // });
    for (auto& block : func.getBlocks()) {
      for (auto& op : block.getOperations())
        if (mlir::dyn_cast<linalg::GenericOp>(op) ||
            mlir::dyn_cast<tensor::CollapseShapeOp>(op) ||
            mlir::dyn_cast<tensor::ExpandShapeOp>(op) ||
            mlir::dyn_cast<tensor::ConcatOp>(op) ||
            mlir::dyn_cast<tensor::ExtractSliceOp>(op) ||
            mlir::dyn_cast<tensor::EmptyOp>(op))
          scheduledCluster.insertOp(&op);
        else if (mlir::dyn_cast<func::ReturnOp>(op) ||
                 mlir::dyn_cast<arith::ConstantOp>(op)) {
          continue;
        } else {
          op.dump();
          assert(0);
        }
    }
    // clusterDAG.constructDAG();
    // clusterDAG.optimizeDAG();

    scheduledCluster.schedule(&ctx, model, archCfg);

    mlir::OpBuilder builder(module);

    unsigned int indexCluster = 0;
    llvm::DenseSet<mlir::Value> constSet;
    func.walk(
        [&](mlir::arith::ConstantOp constOp) { constSet.insert(constOp); });
    for (auto cluster : scheduledCluster) {
      llvm::DenseSet<mlir::Value> clusterInput, clusterOutput;
      for (auto node : *cluster) {
        auto genericOp = llvm::dyn_cast<mlir::linalg::GenericOp>(node);
        assert(genericOp);
        for (auto operand : genericOp.getOperands()) {
          // llvm::errs() << operand << "\n";
          if (!cluster->isMember(operand.getDefiningOp()))
            clusterInput.insert(operand);
        }
        for (auto constValue : constSet) clusterInput.insert(constValue);
        for (auto operand : genericOp.getOutputs()) {
          bool flag = true;
          for (auto use : operand.getUsers()) {
            if (use == genericOp) continue;
            if (cluster->isMember(use)) {
              flag = false;
              break;
            }
          }
          if (flag) clusterOutput.insert(operand);
        }
      }

      builder.setInsertionPointToEnd(module.getBody());
      auto funcType = builder.getFunctionType(
          llvm::to_vector(llvm::map_range(clusterInput,
                                          [](Value v) { return v.getType(); })),
          llvm::to_vector(llvm::map_range(
              clusterOutput, [](Value v) { return v.getType(); })));

      auto clusterFuncOp = builder.create<mlir::func::FuncOp>(
          func.getLoc(),
          std::string("Cluster_") + std::to_string(indexCluster++), funcType);

      // clusterFuncOp.setPrivate();

      mlir::Block* entry = clusterFuncOp.addEntryBlock();
      builder.setInsertionPointToStart(entry);

      IRMapping mapper;
      for (auto [arg, input] : llvm::zip(entry->getArguments(), clusterInput))
        mapper.map(input, arg);

      for (Operation* op : *cluster) {
        builder.clone(*op, mapper);
      }

      llvm::SmallVector<Value> retVals;
      for (Value out : clusterOutput) retVals.push_back(mapper.lookup(out));

      builder.create<func::ReturnOp>(clusterFuncOp.getLoc(), retVals);
    }
    func.setPrivate();
    func.erase();
  }
};

}  // namespace
}  // namespace mlir::accelgen
