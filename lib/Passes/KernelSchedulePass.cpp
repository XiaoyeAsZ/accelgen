

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
#include "accelgen/Utils/DebugUtils.h"
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
    mlir::MLIRContext &ctx = getContext();
    mlir::func::FuncOp func = getOperation();
    mlir::ModuleOp module = func->getParentOfType<ModuleOp>();

    PerfModel model = PerfModel();
    ArchConfig archCfg;

    archCfg.load(configPath);

    // archCfg.bandwidth = 128;
    // archCfg.sramCapacity = 1024 * 1024;
    // archCfg.nSramBank = 32;
    // archCfg.computeResource["mulf_bf16_bf16_bf16"] = 64 * 64 + 64;
    // archCfg.computeResource["mulf_fp32_fp32_fp32"] = 64 * 64 + 64;
    // archCfg.computeResource["addf_bf16_bf16_bf16"] = 64 * 64 + 64;
    // archCfg.computeResource["negf_bf16_bf16"] = 64;
    // archCfg.computeResource["addf_fp32_fp32_fp32"] = 64;
    // archCfg.computeResource["subf_fp32_fp32_fp32"] = 64;
    // archCfg.computeResource["divf_fp32_fp32_fp32"] = 64;
    // archCfg.computeResource["divf_bf16_bf16_bf16"] = 64;
    // archCfg.computeResource["truncf_fp32_bf16"] = 64;
    // archCfg.computeResource["truncf_fp64_bf16"] = 64;
    // archCfg.computeResource["truncf_fp64_fp32"] = 64;
    // archCfg.computeResource["extf_bf16_fp32"] = 64;
    // archCfg.computeResource["maximumf_fp32_fp32_fp32"] = 64;
    // archCfg.computeResource["transpose_bf16_bf16"] = 16 * 16;

    // GenericOpClusterDAG clusterDAG;
    ScheduledGenericOpCluster scheduledCluster("pruning_brute_force");
    // func.walk([&](mlir::linalg::GenericOp genericOp) {
    //   scheduledCluster.insertGenericOp(genericOp);
    // });
    for (auto &block : func.getBlocks()) {
      for (auto &op : block.getOperations())
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

    scheduledCluster.schedule(&ctx, model, archCfg, maxOperation);

    mlir::OpBuilder builder(module);

    unsigned int indexCluster = 0;
    llvm::DenseSet<mlir::Value> constSet;
    func.walk(
        [&](mlir::arith::ConstantOp constOp) { constSet.insert(constOp); });

    for (auto cluster : scheduledCluster) {
      // Collect input and output value of the cluster
      llvm::SmallVector<mlir::Value> clusterInput, clusterOutput;
      for (auto node : cluster->getNodeSetTopOrder()) {
        auto genericOp = llvm::dyn_cast<mlir::linalg::GenericOp>(node);
        assert(genericOp);
        for (auto operand : genericOp.getInputs()) {
          // llvm::errs() << operand << "\n";
          llvm::SmallVector<linalg::GenericOp> previousGenericOps;
          cluster->getPreviousGeneric(operand, previousGenericOps);
          if (previousGenericOps.empty())
            clusterInput.push_back(operand);
        }
        for (auto constValue : constSet)
          clusterInput.push_back(constValue);
        for (auto operand : genericOp.getResults()) {
          llvm::SmallVector<linalg::GenericOp> latterGenericOps;
          cluster->getLatterGeneric(operand, latterGenericOps);
          if (latterGenericOps.empty())
            clusterOutput.push_back(operand);
        }
      }

      auto opsWithTensorOp = cluster->constructClusterWithTensorOp();
      for (auto tensorOp : opsWithTensorOp) {
        if (!mlir::isa<tensor::TensorDialect>(tensorOp->getDialect()))
          continue;
        for (auto ins : tensorOp->getOperands()) {
          if (ins.getDefiningOp() == nullptr &&
              (!llvm::is_contained(clusterInput, ins)))
            clusterInput.push_back(ins);
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
      clusterFuncOp->setAttr(
          "cycles",
          mlir::FloatAttr::get(Float64Type::get(&ctx), cluster->metric.cycles));
      clusterFuncOp->setAttr(
          "externalAccess",
          mlir::FloatAttr::get(Float64Type::get(&ctx),
                               cluster->metric.externalAccess));
      clusterFuncOp->setAttr("sramAccess",
                             mlir::FloatAttr::get(Float64Type::get(&ctx),
                                                  cluster->metric.sramAccess));
      clusterFuncOp->setAttr(
          "flops",
          mlir::FloatAttr::get(Float64Type::get(&ctx), cluster->metric.flops));

      // clusterFuncOp.setPrivate();

      mlir::Block *entry = clusterFuncOp.addEntryBlock();
      builder.setInsertionPointToStart(entry);

      IRMapping mapper;
      // Add mapping for boundray inputs
      for (auto [arg, input] : llvm::zip(entry->getArguments(), clusterInput))
        mapper.map(input, arg);
      // Add mapping for generic outputs
      // ECHO(cluster->getNodeSetTopOrder().size(), "\n")
      for (auto op : cluster->getNodeSetTopOrder()) {
        auto genericOp = llvm::dyn_cast<mlir::linalg::GenericOp>(op);
        assert(genericOp);
        for (auto operand : genericOp.getOutputs()) {
          auto shaped = mlir::dyn_cast<ShapedType>(operand.getType());
          auto emptyOp = builder.create<tensor::EmptyOp>(
              genericOp.getLoc(), shaped.getShape(),
              getElementTypeOrSelf(operand));
          mapper.map(operand, emptyOp.getResult());
        }
      }

      auto opClusterWithTensorOp =
          getTopoOrder(cluster->constructClusterWithTensorOp());

      // ECHO("check clone cluster", "\n")
      for (Operation *op : opClusterWithTensorOp) {
        ECHO("clone", "\n")
        op->dump();
        builder.clone(*op, mapper);
      }

      llvm::SmallVector<Value> retVals;
      for (Value out : clusterOutput)
        retVals.push_back(mapper.lookup(out));

      builder.create<func::ReturnOp>(clusterFuncOp.getLoc(), retVals);
    }

    module.dump();

    func.setPrivate();
    func.erase();
  }
};

} // namespace
} // namespace mlir::accelgen
