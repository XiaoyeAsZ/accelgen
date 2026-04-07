

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
#include "mlir/Dialect/Math/IR/Math.h"

#include "accelgen/Utils/AffineMapUtils.h"
#include "accelgen/Utils/DebugUtils.h"
#include "accelgen/Utils/OperationUtils.h"
#include "accelgen/Utils/ArchUtils.h"

namespace mlir::accelgen {
#define GEN_PASS_DEF_MODELPERFORMANCE
#include "accelgen/Passes/AccelgenPasses.h.inc"

namespace {

class ModelPerformance : public impl::ModelPerformanceBase<ModelPerformance> {
 public:
  using impl::ModelPerformanceBase<ModelPerformance>::ModelPerformanceBase;

  void runOnOperation() final {
    mlir::MLIRContext& ctx = getContext();

    ArchResource archCfg;
    archCfg.load(configPath);

    double_t totalEnergy = 0;
    double_t totalLatency = 0;
    double_t totalFlops = 0;
    double_t totalDramAcc = 0;

    auto module = getOperation();
    module.walk([&](func::FuncOp func) {
      auto sramAccAttr = mlir::dyn_cast<FloatAttr>(func->getAttr("sramAccess"));
      auto dramAccAttr =
          mlir::dyn_cast<FloatAttr>(func->getAttr("externalAccess"));
      auto flopsAttr = mlir::dyn_cast<FloatAttr>(func->getAttr("flops"));
      auto cyclesAttr = mlir::dyn_cast<FloatAttr>(func->getAttr("cycles"));
      assert(sramAccAttr && dramAccAttr && flopsAttr && cyclesAttr);

      auto sramAcc = sramAccAttr.getValueAsDouble();
      auto dramAcc = dramAccAttr.getValueAsDouble();
      auto flops = flopsAttr.getValueAsDouble();
      auto cycles = cyclesAttr.getValueAsDouble();

      totalLatency += (cycles * archCfg.cycle * 1e-6);
      totalFlops += flops;
      totalDramAcc += dramAcc;

      auto sramEnergy = sramAcc * archCfg.sramEnergy;
      auto dramEnergy = dramAcc * archCfg.dramEnergy;
      double_t peEnergy = 0;
      func.walk([&](linalg::GenericOp generic) {
        int64_t totalOps = 1;
        auto bound = generic.getStaticLoopRanges();
        assert(bound.size() == generic.getNumLoops());
        for (auto b : bound) totalOps *= b;
        for (auto& arith : generic.getRegion().front().getOperations()) {
          if (mlir::isa<arith::ArithDialect>(arith.getDialect()) ||
              mlir::isa<math::MathDialect>(arith.getDialect())) {
            std::string resourceName = toString(&arith);
            for (auto operand : arith.getOperands())
              resourceName = resourceName + "_" + toString(operand.getType());
            assert(arith.getResults().size() == 1);
            for (auto result : arith.getResults())
              resourceName = resourceName + "_" + toString(result.getType());

            assert(archCfg.computeEnergy.contains(resourceName));
            peEnergy += totalOps * archCfg.computeEnergy[resourceName];
          }
        }
      });

      totalEnergy += (sramEnergy + dramEnergy + peEnergy);
    });

    auto throughput = totalFlops / totalLatency;
    auto energyEfficiency = totalFlops / totalEnergy;
    auto dramAccess = totalDramAcc;

    llvm::outs() << "===== Model Performance =====\n";
    llvm::outs() << "Latency: " << totalLatency << "\n";
    llvm::outs() << "Energy: " << totalEnergy << "\n";
    llvm::outs() << "Flops: " << totalFlops << "\n";
    llvm::outs() << "Throughput: " << throughput << "\n";
    llvm::outs() << "Energy Efficiency: " << energyEfficiency << "\n";
    llvm::outs() << "DRAM Access: " << dramAccess << "\n";
    llvm::outs() << "=============================\n";
  }
};

}  // namespace
}  // namespace mlir::accelgen
