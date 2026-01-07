

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include <assert.h>
#include <queue>
#include <span>
#include <vector>
#include <span>

#include "accelgen/Passes/KernelSchedulePass.h"

namespace mlir::accelgen {
#define GEN_PASS_DEF_KERNELSCHEDULE
#include "accelgen/Passes/KernelSchedulePass.h.inc"

// bool checkCluster(ScheduledGenericOp* genericOp0,
//                   ScheduledGenericOp* genericOp1) {
//   assert(genericOp0);
//   assert(genericOp1);
//   while (genericOp0->clusterRoot->clusterRoot != genericOp0->clusterRoot)
//     genericOp0->clusterRoot = genericOp0->clusterRoot->clusterRoot;
//   while (genericOp1->clusterRoot->clusterRoot != genericOp1->clusterRoot)
//     genericOp1->clusterRoot = genericOp1->clusterRoot->clusterRoot;
//   return genericOp0->clusterRoot == genericOp1->clusterRoot;
// }

// inline void mergeCluster(ScheduledGenericOp* genericOp0,
//                          ScheduledGenericOp* genericOp1) {
//   genericOp0->clusterRoot->clusterRoot = genericOp1->clusterRoot;
//   return;
// }

GenericOpCluster::GenericOpCluster(mlir::linalg::GenericOp genericOp) {
  nodeSet.insert(genericOp);
  // auto resultValue = genericOp->getResults();
  // assert(resultValue.size() == 1 && "More than one reulst of generic op");
}

/***** GenericOpClusterDAG *****/

GenericOpClusterDAG::GenericOpClusterDAG() {};

GenericOpClusterDAG::~GenericOpClusterDAG() {
  for (auto ptr : clusters) delete ptr;
}

void GenericOpClusterDAG::insertCluster(GenericOpCluster* cluster) {
  clusters.insert(cluster);
}

void GenericOpClusterDAG::constructDAG() {
  for (auto producer : clusters) {
    for (auto consumer : clusters) {
      if (checkDependency(producer, consumer)) {
        producer->consumerSet.insert(consumer);
        consumer->nInD += 1;
      }
    }
  }
}

bool GenericOpClusterDAG::checkDependency(GenericOpCluster* producerCluster,
                                          GenericOpCluster* consumerCluster) {
  // for (mlir::linalg::GenericOp* producerOp : producerCluster->nodeSet) {
  //   for (mlir::linalg::GenericOp* consumerOp : consumerCluster->nodeSet) {
  //     auto results = producerOp->getResults();
  //     assert(results.size() == 1);
  //     auto result = results[0];
  //     for (mlir::Operation* userOp : result.getUsers()) {
  //       auto consumerGenericOp = llvm::dyn_cast<linalg::GenericOp*>(userOp);
  //       assert(consumerGenericOp);
  //       if (consumerGenericOp == consumerOp) return true;
  //     }
  //   }
  // }
  return false;
}

void GenericOpClusterDAG::optimizeDAG() {
  // Top sort
  std::vector<GenericOpCluster*> clusterTopOrder;
  std::queue<GenericOpCluster*> clusterWithoutInd;
  for (auto cluster : clusters)
    if (cluster->nInD == 0) {
      clusterWithoutInd.push(cluster);
      clusterTopOrder.push_back(cluster);
    }

  while (!clusterWithoutInd.empty()) {
    auto cluster = clusterWithoutInd.front();
    clusterWithoutInd.pop();
    for (auto consumer : consumerList[cluster]) {
      consumer->nInD--;
      if (consumer->nInD == 0) {
        clusterWithoutInd.push(cluster);
        clusterTopOrder.push_back(cluster);
      }
    }
  }

  // DP
  assert(clusterTopOrder.size() >= 1);
  unsigned int* cost = new unsigned int[clusterTopOrder.size()];
  cost[0] = clusterTopOrder[0]->cost;
  for (unsigned int i = 1; i < clusterTopOrder.size(); i++) {
    unsigned int minCost = 0x7ffffff;
    unsigned int cutIndex = 0;
    for (unsigned j = 0; j < i; j++) {
      if (cost[j] + pipeCost(clusterTopOrder, j + 1, i) < minCost) {
        minCost = cost[j] + pipeCost(clusterTopOrder, j + 1, i);
        cutIndex = j;
      }
    }
  }
  delete cost;
}
/***************/

namespace {

class KernelSchedule : public impl::KernelScheduleBase<KernelSchedule> {
 public:
  using impl::KernelScheduleBase<KernelSchedule>::KernelScheduleBase;

  void runOnOperation() final {
    mlir::MLIRContext& ctx = getContext();
    mlir::func::FuncOp func = getOperation();
    mlir::ModuleOp module = func->getParentOfType<ModuleOp>();

    GenericOpClusterDAG clusterDAG;
    func.walk([&](mlir::linalg::GenericOp genericOp) {
      clusterDAG.insertCluster(new GenericOpCluster(genericOp));
    });
    clusterDAG.constructDAG();
    clusterDAG.optimizeDAG();

    mlir::OpBuilder builder(module);

    unsigned int indexCluster = 0;
    llvm::DenseSet<mlir::Value> constSet;
    func.walk(
        [&](mlir::arith::ConstantOp constOp) { constSet.insert(constOp); });
    for (auto cluster : clusterDAG.clusters) {
      llvm::DenseSet<mlir::Value> clusterInput, clusterOutput;
      for (auto node : cluster->nodeSet) {
        auto genericOp = llvm::dyn_cast<mlir::linalg::GenericOp>(node);
        assert(genericOp);
        for (auto operand : genericOp.getOperands()) {
          // llvm::errs() << operand << "\n";
          if (cluster->nodeSet.find(operand.getDefiningOp()) ==
              cluster->nodeSet.end())
            clusterInput.insert(operand);
        }
        for (auto constValue : constSet) clusterInput.insert(constValue);
        for (auto operand : genericOp.getOutputs()) {
          bool flag = true;
          for (auto use : operand.getUsers()) {
            if (use == genericOp) continue;
            if (cluster->nodeSet.find(use) != cluster->nodeSet.end()) {
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

      for (Operation* op : cluster->nodeSet) {
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
