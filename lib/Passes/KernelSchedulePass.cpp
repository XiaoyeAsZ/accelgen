

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

#include "accelgen/Passes/KernelSchedulePass.h"

namespace mlir::accelgen {
#define GEN_PASS_DEF_KERNELSCHEDULE
#include "accelgen/Passes/KernelSchedulePass.h.inc"

// ============================
// CLASS: GenericOpCluster
// ============================

GenericOpCluster::GenericOpCluster() {};
GenericOpCluster::GenericOpCluster(linalg::GenericOp *genericOpStart,
                                   linalg::GenericOp *genericOpEnd)
    : nodeSet(genericOpStart, genericOpEnd) {}

bool GenericOpCluster::isMember(mlir::Operation *opToCehck) {
  return nodeSet.find(opToCehck) != nodeSet.end();
}

void GenericOpCluster::attachAttribute(mlir::MLIRContext *ctx) {
  for (auto op : nodeSet) {
    for (auto attr : parameter[op]) {
      op->setAttr(attr.first, mlir::DenseI32ArrayAttr::get(ctx, attr.second));
      op->dump();
    }
  }
}

unsigned int GenericOpCluster::solveBestSchedule() {
  if (nodeSet.size() > 1)
    return 100;
  else {
    auto op = *nodeSet.begin();
    auto &attrMap = parameter[op];
    attrMap["tiling_size"] = llvm::SmallVector<int>{1, 128, 128, 64};
    attrMap["unroll_factor"] = llvm::SmallVector<int>{1, 64, 64, 1};
    return 0;
  }
}

auto GenericOpCluster::begin() { return nodeSet.begin(); }
auto GenericOpCluster::end() { return nodeSet.end(); }

// ============================
// END OF GenericOpCluster
// ============================

// ============================
// CLASS: ScheduledGenericOpCluster
// ============================
ScheduledGenericOpCluster::~ScheduledGenericOpCluster() {
  for (auto cluster : clusters)
    delete cluster;
}

std::vector<linalg::GenericOp> ScheduledGenericOpCluster::getTopSortedNodes() {
  std::vector<linalg::GenericOp> topOrderNodes;
  std::unordered_map<mlir::Operation *, unsigned int> inD;
  for (mlir::Operation *op : genericOps)
    inD.insert({op, 0});
  for (mlir::Operation *op : genericOps) {
    for (auto user : op->getUsers()) {
      if (inD.find(user) != inD.end())
        inD[user]++;
    }
  }
  std::queue<mlir::Operation *> nodesWithoutInD;
  for (auto [op, ind] : inD) {
    if (ind == 0) {
      nodesWithoutInD.push(op);
      if (!mlir::dyn_cast<linalg::GenericOp>(op))
        op->dump();
    }
  }
  while (!nodesWithoutInD.empty()) {
    auto op = nodesWithoutInD.front();
    nodesWithoutInD.pop();
    auto genericOp = mlir::dyn_cast<linalg::GenericOp>(op);
    // op->dump();
    assert(genericOp);
    topOrderNodes.push_back(genericOp);
    for (auto user : op->getUsers()) {
      // if (inD.find(user) != inD.end()) inD[user]--;
      if (inD.find(user) == inD.end())
        continue;
      inD[user]--;
      if (inD[user] == 0) {
        nodesWithoutInD.push(user);
        if (!mlir::dyn_cast<linalg::GenericOp>(user))
          user->dump();
      }
    }
  }
  return topOrderNodes;
}

void ScheduledGenericOpCluster::insertGenericOp(linalg::GenericOp genericOp) {
  genericOps.push_back(genericOp);
}

void ScheduledGenericOpCluster::schedule(mlir::MLIRContext *ctx) {
  std::vector<linalg::GenericOp> genericOpsTopOrder = getTopSortedNodes();

  assert(genericOpsTopOrder.size() >= 1);
  auto dpStatus =
      std::make_unique<unsigned int[]>(genericOpsTopOrder.size() + 1);
  auto cutIndex =
      std::make_unique<unsigned int[]>(genericOpsTopOrder.size() + 1);
  llvm::errs() << genericOpsTopOrder.size() + 1 << "\n";
  dpStatus[0] = 0;
  cutIndex[0] = 0;
  for (unsigned int i = 1; i <= genericOpsTopOrder.size(); i++) {
    // llvm::errs() << i << "\n";
    unsigned int minCost = 0xffffffff;
    unsigned int index = 0;
    for (unsigned j = 1; j <= i; j++) {

      unsigned int mergedCost =
          GenericOpCluster(genericOpsTopOrder.data() + j - 1,
                           genericOpsTopOrder.data() + i)
              .solveBestSchedule();

      if (dpStatus[j - 1] + mergedCost < minCost) {
        minCost = dpStatus[j] + mergedCost;
        index = j;
      }
    }
    dpStatus[i] = minCost;
    cutIndex[i] = index;
  }

  auto p = genericOps.size();
  while (1) {
    auto cluster =
        new GenericOpCluster(genericOpsTopOrder.data() + cutIndex[p] - 1,
                             genericOpsTopOrder.data() + p);
    cluster->solveBestSchedule();
    clusters.push_back(cluster);
    if (cutIndex[p] == 1)
      break;
    p = cutIndex[p] - 1;
  }

  for (auto c : clusters)
    c->attachAttribute(ctx);
}

auto ScheduledGenericOpCluster::begin() { return clusters.begin(); }
auto ScheduledGenericOpCluster::end() { return clusters.end(); }

// ============================
// END OF ScheduledGenericOpCluster
// ============================

namespace {

class KernelSchedule : public impl::KernelScheduleBase<KernelSchedule> {
public:
  using impl::KernelScheduleBase<KernelSchedule>::KernelScheduleBase;

  void runOnOperation() final {
    mlir::MLIRContext &ctx = getContext();
    mlir::func::FuncOp func = getOperation();
    mlir::ModuleOp module = func->getParentOfType<ModuleOp>();

    // GenericOpClusterDAG clusterDAG;
    ScheduledGenericOpCluster scheduledCluster;
    func.walk([&](mlir::linalg::GenericOp genericOp) {
      scheduledCluster.insertGenericOp(genericOp);
    });
    // clusterDAG.constructDAG();
    // clusterDAG.optimizeDAG();

    scheduledCluster.schedule(&ctx);

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
        for (auto constValue : constSet)
          clusterInput.insert(constValue);
        for (auto operand : genericOp.getOutputs()) {
          bool flag = true;
          for (auto use : operand.getUsers()) {
            if (use == genericOp)
              continue;
            if (cluster->isMember(use)) {
              flag = false;
              break;
            }
          }
          if (flag)
            clusterOutput.insert(operand);
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

      mlir::Block *entry = clusterFuncOp.addEntryBlock();
      builder.setInsertionPointToStart(entry);

      IRMapping mapper;
      for (auto [arg, input] : llvm::zip(entry->getArguments(), clusterInput))
        mapper.map(input, arg);

      for (Operation *op : *cluster) {
        builder.clone(*op, mapper);
      }

      llvm::SmallVector<Value> retVals;
      for (Value out : clusterOutput)
        retVals.push_back(mapper.lookup(out));

      builder.create<func::ReturnOp>(clusterFuncOp.getLoc(), retVals);
    }
    func.setPrivate();
    func.erase();
  }
};

} // namespace
} // namespace mlir::accelgen
