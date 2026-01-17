

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
#include "accelgen/Utils/AffineMapUtils.h"

namespace mlir::accelgen {
#define GEN_PASS_DEF_KERNELSCHEDULE
#include "accelgen/Passes/KernelSchedulePass.h.inc"

// ============================
// CLASS: OverallCost
// ============================

unsigned int OverallCost::evaluate(
    std::vector<mlir::Operation*>* topoOrder,
    std::unordered_map<
        mlir::Operation*,
        std::unordered_map<std::string, llvm::SmallVector<int64_t>>>*
        parameter) {
  std::unordered_map<mlir::Operation*, llvm::SmallVector<int64_t, 4>>
      currentCost;  // [0]:flops [1]:externel access [2]:sram access [3]:cycles
  for (auto kv : *parameter)
    currentCost.emplace(kv.first, llvm::SmallVector<int64_t, 4>({1, 1, 1, 1}));
  for (auto riter = topoOrder->rbegin(); riter != topoOrder->rend(); riter++) {
    auto genericOp = mlir::dyn_cast<linalg::GenericOp>(*riter);
    assert(genericOp);

    auto affineMapOutput = genericOp.getIndexingMapsArray().back();
    auto outputDims = std::unordered_set<int64_t>();
    for (auto expr : affineMapOutput.getResults()) {
      if (auto dimExpr = mlir::dyn_cast<mlir::AffineDimExpr>(expr))
        outputDims.insert(dimExpr.getPosition());
    }

    auto dimOrder = (*parameter)[*riter]["outer_order"];
    auto tilingSize = (*parameter)[*riter]["tiling_size"];
    auto loopBound = (*parameter)[*riter]["loop_bound"];
    auto iteratorTypes = genericOp.getIteratorTypesArray();

    int64_t firstReductionDimIndex = -1;
    for (auto item : llvm::enumerate(dimOrder)) {
      if (iteratorTypes[item.value()] == mlir::utils::IteratorType::reduction)
        firstReductionDimIndex = item.index();
    }

    auto analysisTileSize = std::unordered_map<int64_t, int64_t>();
    for (auto item : llvm::enumerate(dimOrder)) {
      if (outputDims.find(item.value()) != outputDims.end()) {
        if (item.index() < firstReductionDimIndex)
          analysisTileSize[item.value()] = tilingSize[item.value()];
        else
          analysisTileSize[item.value()] = loopBound[item.value()];
      }
    }
  }
}

// ============================
// END OF OverallCost
// ============================

// ============================
// CLASS: BruteForceSolver
// ============================
unsigned int BruteForceSolver::solve(
    std::vector<mlir::Operation*>* topoOrder,
    std::unordered_map<
        mlir::Operation*,
        std::unordered_map<std::string, llvm::SmallVector<int64_t>>>* parameter,
    CostModelInterface* evaluator) {
  return evaluator->evaluate(topoOrder, parameter);
}
// ============================
// END OF BruteForceSolver
// ============================

// ============================
// CLASS: GenericOpCluster
// ============================

GenericOpCluster::GenericOpCluster() {};
GenericOpCluster::GenericOpCluster(linalg::GenericOp* genericOpStart,
                                   linalg::GenericOp* genericOpEnd)
    : nodeSet(genericOpStart, genericOpEnd),
      nodeSetTopOrder(genericOpStart, genericOpEnd) {
  for (auto iter = genericOpStart; iter != genericOpEnd; iter++) {
    parameter[*iter]["loop_bound"] = genericOpStart->getStaticLoopRanges();
    parameter[*iter]["tiling_size"] =
        llvm::SmallVector<int64_t>((*iter).getNumLoops(), 1);
    parameter[*iter]["unroll_factor"] =
        llvm::SmallVector<int64_t>((*iter).getNumLoops(), 1);
    parameter[*iter]["outer_order"] =
        llvm::SmallVector<int64_t>((*iter).getNumLoops());
    std::iota(parameter[*iter]["outer_order"].begin(),
              parameter[*iter]["outer_order"].end(), 0);
    parameter[*iter]["inner_order"] =
        llvm::SmallVector<int64_t>((*iter).getNumLoops());
    std::iota(parameter[*iter]["inner_order"].begin(),
              parameter[*iter]["inner_order"].end(), 0);
  }
}

bool GenericOpCluster::isMember(mlir::Operation* opToCehck) {
  return nodeSet.find(opToCehck) != nodeSet.end();
}

unsigned int GenericOpCluster::solveBestSchedule(
    CostModelInterface* evaluator, ParameterSolvingInterface* solver) {
  return solver->solve(&nodeSetTopOrder, &parameter, evaluator);
}

void GenericOpCluster::attachAttribute(mlir::MLIRContext* ctx) {
  for (auto op : nodeSet) {
    for (auto namedAttr : parameter[op]) {
      llvm::SmallVector<mlir::Attribute> attrVec;
      for (auto attr : namedAttr.second)
        attrVec.push_back(mlir::IntegerAttr::get(
            mlir::IntegerType::get(ctx, 32, mlir::IntegerType::Unsigned),
            attr));
      op->setAttr(namedAttr.first, mlir::ArrayAttr::get(ctx, attrVec));
    }
  }
}

void GenericOpCluster::evaluate() {
  for (auto kv : parameter) {
    metric[kv.first]["factor"] = 1;
    metric[kv.first]["flops"] = 0;
    metric[kv.first]["external_access"] = 0;
    metric[kv.first]["sram_access"] = 0;
    metric[kv.first]["cycles"] = 0;
  }
  for (auto riter = nodeSetTopOrder.rbegin(); riter != nodeSetTopOrder.rend();
       riter++) {
    auto genericOp = mlir::dyn_cast<linalg::GenericOp>(*riter);
    assert(genericOp);

    auto indexingMaps = genericOp.getIndexingMapsArray();

    // auto dimOrder = (*parameter)[*riter]["outer_order"];
    auto tilingSize = parameter[*riter]["tiling_size"];
    auto loopBound = parameter[*riter]["loop_bound"];
    auto unrollFactor = parameter[*riter]["unroll_factor"];
    auto iteratorTypes = genericOp.getIteratorTypesArray();

    auto analysisTileSize = llvm::SmallVector<int64_t>(genericOp.getNumLoops());
    for (size_t i = 0; i < analysisTileSize.size(); i++) {
      switch (iteratorTypes[i]) {
        case mlir::utils::IteratorType::reduction:
          analysisTileSize[i] = loopBound[i];
          break;
        case mlir::utils::IteratorType::parallel:
          analysisTileSize[i] = tilingSize[i];
          break;
        default:
          llvm::errs() << "Unknown iterator types.\n";
          break;
      }
    }

    auto flops =
        std::accumulate(analysisTileSize.begin(), analysisTileSize.end(),
                        int64_t(1), std::multiplies<>());

    auto externalAccess = 0;
    for (auto [index, operand] : llvm::enumerate(genericOp.getInputs())) {
      auto producer = operand.getDefiningOp();
      assert(producer);
      // Input from
      if (!isMember(producer)) {
        auto accessDims = getAffineMapAccessDims(indexingMaps[index]);
        int64_t partial = 1;
        for (auto dim : accessDims) partial *= analysisTileSize[dim];
        externalAccess += partial;
      }
    }

    for (auto [index, result] : llvm::enumerate(genericOp.getResults())) {
      bool flag = true;
      for (auto user : result.getUsers()) {
        if (isMember(user)) {
          flag = false;
          break;
        }
      }
      if (flag) {
        auto accessDims = getAffineMapAccessDims(
            indexingMaps[genericOp.getInputs().size() + index]);
        int64_t partial = 1;
        for (auto dim : accessDims) partial *= analysisTileSize[dim];
        externalAccess += partial;
      }
    }

    int64_t cycles = 1;
    assert(analysisTileSize.size() == unrollFactor.size());
    for (size_t i = 0; i < analysisTileSize.size(); i++) {
      cycles *= (analysisTileSize[i] / unrollFactor[i]);
    }

    metric[*riter]["flops"] = flops;
    metric[*riter]["external_access"] = externalAccess;
    metric[*riter]["cycles"] = cycles;

    for (auto [index, operand] : llvm::enumerate(genericOp.getInputs())) {
      auto op = operand.getDefiningOp();
      assert(op);
      if (!isMember(op))
        continue;
      else {
        int64_t factor = 1;
        auto accessDims = getAffineMapAccessDims(indexingMaps[index]);
        for (size_t i = 0; i < accessDims.size(); i++)
          factor *= analysisTileSize[accessDims[i]] / tilingSize[accessDims[i]];
        if (metric[op]["factor"] == 1)
          metric[op]["factor"] = factor * metric[genericOp]["factor"];
      }
    }
  }
}

void GenericOpCluster::factorForwardHelp(mlir::Operation* op, int64_t factor) {
  if (factor == 1) return;
  metric[op]["flops"] *= factor;
  metric[op]["external_access"] *= factor;
  metric[op]["sram_access"] *= factor;
  metric[op]["cycles"] *= factor;
  auto genericOp = mlir::dyn_cast<linalg::GenericOp>(op);
  assert(genericOp);
  for (auto operand : genericOp.getInputs()) {
    auto op = operand.getDefiningOp();
    if (isMember(op)) factorForwardHelp(op, factor);
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

ScheduledGenericOpCluster::ScheduledGenericOpCluster(llvm::StringRef evaluator,
                                                     llvm::StringRef solver) {
  this->evaluator = llvm::StringSwitch<CostModelInterface*>(evaluator)
                        .Case("overall", new OverallCost())
                        .Default(nullptr);

  this->solver = llvm::StringSwitch<ParameterSolvingInterface*>(solver)
                     .Case("brute_force", new BruteForceSolver())
                     .Default(nullptr);
}

ScheduledGenericOpCluster::~ScheduledGenericOpCluster() {
  for (auto cluster : clusters) delete cluster;
  delete evaluator;
  delete solver;
}

std::vector<linalg::GenericOp> ScheduledGenericOpCluster::getTopSortedNodes() {
  std::vector<linalg::GenericOp> topOrderNodes;
  std::unordered_map<mlir::Operation*, unsigned int> inD;
  for (mlir::Operation* op : genericOps) inD.insert({op, 0});
  for (mlir::Operation* op : genericOps) {
    for (auto user : op->getUsers()) {
      if (inD.find(user) != inD.end()) inD[user]++;
    }
  }
  std::queue<mlir::Operation*> nodesWithoutInD;
  for (auto [op, ind] : inD) {
    if (ind == 0) {
      nodesWithoutInD.push(op);
      if (!mlir::dyn_cast<linalg::GenericOp>(op)) op->dump();
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
      if (inD.find(user) == inD.end()) continue;
      inD[user]--;
      if (inD[user] == 0) {
        nodesWithoutInD.push(user);
        if (!mlir::dyn_cast<linalg::GenericOp>(user)) user->dump();
      }
    }
  }
  return topOrderNodes;
}

void ScheduledGenericOpCluster::insertGenericOp(linalg::GenericOp genericOp) {
  genericOps.push_back(genericOp);
}

void ScheduledGenericOpCluster::schedule(mlir::MLIRContext* ctx) {
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
              .solveBestSchedule(evaluator, solver);

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
    cluster->solveBestSchedule(evaluator, solver);
    clusters.push_back(cluster);
    if (cutIndex[p] == 1) break;
    p = cutIndex[p] - 1;
  }

  for (auto c : clusters) c->attachAttribute(ctx);
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
    mlir::MLIRContext& ctx = getContext();
    mlir::func::FuncOp func = getOperation();
    mlir::ModuleOp module = func->getParentOfType<ModuleOp>();

    // GenericOpClusterDAG clusterDAG;
    ScheduledGenericOpCluster scheduledCluster("overall", "brute_force");
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
