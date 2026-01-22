#ifndef KERNEL_SCHEDULE_PASS_H
#define KERNEL_SCHEDULE_PASS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include <any>
#include <map>
#include <numeric>
#include <unordered_set>
#include <vector>

namespace mlir {
namespace accelgen {

#define GEN_PASS_DECL
#include "accelgen/Passes/KernelSchedulePass.h.inc"

using ParameterVariant = std::variant<llvm::SmallVector<int64_t>, int64_t>;
using ParameterPointerVariant =
    std::variant<llvm::SmallVector<int64_t>*, int64_t*>;

class ParameterWrapper {
 public:
  ParameterWrapper();
  std::string parName;
  void* data;
};

class Int64Parameter : public ParameterWrapper {
 public:
  int64_t data;
  int64_t lowBound;
  int64_t upBound;
};

class TileParameter {
 public:
  std::unordered_map<mlir::Operation*, llvm::SmallVector<int64_t>> order;
  std::unordered_map<mlir::Operation*, llvm::SmallVector<ParameterWrapper*>>
      mapping;
  std::vector<ParameterWrapper*> parameterVec;
};

class GenericOpCluster {
 public:
  GenericOpCluster();
  GenericOpCluster(linalg::GenericOp* genericOpStart,
                   linalg::GenericOp* genericOpEnd);
  virtual ~GenericOpCluster() = default;

  bool isMember(mlir::Operation* opToCehck);

  void attachAttribute(mlir::MLIRContext* ctx);

  void evaluate();

  auto begin();
  auto end();

  void clearParameter();

  std::vector<mlir::Operation*>& getNodeSetTopOrder();
  std::unordered_map<
      mlir::Operation*,
      std::unordered_map<std::string, llvm::SmallVector<int64_t>>>&
  getParameter();
  // auto getMetric();

  TileParameter extractDimRelation();

 private:
  unsigned int nInD = 0;
  unsigned int nCycles = 0;
  unsigned int memAccess = 0;
  unsigned int cost = 0;
  std::unordered_set<mlir::Operation*> nodeSet;
  std::vector<mlir::Operation*> nodeSetTopOrder;
  std::unordered_map<
      mlir::Operation*,
      std::unordered_map<std::string, llvm::SmallVector<int64_t>>>
      parameter;
  // std::unordered_map<mlir::Operation*, std::unordered_map<std::string,
  // int64_t>>

  //     metric;

  void factorForwardHelp(mlir::Operation* op, int64_t factor);
};

// class GenericOpClusterBruteForce : public GenericOpCluster {
//  public:
//   using GenericOpCluster::GenericOpCluster;
//   unsigned int solveBestSchedule() override;
// };

class ArchConfig {
 public:
  size_t bandwidth;     // GB/s
  size_t sramCapacity;  // B
  size_t mulCnt;        // #
};

class EvaluationMetric {
 public:
  double_t throughput;
};

class PerfModel {
 public:
  EvaluationMetric evaluate(GenericOpCluster& cluster, ArchConfig& cfg);
};

class ParameterSolvingInterface {
 public:
  ParameterSolvingInterface() = default;
  //   ParameterSolvingInterface(GenericOpCluster& cluster);
  virtual ~ParameterSolvingInterface() = default;

  //   virtual unsigned int solve(
  //       std::vector<mlir::Operation*>* topoOrder,
  //       std::unordered_map<
  //           mlir::Operation*,
  //           std::unordered_map<std::string, llvm::SmallVector<int64_t>>>*
  //           parameter,
  //       CostModelInterface* evaluator) = 0;

  virtual unsigned int solve(GenericOpCluster& cluster, PerfModel& model,
                             ArchConfig& archCfg) = 0;

  //  private:
  //   GenericOpCluster& cluster;
};

class ParameterGenerator {
 public:
  void addVariable(std::vector<ParameterVariant> values) {
    std::vector<ParameterVariant> v;
    for (auto& x : values) v.emplace_back(x);
    candidates.push_back(std::move(v));
    indices.push_back(0);
  }

  bool hasNext() const { return _hasNext; }

  void next() {
    // auto result = current();
    for (auto x : indices) llvm::errs() << x << " ";
    llvm::errs() << "\n";
    advance();
    return;
  }

  size_t size() { return candidates.size(); }

  std::vector<std::vector<ParameterVariant>> candidates;
  std::vector<size_t> indices;

 private:
  bool _hasNext = true;

  std::vector<ParameterVariant> current() const {
    std::vector<ParameterVariant> res;
    for (size_t i = 0; i < candidates.size(); i++) {
      // llvm::errs() << i << " " << indices[i] << " " << candidates.size()
      //              << "\n";
      res.push_back(candidates[i][indices[i]]);
    }
    return res;
  }

  void advance() {
    for (int i = indices.size() - 1; i >= 0; i--) {
      indices[i]++;
      if (indices[i] < candidates[i].size()) return;
      indices[i] = 0;
      if (i == 0) {
        _hasNext = false;
        return;
      }
    }
  }
};

class BruteForceSolver : public ParameterSolvingInterface {
 public:
  unsigned int solve(GenericOpCluster& cluster, PerfModel& model,
                     ArchConfig& archCfg) override;

 private:
  void generateParSet(linalg::GenericOp genericOp);
  bool checkConstraint(GenericOpCluster& cluster, ArchConfig& archCfg);
  inline void assignParameter(ParameterVariant& v, ParameterPointerVariant& p);
};

class PruningSolver : public ParameterSolvingInterface {
 public:
  unsigned int solve(GenericOpCluster& cluster, PerfModel& model,
                     ArchConfig& archCfg) override;
};

class ScheduledGenericOpCluster {
 public:
  ScheduledGenericOpCluster() = default;
  ScheduledGenericOpCluster(llvm::StringRef solver);
  ~ScheduledGenericOpCluster();
  void insertGenericOp(linalg::GenericOp genericOp);
  void schedule(mlir::MLIRContext* ctx, PerfModel& model, ArchConfig& archCfg);

  auto begin();
  auto end();

 private:
  std::vector<linalg::GenericOp> genericOps;
  std::vector<GenericOpCluster*> clusters;
  std::vector<linalg::GenericOp> getTopSortedNodes();

  ParameterSolvingInterface* solver;
};

// class GenericOpClusterDAG {
//  public:
//   std::unordered_set<linalg::GenericOp> genericOps;
//   std::unordered_set<GenericOpCluster*> clusters;
//   std::unordered_map<GenericOpCluster*, std::vector<GenericOpCluster*>>
//       consumerList;
//   GenericOpClusterDAG();
//   ~GenericOpClusterDAG();
//   void insertCluster(GenericOpCluster* cluster);
//   void constructDAG();
//   void optimizeDAG();

//  private:
//   bool checkDependency(GenericOpCluster* cluster0, GenericOpCluster*
//   cluster1);

//   unsigned int pipeCost(std::vector<GenericOpCluster*> clusters, unsigned int
//   s,
//                         unsigned int e);
// };

// bool checkCluster(ScheduledGenericOp* genericOp0,
//                   ScheduledGenericOp* genericOp1);

// void mergeCluster(ScheduledGenericOp* genericOp0,
//                   ScheduledGenericOp* genericOp1);

}  // namespace accelgen

}  // namespace mlir

#endif  // KERNEL_SCHEDULE_PASS_H