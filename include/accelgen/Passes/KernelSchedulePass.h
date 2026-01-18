#ifndef KERNEL_SCHEDULE_PASS_H
#define KERNEL_SCHEDULE_PASS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include <numeric>
#include <unordered_set>
#include <vector>
#include <map>
#include <any>

namespace mlir {
namespace accelgen {

#define GEN_PASS_DECL
#include "accelgen/Passes/KernelSchedulePass.h.inc"

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

  auto getParameter();

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
  std::unordered_map<mlir::Operation*, std::unordered_map<std::string, int64_t>>
      metric;

  void factorForwardHelp(mlir::Operation* op, int64_t factor);
};

// class GenericOpClusterBruteForce : public GenericOpCluster {
//  public:
//   using GenericOpCluster::GenericOpCluster;
//   unsigned int solveBestSchedule() override;
// };

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

  virtual unsigned int solve(GenericOpCluster* cluster) = 0;

  //  private:
  //   GenericOpCluster& cluster;
};

template <typename T>
class ParameterGenerator {
 public:
  void addVariable(std::vector<T> values) {
    std::vector<std::any> v;
    for (auto& x : values) v.emplace_back(x);
    candidates_.push_back(std::move(v));
    indices_.push_back(0);
  }

  bool hasNext() const { return hasNext_; }

  std::vector<std::any> next() {
    auto result = current();
    advance();
    return result;
  }

 private:
  std::vector<std::vector<std::any>> candidates_;
  std::vector<size_t> indices_;
  bool hasNext_ = true;

  std::vector<std::any> current() const {
    std::vector<std::any> res;
    for (size_t i = 0; i < candidates_.size(); i++) {
      res.push_back(candidates_[i][indices_[i]]);
    }
    return res;
  }

  void advance() {
    for (int i = indices_.size() - 1; i >= 0; i--) {
      indices_[i]++;
      if (indices_[i] < candidates_[i].size()) return;
      indices_[i] = 0;
      if (i == 0) {
        hasNext_ = false;
        return;
      }
    }
  }
};

class BruteForceSolver : public ParameterSolvingInterface {
 public:
  unsigned int solve(GenericOpCluster* cluster) override;

 private:
  void generateParSet(linalg::GenericOp genericOp);
  bool checkConstraint();
};

class ScheduledGenericOpCluster {
 public:
  ScheduledGenericOpCluster() = default;
  ScheduledGenericOpCluster(llvm::StringRef solver);
  ~ScheduledGenericOpCluster();
  void insertGenericOp(linalg::GenericOp genericOp);
  void schedule(mlir::MLIRContext* ctx);

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