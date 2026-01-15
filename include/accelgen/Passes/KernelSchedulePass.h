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

  virtual unsigned int solveBestSchedule() = 0;

  void attachAttribute(mlir::MLIRContext* ctx);

  auto begin();
  auto end();

 protected:
  unsigned int nInD = 0;
  unsigned int nCycles = 0;
  unsigned int memAccess = 0;
  unsigned int cost = 0;
  std::unordered_set<mlir::Operation*> nodeSet;
  std::vector<mlir::Operation*> nodeSetTopOrder;
  std::unordered_map<
      mlir::Operation*,
      std::unordered_map<std::string, llvm::SmallVector<unsigned int>>>
      parameter;
};

class GenericOpClusterBruteForce : public GenericOpCluster {
 public:
  using GenericOpCluster::GenericOpCluster;
  unsigned int solveBestSchedule() override;
};

template <typename _Cluster>
class ScheduledGenericOpCluster {
 public:
  ~ScheduledGenericOpCluster();
  void insertGenericOp(linalg::GenericOp genericOp);
  void schedule(mlir::MLIRContext* ctx);

  auto begin();
  auto end();

 private:
  std::vector<linalg::GenericOp> genericOps;
  std::vector<_Cluster*> clusters;
  std::vector<linalg::GenericOp> getTopSortedNodes();
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