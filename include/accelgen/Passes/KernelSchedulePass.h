#ifndef KERNEL_SCHEDULE_PASS_H
#define KERNEL_SCHEDULE_PASS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include <numeric>
#include <unordered_set>
#include <vector>

namespace mlir {
namespace accelgen {

#define GEN_PASS_DECL
#include "accelgen/Passes/KernelSchedulePass.h.inc"

class GenericOpCluster {
public:
  // std::unordered_set<GenericOpCluster *> consumerSet;
  // std::unordered_map<
  //     mlir::Operation*,
  //     std::unordered_map<llvm::StringRef, llvm::SmallVector<unsigned int>>>
  //     parameter;
  llvm::DenseMap<
      mlir::Operation *,
      llvm::DenseMap<llvm::StringRef, llvm::SmallVector<int>>>
      parameter;

  GenericOpCluster();
  GenericOpCluster(linalg::GenericOp *genericOpStart,
                   linalg::GenericOp *genericOpEnd);

  bool isMember(mlir::Operation *opToCehck);

  unsigned int solveBestSchedule();

  void attachAttribute(mlir::MLIRContext* ctx);

  auto begin();
  auto end();

private:
  unsigned int nInD = 0;
  unsigned int nCycles = 0;
  unsigned int memAccess = 0;
  unsigned int cost = 0;
  std::unordered_set<mlir::Operation *> nodeSet;
};

class ScheduledGenericOpCluster {
public:
  ~ScheduledGenericOpCluster();
  void insertGenericOp(linalg::GenericOp genericOp);
  void schedule(mlir::MLIRContext* ctx);

  auto begin();
  auto end();

private:
  std::vector<linalg::GenericOp> genericOps;
  std::vector<GenericOpCluster *> clusters;
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

} // namespace accelgen

} // namespace mlir

#endif // KERNEL_SCHEDULE_PASS_H