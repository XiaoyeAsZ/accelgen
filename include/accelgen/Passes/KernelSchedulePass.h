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

// enum LinkType { PIPE, SEQ };

// struct ScheduledGenericOp {
//   mlir::linalg::GenericOp& genericOp;
//   llvm::SmallVector<int> tilingFactor;
//   llvm::SmallVector<int> unrollingFactor;
//   llvm::SmallVector<int> externalOrder;
//   llvm::SmallVector<int> onchipOrder;

//   ScheduledGenericOp* clusterRoot;

//   ScheduledGenericOp(mlir::linalg::GenericOp& genericOp)
//       : genericOp(genericOp), clusterRoot(this) {
//     unsigned int nDim = genericOp.getIndexingMapsArray()[0].getNumDims();
//     tilingFactor.resize(nDim, 1);
//     unrollingFactor.resize(nDim, 1);
//     externalOrder.resize(nDim);
//     onchipOrder.resize(nDim);
//     std::iota(externalOrder.begin(), externalOrder.end(), 0);
//     std::iota(onchipOrder.begin(), externalOrder.end(), 0);
//   }
// };

class GenericOpCluster {
 public:
  unsigned int nInD = 0;
  unsigned int nCycles = 0;
  unsigned int memAccess = 0;
  unsigned int cost = 0;
  std::unordered_set<mlir::Operation*> nodeSet;
  // std::unordered_set<GenericOpCluster *> consumerSet;
  std::unordered_map<
      mlir::Operation*,
      std::unordered_map<llvm::StringRef, llvm::SmallVector<unsigned int>>>
      parameter;

  GenericOpCluster(mlir::linalg::GenericOp genericOp);

  void solveBestSchedule();
};

class GenericOpClusterDAG {
 public:
  std::unordered_set<linalg::GenericOp> genericOps;
  std::unordered_set<GenericOpCluster*> clusters;
  std::unordered_map<GenericOpCluster*, std::vector<GenericOpCluster*>>
      consumerList;
  GenericOpClusterDAG();
  ~GenericOpClusterDAG();
  void insertCluster(GenericOpCluster* cluster);
  void constructDAG();
  void optimizeDAG();

 private:
  bool checkDependency(GenericOpCluster* cluster0, GenericOpCluster* cluster1);

  unsigned int pipeCost(std::vector<GenericOpCluster*> clusters, unsigned int s,
                        unsigned int e);
};

// bool checkCluster(ScheduledGenericOp* genericOp0,
//                   ScheduledGenericOp* genericOp1);

// void mergeCluster(ScheduledGenericOp* genericOp0,
//                   ScheduledGenericOp* genericOp1);

}  // namespace accelgen

}  // namespace mlir

#endif  // KERNEL_SCHEDULE_PASS_H