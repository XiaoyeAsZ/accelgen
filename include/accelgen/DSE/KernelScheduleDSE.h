#ifndef KERNEL_SCHEDULE_DSE_H
#define KERNEL_SCHEDULE_DSE_H

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

#endif