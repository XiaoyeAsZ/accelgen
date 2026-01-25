#ifndef KERNEL_SCHEDULE_DSE_H
#define KERNEL_SCHEDULE_DSE_H

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Pass/Pass.h"
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "accelgen/Utils/Wrapper.h"

namespace mlir {
namespace accelgen {

enum DimRelationType { EQUAL, COMPOSE, ENDPOINT };

class Parameter {
public:
  int64_t parameter;
  int64_t bound;
  bool isConst;
  Parameter *srcParameter;

  virtual ~Parameter() = default;
  bool set(int64_t value) {
    if (isConst && value != parameter)
      return false;
    parameter = value;
    isConst = true;
    return true;
  };
  int64_t value() { return parameter; };
  // bool isConst() { return isConst; };

  virtual Parameter *forward() = 0;
  virtual bool backward() = 0;
};

class EndpointParameter : public Parameter {
public:
  int64_t *dim;
  EndpointParameter(int64_t *dim) : dim(dim) {};
  Parameter *forward() {
    if (!isConst)
      return nullptr;
    if (!srcParameter)
      return this;
    if (auto ptr = srcParameter->forward())
      return ptr;
    else
      return this;
  }
  bool backward() {
    assert(isConst);
    *dim = parameter;
    return true;
  }
};

class EqualDeduceParameter : public Parameter {
public:
  std::vector<Parameter *> dims;

  EqualDeduceParameter(std::vector<Parameter *> dims) : dims(dims) {};

  Parameter *forward() {
    bool flag = false;
    for (auto p : dims) {
      if (p->isConst) {
        flag = true;
        break;
      }
    }
    if (flag) {
      if (!srcParameter)
        return this;
      if (auto ptr = srcParameter->forward())
        return ptr;
      else
        return this;
    } else
      return nullptr;
  }
  bool backward() {
    for (auto p : dims) {
      assert(isConst);
      p->set(parameter);
      if (!p->backward())
        return false;
    }
    return true;
  }
};

class CollapseDeduceParameter : public Parameter {
public:
  std::vector<Parameter *> dims;
  CollapseDeduceParameter(std::vector<Parameter *> dims) : dims(dims) {};

  Parameter *forward() { return nullptr; }
  bool backward() {
    auto totalFactor = parameter;
    for (auto p = dims.rbegin(); p != dims.rend(); p++) {
      int64_t n = totalFactor / (*p)->bound;
      int64_t r = totalFactor % (*p)->bound;

      if (n != 0 && r != 0 || n == 0 && r == 0)
        return false;
      auto tilingSize = (r == 0) ? (*p)->bound : r;
      (*p)->set(tilingSize);
    }
  }
};

class DimRelationNetwork {
public:
  std::unordered_map<int64_t *, int64_t> constDims;
  std::vector<Parameter *> parameterVec;
  std::unordered_map<int64_t *, Parameter *> dimMapping;

  Parameter *addFreeMapping(int64_t *dimTarget);
  Parameter *addEqualMapping(int64_t *dimTarget, int64_t *dimSrc);
  Parameter *addCollapseMapping(std::vector<int64_t *> dimTarget,
                                int64_t *dimSrc);
  bool setConstParameter(int64_t *dimTarget, int64_t value);

  Parameter *forward(Parameter *cur);
};

class GenericOpCluster {
public:
  GenericOpCluster();
  GenericOpCluster(linalg::GenericOp *genericOpStart,
                   linalg::GenericOp *genericOpEnd);
  virtual ~GenericOpCluster() = default;

  bool isMember(mlir::Operation *opToCehck);

  void attachAttribute(mlir::MLIRContext *ctx);

  void evaluate();

  auto begin();
  auto end();

  void clearParameter();

  std::vector<mlir::Operation *> &getNodeSetTopOrder();
  std::unordered_map<
      mlir::Operation *,
      std::unordered_map<std::string, llvm::SmallVector<int64_t>>> &
  getParameter();
  // auto getMetric();

  DimRelationNetwork extractDimRelation();

private:
  unsigned int nInD = 0;
  unsigned int nCycles = 0;
  unsigned int memAccess = 0;
  unsigned int cost = 0;
  std::unordered_set<mlir::Operation *> nodeSet;
  std::vector<mlir::Operation *> nodeSetTopOrder;
  std::unordered_map<
      mlir::Operation *,
      std::unordered_map<std::string, llvm::SmallVector<int64_t>>>
      parameter;

  void factorForwardHelp(mlir::Operation *op, int64_t factor);
};

class ArchConfig {
public:
  size_t bandwidth;    // GB/s
  size_t sramCapacity; // B
  size_t mulCnt;       // #
};

class EvaluationMetric {
public:
  double_t throughput;
};

class PerfModel {
public:
  EvaluationMetric evaluate(GenericOpCluster &cluster, ArchConfig &cfg);
};

class ParameterSolvingInterface {
public:
  ParameterSolvingInterface() = default;
  //   ParameterSolvingInterface(GenericOpCluster& cluster);
  virtual ~ParameterSolvingInterface() = default;

  virtual unsigned int solve(GenericOpCluster &cluster, PerfModel &model,
                             ArchConfig &archCfg) = 0;

  //  private:
  //   GenericOpCluster& cluster;
};

class PruningSolver : public ParameterSolvingInterface {
public:
  unsigned int solve(GenericOpCluster &cluster, PerfModel &model,
                     ArchConfig &archCfg) override;

  void generateCandidateNetworks(
      std::vector<mlir::Operation *>::iterator curp,
      std::vector<mlir::Operation *>::iterator endp,
      std::unordered_map<mlir::Operation *, llvm::SmallVector<int64_t>>
          *curOrder,
      DimRelationNetwork *curNetwork,
      std::vector<
          std::unordered_map<mlir::Operation *, llvm::SmallVector<int64_t>>>
          *candidateOrder,
      std::vector<DimRelationNetwork> *candidateNetworks);

private:
  std::vector<llvm::SmallVector<int64_t>> generatePermutation(size_t n);
  std::vector<llvm::SmallVector<int64_t>> generateBitMask(size_t n);
};

class ScheduledGenericOpCluster {
public:
  ScheduledGenericOpCluster() = default;
  ScheduledGenericOpCluster(llvm::StringRef solver);
  ~ScheduledGenericOpCluster();
  void insertGenericOp(linalg::GenericOp genericOp);
  void schedule(mlir::MLIRContext *ctx, PerfModel &model, ArchConfig &archCfg);

  auto begin();
  auto end();

private:
  std::vector<linalg::GenericOp> genericOps;
  std::vector<GenericOpCluster *> clusters;
  std::vector<linalg::GenericOp> getTopSortedNodes();

  ParameterSolvingInterface *solver;
};

} // namespace accelgen
} // namespace mlir

#endif