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

class Parameter {
 public:
  enum class DeduceType { EQUAL, COLLAPSE, ENDPOINT };

  Parameter() = default;
  virtual ~Parameter() = default;

  bool set(int64_t value) {
    if (_valid && value != _parameter) return false;
    _parameter = value;
    _valid = true;
    return true;
  };

  int64_t value() { return _parameter; };
  int64_t bound() { return _bound; };
  Parameter* source() { return _srcParameter; };
  bool valid() { return _valid; };
  void clearValue() { _valid = false; }

  virtual bool forward() = 0;
  virtual bool backward() = 0;
  virtual std::vector<Parameter*> getDeducedParameters() = 0;

  DeduceType getDeduceType() const { return deduceType; }

 private:
  DeduceType deduceType;
  int64_t _parameter;
  int64_t _bound;
  bool _valid;
  Parameter* _srcParameter;
};

class EndpointParameter : public Parameter {
 public:
  int64_t* dim;
  EndpointParameter(int64_t* dim) : dim(dim) {};
  std::vector<Parameter*> getDeducedParameters() override {
    return std::vector<Parameter*>({nullptr});
  }
  bool forward() override { return true; }
  bool backward() override {
    assert(this->valid());
    return true;
  }

  static bool classof(const Parameter* p) {
    return p->getDeduceType() == DeduceType::ENDPOINT;
  }
};

class EqualDeduceParameter : public Parameter {
 public:
  std::vector<Parameter*> dims;

  EqualDeduceParameter(std::vector<Parameter*> dims) : dims(dims) {};

  std::vector<Parameter*> getDeducedParameters() override { return dims; }

  bool forward() override {
    bool flag = false;
    int64_t deducedValue = -1;
    for (auto producer : dims) {
      if (producer->valid()) {
        if (!flag) {
          deducedValue = producer->value();
          flag = true;
        } else {
          if (producer->value() != deducedValue) return false;
        }
      }
    }
    this->set(deducedValue);
    return true;
  }

  bool backward() override {
    assert(this->valid());
    for (auto p : dims) {
      if (!p->valid())
        p->set(this->value());
      else if (p->value() != p->value())
        return false;
    }
    return true;
  }

  static bool classof(const Parameter* p) {
    return p->getDeduceType() == DeduceType::EQUAL;
  }
};

class CollapseDeduceParameter : public Parameter {
 public:
  std::vector<Parameter*> dims;
  CollapseDeduceParameter(std::vector<Parameter*> dims) : dims(dims) {};

  std::vector<Parameter*> getDeducedParameters() override { return dims; }

  bool forward() override {
    for (auto producer : dims) {
      if (!producer->valid()) return true;
    }
    llvm::SmallVector<int64_t> bitMask;
    for (auto producer : dims) {
      bitMask.push_back(producer->value() == producer->bound());
    }

    int64_t state = 0;
    for (auto producer : dims) {
      switch (state) {
        case 0: {
          if (producer->value() == 1)
            state = 0;
          else
            state = 1;
          break;
        }
        case 1: {
          if (producer->value() == producer->bound())
            state = 1;
          else
            return false;
          break;
        }
        default:
          assert(0);
      }
    }
    if (state != 1) return false;

    int64_t outputTile = 1;
    for (auto producer : dims) outputTile *= producer->value();
    this->set(outputTile);
    return true;
  }

  bool backward() override {
    assert(this->valid());
    auto factor = this->value();
    auto iter = dims.rbegin();
    for (; iter != dims.rend(); iter++) {
      auto d = factor / (*iter)->bound();
      auto r = factor % (*iter)->bound();
      if (d > 0) {
        if (r != 0) return false;
        factor = d;
        (*iter)->set((*iter)->bound());
      } else if (d == 0) {
        (*iter)->set(r);
        iter++;
        break;
      } else
        assert(0);
    }
    for (; iter != dims.rend(); iter++) {
      (*iter)->set(1);
    }
    return true;
  }

  static bool classof(const Parameter* p) {
    return p->getDeduceType() == DeduceType::COLLAPSE;
  }
};

class DimRelationNetwork {
 public:
  std::unordered_map<int64_t*, int64_t> constDims;
  std::unordered_map<Parameter*, int64_t> parValue;
  std::vector<Parameter*> parameterVec;
  std::unordered_map<int64_t*, Parameter*> dimMapping;

  Parameter* addFreeMapping(int64_t* dimTarget);
  Parameter* addEqualMapping(int64_t* dimTarget, int64_t* dimSrc);
  Parameter* addCollapseMapping(std::vector<int64_t*> dimTarget,
                                int64_t* dimSrc);
  bool setConstParameter(int64_t* dimTarget, int64_t value);

  // Parameter* forward(Parameter* cur);

  bool forward();
  bool backward();

  void addConstDim(int64_t* dim, int64_t value);
  void removeConstDim(int64_t* dim);

  int64_t getDimValue(int64_t* dim);

  std::vector<Parameter*> getFreeParameter();

  void clearValue();
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

  std::unordered_set<mlir::Operation*>::iterator begin();
  std::unordered_set<mlir::Operation*>::iterator end();

  void clearParameter();

  std::vector<mlir::Operation*>& getNodeSetTopOrder();
  std::unordered_map<
      mlir::Operation*,
      std::unordered_map<std::string, llvm::SmallVector<int64_t>>>&
  getParameter();
  // auto getMetric();

  DimRelationNetwork extractDimRelation();

  void applyOrderTiling(
      std::unordered_map<mlir::Operation*, llvm::SmallVector<int64_t>>& order,
      DimRelationNetwork& network);

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

  void factorForwardHelp(mlir::Operation* op, int64_t factor);
};

class ArchConfig {
 public:
  size_t bandwidth;     // GB/s
  size_t sramCapacity;  // B
  size_t mulCnt;        // #
};

class EvaluationMetric {
 public:
  double_t throughput;
  double_t computeDensity;
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

  virtual unsigned int solve(GenericOpCluster& cluster, PerfModel& model,
                             ArchConfig& archCfg) = 0;

  //  private:
  //   GenericOpCluster& cluster;
};

class PruningSolver : public ParameterSolvingInterface {
 public:
  unsigned int solve(GenericOpCluster& cluster, PerfModel& model,
                     ArchConfig& archCfg) override;

  void generateCandidateNetworks(
      std::vector<mlir::Operation*>::iterator curp,
      std::vector<mlir::Operation*>::iterator endp, GenericOpCluster& cluster,
      std::unordered_map<mlir::Operation*, llvm::SmallVector<int64_t>>&
          curOrder,
      DimRelationNetwork& curNetwork,
      std::vector<
          std::unordered_map<mlir::Operation*, llvm::SmallVector<int64_t>>>&
          candidateOrder,
      std::vector<DimRelationNetwork>& candidateNetworks);

 private:
  std::vector<llvm::SmallVector<int64_t>> generatePermutation(size_t n);
  std::vector<llvm::SmallVector<int64_t>> generateBitMask(size_t n);

  bool checkOrder(GenericOpCluster& cluster, linalg::GenericOp& op,
                  llvm::SmallVector<int64_t>& order,
                  llvm::SmallVector<int64_t>& mask);

  bool checkReuseDistance(llvm::SmallVector<int64_t>& order,
                          llvm::SmallVector<int64_t>& dims,
                          llvm::SmallVector<int64_t>& mask);
};

class ScheduledGenericOpCluster {
 public:
  ScheduledGenericOpCluster() = default;
  ScheduledGenericOpCluster(llvm::StringRef solver);
  ~ScheduledGenericOpCluster();
  void insertGenericOp(linalg::GenericOp genericOp);
  void schedule(mlir::MLIRContext* ctx, PerfModel& model, ArchConfig& archCfg);

  std::vector<mlir::accelgen::GenericOpCluster*>::iterator begin();
  std::vector<mlir::accelgen::GenericOpCluster*>::iterator end();

 private:
  std::vector<linalg::GenericOp> genericOps;
  std::vector<GenericOpCluster*> clusters;
  std::vector<linalg::GenericOp> getTopSortedNodes();

  ParameterSolvingInterface* solver;
};

}  // namespace accelgen
}  // namespace mlir

#endif