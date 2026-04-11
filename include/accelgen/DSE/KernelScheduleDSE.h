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

class Dimension {
public:
  mlir::Operation *op;
  int64_t dim;

  Dimension(mlir::Operation *op, int64_t dim) : op(op), dim(dim) {};

  bool operator<(const Dimension &other) const {
    if (op != other.op)
      return op < other.op;
    return dim < other.dim;
  }

  bool operator==(const Dimension &other) const {
    return op == other.op && dim == other.dim;
  }

  struct Hash {
    std::size_t operator()(const Dimension &d) const noexcept {
      size_t h1 = std::hash<mlir::Operation *>()(d.op);
      size_t h2 = std::hash<int64_t>()(d.dim);

      return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
    }
  };
};

class Parameter;
class Relation;

class Parameter {
public:
  Parameter(int64_t bound);
  Parameter(const Parameter &src);

  void addRelation(Relation *rel);
  bool set(int64_t value);
  bool setConst(int64_t value);
  void unsetConst();
  void inValid();
  bool isConst();
  bool valid();
  int64_t value();
  int64_t bound();
  const std::vector<Relation *> &relations();

private:
  bool _valid;
  bool _const;
  int64_t _value;
  int64_t _bound;
  std::vector<Relation *> _relations;
};

class Relation {
public:
  enum class DeduceType { EQUAL, COLLAPSE };
  Relation() = default;
  Relation(const Relation &src);
  Relation(DeduceType type, llvm::ArrayRef<Parameter *> dims);
  const std::vector<Parameter *> &deducedParameters();
  bool forward(int64_t value);
  DeduceType type();

private:
  DeduceType _relType;
  std::vector<Parameter *> _deducedParameters;
};

class DimensionRelationNetwork {
public:
  bool addDimension(const Dimension &dim, int64_t bound);
  bool addEqualRelation(const Dimension &dimSrc, const Dimension &dimTarget);
  bool addCollapseRelation(const Dimension &dimSrc,
                           llvm::ArrayRef<Dimension> dimTarget);
  bool setConstDimension(const Dimension &dim, int64_t value);
  bool removeConstDimension(const Dimension &dim);
  bool isConst(const Dimension &dim);
  bool forward();
  int64_t getValue(const Dimension &dim);
  std::vector<int64_t> getUndeterminedParsBound();
  bool setUndeterminedPars(const std::vector<int64_t> &undeterminedPars);
  const std::unordered_map<Dimension, int64_t, Dimension::Hash> &
  getDimensionValueMapping();
  void clearUndeterminedPars();

  DimensionRelationNetwork() = default;
  ~DimensionRelationNetwork();
  DimensionRelationNetwork(const DimensionRelationNetwork &src);

public:
  std::map<Dimension, Parameter *> dimMapping;
  std::unordered_map<Dimension, int64_t, Dimension::Hash> dimValue;
  std::vector<Relation *> rels;
  std::vector<Parameter *> pars;
  llvm::MapVector<Parameter *, size_t> _indPars;
  // std::unordered_map<Parameter*, size_t> _indPars;
};

// class Parameter {
//  public:
//   enum class DeduceType { EQUAL, COLLAPSE, ENDPOINT };

//   Parameter() = default;
//   virtual ~Parameter() = default;

//   bool set(int64_t value) {
//     if (_valid && value != _parameter) return false;
//     _parameter = value;
//     _valid = true;
//     return true;
//   };

//   int64_t value() { return _parameter; };
//   int64_t bound() { return _bound; };
//   Parameter* source() { return _srcParameter; };
//   bool valid() { return _valid; };
//   void clearValue() { _valid = false; }

//   virtual bool forward() = 0;
//   virtual bool backward() = 0;
//   virtual std::vector<Parameter*> getDeducedParameters() = 0;

//   DeduceType getDeduceType() const { return deduceType; }

//  private:
//   DeduceType deduceType;
//   int64_t _parameter;
//   int64_t _bound;
//   bool _valid;
//   Parameter* _srcParameter;
// };

// class EndpointParameter : public Parameter {
//  public:
//   Dimension dim;
//   EndpointParameter(Dimension dim) : dim(dim) {};
//   std::vector<Parameter*> getDeducedParameters() override {
//     return std::vector<Parameter*>({nullptr});
//   }
//   bool forward() override { return true; }
//   bool backward() override {
//     assert(this->valid());
//     return true;
//   }

//   static bool classof(const Parameter* p) {
//     return p->getDeduceType() == DeduceType::ENDPOINT;
//   }
// };

// class EqualDeduceParameter : public Parameter {
//  public:
//   std::vector<Parameter*> dims;

//   EqualDeduceParameter(std::vector<Parameter*> dims) : dims(dims) {};

//   std::vector<Parameter*> getDeducedParameters() override { return dims; }

//   void addDims(Parameter* dim) { dims.push_back(dim); }

//   bool forward() override {
//     bool flag = false;
//     int64_t deducedValue = -1;
//     for (auto producer : dims) {
//       if (producer->valid()) {
//         if (!flag) {
//           deducedValue = producer->value();
//           flag = true;
//         } else {
//           if (producer->value() != deducedValue) return false;
//         }
//       }
//     }
//     this->set(deducedValue);
//     return true;
//   }

//   bool backward() override {
//     assert(this->valid());
//     for (auto p : dims) {
//       if (!p->valid())
//         p->set(this->value());
//       else if (p->value() != p->value())
//         return false;
//     }
//     return true;
//   }

//   static bool classof(const Parameter* p) {
//     return p->getDeduceType() == DeduceType::EQUAL;
//   }
// };

// class CollapseDeduceParameter : public Parameter {
//  public:
//   std::vector<Parameter*> dims;
//   CollapseDeduceParameter(std::vector<Parameter*> dims) : dims(dims) {};

//   std::vector<Parameter*> getDeducedParameters() override { return dims; }

//   bool forward() override {
//     for (auto producer : dims) {
//       if (!producer->valid()) return true;
//     }
//     llvm::SmallVector<int64_t> bitMask;
//     for (auto producer : dims) {
//       bitMask.push_back(producer->value() == producer->bound());
//     }

//     int64_t state = 0;
//     for (auto producer : dims) {
//       switch (state) {
//         case 0: {
//           if (producer->value() == 1)
//             state = 0;
//           else
//             state = 1;
//           break;
//         }
//         case 1: {
//           if (producer->value() == producer->bound())
//             state = 1;
//           else
//             return false;
//           break;
//         }
//         default:
//           assert(0);
//       }
//     }
//     if (state != 1) return false;

//     int64_t outputTile = 1;
//     for (auto producer : dims) outputTile *= producer->value();
//     this->set(outputTile);
//     return true;
//   }

//   bool backward() override {
//     assert(this->valid());
//     auto factor = this->value();
//     auto iter = dims.rbegin();
//     for (; iter != dims.rend(); iter++) {
//       auto d = factor / (*iter)->bound();
//       auto r = factor % (*iter)->bound();
//       if (d > 0) {
//         if (r != 0) return false;
//         factor = d;
//         (*iter)->set((*iter)->bound());
//       } else if (d == 0) {
//         (*iter)->set(r);
//         iter++;
//         break;
//       } else
//         assert(0);
//     }
//     for (; iter != dims.rend(); iter++) {
//       (*iter)->set(1);
//     }
//     return true;
//   }

//   static bool classof(const Parameter* p) {
//     return p->getDeduceType() == DeduceType::COLLAPSE;
//   }
// };

// class DimRelationNetwork {
//  public:
//   // std::unordered_map<int64_t *, int64_t> constDims;
//   // std::unordered_map<mlir::Operation *, std::unordered_map<int64_t,
//   int64_t>>
//   //     constDims;
//   // std::unordered_map<Parameter*, int64_t> parValue;
//   std::vector<Parameter*> parameterVec;
//   // std::unordered_map<int64_t *, Parameter *> dimMapping;
//   std::map<Dimension, Parameter*> dimMapping;

//   Parameter* addFreeMapping(Dimension dimTarget);
//   Parameter* addEqualMapping(Dimension dimTarget, Dimension dimSrc);
//   Parameter* addCollapseMapping(std::vector<Dimension> dimTarget,
//                                 Dimension dimSrc);
//   Parameter* addExpandMapping(Dimension dimTarget,
//                               std::vector<Dimension> dimSrc);
//   bool setConstParameter(Dimension dimTarget, int64_t value);

//   // Parameter* forward(Parameter* cur);

//   bool forward();
//   bool backward();

//   void setDim(Dimension dim, int64_t value);
//   void clearDim(Dimension dim);

//   int64_t getDimValue(Dimension dim);

//   bool exist(Dimension dim);

//   std::vector<Parameter*> getFreeParameter();

//   void clearValue();
// };

class ArchConfig {
public:
  size_t cycle;
  size_t bandwidth;       // GB/s
  size_t sramWidth = 512; // bit
  size_t sramDepth = 64;
  size_t nSramBank;
  size_t sramCapacity;                                     // B
  std::unordered_map<std::string, size_t> computeResource; // #

  void load(llvm::StringRef cfgPath);
};

class EvaluationMetric {
public:
  static EvaluationMetric merge(const EvaluationMetric &a,
                                const EvaluationMetric &b) {
    EvaluationMetric mergedMetric;
    if (a.isValid && b.isValid)
      mergedMetric.isValid = true;
    else
      mergedMetric.isValid = false;
    mergedMetric.flops = a.flops + b.flops;
    mergedMetric.cycles = a.cycles + b.cycles;
    mergedMetric.sramAccess = a.sramAccess + b.sramAccess;
    mergedMetric.externalAccess = a.externalAccess + b.externalAccess;

    return mergedMetric;
  }

  double_t eval() {
    if (!isValid)
      return 0;
    auto throughput = std::max(flops, 1.0) / (cycles + 1e-9);
    auto externalDensity = std::max(flops, 1.0) / (externalAccess + 1e-9);
    auto sramDensity = std::max(flops, 1.0) / (sramAccess + 1e-9);
    return ratioMetric["throughput"] * throughput +
           ratioMetric["externalDensity"] * externalDensity +
           ratioMetric["sramDensity"] * sramDensity;
  }

  double_t externalDensity() { return flops / (externalAccess + 1e-9); }
  double_t sramDensity() { return flops / (sramAccess + 1e-9); }
  double_t throughput() { return flops / (cycles + 1e-9); }

  // Whetehr parameter is under arch constraint
  bool isValid;
  double_t ratioThroughput = 0.8;
  double_t target = 0;
  // double_t throughput = 0;
  double_t computeDensity = 0;
  double_t cycles = 0;
  double_t externalAccess = 0;
  double_t sramAccess = 0;
  double_t flops = 0;
  llvm::DenseMap<llvm::StringRef, int64_t> resourceCnt;
  llvm::DenseMap<llvm::StringRef, int64_t> eventCnt;
  llvm::DenseMap<llvm::StringRef, double_t> ratioMetric;
  EvaluationMetric() {
    ratioMetric["throughput"] = 0.9997;
    ratioMetric["externalDensity"] = 0.0002;
    ratioMetric["sramDensity"] = 0.0001;
    isValid = false;
    flops = 0;
    cycles = 0;
    externalAccess = 0;
    sramAccess = 0;
  };
  // EvaluationMetric(double_t throughput, double_t computeDensity,
  //                  double_t externalAccess, double_t flops)
  //     : throughput(throughput),
  //       computeDensity(computeDensity),
  //       externalAccess(externalAccess),
  //       flops(flops) {}
};

class GenericOpCluster {
public:
  GenericOpCluster();
  GenericOpCluster(linalg::GenericOp *genericOpStart,
                   linalg::GenericOp *genericOpEnd);
  // GenericOpCluster(const GenericOpCluster &src);
  virtual ~GenericOpCluster() = default;

  bool isMember(mlir::Operation *opToCehck);

  void attachAttribute(mlir::MLIRContext *ctx);

  void evaluate();

  std::unordered_set<mlir::Operation *>::iterator begin();
  std::unordered_set<mlir::Operation *>::iterator end();

  void clearParameter();

  std::vector<mlir::Operation *> &getNodeSetTopOrder();
  std::unordered_map<
      mlir::Operation *,
      std::unordered_map<std::string, llvm::SmallVector<int64_t>>> &
  getParameter();
  // auto getMetric();

  DimensionRelationNetwork extractDimRelation();

  void applyTiling(
      const std::unordered_map<Dimension, int64_t, Dimension::Hash> &tiling);
  void applyOrder(
      const std::unordered_map<Operation *, llvm::SmallVector<int64_t>> &order);
  bool checkOrder();
  bool checkReuseDistance(const llvm::SmallVector<int64_t> &order,
                          const llvm::SmallVector<int64_t> &dims,
                          const llvm::SmallVector<int64_t> &mask);
  bool checkArchConstraint(const ArchConfig &cfg);
  bool checkOpOrder(mlir::Operation *op);

  bool checkConnectivity();

  void
  getPreviousGeneric(mlir::Value operand,
                     llvm::SmallVector<linalg::GenericOp> &previousGenerics,
                     bool ignoreNonMember = true);

  void getLatterGeneric(mlir::Value operand,
                        llvm::SmallVector<linalg::GenericOp> &latterGenerics,
                        bool ignoreNonMember = true);

  std::vector<mlir::Operation *> constructClusterWithTensorOp();

  std::vector<mlir::Operation *> getTopoOrderALSP();

  void setMetric(const EvaluationMetric &metric);

  EvaluationMetric metric;

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
  std::vector<int64_t> getDimsInOrder(linalg::GenericOp op,
                                      mlir::Value operand);

  void constructClusterHelp(mlir::Value operand,
                            llvm::SmallVector<mlir::Operation *> &path,
                            llvm::DenseSet<mlir::Operation *> &ops);
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

  virtual EvaluationMetric solve(GenericOpCluster &cluster, PerfModel &model,
                                 ArchConfig &archCfg) = 0;

  //  private:
  //   GenericOpCluster& cluster;
};

class PruningSolver : public ParameterSolvingInterface {
public:
  EvaluationMetric solve(GenericOpCluster &cluster, PerfModel &model,
                         ArchConfig &archCfg) override;

  void generateCandidateNetworks(
      std::vector<mlir::Operation *>::iterator curp,
      std::vector<mlir::Operation *>::iterator endp, GenericOpCluster &cluster,
      std::unordered_map<mlir::Operation *, llvm::SmallVector<int64_t>>
          &curOrder,
      DimensionRelationNetwork &curNetwork,

      std::vector<
          std::unordered_map<mlir::Operation *, llvm::SmallVector<int64_t>>>
          &candidateOrder,
      std::vector<DimensionRelationNetwork> &candidateNetworks);

  void generateCandidateOrders(
      GenericOpCluster &cluster, std::vector<mlir::Operation *>::iterator curOp,
      std::unordered_map<mlir::Operation *, llvm::SmallVector<int64_t>>
          &candidate,
      std::vector<std::unordered_map<mlir::Operation *,
                                     llvm::SmallVector<int64_t>>> &orders);

private:
  std::vector<llvm::SmallVector<int64_t>> generatePermutation(size_t n);
  std::vector<llvm::SmallVector<int64_t>> generateBitMask(size_t n);

  bool checkOrder(GenericOpCluster &cluster, linalg::GenericOp &op,
                  llvm::SmallVector<int64_t> &order,
                  llvm::SmallVector<int64_t> &mask);

  bool checkReuseDistance(llvm::SmallVector<int64_t> &order,
                          llvm::SmallVector<int64_t> &dims,
                          llvm::SmallVector<int64_t> &mask);

  void inferUnrollFactor(GenericOpCluster &cluster, ArchConfig &archCfg);
};

class ScheduledGenericOpCluster {
public:
  ScheduledGenericOpCluster() = default;
  ScheduledGenericOpCluster(llvm::StringRef solver);
  ~ScheduledGenericOpCluster();
  void insertOp(mlir::Operation *op);
  void insertGenericOp(linalg::GenericOp genericOp);
  void schedule(mlir::MLIRContext *ctx, PerfModel &model, ArchConfig &archCfg,
                int64_t maxSubgraphOp);

  std::vector<mlir::accelgen::GenericOpCluster *>::iterator begin();
  std::vector<mlir::accelgen::GenericOpCluster *>::iterator end();

private:
  std::vector<mlir::Operation *> ops;
  std::vector<linalg::GenericOp> genericOps;
  std::vector<GenericOpCluster *> clusters;
  std::vector<linalg::GenericOp> getTopSortedNodes();

  ParameterSolvingInterface *solver;
};

} // namespace accelgen
} // namespace mlir

#endif