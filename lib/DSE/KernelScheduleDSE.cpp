#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include <assert.h>
#include <numeric>
#include <queue>
#include <span>
#include <variant>
#include <vector>

#include "accelgen/DSE/KernelScheduleDSE.h"
#include "accelgen/Utils/AffineMapUtils.h"
#include "accelgen/Utils/DSEUtils.h"
#include "accelgen/Utils/OperationUtils.h"

namespace mlir::accelgen {

Parameter::Parameter(int64_t bound)
    : _bound(bound), _valid(false), _const(false) {}
Parameter::Parameter(const Parameter &src)
    : _const(src._const), _value(src._value), _bound(src._bound) {}
void Parameter::addRelation(Relation *rel) { _relations.push_back(rel); }

bool Parameter::set(int64_t value) {
  if ((_valid | _const) && value != _value)
    return false;
  this->_value = value;
  this->_valid = true;
  return true;
}

bool Parameter::setConst(int64_t value) {
  if (_const && value != _value)
    return false;
  _const = true;
  _valid = true;
  this->_value = value;
  return true;
}

void Parameter::unsetConst() {
  _const = false;
  _valid = false;
}

bool Parameter::isConst() { return _const; }
bool Parameter::valid() { return _valid; }

const std::vector<Relation *> &Parameter::relations() { return _relations; }

int64_t Parameter::value() { return _value; }
int64_t Parameter::bound() { return _bound; }

Relation::Relation(Relation::DeduceType type, llvm::ArrayRef<Parameter *> dims)
    : _relType(type) {
  for (auto dim : dims)
    _deducedParameters.push_back(dim);
}

const std::vector<Parameter *> &Relation::deducedParameters() {
  return _deducedParameters;
};

bool Relation::forward(int64_t value) {
  switch (this->_relType) {
  case DeduceType::EQUAL: {
    for (auto &p : this->_deducedParameters) {
      if (!p->set(value))
        return false;
    }
  } break;
  case DeduceType::COLLAPSE: {
    auto factor = value;
    auto iter = _deducedParameters.rbegin();
    for (; iter != _deducedParameters.rend(); iter++) {
      auto d = factor / (*iter)->bound();
      auto r = factor % (*iter)->bound();
      if (d > 0) {
        if (r != 0)
          return false;
        factor = d;
        if (!((*iter)->set((*iter)->bound())))
          return false;
      } else if (d == 0) {
        if (!((*iter)->set(r)))
          return false;
        iter++;
        break;
      } else
        assert(0);
    }
    for (; iter != _deducedParameters.rend(); iter++) {
      if (!((*iter)->set(1)))
        return false;
    }
  } break;
  default:
    assert(0);
    break;
  }
  return true;
}
Relation::Relation(const Relation &src) : _relType(src._relType) {}
Relation::DeduceType Relation::type() { return _relType; }

// ============================
// CLASS: DimensionRelationNetwork
// ============================

DimensionRelationNetwork::~DimensionRelationNetwork() {
  for (auto r : rels)
    delete r;
  for (auto p : pars)
    delete p;
}

DimensionRelationNetwork::DimensionRelationNetwork(
    const DimensionRelationNetwork &src) {
  std::unordered_map<Parameter *, Parameter *> parMap;
  for (auto p : src.pars)
    parMap[p] = new Parameter(*p);
  for (auto &[op, np] : parMap) {
    for (auto &orel : op->relations()) {
      std::vector<Parameter *> pars;
      for (auto p : orel->deducedParameters())
        pars.push_back(parMap[p]);
      auto nrel = new Relation(orel->type(), pars);
      np->addRelation(nrel);
      rels.push_back(nrel);
    }
  }
  for (auto &[d, p] : src.dimMapping)
    this->dimMapping[d] = parMap[p];
  for (auto &[op, np] : parMap)
    pars.push_back(np);
  for (auto &[p, id] : src._indPars)
    this->_indPars[parMap[p]] = id;
}

bool DimensionRelationNetwork::addDimension(const Dimension &dim,
                                            int64_t bound) {
  if (dimMapping.find(dim) != dimMapping.end())
    return false;
  auto par = new Parameter(bound);
  pars.push_back(par);
  dimMapping[dim] = par;
  _indPars[par] = 0;
  return true;
}

bool DimensionRelationNetwork::addEqualRelation(const Dimension &dimSrc,
                                                const Dimension &dimTarget) {
  if (dimMapping.find(dimSrc) == dimMapping.end() ||
      dimMapping.find(dimTarget) == dimMapping.end())
    return false;
  auto rel = new Relation(Relation::DeduceType::EQUAL, {dimMapping[dimTarget]});
  _indPars[dimMapping[dimTarget]]++;
  dimMapping[dimSrc]->addRelation(rel);
  rels.push_back(rel);
  return true;
}

bool DimensionRelationNetwork::addCollapseRelation(
    const Dimension &dimSrc, llvm::ArrayRef<Dimension> dimTarget) {
  if (dimMapping.find(dimSrc) == dimMapping.end())
    return false;
  std::vector<Parameter *> pars;
  for (auto dim : dimTarget) {
    if (dimMapping.find(dim) == dimMapping.end())
      return false;
    pars.push_back(dimMapping[dim]);
    _indPars[dimMapping[dim]]++;
  }
  auto rel = new Relation(Relation::DeduceType::COLLAPSE, pars);
  dimMapping[dimSrc]->addRelation(rel);
  rels.push_back(rel);
  return true;
}

bool DimensionRelationNetwork::setConstDimension(const Dimension &dim,
                                                 int64_t value) {
  if (dimMapping.find(dim) == dimMapping.end())
    return false;
  dimMapping[dim]->set(value);
  return true;
}

bool DimensionRelationNetwork::forward() {
  std::map<Parameter *, size_t> ind(_indPars);
  std::queue<Parameter *> q;
  for (auto [p, d] : ind) {
    if (d == 0)
      q.push(p);
  }
  while (!q.empty()) {
    auto p = q.front();
    q.pop();
    assert(p->valid());
    for (auto dr : p->relations()) {
      if (!dr->forward(p->value()))
        return false;
      for (auto dp : dr->deducedParameters()) {
        ind[dp]--;
        if (ind[dp] == 0)
          q.push(dp);
      }
    }
  }
  return true;
}

int64_t DimensionRelationNetwork::getValue(const Dimension &dim) {
  assert(dimMapping.find(dim) != dimMapping.end());
  assert(dimMapping[dim]->valid());
  return dimMapping[dim]->value();
}

std::vector<int64_t> DimensionRelationNetwork::getUndeterminedParsBound() {
  std::vector<int64_t> bounds;
  for (auto &[p, d] : _indPars) {
    if (d == 0 && (!p->isConst()))
      bounds.push_back(p->bound());
  }
  return bounds;
}

bool DimensionRelationNetwork::isConst(const Dimension &dim) {
  assert(dimMapping.find(dim) != dimMapping.end());
  return dimMapping[dim]->isConst();
}

bool DimensionRelationNetwork::setUndeterminedPars(
    const std::vector<int64_t> &undeterminedPars) {
  std::vector<Parameter *> ups;
  for (auto &[p, d] : _indPars) {
    if (d == 0 && (!p->isConst()))
      ups.push_back(p);
  }
  for (auto [p, v] : llvm::zip(ups, undeterminedPars))
    p->set(v);
}

const std::unordered_map<Dimension, int64_t, Dimension::Hash> &
DimensionRelationNetwork::getDimensionValueMapping() {
  dimValue.clear();
  for (auto &[d, p] : dimMapping)
    dimValue[d] = p->value();
  return dimValue;
}

bool DimensionRelationNetwork::removeConstDimension(const Dimension &dim) {
  assert(dimMapping.find(dim) != dimMapping.end());
  dimMapping[dim]->unsetConst();
}

// ============================
// END OF DimensionRelationNetwork
// ============================

// ============================
// CLASS: DimRelationNetwork
// ============================

// Parameter* DimRelationNetwork::addFreeMapping(Dimension dimTarget) {
//   if (dimMapping.find(dimTarget) != dimMapping.end())
//     return dimMapping[dimTarget];
//   Parameter* par = new EndpointParameter(dimTarget);
//   parameterVec.push_back(par);
//   dimMapping[dimTarget] = par;
//   return par;
// }

// Parameter* DimRelationNetwork::addEqualMapping(Dimension dimTarget,
//                                                Dimension dimSrc) {
//   assert(dimMapping.find(dimTarget) != dimMapping.end());
//   Parameter* endpointTarget = dimMapping[dimTarget];
//   assert(dimMapping.find(dimSrc) != dimMapping.end());
//   auto endpointSrc = dimMapping[dimSrc];
//   assert(dimMapping[dimTarget]->source() == nullptr);
//   if (endpointSrc->source() == nullptr) {
//     auto par = new EqualDeduceParameter(
//         std::vector<Parameter*>({endpointTarget, endpointSrc}));
//     parameterVec.push_back(par);
//     return par;
//   } else if (auto parSrc =
//                  mlir::dyn_cast<EqualDeduceParameter>(endpointSrc->source()))
//                  {
//     parSrc->addDims(dimMapping[dimTarget]);
//     return parSrc;
//   } else if (auto parSrc = mlir::dyn_cast<CollapseDeduceParameter>(
//                  endpointSrc->source())) {
//     Parameter* par = new EqualDeduceParameter(
//         std::vector<Parameter*>({endpointTarget, endpointSrc}));
//     std::replace(parSrc->dims.begin(), parSrc->dims.end(), endpointSrc, par);
//     return par;
//   } else
//     assert(0);
// }

// Parameter* DimRelationNetwork::addCollapseMapping(
//     std::vector<Dimension> dimTarget, Dimension dimSrc) {
//   std::vector<Parameter*> parTarget;
//   for (auto dim : dimTarget) {
//     assert(dimMapping.find(dim) != dimMapping.end());
//     EndpointParameter* endpoint =
//         mlir::dyn_cast<EndpointParameter>(dimMapping[dim]);
//     if (endpoint->source() == nullptr)
//       parTarget.push_back(endpoint);
//     else if (mlir::dyn_cast<EqualDeduceParameter>(endpoint->source()))
//       parTarget.push_back(endpoint->source());
//     else
//       assert(0);
//   }
//   Parameter* parCollapse = new CollapseDeduceParameter(parTarget);
//   parameterVec.push_back(parCollapse);
//   assert(dimMapping.find(dimSrc) != dimMapping.end());
//   Parameter* parSrc =
//       mlir::dyn_cast<EqualDeduceParameter>(dimMapping[dimSrc]->source());
//   if (!parSrc) parSrc = dimMapping[dimSrc];
//   auto parEqual =
//       new EqualDeduceParameter(std::vector<Parameter*>({parCollapse,
//       parSrc}));
//   parameterVec.push_back(parEqual);
//   return parEqual;
// }

// Parameter* DimRelationNetwork::addExpandMapping(Dimension dimTarget,
//                                                 std::vector<Dimension>
//                                                 dimSrc) {
//   std::vector<Parameter*> parTarget;
//   assert(dimMapping.find(dimTarget) != dimMapping.end());
//   Parameter* par;
//   if (dimMapping[dimTarget]->source() == nullptr)
//     par = dimMapping[dimTarget];
//   else if (mlir::dyn_cast<EqualDeduceParameter>(
//                dimMapping[dimTarget]->source()))
//     par = dimMapping[dimTarget]->source();
//   else
//     assert(0);

//   for (auto dim : dimSrc) {
//     assert(dimMapping.find(dim) != dimMapping.end());
//     EndpointParameter* endpoint =
//         mlir::dyn_cast<EndpointParameter>(dimMapping[dim]);
//     if (endpoint->source() == nullptr)
//       parTarget.push_back(endpoint);
//     else if (mlir::dyn_cast<EqualDeduceParameter>(endpoint->source()))
//       parTarget.push_back(endpoint->source());
//     else
//       assert(0);
//   }
//   Parameter* parCollapse = new CollapseDeduceParameter(parTarget);
//   parameterVec.push_back(parCollapse);
//   assert(dimMapping.find(dimSrc) != dimMapping.end());
//   Parameter* parSrc =
//       mlir::dyn_cast<EqualDeduceParameter>(dimMapping[dimSrc]->source());
//   if (!parSrc) parSrc = dimMapping[dimSrc];
//   auto parEqual =
//       new EqualDeduceParameter(std::vector<Parameter*>({parCollapse,
//       parSrc}));
//   parameterVec.push_back(parEqual);
//   return parEqual;
// }

// bool DimRelationNetwork::setConstParameter(int64_t *dimTarget, int64_t value)
// {
//   assert(dimMapping.find(dimTarget) != dimMapping.end());
//   auto parPtr = dimMapping[dimTarget];
//   if (!parPtr->set(value))
//     return false;
//   while (parPtr->source() &&
//          mlir::dyn_cast<EqualDeduceParameter>(parPtr->source())) {
//     if (!parPtr->source()->set(value))
//       return false;
//     parPtr = parPtr->source();
//   }
// }

// void DimRelationNetwork::setDim(Dimension dim, int64_t value) {
//   assert(dimMapping.find(dim) != dimMapping.end());
//   dimMapping[dim]->set(value);
// }

// void DimRelationNetwork::clearDim(Dimension dim) {
//   assert(dimMapping.find(dim) != dimMapping.end());
//   dimMapping[dim]->clearValue();
// }

// std::vector<Parameter*> DimRelationNetwork::getFreeParameter() {
//   std::vector<Parameter*> result;
//   for (auto par : parameterVec) {
//     if (par->source() == nullptr && (!par->valid())) result.push_back(par);
//   }
//   return result;
// }

// bool DimRelationNetwork::forward() {
//   clearValue();
//   std::map<Parameter*, size_t> inD;
//   for (auto p : parameterVec) inD[p] = 0;
//   for (auto p : parameterVec) {
//     if (p->source()) inD[p->source()]++;
//   }
//   std::queue<Parameter*> q;
//   for (auto [p, i] : inD) {
//     if (i == 0) q.push(p);
//   }
//   while (!q.empty()) {
//     auto curPar = q.front();
//     q.pop();
//     if (!curPar->forward()) return false;
//     if (curPar->source()) q.push(curPar->source());
//   }
//   return true;
// }

// bool DimRelationNetwork::backward() {
//   std::queue<Parameter*> q;
//   for (auto p : parameterVec) {
//     if (p->source() == nullptr) q.push(p);
//   }
//   while (!q.empty()) {
//     auto curPar = q.front();
//     q.pop();
//     if (!curPar->backward()) return false;
//     auto deducedPars = curPar->getDeducedParameters();
//     for (auto d : deducedPars)
//       if (d) q.push(d);
//   }
//   return true;
// }

// void DimRelationNetwork::clearValue() {
//   for (auto p : parameterVec) p->clearValue();
// }

// int64_t DimRelationNetwork::getDimValue(Dimension dim) {
//   assert(dimMapping.find(dim) != dimMapping.end());
//   return dimMapping[dim]->value();
// }

// bool DimRelationNetwork::exist(Dimension dim) {
//   return dimMapping.find(dim) != dimMapping.end();
// }

// ============================
// END OF DimRelationNetwork
// ============================

// ============================
// CLASS: GenericOpCluster
// ============================

GenericOpCluster::GenericOpCluster() {};
GenericOpCluster::GenericOpCluster(linalg::GenericOp *genericOpStart,
                                   linalg::GenericOp *genericOpEnd)
    : nodeSet(genericOpStart, genericOpEnd),
      nodeSetTopOrder(genericOpStart, genericOpEnd) {
  for (auto iter = genericOpStart; iter != genericOpEnd; iter++) {
    parameter[*iter]["loop_bound"] = iter->getStaticLoopRanges();
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

bool GenericOpCluster::isMember(mlir::Operation *opToCehck) {
  return nodeSet.find(opToCehck) != nodeSet.end();
}

void GenericOpCluster::attachAttribute(mlir::MLIRContext *ctx) {
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

std::unordered_set<mlir::Operation *>::iterator GenericOpCluster::begin() {
  return nodeSet.begin();
}
std::unordered_set<mlir::Operation *>::iterator GenericOpCluster::end() {
  return nodeSet.end();
}

std::vector<mlir::Operation *> &GenericOpCluster::getNodeSetTopOrder() {
  return nodeSetTopOrder;
}
std::unordered_map<
    mlir::Operation *,
    std::unordered_map<std::string, llvm::SmallVector<int64_t>>> &
GenericOpCluster::getParameter() {
  return parameter;
}

void GenericOpCluster::clearParameter() {
  for (auto [op, par] : parameter)
    par.clear();
}

// auto GenericOpCluster::getMetric() { return &metric; }

DimensionRelationNetwork GenericOpCluster::extractDimRelation() {
  DimensionRelationNetwork network;
  // Init all dims
  for (auto op : nodeSetTopOrder) {
    for (auto [index, dim] : llvm::enumerate(parameter[op]["tiling_size"]))
      network.addDimension(Dimension(op, index),
                           parameter[op]["loop_bound"][index]);
  }
  for (auto op : nodeSetTopOrder) {
    auto genericOp = mlir::dyn_cast<linalg::GenericOp>(op);
    assert(genericOp);
    for (auto [indexOperand, operand] :
         llvm::enumerate(genericOp.getInputs())) {
      auto accessDims = getAffineMapAccessDims(
          genericOp.getIndexingMapsArray()[indexOperand]);
      if (!operand.getDefiningOp())
        continue;
      mlir::TypeSwitch<mlir::Operation *>(operand.getDefiningOp())
          .Case<linalg::GenericOp>([&](linalg::GenericOp generic) {
            if (isMember(generic)) {
              auto accessDimsProducer =
                  getAffineMapAccessDims(generic.getIndexingMapsArray().back());
              for (auto [target, src] :
                   llvm::zip(accessDims, accessDimsProducer))
                network.addEqualRelation(Dimension(op, target),
                                         Dimension(generic, src));
            }
          })
          .Case<tensor::CollapseShapeOp>([&](tensor::CollapseShapeOp collapse) {
            auto producer = mlir::dyn_cast<linalg::GenericOp>(
                collapse.getSrc().getDefiningOp());
            assert(producer);

            auto reassociationMap = collapse.getReassociationMaps();
            auto accessDimsProducer =
                getAffineMapAccessDims(producer.getIndexingMapsArray().back());

            for (size_t d = 0; d < reassociationMap.size(); d++) {
              auto collapseDims = getAffineMapAccessDims(reassociationMap[d]);
              std::vector<Dimension> dimPtr;
              for (auto cd : collapseDims)
                dimPtr.push_back(Dimension(producer, accessDimsProducer[cd]));
              network.addCollapseRelation(Dimension(op, accessDims[d]), dimPtr);
            }
          })
          .Case<tensor::ExpandShapeOp>([&](tensor::ExpandShapeOp expand) {
            auto producer = mlir::dyn_cast<linalg::GenericOp>(
                expand.getSrc().getDefiningOp());
            assert(producer);

            auto reassociationMap = expand.getReassociationMaps();
            auto accessDimsProducer =
                getAffineMapAccessDims(producer.getIndexingMapsArray().back());

            for (size_t d = 0; d < reassociationMap.size(); d++) {
              auto expandDims = getAffineMapAccessDims(reassociationMap[d]);
              std::vector<Dimension> dimPtr;
              for (auto d : expandDims)
                dimPtr.push_back(Dimension(op, accessDims[d]));
              network.addCollapseRelation(
                  Dimension(producer, accessDimsProducer[d]), dimPtr);
            }
          })
          .Case<arith::ConstantOp>([&](arith::ConstantOp constant) {})
          .Default([](mlir::Operation *placeholder) {
            placeholder->dump();
            assert(0);
          });
    }
  }
  return network;
}

void GenericOpCluster::applyOrderTiling(
    const std::unordered_map<mlir::Operation *, llvm::SmallVector<int64_t>>
        &order,
    const std::unordered_map<Dimension, int64_t, Dimension::Hash> &tiling) {
  for (auto [op, order] : order)
    parameter[op]["outer_order"] = order;
  for (auto [op, par] : parameter) {
    for (auto [index, dim] : llvm::enumerate(par["tiling_size"]))
      dim = tiling.at(Dimension(op, index));
  }
}

// ============================
// END OF GenericOpCluster
// ============================

// ============================
// CLASS: PerfModel
// ============================

EvaluationMetric PerfModel::evaluate(GenericOpCluster &cluster,
                                     ArchConfig &cfg) {
  auto parameter = cluster.getParameter();
  std::unordered_map<mlir::Operation *,
                     std::unordered_map<std::string, double_t>>
      statisticDat;
  for (auto kv : parameter) {
    statisticDat[kv.first]["factor"] = 1;
    statisticDat[kv.first]["flops"] = 0;
    statisticDat[kv.first]["external_access"] = 0;
    statisticDat[kv.first]["sram_access"] = 0;
    statisticDat[kv.first]["cycles"] = 0;
  }
  std::vector<mlir::Operation *> nodeSetTopOrder = cluster.getNodeSetTopOrder();
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
        // llvm::errs() << "Unknown iterator types.\n";
        break;
      }
    }

    auto flops =
        std::accumulate(analysisTileSize.begin(), analysisTileSize.end(),
                        int64_t(1), std::multiplies<>());

    auto externalAccess = 0;
    for (auto [index, operand] : llvm::enumerate(genericOp.getInputs())) {
      auto producer = operand.getDefiningOp();
      // Input from
      if (!cluster.isMember(producer)) {
        auto accessDims = getAffineMapAccessDims(indexingMaps[index]);
        int64_t partial = 1;
        for (auto dim : accessDims)
          partial *= analysisTileSize[dim];
        externalAccess += partial;
      }
    }

    for (auto [index, result] : llvm::enumerate(genericOp.getResults())) {
      bool flag = true;
      for (auto user : result.getUsers()) {
        if (cluster.isMember(user)) {
          flag = false;
          break;
        }
      }
      if (flag) {
        auto accessDims = getAffineMapAccessDims(
            indexingMaps[genericOp.getInputs().size() + index]);
        int64_t partial = 1;
        for (auto dim : accessDims)
          partial *= analysisTileSize[dim];
        externalAccess += partial;
      }
    }

    int64_t cycles = 1;
    assert(analysisTileSize.size() == unrollFactor.size());
    for (size_t i = 0; i < analysisTileSize.size(); i++) {
      cycles *= (analysisTileSize[i] / unrollFactor[i]);
    }

    statisticDat[*riter]["flops"] = flops;
    statisticDat[*riter]["external_access"] = externalAccess;
    statisticDat[*riter]["cycles"] = cycles;

    for (auto [index, operand] : llvm::enumerate(genericOp.getInputs())) {
      auto op = operand.getDefiningOp();
      if (!cluster.isMember(op))
        continue;
      else {
        int64_t factor = 1;
        auto accessDims = getAffineMapAccessDims(indexingMaps[index]);
        for (size_t i = 0; i < accessDims.size(); i++)
          factor *= analysisTileSize[accessDims[i]] / tilingSize[accessDims[i]];
        if (statisticDat[op]["factor"] == 1)
          statisticDat[op]["factor"] =
              factor * statisticDat[genericOp]["factor"];
      }
    }
  }

  EvaluationMetric metric;

  double_t externalAccess = 0;
  double_t bottleneckCycles = 0;
  double_t flops = 0;
  for (auto [op, namedMetric] : statisticDat) {
    externalAccess += namedMetric["external_access"] * namedMetric["factor"];
    bottleneckCycles = std::max(bottleneckCycles,
                                namedMetric["cycles"] * namedMetric["factor"]);
    flops += namedMetric["flops"] * namedMetric["factor"];
  }
  bottleneckCycles =
      std::max(bottleneckCycles, ceil(externalAccess / cfg.bandwidth));
  double_t throughput = flops / bottleneckCycles;

  metric.throughput = throughput;

  return metric;
}

// ============================
// END OF PerfModel
// ============================

// ============================
// CLASS: BruteForceSolver
// ============================

// inline void BruteForceSolver::assignParameter(ParameterVariant& v,
//                                               ParameterPointerVariant& p) {
//   std::visit(
//       [&](auto&& value, auto&& ptr) {
//         using V = std::decay_t<decltype(value)>;
//         using P = std::decay_t<decltype(ptr)>;
//         if constexpr (std::is_same_v<P, V*>) {
//           *ptr = value;
//         } else {
//           llvm_unreachable("Type mismatch!");
//         }
//       },
//       v, p);
// }

// bool BruteForceSolver::checkConstraint(GenericOpCluster& cluster,
//                                        ArchConfig& archCfg) {
//   return false;
// }

// unsigned int BruteForceSolver::solve(GenericOpCluster& cluster,
//                                      PerfModel& model, ArchConfig& archCfg) {
//   auto parGen = ParameterGenerator();
//   std::vector<ParameterPointerVariant> parameterMap;
//   for (auto [op, namedPar] : cluster.getParameter()) {
//     auto bound = namedPar["loop_bound"];
//     auto tilingSize = namedPar["tiling_size"];

//     assert(bound.size() == tilingSize.size());

//     for (size_t dim = 0; dim < tilingSize.size(); dim++) {
//       parameterMap.push_back(&tilingSize[dim]);
//       std::vector<ParameterVariant> candidate;
//       for (int64_t i = 1; i < bound[dim]; i *= 2) {
//         candidate.push_back(i);
//       }
//       parGen.addVariable(candidate);
//     }

//     auto unrollFactor = namedPar["unroll_factor"];
//     for (size_t dim = 0; dim < unrollFactor.size(); dim++) {
//       parameterMap.push_back(&unrollFactor[dim]);
//       std::vector<ParameterVariant> candidate;
//       for (int64_t i = 1; i < bound[dim]; i *= 2) {
//         candidate.push_back(i);
//       }
//       parGen.addVariable(candidate);
//     }

//     llvm::SmallVector<int64_t> outerOrder;
//     std::iota(outerOrder.begin(), outerOrder.end(), 0);
//     std::vector<ParameterVariant> outerOrderSet;
//     do {
//       outerOrderSet.push_back(outerOrder);
//     } while (std::next_permutation(outerOrder.begin(), outerOrder.end()));
//     parameterMap.push_back(&namedPar["outer_order"]);
//     parGen.addVariable(outerOrderSet);

//     llvm::SmallVector<int64_t> innerOrder;
//     std::iota(innerOrder.begin(), innerOrder.end(), 0);
//     std::vector<ParameterVariant> innerOrderSet;
//     do {
//       innerOrderSet.push_back(innerOrder);
//     } while (std::next_permutation(innerOrder.begin(), innerOrder.end()));
//     parameterMap.push_back(&namedPar["inner_order"]);
//     parGen.addVariable(innerOrderSet);
//   }

//   llvm::errs() << parGen.size() << "\n";
//   for (auto x : parGen.candidates) llvm::errs() << x.size() << " ";
//   llvm::errs() << "\n";

//   int64_t bestThroughput = 0;
//   std::vector<ParameterVariant> parRecord;
//   while (parGen.hasNext()) {
//     parGen.next();
//     // for (auto [par, parPointer] : llvm::zip(parVec, parameterMap)) {
//     //   assignParameter(par, parPointer);
//     // }

//     if (!checkConstraint(cluster, archCfg)) continue;
//     // cluster->evaluate();
//     // auto metric = cluster->getMetric();

//     auto metric = model.evaluate(cluster, archCfg);

//     if (metric.throughput > bestThroughput) {
//       bestThroughput = metric.throughput;
//       // parRecord = parVec;
//     }
//   }

//   return bestThroughput;
// }

// void BruteForceSolver::generateParSet(linalg::GenericOp genericOp) {
//   std::vector<int64_t> order;
//   std::iota(order.begin(), order.end(), 0);
//   std::vector<std::vector<int64_t>> orderSet;
//   do {
//     orderSet.push_back(order);
//   } while (std::next_permutation(order.begin(), order.end()));
// }

// ============================
// END OF BruteForceSolver
// ============================

// ============================
// CLASS: PruningSolver
// ============================

unsigned int PruningSolver::solve(GenericOpCluster &cluster, PerfModel &model,
                                  ArchConfig &archCfg) {
  llvm::errs() << "Start solving, subgraph size : "
               << cluster.getNodeSetTopOrder().size() << "\n================\n";
  for (auto op : cluster.getNodeSetTopOrder())
    op->dump();
  llvm::errs() << "=========================\n";
  std::unordered_map<mlir::Operation *, llvm::SmallVector<int64_t>> curOrder;
  DimensionRelationNetwork curNetwork = cluster.extractDimRelation();
  std::vector<std::unordered_map<mlir::Operation *, llvm::SmallVector<int64_t>>>
      candidateOrder;
  std::vector<DimensionRelationNetwork> candidateNetworks;

  auto iter = cluster.getNodeSetTopOrder().begin();
  auto iterEnd = cluster.getNodeSetTopOrder().end();
  // llvm::errs() << "start solving\n";
  generateCandidateNetworks(iter, iterEnd, cluster, curOrder, curNetwork,
                            candidateOrder, candidateNetworks);

  // llvm::errs() << "finish networks generating, total : "
  //              << candidateNetworks.size() << "\n";
  std::vector<double_t> densityRcd;
  llvm::errs() << candidateNetworks.size() << " Candidates to search\n";
  for (auto [order, network] : llvm::zip(candidateOrder, candidateNetworks)) {
    auto parBounds = network.getUndeterminedParsBound();
    llvm::errs() << parBounds.size() << " Parameters to search\n";
    auto parGen = ParameterGenerator();
    for (auto bd : parBounds) {
      std::vector<int64_t> tiling;
      for (size_t t = 1; t <= bd; t *= 2)
        tiling.push_back(t);
      parGen.addVariable(tiling);
    }

    bool flag = false;
    double_t metric = 0;
    std::vector<int64_t> bestTilingSize;
    while (parGen.hasNext()) {
      auto tilingVec = parGen.next();
      network.setUndeterminedPars(tilingVec);
      if (network.forward())
        flag = true;
      else
        continue;
      cluster.applyOrderTiling(order, network.getDimensionValueMapping());
      double_t density = model.evaluate(cluster, archCfg).computeDensity;
      if (density > metric) {
        metric = density;
        bestTilingSize = tilingVec;
      }
    }
    if (flag) {
      network.setUndeterminedPars(bestTilingSize);
      densityRcd.push_back(metric);
    }
  }

  auto maxE = std::max_element(densityRcd.begin(), densityRcd.end());
  size_t indexMaxDensity = std::distance(densityRcd.begin(), maxE);

  cluster.applyOrderTiling(
      candidateOrder[indexMaxDensity],
      candidateNetworks[indexMaxDensity].getDimensionValueMapping());

  llvm::errs() << "Finish Solving \n";
  return *maxE;
}

void PruningSolver::generateCandidateNetworks(
    std::vector<mlir::Operation *>::iterator curp,
    std::vector<mlir::Operation *>::iterator endp, GenericOpCluster &cluster,
    std::unordered_map<mlir::Operation *, llvm::SmallVector<int64_t>> &curOrder,
    DimensionRelationNetwork &curNetwork,

    std::vector<std::unordered_map<mlir::Operation *,
                                   llvm::SmallVector<int64_t>>> &candidateOrder,
    std::vector<DimensionRelationNetwork> &candidateNetworks) {

  if (curp == endp) {
    candidateOrder.push_back(curOrder);
    candidateNetworks.push_back(curNetwork);
    // llvm::errs() << "push one network\n";
    return;
  }
  assert(cluster.getParameter().find(*curp) != cluster.getParameter().end());

  auto genericOp = mlir::dyn_cast<linalg::GenericOp>(*curp);
  assert(genericOp);
  auto nDim = genericOp.getNumLoops();

  auto orders = generatePermutation(nDim);
  auto masks = generateBitMask(nDim);
  for (auto order : orders) {
    curOrder[*curp] = order;
    for (auto mask : masks) {
      // llvm::interleave(order, llvm::errs(), ", ");
      // llvm::interleave(mask, llvm::errs(), ", ");

      if (!checkOrder(cluster, genericOp, order, mask))
        continue;

      bool flag = true;
      std::vector<Dimension> newConstDims;
      for (auto [index, maskBit] : llvm::enumerate(mask)) {
        if (maskBit) {
          auto td = Dimension(*curp, index);
          // llvm::errs() << curNetwork.isConst(td) << " "
          //              << curNetwork.getValue(td) << " "
          //              << cluster.getParameter()[*curp]["loop_bound"][index]
          //              << "\n";
          if (curNetwork.isConst(td) &&
              curNetwork.getValue(td) !=
                  cluster.getParameter()[*curp]["loop_bound"][index]) {
            flag = false;
            break;
          } else
            newConstDims.push_back(td);
        }
      }

      if (flag) {
        for (auto [index, dim] : llvm::enumerate(newConstDims)) {
          curNetwork.setConstDimension(
              dim, cluster.getParameter()[*curp]["loop_bound"][index]);
        }
      } else
        continue;

      generateCandidateNetworks(std::next(curp), endp, cluster, curOrder,
                                curNetwork, candidateOrder, candidateNetworks);

      for (auto &dim : newConstDims)
        curNetwork.removeConstDimension(dim);
    }
    curOrder.erase(*curp);
  }
}

std::vector<llvm::SmallVector<int64_t>>
PruningSolver::generatePermutation(size_t n) {
  std::vector<llvm::SmallVector<int64_t>> orders;
  llvm::SmallVector<int64_t> order(n);
  std::iota(order.begin(), order.end(), 0);
  do {
    orders.push_back(order);
  } while (std::next_permutation(order.begin(), order.end()));
  return orders;
}

std::vector<llvm::SmallVector<int64_t>>
PruningSolver::generateBitMask(size_t n) {
  std::vector<llvm::SmallVector<int64_t>> result;
  for (size_t mask = 0; mask < (1 << n); ++mask) {
    llvm::SmallVector<int64_t> v(n);
    for (size_t i = 0; i < n; ++i) {
      v[i] = (mask >> i) & 1;
    }
    result.push_back(v);
  }
  return result;
}

bool PruningSolver::checkReuseDistance(llvm::SmallVector<int64_t> &order,
                                       llvm::SmallVector<int64_t> &dims,
                                       llvm::SmallVector<int64_t> &mask) {
  assert(order.size() == mask.size());
  std::unordered_set<int64_t> dimsSet(dims.data(), dims.data() + dims.size());
  for (size_t i = 0; i < order.size(); i++) {
    bool flag = false;
    for (size_t j = i + 1; j < order.size(); j++) {
      if (dimsSet.find(order[j]) != dimsSet.end()) {
        flag = true;
        break;
      }
    }
    if (flag) {
      if (mask[i])
        break;
      for (size_t j = i + 1; j < order.size(); j++) {
        if (!mask[j])
          return false;
      }
    } else
      break;
  }
  return true;
}

bool PruningSolver::checkOrder(GenericOpCluster &cluster, linalg::GenericOp &op,
                               llvm::SmallVector<int64_t> &order,
                               llvm::SmallVector<int64_t> &mask) {
  // Check output reuse distance
  {
    assert(op.getOutputs().size() == 1);
    auto outputDims = getAffineMapAccessDims(op.getIndexingMapsArray().back());
    if (!checkReuseDistance(order, outputDims, mask))
      return false;
  }

  // Check input reuse distance
  for (auto [index, operand] : llvm::enumerate(op.getInputs())) {
    auto producers = getProducerGeneric(operand);
    bool flag = false;
    for (auto p : producers) {
      if (cluster.isMember(p)) {
        flag = true;
        break;
      }
    }
    if (flag) {
      auto inputDims = getAffineMapAccessDims(op.getIndexingMapsArray()[index]);
      if (!checkReuseDistance(order, inputDims, mask))
        return false;
    }
  }

  return true;
}

// ============================
// END OF PruningSolver
// ============================

// ============================
// CLASS: ScheduledGenericOpCluster
// ============================

ScheduledGenericOpCluster::ScheduledGenericOpCluster(llvm::StringRef solver) {
  this->solver = llvm::StringSwitch<ParameterSolvingInterface *>(solver)
                     .Case("brute_force", new PruningSolver())
                     .Case("pruning_brute_force", new PruningSolver())
                     .Default(nullptr);
}

ScheduledGenericOpCluster::~ScheduledGenericOpCluster() {
  for (auto cluster : clusters)
    delete cluster;
  delete solver;
}

std::vector<linalg::GenericOp> ScheduledGenericOpCluster::getTopSortedNodes() {
  std::vector<linalg::GenericOp> topOrderNodes;
  std::unordered_map<mlir::Operation *, unsigned int> inD;
  for (mlir::Operation *op : genericOps)
    inD.insert({op, 0});
  for (mlir::Operation *op : genericOps) {
    for (auto user : op->getUsers()) {
      if (inD.find(user) != inD.end())
        inD[user]++;
    }
  }
  std::queue<mlir::Operation *> nodesWithoutInD;
  for (auto [op, ind] : inD) {
    if (ind == 0) {
      nodesWithoutInD.push(op);
      if (!mlir::dyn_cast<linalg::GenericOp>(op))
        op->dump();
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
      if (inD.find(user) == inD.end())
        continue;
      inD[user]--;
      if (inD[user] == 0) {
        nodesWithoutInD.push(user);
        if (!mlir::dyn_cast<linalg::GenericOp>(user))
          user->dump();
      }
    }
  }
  return topOrderNodes;
}

void ScheduledGenericOpCluster::insertOp(mlir::Operation *op) {
  ops.push_back(op);
}

void ScheduledGenericOpCluster::insertGenericOp(linalg::GenericOp genericOp) {
  genericOps.push_back(genericOp);
}

void ScheduledGenericOpCluster::schedule(mlir::MLIRContext *ctx,
                                         PerfModel &model,
                                         ArchConfig &archCfg) {
  std::vector<mlir::Operation *> opsTopOrder = getTopoOrder(ops);
  std::vector<linalg::GenericOp> genericOpsTopOrder;
  for (auto &op : opsTopOrder) {
    if (auto generic = mlir::dyn_cast<linalg::GenericOp>(op);
        (!op->hasAttr("accelgen.const_broadcast")) && generic)
      genericOpsTopOrder.push_back(generic);
  }

  assert(genericOpsTopOrder.size() >= 1);
  auto dpStatus =
      std::make_unique<unsigned int[]>(genericOpsTopOrder.size() + 1);
  auto cutIndex =
      std::make_unique<unsigned int[]>(genericOpsTopOrder.size() + 1);
  // llvm::errs() << genericOpsTopOrder.size() + 1 << "\n";
  dpStatus[0] = 0;
  cutIndex[0] = 0;
  for (unsigned int i = 1; i <= genericOpsTopOrder.size(); i++) {
    // llvm::errs() << "outer" << i << "\n";
    unsigned int minCost = 0xffffffff;
    unsigned int index = 0;
    for (unsigned j = 1; j <= i; j++) {
      // llvm::errs() << "inner" << j << "\n";
      auto mergedCluster = GenericOpCluster(genericOpsTopOrder.data() + j - 1,
                                            genericOpsTopOrder.data() + i);

      unsigned int mergedCost = solver->solve(mergedCluster, model, archCfg);
      // GenericOpCluster(genericOpsTopOrder.data() + j - 1,
      //                  genericOpsTopOrder.data() + i)
      //     .solveBestSchedule(evaluator, solver);

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
    // cluster->solveBestSchedule(evaluator, solver);
    solver->solve(*cluster, model, archCfg);
    clusters.push_back(cluster);
    if (cutIndex[p] == 1)
      break;
    p = cutIndex[p] - 1;
  }

  for (auto c : clusters)
    c->attachAttribute(ctx);
}

std::vector<mlir::accelgen::GenericOpCluster *>::iterator
ScheduledGenericOpCluster::begin() {
  return clusters.begin();
}

std::vector<mlir::accelgen::GenericOpCluster *>::iterator
ScheduledGenericOpCluster::end() {
  return clusters.end();
}

// ============================
// END OF ScheduledGenericOpCluster
// ============================

} // namespace mlir::accelgen