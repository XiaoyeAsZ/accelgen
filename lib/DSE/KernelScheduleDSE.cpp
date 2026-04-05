#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Parallel.h"
#include <assert.h>
#include <chrono>
#include <iostream>
#include <numeric>
#include <queue>
#include <span>
#include <variant>
#include <vector>
#include "mlir/Dialect/Math/IR/Math.h"

#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>

#include "accelgen/DSE/KernelScheduleDSE.h"
#include "accelgen/Utils/AffineMapUtils.h"
#include "accelgen/Utils/DSEUtils.h"
#include "accelgen/Utils/DebugUtils.h"
#include "accelgen/Utils/OperationUtils.h"

namespace mlir::accelgen {

void ArchConfig::load(llvm::StringRef cfgPath) {
  std::ifstream f(cfgPath.str());
  nlohmann::json config = nlohmann::json::parse(f);

  this->cycle = config["system"]["cycle"];
  this->bandwidth = config["dram"]["bandwidth"];
  this->nSramBank = config["sram"]["sram_bank"];
  this->sramWidth = config["sram"]["sram_width"];
  this->sramDepth = config["sram"]["sram_depth"];
  this->sramCapacity = this->nSramBank * this->sramWidth / 8 * this->sramDepth;

  auto comps = config["compute"];

  for (auto& [key, val] : comps.items()) {
    int64_t value = val["num"].get<int64_t>();
    this->computeResource[key] = value;
  }
}

Parameter::Parameter(int64_t bound)
    : _bound(bound), _valid(false), _const(false) {}
Parameter::Parameter(const Parameter& src)
    : _const(src._const), _value(src._value), _bound(src._bound) {}
void Parameter::addRelation(Relation* rel) { _relations.push_back(rel); }

bool Parameter::set(int64_t value) {
  if ((_valid | _const) && value != _value) return false;
  this->_value = value;
  this->_valid = true;
  return true;
}

bool Parameter::setConst(int64_t value) {
  if (_const && value != _value) return false;
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
void Parameter::inValid() { _valid = false; }

const std::vector<Relation*>& Parameter::relations() { return _relations; }

int64_t Parameter::value() { return _value; }
int64_t Parameter::bound() { return _bound; }

Relation::Relation(Relation::DeduceType type, llvm::ArrayRef<Parameter*> dims)
    : _relType(type) {
  for (auto dim : dims) _deducedParameters.push_back(dim);
}

const std::vector<Parameter*>& Relation::deducedParameters() {
  return _deducedParameters;
};

bool Relation::forward(int64_t value) {
  switch (this->_relType) {
    case DeduceType::EQUAL: {
      for (auto& p : this->_deducedParameters) {
        if (!p->set(value)) return false;
      }
    } break;
    case DeduceType::COLLAPSE: {
      auto factor = value;
      auto iter = _deducedParameters.rbegin();
      for (; iter != _deducedParameters.rend(); iter++) {
        auto d = factor / (*iter)->bound();
        auto r = factor % (*iter)->bound();
        if (d > 0) {
          if (r != 0) return false;
          factor = d;
          if (!((*iter)->set((*iter)->bound()))) return false;
        } else if (d == 0) {
          if (!((*iter)->set(r))) return false;
          iter++;
          break;
        } else
          assert(0);
      }
      for (; iter != _deducedParameters.rend(); iter++) {
        if (!((*iter)->set(1))) return false;
      }
    } break;
    default:
      assert(0);
      break;
  }
  return true;
}
Relation::Relation(const Relation& src) : _relType(src._relType) {}
Relation::DeduceType Relation::type() { return _relType; }

// ============================
// CLASS: DimensionRelationNetwork
// ============================

DimensionRelationNetwork::~DimensionRelationNetwork() {
  for (auto r : rels) delete r;
  for (auto p : pars) delete p;
}

DimensionRelationNetwork::DimensionRelationNetwork(
    const DimensionRelationNetwork& src) {
  std::unordered_map<Parameter*, Parameter*> parMap;
  for (auto p : src.pars) parMap[p] = new Parameter(*p);
  for (auto& [op, np] : parMap) {
    for (auto& orel : op->relations()) {
      std::vector<Parameter*> npars;
      for (auto p : orel->deducedParameters()) npars.push_back(parMap[p]);
      auto nrel = new Relation(orel->type(), npars);
      np->addRelation(nrel);
      rels.push_back(nrel);
    }
  }
  for (auto& [d, p] : src.dimMapping) this->dimMapping[d] = parMap[p];
  for (auto& op : src.pars) this->pars.push_back(parMap[op]);
  for (auto& [p, id] : src._indPars) this->_indPars[parMap[p]] = id;
}

bool DimensionRelationNetwork::addDimension(const Dimension& dim,
                                            int64_t bound) {
  if (dimMapping.find(dim) != dimMapping.end()) return false;
  auto par = new Parameter(bound);
  pars.push_back(par);
  dimMapping[dim] = par;
  _indPars[par] = 0;
  return true;
}

bool DimensionRelationNetwork::addEqualRelation(const Dimension& dimSrc,
                                                const Dimension& dimTarget) {
  if (dimMapping.find(dimSrc) == dimMapping.end() ||
      dimMapping.find(dimTarget) == dimMapping.end())
    assert(0);
  auto rel = new Relation(Relation::DeduceType::EQUAL, {dimMapping[dimTarget]});
  _indPars[dimMapping[dimTarget]]++;
  dimMapping[dimSrc]->addRelation(rel);
  rels.push_back(rel);
  return true;
}

bool DimensionRelationNetwork::addCollapseRelation(
    const Dimension& dimSrc, llvm::ArrayRef<Dimension> dimTarget) {
  if (dimMapping.find(dimSrc) == dimMapping.end()) assert(0);
  std::vector<Parameter*> pars;
  for (auto dim : dimTarget) {
    if (dimMapping.find(dim) == dimMapping.end()) assert(0);
    pars.push_back(dimMapping[dim]);
    _indPars[dimMapping[dim]]++;
  }
  auto rel = new Relation(Relation::DeduceType::COLLAPSE, pars);
  dimMapping[dimSrc]->addRelation(rel);
  rels.push_back(rel);
  return true;
}

bool DimensionRelationNetwork::setConstDimension(const Dimension& dim,
                                                 int64_t value) {
  if (dimMapping.find(dim) == dimMapping.end()) return false;
  dimMapping[dim]->set(value);
  return true;
}

bool DimensionRelationNetwork::forward() {
  llvm::MapVector<Parameter*, size_t> ind(_indPars);
  std::queue<Parameter*> q;
  for (auto [p, d] : ind) {
    if (d == 0) {
      q.push(p);
    }
  }
  while (!q.empty()) {
    auto p = q.front();
    q.pop();
    assert(p->valid());
    for (auto dr : p->relations()) {
      if (!dr->forward(p->value())) return false;
      for (auto dp : dr->deducedParameters()) {
        ind[dp]--;
        if (ind[dp] == 0) {
          q.push(dp);
        }
      }
    }
  }

  return true;
}

int64_t DimensionRelationNetwork::getValue(const Dimension& dim) {
  assert(dimMapping.find(dim) != dimMapping.end());
  assert(dimMapping[dim]->valid());
  return dimMapping[dim]->value();
}

std::vector<int64_t> DimensionRelationNetwork::getUndeterminedParsBound() {
  // std::vector<int64_t> bounds;
  // for (auto &[p, d] : _indPars) {
  //   if (d == 0 && (!p->isConst()))
  //     bounds.push_back(p->bound());
  // }
  // return bounds;
  std::vector<int64_t> bounds;
  for (auto& p : pars) {
    if (_indPars[p] == 0 && (!p->isConst())) bounds.push_back(p->bound());
    // else {
    //   ECHO("ind == 0", "\n")
    //   ECHO(_indPars[p], "\n")
    //   ECHO("p const", "\n")
    //   ECHO(p->isConst(), "\n")
    // }
  }
  return bounds;
}

bool DimensionRelationNetwork::isConst(const Dimension& dim) {
  assert(dimMapping.find(dim) != dimMapping.end());
  return dimMapping[dim]->isConst();
}

bool DimensionRelationNetwork::setUndeterminedPars(
    const std::vector<int64_t>& undeterminedPars) {
  clearUndeterminedPars();

  // for (auto xx : undeterminedPars) {
  //   llvm::errs() << xx << ",";
  // }
  // llvm::errs() << "\n";
  // std::vector<Parameter *> ups;
  // for (auto &[p, d] : _indPars) {
  //   if (d == 0 && (!p->isConst()))
  //     ups.push_back(p);
  // }
  // for (auto [p, v] : llvm::zip(ups, undeterminedPars)) {
  //   assert(p->set(v));
  std::vector<Parameter*> ups;
  for (auto& p : pars) {
    if (_indPars[p] == 0 && (!p->isConst())) ups.push_back(p);
  }
  assert(ups.size() == undeterminedPars.size());
  for (auto [p, v] : llvm::zip(ups, undeterminedPars)) {
    assert(p->set(v));
  }
}

void DimensionRelationNetwork::clearUndeterminedPars() {
  for (auto& p : pars) {
    if (!p->isConst()) p->inValid();
  }
}

const std::unordered_map<Dimension, int64_t, Dimension::Hash>&
DimensionRelationNetwork::getDimensionValueMapping() {
  dimValue.clear();
  for (auto& [d, p] : dimMapping) {
    dimValue[d] = p->value();
  }

  return dimValue;
}

bool DimensionRelationNetwork::removeConstDimension(const Dimension& dim) {
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
GenericOpCluster::GenericOpCluster(linalg::GenericOp* genericOpStart,
                                   linalg::GenericOp* genericOpEnd)
    : nodeSet(genericOpStart, genericOpEnd),
      nodeSetTopOrder(genericOpStart, genericOpEnd) {
  nodeSetTopOrder = this->getTopoOrderALSP();
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

// GenericOpCluster::GenericOpCluster(const GenericOpCluster &src){

// }

bool GenericOpCluster::isMember(mlir::Operation* opToCehck) {
  return nodeSet.find(opToCehck) != nodeSet.end();
}

void GenericOpCluster::attachAttribute(mlir::MLIRContext* ctx) {
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

std::unordered_set<mlir::Operation*>::iterator GenericOpCluster::begin() {
  return nodeSet.begin();
}
std::unordered_set<mlir::Operation*>::iterator GenericOpCluster::end() {
  return nodeSet.end();
}

std::vector<mlir::Operation*>& GenericOpCluster::getNodeSetTopOrder() {
  return nodeSetTopOrder;
}
std::unordered_map<mlir::Operation*,
                   std::unordered_map<std::string, llvm::SmallVector<int64_t>>>&
GenericOpCluster::getParameter() {
  return parameter;
}

void GenericOpCluster::clearParameter() {
  for (auto [op, par] : parameter) par.clear();
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
    // ECHO("now extracting", "\n")
    // op->dump();
    auto genericOp = mlir::dyn_cast<linalg::GenericOp>(op);
    assert(genericOp);
    for (auto [indexOperand, operand] :
         llvm::enumerate(genericOp.getInputs())) {
      // ECHO("operand:", "\n")
      // ECHO(indexOperand, "\n")
      auto accessDims = getAffineMapAccessDims(
          genericOp.getIndexingMapsArray()[indexOperand]);

      // Check whether this operand is DRAM access and used by other cluster
      // member, if so make sure same tiling size
      llvm::SmallVector<linalg::GenericOp> previousGeneric;
      getPreviousGeneric(operand, previousGeneric);
      if (previousGeneric.empty()) {
        for (auto use : operand.getUsers()) {
          // ECHO("check user", "\n")
          // use->dump();

          // Temporal fix, avoid deadlock
          // TODO : add suport for processing affinemap 0 broadcast
          bool hasDeadlock = false;
          for (auto ins : genericOp.getInputs()) {
            llvm::SmallVector<linalg::GenericOp> producers;
            getPreviousGeneric(ins, producers);
            for (auto p : producers) {
              if (p == use) {
                hasDeadlock = true;
                break;
              }
            }
          }
          if (hasDeadlock) continue;

          if (isMember(use) &&
              (std::distance(nodeSetTopOrder.begin(),
                             std::find(nodeSetTopOrder.begin(),
                                       nodeSetTopOrder.end(), use)) <
               std::distance(nodeSetTopOrder.begin(),
                             std::find(nodeSetTopOrder.begin(),
                                       nodeSetTopOrder.end(), op)))) {
            auto genericSharedOp = mlir::dyn_cast<linalg::GenericOp>(use);
            assert(genericSharedOp);
            auto insSharedOp = genericSharedOp.getInputs();
            auto it =
                std::find(insSharedOp.begin(), insSharedOp.end(), operand);
            assert(it != insSharedOp.end());
            auto indexOperandSharedOp = std::distance(insSharedOp.begin(), it);
            auto accessDimsSharedOp = getAffineMapAccessDims(
                genericSharedOp.getIndexingMapsArray()[indexOperandSharedOp]);

            for (auto [src, target] :
                 llvm::zip(accessDims, accessDimsSharedOp)) {
              // ECHO("add equal rel for shapred input", "\n")
              network.addEqualRelation(Dimension(op, src),
                                       Dimension(use, target));
            }
          }
        }
      }

      if (!operand.getDefiningOp()) continue;
      mlir::TypeSwitch<mlir::Operation*>(operand.getDefiningOp())
          .Case<linalg::GenericOp>([&](linalg::GenericOp generic) {
            if (isMember(generic)) {
              auto accessDimsProducer =
                  getAffineMapAccessDims(generic.getIndexingMapsArray().back());
              for (auto [src, target] :
                   llvm::zip(accessDims, accessDimsProducer)) {
                // ECHO("add equal rel for producer consumer", "\n")
                // assert(0 && "wrong src -> target");
                network.addEqualRelation(Dimension(op, src),
                                         Dimension(generic, target));
              }
            }
          })
          .Case<tensor::CollapseShapeOp>([&](tensor::CollapseShapeOp collapse) {
            if (collapse.getSrc().getDefiningOp() == nullptr) return;

            // auto producer = mlir::dyn_cast<linalg::GenericOp>(
            //     collapse.getSrc().getDefiningOp());
            // assert(producer);

            if (auto producerGeneric = mlir::dyn_cast<linalg::GenericOp>(
                    collapse.getSrc().getDefiningOp())) {
              if (isMember(producerGeneric)) {
                auto reassociationMap = collapse.getReassociationMaps();
                auto accessDimsProducer = getAffineMapAccessDims(
                    producerGeneric.getIndexingMapsArray().back());

                for (size_t d = 0; d < reassociationMap.size(); d++) {
                  auto collapseDims =
                      getAffineMapAccessDims(reassociationMap[d]);
                  if (collapseDims.size() > 1) {
                    std::vector<Dimension> dimPtr;
                    for (auto cd : collapseDims)
                      dimPtr.push_back(
                          Dimension(producerGeneric, accessDimsProducer[cd]));
                    network.addCollapseRelation(Dimension(op, accessDims[d]),
                                                dimPtr);
                  } else {
                    assert(collapseDims.size() == 1);
                    network.addEqualRelation(
                        Dimension(producerGeneric,
                                  accessDimsProducer[collapseDims[0]]),
                        Dimension(op, accessDims[d]));
                  }
                }
              }
            } else if (auto producerConcat = mlir::dyn_cast<tensor::ConcatOp>(
                           collapse.getSrc().getDefiningOp())) {
              for (auto insConcat : producerConcat.getInputs()) {
                auto producerGeneric =
                    insConcat.getDefiningOp<linalg::GenericOp>();
                if (isMember(producerGeneric)) {
                  auto concatDim = producerConcat.getDim();

                  assert(producerGeneric.getResults().size() == 1);
                  auto producerAccDims = getAffineMapAccessDims(
                      producerGeneric.getIndexingMapsArray()
                          [producerGeneric.getInputs().size()]);
                  auto consumerAccDims = getAffineMapAccessDims(
                      genericOp.getIndexingMapsArray()[indexOperand]);

                  auto reassMaps = collapse.getReassociationMaps();

                  int64_t tmp = 0;
                  int64_t rcd = 0;
                  for (auto [indexReassMap, itemReassMap] :
                       llvm::enumerate(reassMaps)) {
                    rcd = indexReassMap;
                    if (tmp >= concatDim) break;
                    auto collapseDims = getAffineMapAccessDims(itemReassMap);
                    llvm::SmallVector<Dimension> dimPtr;
                    for (auto d : collapseDims) {
                      dimPtr.push_back(Dimension(producerGeneric, d));
                    }
                    network.addCollapseRelation(
                        Dimension(genericOp, consumerAccDims[indexReassMap]),
                        dimPtr);
                    tmp += collapseDims.size();
                  }

                  for (int64_t d = concatDim;
                       d < producerConcat.getResultType().getShape().size();
                       d++) {
                    network.setConstDimension(
                        Dimension(producerGeneric, d),
                        parameter[producerGeneric]["loop_bound"][d]);
                  }
                  for (int64_t d = rcd;
                       d < collapse.getResultType().getShape().size(); d++) {
                    network.setConstDimension(
                        Dimension(genericOp, consumerAccDims[d]),
                        parameter[genericOp]["loop_bound"][consumerAccDims[d]]);
                  }
                }
              }
            } else
              assert(0);
          })
          .Case<tensor::ExpandShapeOp>([&](tensor::ExpandShapeOp expand) {
            if (expand.getSrc().getDefiningOp() == nullptr) return;
            auto producer = mlir::dyn_cast<linalg::GenericOp>(
                expand.getSrc().getDefiningOp());
            assert(producer);

            if (isMember(producer)) {
              auto reassociationMap = expand.getReassociationMaps();
              auto accessDimsProducer = getAffineMapAccessDims(
                  producer.getIndexingMapsArray().back());

              for (size_t d = 0; d < reassociationMap.size(); d++) {
                auto expandDims = getAffineMapAccessDims(reassociationMap[d]);
                if (expandDims.size() > 1) {
                  std::vector<Dimension> dimPtr;
                  for (auto d : expandDims)
                    dimPtr.push_back(Dimension(op, accessDims[d]));
                  // ECHO("add collapse rel", "\n")
                  network.addCollapseRelation(
                      Dimension(producer, accessDimsProducer[d]), dimPtr);

                } else {
                  assert(expandDims.size() == 1);
                  // ECHO("add equal rel for producer consumer", "\n")
                  network.addEqualRelation(
                      Dimension(producer, accessDimsProducer[d]),
                      Dimension(op, accessDims[expandDims[0]]));
                }
              }
            }
          })
          .Case<tensor::ExtractSliceOp>([&](tensor::ExtractSliceOp extract) {
            auto producer =
                extract.getSource().getDefiningOp<linalg::GenericOp>();
            assert(producer);

            if (isMember(producer)) {
              auto srcShape =
                  mlir::dyn_cast<RankedTensorType>(extract.getSourceType())
                      .getShape();
              auto sliceShape =
                  mlir::dyn_cast<RankedTensorType>(extract.getType())
                      .getShape();
              size_t firstExtractDim;
              for (size_t di = 0; di < srcShape.size(); di++) {
                if (srcShape[di] != sliceShape[di]) {
                  firstExtractDim = di;
                  break;
                }
              }
              assert(producer.getResults().size() == 1);
              auto producerAccessDims = getAffineMapAccessDims(
                  producer.getIndexingMapsArray().back());

              std::vector<int64_t> equalDims;
              std::vector<int64_t> constDims;
              for (size_t di = 0; di < srcShape.size(); di++) {
                if (di < firstExtractDim) {
                  network.addEqualRelation(
                      Dimension(producer, producerAccessDims[di]),
                      Dimension(op, accessDims[di]));
                } else {
                  network.setConstDimension(
                      Dimension(producer, producerAccessDims[di]),
                      parameter[producer]["loop_bound"]
                               [producerAccessDims[di]]);
                  network.setConstDimension(
                      Dimension(op, accessDims[di]),
                      parameter[op]["loop_bound"][accessDims[di]]);
                }
              }
            }
          })
          .Case<tensor::ConcatOp>([&](tensor::ConcatOp concat) {
            auto concatDim = concat.getDim();
            auto concatShape =
                mlir::dyn_cast<RankedTensorType>(concat.getType()).getShape();

            for (auto [idx, prevOperand] :
                 llvm::enumerate(concat.getInputs())) {
              if (auto prevOp =
                      prevOperand.getDefiningOp<linalg::GenericOp>()) {
                assert(prevOp);
                if (isMember(prevOp)) {
                  assert(prevOp.getResults().size() == 1);
                  auto producerAccessDims = getAffineMapAccessDims(
                      prevOp.getIndexingMapsArray().back());

                  for (size_t di = 0; di < concatShape.size(); di++) {
                    if (di < concatDim) {
                      network.addEqualRelation(
                          Dimension(prevOp, producerAccessDims[di]),
                          Dimension(op, accessDims[di]));
                    } else {
                      network.setConstDimension(
                          Dimension(prevOp, producerAccessDims[di]),
                          parameter[prevOp]["loop_bound"]
                                   [producerAccessDims[di]]);
                      network.setConstDimension(
                          Dimension(op, accessDims[di]),
                          parameter[op]["loop_bound"][accessDims[di]]);
                    }
                  }
                }
              } else if (auto prevOp =
                             prevOperand
                                 .getDefiningOp<tensor::ExtractSliceOp>()) {
                auto prevGeneric =
                    prevOp.getSource().getDefiningOp<linalg::GenericOp>();
                assert(prevGeneric);

                if (isMember(prevGeneric)) {
                  auto srcShape =
                      mlir::dyn_cast<RankedTensorType>(prevOp.getSourceType())
                          .getShape();
                  auto sliceShape =
                      mlir::dyn_cast<RankedTensorType>(prevOp.getType())
                          .getShape();
                  size_t firstExtractDim;
                  for (size_t di = 0; di < srcShape.size(); di++) {
                    if (srcShape[di] != sliceShape[di]) {
                      firstExtractDim = di;
                      break;
                    }
                  }
                  auto firstConstDim = std::min(firstExtractDim, concatDim);
                  auto producerAccessDims = getAffineMapAccessDims(
                      prevGeneric.getIndexingMapsArray().back());

                  for (size_t di = 0; di < concatShape.size(); di++) {
                    if (di < firstConstDim) {
                      network.addEqualRelation(
                          Dimension(prevGeneric, producerAccessDims[di]),
                          Dimension(op, accessDims[di]));
                    } else {
                      network.setConstDimension(
                          Dimension(prevGeneric, producerAccessDims[di]),
                          parameter[prevGeneric]["loop_bound"]
                                   [producerAccessDims[di]]);
                      network.setConstDimension(
                          Dimension(op, accessDims[di]),
                          parameter[op]["loop_bound"][accessDims[di]]);
                    }
                  }
                }
              } else if (prevOperand.getDefiningOp() == nullptr) {
              } else
                assert(0);
            }
          })
          .Case<arith::ConstantOp>([&](arith::ConstantOp constant) {})
          .Default([](mlir::Operation* placeholder) {
            placeholder->dump();
            assert(0);
          });
    }
  }
  return network;
}

void GenericOpCluster::applyTiling(
    const std::unordered_map<Dimension, int64_t, Dimension::Hash>& tiling) {
  for (auto& [op, par] : parameter) {
    for (size_t d = 0; d < par["tiling_size"].size(); d++) {
      par["tiling_size"][d] = tiling.at(Dimension(op, d));
    }
  }
}

void GenericOpCluster::applyOrder(
    const std::unordered_map<Operation*, llvm::SmallVector<int64_t>>& order) {
  // for (auto& [op, par] : parameter) {
  //   assert(order.find(op) != order.end());
  //   par["outer_order"] = order.at(op);
  // }
  for (auto& [op, ord] : order) {
    assert(parameter.find(op) != parameter.end());
    parameter[op]["outer_order"] = ord;
  }
}

// bool GenericOpCluster::checkOrder() {
//   for (auto &op : nodeSetTopOrder) {
//     auto generic = mlir::dyn_cast<linalg::GenericOp>(op);
//     assert(generic);

//     auto order = parameter[op]["outer_order"];
//     auto tiling = parameter[op]["tiling_size"];
//     auto bound = parameter[op]["loop_bound"];

//     llvm::SmallVector<int64_t> mask;
//     for (auto [t, b] : llvm::zip(tiling, bound)) {
//       if (t == b)
//         mask.push_back(1);
//       else
//         mask.push_back(0);
//     }

//     // Check output reuse distance
//     assert(generic.getOutputs().size() == 1);
//     auto outputDims =
//         getAffineMapAccessDims(generic.getIndexingMapsArray().back());
//     if (!checkReuseDistance(order, outputDims, mask)) {
//       // llvm::errs() << "distance fail\n";
//       return false;
//     }

//     // Check input reuse distance
//     for (auto [index, operand] : llvm::enumerate(generic.getInputs())) {
//       auto producers = getProducerGeneric(operand);
//       bool isProducerConsumerRel = false;
//       for (auto p : producers) {
//         if (isMember(p)) {
//           isProducerConsumerRel = true;
//           break;
//         }
//       }
//       bool isSharedInputRel = false;
//       std::vector<mlir::Operation *> opSharedInputs;
//       for (auto use : operand.getUsers()) {
//         if (isMember(use) && (use != op)) {
//           isSharedInputRel = true;
//           opSharedInputs.push_back(use);
//         }
//       }
//       assert(!(isProducerConsumerRel && isSharedInputRel));
//       if (isProducerConsumerRel) {
//         auto inputDims =
//             getAffineMapAccessDims(generic.getIndexingMapsArray()[index]);
//         if (!checkReuseDistance(order, inputDims, mask)) {
//           // llvm::errs() << "pc fail\n";
//           return false;
//         }
//       } else if (isSharedInputRel) {
//         for (auto sg : opSharedInputs) {
//           auto sharedGeneric = mlir::dyn_cast<linalg::GenericOp>(sg);
//           assert(sg);
//           if (parameter[sg]["outer_order"] != order) {
//             // llvm::errs() << "so fail\n";
//             return false;
//           }
//         }
//       }
//     }
//   }
//   return true;
// }

bool GenericOpCluster::checkOrder() {
  for (auto& op : nodeSetTopOrder) {
    auto generic = mlir::dyn_cast<linalg::GenericOp>(op);
    assert(generic);

    auto order = parameter[op]["outer_order"];
    auto tiling = parameter[op]["tiling_size"];
    auto bound = parameter[op]["loop_bound"];

    llvm::SmallVector<int64_t> mask;
    for (auto [t, b] : llvm::zip(tiling, bound)) {
      if (t == b)
        mask.push_back(1);
      else
        mask.push_back(0);
    }

    // Check input reuse distance
    for (auto [index, operand] : llvm::enumerate(generic.getInputs())) {
      auto producers = getProducerGeneric(operand);
      bool isSharedInputRel = false;
      std::vector<mlir::Operation*> opSharedInputs;
      for (auto use : operand.getUsers()) {
        if (isMember(use) && (use != op)) {
          isSharedInputRel = true;
          opSharedInputs.push_back(use);
        }
      }
      for (auto sg : opSharedInputs) {
        auto sharedGeneric = mlir::dyn_cast<linalg::GenericOp>(sg);
        assert(sg);
        if (parameter[sg]["outer_order"] != order) {
          // llvm::errs() << "so fail\n";
          return false;
        }
      }
    }
  }
  return true;
}

bool GenericOpCluster::checkOpOrder(mlir::Operation* op) {
  auto generic = mlir::dyn_cast<linalg::GenericOp>(op);
  assert(generic);

  auto tiling = parameter[op]["tiling_size"];
  auto bound = parameter[op]["loop_bound"];
  auto order = parameter[op]["outer_order"];

  llvm::SmallVector<int64_t> mask;
  for (auto [t, b] : llvm::zip(tiling, bound)) {
    if (t == b)
      mask.push_back(1);
    else
      mask.push_back(0);
  }
  // ECHO("checkone", "\n")
  // ECHO_LIST(tiling, ",")
  // ECHO_LIST(order, ",")

  // Check output reuse distance
  assert(generic.getOutputs().size() == 1);
  auto outputDims =
      getAffineMapAccessDims(generic.getIndexingMapsArray().back());
  if (!checkReuseDistance(order, outputDims, mask)) {
    // llvm::errs() << "distance fail\n";
    return false;
  }

  // ECHO("out finish", "\n")

  // Check input reuse distance
  for (auto [index, operand] : llvm::enumerate(generic.getInputs())) {
    // Producer-Consumer order relation
    if (auto producer = operand.getDefiningOp<linalg::GenericOp>()) {
      if (isMember(producer)) {
        // Check : continous access
        // ECHO("continous", "\n")
        auto inputDims =
            getAffineMapAccessDims(generic.getIndexingMapsArray()[index]);
        if (!checkReuseDistance(order, inputDims, mask)) {
          return false;
        }
        // Check : the same access order
        // ECHO("same order", "\n")
        std::vector<int64_t> operandDimsOrder =
            getDimsInOrder(generic, operand);
        std::vector<int64_t> producerOperandDimsOrder =
            getDimsInOrder(producer, operand);
        // ECHO_LIST(operandDimsOrder, ",")
        // ECHO_LIST(producerOperandDimsOrder, ",")
        if (operandDimsOrder != producerOperandDimsOrder) return false;
      }
    } else if (auto producer =
                   operand.getDefiningOp<tensor::CollapseShapeOp>()) {
      if (producer.getSrc().getDefiningOp() == nullptr) continue;

      // Check : the same access order
      if (auto producerGeneric =
              producer.getSrc().getDefiningOp<linalg::GenericOp>()) {
        if (isMember(producerGeneric)) {
          // Check : continous access
          auto inputDims =
              getAffineMapAccessDims(generic.getIndexingMapsArray()[index]);
          if (!checkReuseDistance(order, inputDims, mask)) {
            return false;
          }

          std::vector<int64_t> operandDimsOrder =
              getDimsInOrder(generic, operand);
          std::vector<int64_t> producerOperandDimsOrder =
              getDimsInOrder(producerGeneric, producer.getSrc());
          std::vector<int64_t> operandDimsOrderExpand;
          for (auto d : operandDimsOrder) {
            for (auto dd :
                 getAffineMapAccessDims(producer.getReassociationMaps()[d])) {
              operandDimsOrderExpand.push_back(dd);
            }
          }
          if (operandDimsOrderExpand != producerOperandDimsOrder) return false;
        }
      } else if (auto producerConcat =
                     producer.getSrc().getDefiningOp<tensor::ConcatOp>()) {
        for (auto [indexIns, itemIns] :
             llvm::enumerate(producerConcat.getInputs())) {
          auto producerGeneric = itemIns.getDefiningOp<linalg::GenericOp>();
          if (isMember(producerGeneric)) {
            std::vector<int64_t> operandDimsOrder =
                getDimsInOrder(generic, operand);
            std::vector<int64_t> producerOperandDimsOrder =
                getDimsInOrder(producerGeneric, itemIns);
            std::vector<int64_t> operandDimsOrderExpand;
            for (auto d : operandDimsOrder) {
              for (auto dd :
                   getAffineMapAccessDims(producer.getReassociationMaps()[d])) {
                operandDimsOrderExpand.push_back(dd);
              }
            }
            if (operandDimsOrderExpand != producerOperandDimsOrder)
              return false;
          }
        }
      } else
        assert(0);

    } else if (auto producer = operand.getDefiningOp<tensor::ExpandShapeOp>()) {
      if (producer.getSrc().getDefiningOp() == nullptr) continue;
      auto producerGeneric =
          producer.getSrc().getDefiningOp<linalg::GenericOp>();
      assert(producerGeneric);
      if (isMember(producerGeneric)) {
        // Check : continous access
        auto inputDims =
            getAffineMapAccessDims(generic.getIndexingMapsArray()[index]);
        if (!checkReuseDistance(order, inputDims, mask)) {
          return false;
        }
        // Check : the same access order
        std::vector<int64_t> operandDimsOrder =
            getDimsInOrder(generic, operand);
        std::vector<int64_t> producerOperandDimsOrder =
            getDimsInOrder(producerGeneric, producer.getSrc());
        std::vector<int64_t> producerOperandDimsOrderExpand;
        for (auto d : producerOperandDimsOrder) {
          for (auto dd :
               getAffineMapAccessDims(producer.getReassociationMaps()[d])) {
            producerOperandDimsOrderExpand.push_back(dd);
          }
        }
        if (producerOperandDimsOrderExpand != operandDimsOrder) return false;
      }

    } else if (auto producer = operand.getDefiningOp<tensor::ConcatOp>()) {
      auto producerOperands = producer.getInputs();
      assert(producerOperands.size() == 2);
      for (auto proOperand : producerOperands) {
        if (auto producerGeneric =
                proOperand.getDefiningOp<linalg::GenericOp>()) {
          if (isMember(producerGeneric)) {
            // Check : continous access
            auto inputDims =
                getAffineMapAccessDims(generic.getIndexingMapsArray()[index]);
            if (!checkReuseDistance(order, inputDims, mask)) {
              return false;
            }
            // Check : the same access order
            std::vector<int64_t> operandDimsOrder =
                getDimsInOrder(generic, operand);
            std::vector<int64_t> producerOperandDimsOrder =
                getDimsInOrder(producerGeneric, proOperand);
            if (operandDimsOrder != producerOperandDimsOrder) return false;
          }
        } else if (auto prevOp =
                       proOperand.getDefiningOp<tensor::ExtractSliceOp>()) {
          auto producerGeneric =
              prevOp.getSource().getDefiningOp<linalg::GenericOp>();
          assert(producerGeneric);
          if (isMember(producerGeneric)) {
            // Check : continous access
            auto inputDims =
                getAffineMapAccessDims(generic.getIndexingMapsArray()[index]);
            if (!checkReuseDistance(order, inputDims, mask)) {
              return false;
            }
            // Check : the same access order
            std::vector<int64_t> operandDimsOrder =
                getDimsInOrder(generic, operand);
            std::vector<int64_t> producerOperandDimsOrder =
                getDimsInOrder(producerGeneric, prevOp.getSource());
            if (operandDimsOrder != producerOperandDimsOrder) return false;
          }
        }
      }
    } else if (auto producer =
                   operand.getDefiningOp<tensor::ExtractSliceOp>()) {
      auto producerGeneric =
          producer.getSource().getDefiningOp<linalg::GenericOp>();
      assert(producerGeneric);
      if (isMember(producerGeneric)) {
        // Check : continous access
        auto inputDims =
            getAffineMapAccessDims(generic.getIndexingMapsArray()[index]);
        if (!checkReuseDistance(order, inputDims, mask)) {
          return false;
        }
        // Check : the same access order
        std::vector<int64_t> operandDimsOrder =
            getDimsInOrder(generic, operand);
        std::vector<int64_t> producerOperandDimsOrder =
            getDimsInOrder(producerGeneric, producer.getSource());
        if (operandDimsOrder != producerOperandDimsOrder) return false;
      }
    } else if (auto producer = operand.getDefiningOp() == nullptr) {
      continue;
    } else
      assert(0);
  }

  // ECHO("input finish", "\n")

  // TODO : Shared input relation
  for (auto [index, operand] : llvm::enumerate(generic.getInputs())) {
    for (auto use : operand.getUsers()) {
      if (isMember(use) &&
          std::distance(
              nodeSetTopOrder.begin(),
              std::find(nodeSetTopOrder.begin(), nodeSetTopOrder.end(), use)) <
              std::distance(nodeSetTopOrder.begin(),
                            std::find(nodeSetTopOrder.begin(),
                                      nodeSetTopOrder.end(), op))) {
        // ECHO("here", "\n")
        auto sharedGeneric = mlir::dyn_cast<linalg::GenericOp>(use);
        assert(sharedGeneric);
        // Check : the same access order
        // ECHO("ck", "\n")
        // ECHO_LIST(parameter[generic]["outer_order"], ",")
        // ECHO_LIST(parameter[sharedGeneric]["outer_order"], ",")

        auto genericDims = parameter[generic]["outer_order"];
        assert(sharedGeneric.getResults().size() == 1);
        auto sharedGenericDims = parameter[sharedGeneric]["outer_order"];

        // for (auto [d, m] : llvm::enumerate(mask)) {
        //   if (m) {
        //     genericDims.erase(
        //         std::remove(genericDims.begin(), genericDims.end(), d),
        //         genericDims.end());
        //     sharedGenericDims.erase(std::remove(sharedGenericDims.begin(),
        //                                         sharedGenericDims.end(), d),
        //                             sharedGenericDims.end());
        //   }
        // }

        if (genericDims != sharedGenericDims) {
          // if (flagxxx) {
          //   ECHO_LIST(order0, ",")
          //   ECHO_LIST(order1, ",")
          //   ECHO_LIST(operandDimsOrder, ",")
          //   ECHO_LIST(sharedOperandDimsOrder, ",")
          // }
          return false;

        } else {
          // generic.dump();
          // sharedGeneric.dump();
          // ECHO_LIST(parameter[generic]["outer_order"], ",")
          // ECHO_LIST(parameter[sharedGeneric]["outer_order"], ",")
          // ECHO_LIST(operandDimsOrder, ",")
          // ECHO_LIST(sharedOperandDimsOrder, ",")
        }
      }
    }
  }

  // ECHO("true", "\n")

  return true;
}

bool GenericOpCluster::checkReuseDistance(
    const llvm::SmallVector<int64_t>& order,
    const llvm::SmallVector<int64_t>& dims,
    const llvm::SmallVector<int64_t>& mask) {
  assert(order.size() == mask.size());
  // llvm::interleave(order, llvm::errs(), ",");
  // llvm::errs() << "\n";
  // llvm::interleave(dims, llvm::errs(), ",");
  // llvm::errs() << "\n";
  // llvm::interleave(mask, llvm::errs(), ",");
  // llvm::errs() << "\n";
  std::unordered_set<int64_t> dimsSet(dims.data(), dims.data() + dims.size());

  // If one dim tiling=bound, free constraint
  for (auto d : dims) {
    if (mask[d]) dimsSet.erase(d);
  }
  // ECHO_LIST(dimsSet, ",")
  for (size_t i = 0; i < order.size(); i++) {
    // ECHO("checking ", "\n")
    // ECHO(i, "\n")
    if (dimsSet.find(order[i]) != dimsSet.end()) continue;
    bool flag = false;
    for (size_t j = i + 1; j < order.size(); j++) {
      if (dimsSet.find(order[j]) != dimsSet.end()) {
        flag = true;
        break;
      }
    }
    // ECHO(flag, "\n")
    if (flag) {
      if (mask[order[i]])
        continue;
      else
        return false;
      // for (size_t j = i + 1; j < order.size(); j++) {
      //   if (!mask[j])
      //     return false;
      // }
    } else
      break;
  }
  return true;
}

bool GenericOpCluster::checkArchConstraint(const ArchConfig& cfg) {
  // #Check : Sram capacity
  size_t sramOcc = 0;
  llvm::DenseSet<mlir::Value> usedValues;
  for (auto op : nodeSetTopOrder) {
    auto generic = mlir::dyn_cast<linalg::GenericOp>(op);
    assert(generic);
    auto tiling = parameter[op]["tiling_size"];
    // ECHO_LIST(tiling, ",")
    size_t sramUsage = 0;
    for (auto [operand, affinemap] :
         llvm::zip(generic.getOperands(), generic.getIndexingMapsArray())) {
      if (usedValues.contains(operand)) continue;
      bool isPipelineBuffer = false;
      for (auto user : operand.getUsers()) {
        if (isMember(user) &&
            argwhere(nodeSetTopOrder, user) < argwhere(nodeSetTopOrder, op)) {
          isPipelineBuffer = true;
          break;
        }
      }
      if (isPipelineBuffer) continue;
      usedValues.insert(operand);
      auto tensorType = mlir::dyn_cast<TensorType>(operand.getType());
      assert(tensorType);
      auto shape = tensorType.getShape();
      size_t valueBitwdith = tensorType.getElementTypeBitWidth();
      assert(valueBitwdith % 8 == 0);
      size_t valueBytes = valueBitwdith / 8;
      auto accessDims = getAffineMapAccessDims(affinemap);
      size_t sramSize = valueBytes;
      for (auto d : accessDims) sramSize *= tiling[d];
      sramUsage += sramSize * 2;  // Double buffer
    }
    sramOcc += sramUsage;
    // ECHO_LIST(tiling, ",")
  }
  // ECHO(sramOcc, "\n")
  // ECHO(cfg.sramCapacity, "\n")

  if (!(sramOcc <= cfg.sramCapacity)) {
    // ECHO("check failure!!!", "\n")
    return false;
  } else
    return true;
}

std::vector<int64_t> GenericOpCluster::getDimsInOrder(linalg::GenericOp op,
                                                      mlir::Value operand) {
  OpOperand* opOperand = nullptr;
  assert(op.getResults().size() == 1);
  if (op.getResults()[0] == operand)
    opOperand = &op->getOpOperands().back();
  else {
    for (OpOperand& o : op->getOpOperands()) {
      if (o.get() == operand) {
        opOperand = &o;
        break;
      }
    }
  }

  assert(opOperand);
  auto accDims = getAffineMapAccessDims(op.getMatchingIndexingMap(opOperand));
  auto accDimsSet = llvm::DenseSet<int64_t>(accDims.begin(), accDims.end());
  std::vector<int64_t> orders;
  for (auto d : parameter[op]["outer_order"]) {
    if (accDimsSet.contains(d)) orders.push_back(d);
  }

  // Compare accDims and order to find how this operand is accessed. e.g [0, 1,
  // 2, 4] & [0, 2, 1, 4] -> [0, 2, 1, 3]
  llvm::DenseMap<int64_t, size_t> pos;
  for (auto [idxDim, dim] : llvm::enumerate(accDims)) pos[dim] = idxDim;
  std::vector<int64_t> idx;
  for (const auto& v : orders) idx.push_back(pos[v]);

  return idx;
}

bool GenericOpCluster::checkConnectivity() {
  MergedSet<mlir::Operation*> connectivity;
  for (auto& op : nodeSetTopOrder) connectivity.insert(op);
  for (auto& op : nodeSetTopOrder) {
    // ECHO("check op", "\n")
    // op->dump();
    auto generic = mlir::dyn_cast<linalg::GenericOp>(op);
    assert(generic);
    for (auto operand : generic.getInputs()) {
      // auto producers = getProducerGeneric(operand);

      llvm::SmallVector<linalg::GenericOp> producers;
      getPreviousGeneric(operand, producers);
      for (auto p : producers) {
        assert(isMember(p));

        if (std::distance(
                nodeSetTopOrder.begin(),
                std::find(nodeSetTopOrder.begin(), nodeSetTopOrder.end(), p)) <
            std::distance(nodeSetTopOrder.begin(),
                          std::find(nodeSetTopOrder.begin(),
                                    nodeSetTopOrder.end(), generic))) {
          // ECHO("add connect for pc", "\n")
          // p.dump();
          connectivity.merge(p, generic);
        }
      }

      // llvm::SmallVector<linalg::GenericOp> consumers;
      // getLatterGeneric(operand, consumers);
      // TODO : Optimize one value is shared by multi op with tensor op
      // connection
      for (auto user : operand.getUsers()) {
        if (isMember(user)) connectivity.merge(user, generic);
      }
      // for (auto consumer : consumers) {
      //   assert(isMember(consumer));
      //   if (std::distance(nodeSetTopOrder.begin(),
      //                     std::find(nodeSetTopOrder.begin(),
      //                               nodeSetTopOrder.end(), consumer)) <
      //       std::distance(nodeSetTopOrder.begin(),
      //                     std::find(nodeSetTopOrder.begin(),
      //                               nodeSetTopOrder.end(), generic))) {
      //     ECHO("add connect for shared input", "\n")
      //     consumer.dump();
      //     connectivity.merge(consumer, generic);
      //   }
      // }
    }
  }
  if (connectivity.getNumSet() != 1) return false;

  auto opsWithTensorOp = constructClusterWithTensorOp();
  for (auto tensorOp : opsWithTensorOp) {
    if (!mlir::isa<mlir::tensor::TensorDialect>(tensorOp->getDialect()))
      continue;

    for (auto operand : tensorOp->getOperands()) {
      llvm::SmallVector<linalg::GenericOp> allPrevGeneric, memberPrevGeneric;
      getPreviousGeneric(operand, allPrevGeneric);
      getPreviousGeneric(operand, memberPrevGeneric, false);

      if (allPrevGeneric.size() != memberPrevGeneric.size()) return false;
    }
  }

  return true;
}

void GenericOpCluster::getPreviousGeneric(
    mlir::Value operand, llvm::SmallVector<linalg::GenericOp>& previousGenerics,
    bool ignoreNonMember) {
  auto preOp = operand.getDefiningOp();

  // Func arg
  if (preOp == nullptr) return;
  // Previous generic, push directly
  if (auto generic = mlir::dyn_cast<linalg::GenericOp>(preOp)) {
    if (llvm::is_contained(previousGenerics, generic)) return;

    if (ignoreNonMember) {
      if (isMember(generic)) previousGenerics.push_back(generic);
    } else
      previousGenerics.push_back(generic);

    return;
  }
  // Previous tensor op, recursively check
  if (mlir::isa<tensor::TensorDialect>(preOp->getDialect())) {
    for (auto tensorOperand : preOp->getOperands()) {
      getPreviousGeneric(tensorOperand, previousGenerics, ignoreNonMember);
    }
  }
}

void GenericOpCluster::getLatterGeneric(
    mlir::Value operand, llvm::SmallVector<linalg::GenericOp>& latterGenerics,
    bool ignoreNonMember) {
  for (auto user : operand.getUsers()) {
    if (auto generic = mlir::dyn_cast<linalg::GenericOp>(user)) {
      if (llvm::is_contained(latterGenerics, generic)) continue;
      if (ignoreNonMember) {
        if (isMember(generic)) latterGenerics.push_back(generic);
      } else
        latterGenerics.push_back(generic);
    } else if (mlir::isa<tensor::TensorDialect>(user->getDialect())) {
      auto tensorOpResult = user->getResults();
      assert(tensorOpResult.size() == 1);
      getLatterGeneric(tensorOpResult[0], latterGenerics, ignoreNonMember);
    }
  }
}

std::vector<mlir::Operation*> GenericOpCluster::constructClusterWithTensorOp() {
  llvm::DenseSet<mlir::Operation*> allOps;
  for (auto op : nodeSetTopOrder) {
    auto generic = mlir::dyn_cast<linalg::GenericOp>(op);
    assert(generic);
    allOps.insert(generic);
    for (auto result : generic.getResults()) {
      llvm::SmallVector<mlir::Operation*> path;
      llvm::DenseSet<mlir::Operation*> opOnAllPath;
      constructClusterHelp(result, path, opOnAllPath);
      for (auto o : opOnAllPath) allOps.insert(o);
    }
  }
  auto opVec = std::vector<mlir::Operation*>(allOps.begin(), allOps.end());
  return opVec;
}

void GenericOpCluster::constructClusterHelp(
    mlir::Value operand, llvm::SmallVector<mlir::Operation*>& path,
    llvm::DenseSet<mlir::Operation*>& ops) {
  for (auto user : operand.getUsers()) {
    if (auto generic = mlir::dyn_cast<linalg::GenericOp>(user)) {
      if (isMember(generic)) {
        for (auto op : path) {
          // ECHO("inserting, ", "\n")
          // op->dump();
          ops.insert(op);
        }
      }
    }
    if (mlir::isa<tensor::TensorDialect>(user->getDialect())) {
      auto tensorOpResult = user->getResults();
      assert(tensorOpResult.size() == 1);
      path.push_back(user);
      constructClusterHelp(user->getResults()[0], path, ops);
      path.pop_back();
    }
  }
}

std::vector<mlir::Operation*> GenericOpCluster::getTopoOrderALSP() {
  std::vector<mlir::Operation*> topOrderNodes(nodeSetTopOrder.size());
  std::unordered_map<mlir::Operation*, unsigned int> inD;
  std::unordered_map<mlir::Operation*, unsigned int> outD;
  for (mlir::Operation* op : nodeSetTopOrder) {
    inD[op] = 0;
    outD[op] = 0;
  }
  for (mlir::Operation* op : nodeSetTopOrder) {
    for (auto result : op->getResults()) {
      llvm::SmallVector<linalg::GenericOp> latterGenericOps;
      getLatterGeneric(result, latterGenericOps);
      for (auto user : latterGenericOps) {
        assert(inD.find(user) != inD.end());
        inD[user]++;
        outD[op]++;
      }
    }
  }

  // for (auto [op, outd] : outD) {
  //   op->dump();
  //   ECHO(outd, "\n")
  // }

  auto cmp = [&](mlir::Operation* a, mlir::Operation* b) {
    return inD[a] > inD[b];
  };
  std::priority_queue<mlir::Operation*, std::vector<mlir::Operation*>,
                      decltype(cmp)>
      nodesWithoutOutD(cmp);
  std::vector<mlir::Operation*> tmp;
  for (auto [op, outd] : outD) {
    if (outd == 0) {
      tmp.push_back(op);
    }
  }
  for (auto op : tmp) {
    // ECHO("add init", "\n")
    // op->dump();
    nodesWithoutOutD.push(op);
  }
  size_t index = topOrderNodes.size() - 1;
  while (!nodesWithoutOutD.empty()) {
    auto op = nodesWithoutOutD.top();
    // ECHO("now proces", "\n")
    // op->dump();
    // ECHO(inD[op], "\n")

    nodesWithoutOutD.pop();
    topOrderNodes[index--] = op;

    std::vector<mlir::Operation*> nodes;
    for (auto operand : op->getOperands()) {
      // if (inD.find(user) != inD.end()) inD[user]--;
      // ECHO("check one operand", "\n")
      llvm::SmallVector<linalg::GenericOp> previousGenericOps;
      getPreviousGeneric(operand, previousGenericOps);
      // ECHO("imp", "\n")
      // for (auto op : previousGenericOps) op.dump();
      for (auto prev : previousGenericOps) {
        assert(inD.find(prev) != inD.end());
        outD[prev]--;
        if (outD[prev] == 0) {
          // ECHO("add 0", "\n")
          // prev.dump();
          nodes.push_back(prev);
        }
      }
    }
    for (auto op : nodes) {
      nodesWithoutOutD.push(op);
    }
  }

  // for (auto [x, y] : outD) {
  //   if (y != 0) {
  //     x->dump();
  //     ECHO(y, "\n")
  //     assert(0);
  //   }
  // }
  return topOrderNodes;
}

void GenericOpCluster::setMetric(const EvaluationMetric& metric) {
  this->metric = metric;
}
// ============================
// END OF GenericOpCluster
// ============================

// ============================
// CLASS: PerfModel
// ============================

EvaluationMetric PerfModel::evaluate(GenericOpCluster& cluster,
                                     ArchConfig& cfg) {
  auto parameter = cluster.getParameter();

  EvaluationMetric metric;

  std::unordered_map<mlir::Operation*,
                     std::unordered_map<std::string, double_t>>
      statisticDat;
  for (auto kv : parameter) {
    statisticDat[kv.first]["factor"] = 1;
    statisticDat[kv.first]["flops"] = 0;
    statisticDat[kv.first]["external_access"] = 0;
    statisticDat[kv.first]["sram_access"] = 0;
    statisticDat[kv.first]["cycles"] = 0;
  }
  std::vector<mlir::Operation*> nodeSetTopOrder = cluster.getNodeSetTopOrder();

  // Analyze factor and flops for each pipeline stage
  for (auto riter = nodeSetTopOrder.rbegin(); riter != nodeSetTopOrder.rend();
       riter++) {
    auto genericOp = mlir::dyn_cast<linalg::GenericOp>(*riter);
    assert(genericOp);
    auto iteratorTypes = genericOp.getIteratorTypesArray();
    auto indexingMaps = genericOp.getIndexingMapsArray();
    auto loopBound = parameter[*riter]["loop_bound"];
    auto tilingSize = parameter[*riter]["tiling_size"];
    auto unrollFactor = parameter[*riter]["unroll_factor"];

    // ECHO("tiling", "\n")
    // for (auto xx : tilingSize) {
    //   ECHO(xx, "\n")
    // }
    // ECHO("bound", "\n")
    // for (auto xx : loopBound) {
    //   ECHO(xx, "\n")
    // }
    // ECHO("unroll", "\n")
    // for (auto xx : unrollFactor) {
    //   ECHO(xx, "\n")
    // }
    // for (auto xx : iteratorTypes) {
    //   if (xx == utils::IteratorType::parallel) {
    //     ECHO("parallel", "\n")
    //   } else {
    //     ECHO("red", "\n")
    //   }
    // }

    // Analysis tile : least tile size for producing a output tile
    auto analysisTileSize = llvm::SmallVector<int64_t>(genericOp.getNumLoops());
    for (auto [idxAnaTileSize, itemAnaTileSize] :
         llvm::enumerate(analysisTileSize)) {
      switch (iteratorTypes[idxAnaTileSize]) {
        case mlir::utils::IteratorType::reduction:
          analysisTileSize[idxAnaTileSize] = loopBound[idxAnaTileSize];
          break;
        case mlir::utils::IteratorType::parallel:
          analysisTileSize[idxAnaTileSize] = tilingSize[idxAnaTileSize];
          break;
        default:
          break;
      }
    }

    // Analyze analysis tile size with tiling size to get how many tiles are
    // need for each inputs, i.e. factor, forward this factor to producer
    // generic ops
    for (auto [idxIns, itemIns] : llvm::enumerate(genericOp.getInputs())) {
      llvm::SmallVector<linalg::GenericOp> previousGenerics;
      cluster.getPreviousGeneric(itemIns, previousGenerics);

      for (auto [idxPrevGeneric, itemPrevGeneric] :
           llvm::enumerate(previousGenerics)) {
        int64_t factor = 1;
        auto accessDims = getAffineMapAccessDims(indexingMaps[idxIns]);
        for (auto [idxAccDim, itemAccDim] : llvm::enumerate(accessDims)) {
          // assert(analysisTileSize[itemAccDim] % tilingSize[itemAccDim] == 0);
          factor *= int64_t(ceil(analysisTileSize[itemAccDim] /
                                 double_t(tilingSize[itemAccDim])));
        }
        if (factor * statisticDat[genericOp]["factor"] >
            statisticDat[itemPrevGeneric]["factor"])
          statisticDat[itemPrevGeneric]["factor"] =
              factor * statisticDat[genericOp]["factor"];
      }
    }

    // FLops for each stage = flops of one analysis tile * n tile (factor)
    auto flops =
        std::accumulate(analysisTileSize.begin(), analysisTileSize.end(),
                        int64_t(1), std::multiplies<>());
    statisticDat[*riter]["flops"] = flops * statisticDat[*riter]["factor"];

    // Cycles for each stage
    double_t cycles = 1;
    // for (auto xx : analysisTileSize) {
    //   ECHO(xx, "\n")
    // }
    assert(analysisTileSize.size() == unrollFactor.size());
    for (auto [dimTileSize, dimUnrollFactor] :
         llvm::zip(analysisTileSize, unrollFactor)) {
      // ECHO(dimTileSize, "\n")
      // ECHO(dimUnrollFactor, "\n")
      cycles *= ceil(double_t(dimTileSize) / dimUnrollFactor);
    }
    statisticDat[*riter]["cycles"] = cycles * statisticDat[*riter]["factor"];
  }

  double_t externalAccess = 0;
  llvm::DenseMap<mlir::Value, double_t> externalOperandAccess;
  for (auto& op : cluster.getNodeSetTopOrder()) {
    auto genericOp = mlir::dyn_cast<linalg::GenericOp>(op);
    assert(genericOp);

    auto iteratorTypes = genericOp.getIteratorTypesArray();
    auto indexingMaps = genericOp.getIndexingMapsArray();
    auto loopBound = parameter[genericOp]["loop_bound"];
    auto tilingSize = parameter[genericOp]["tiling_size"];
    auto outerOrder = cluster.getParameter()[op]["outer_order"];

    // Analysis tile : least tile size for producing a output tile
    auto analysisTileSize = llvm::SmallVector<int64_t>(genericOp.getNumLoops());
    for (auto [idxAnaTileSize, itemAnaTileSize] :
         llvm::enumerate(analysisTileSize)) {
      switch (iteratorTypes[idxAnaTileSize]) {
        case mlir::utils::IteratorType::reduction:
          analysisTileSize[idxAnaTileSize] = loopBound[idxAnaTileSize];
          break;
        case mlir::utils::IteratorType::parallel:
          analysisTileSize[idxAnaTileSize] = tilingSize[idxAnaTileSize];
          break;
        default:
          break;
      }
    }

    // Get dim mask to check whetehr one value is accessed one pass
    llvm::SmallVector<int64_t> dimMask;
    for (auto [t, b] : llvm::zip(tilingSize, loopBound)) {
      if (t == b)
        dimMask.push_back(1);
      else
        dimMask.push_back(0);
    }

    // Analyze for each input operand
    for (auto [idxIns, itemIns] : llvm::enumerate(genericOp.getInputs())) {
      llvm::SmallVector<linalg::GenericOp> previousGenericOps;
      cluster.getPreviousGeneric(itemIns, previousGenericOps);
      // Value is internal
      if (previousGenericOps.size() != 0) continue;

      auto accDims = getAffineMapAccessDims(indexingMaps[idxIns]);

      llvm::SmallVector<int64_t> accTileSize;
      for (auto [idxDim, itemDim] : llvm::enumerate(accDims)) {
        if (idxDim == accDims.size() - 1) {
          // For those tile less that 64B (AXI burst length), pad them
          int64_t nByte =
              getElementTypeOrSelf(itemIns).getIntOrFloatBitWidth() / 8;
          accTileSize.push_back(std::max(64 / nByte, tilingSize[itemDim]));
        } else
          accTileSize.push_back(tilingSize[itemDim]);
      }
      // Calculate how many times needed for proceesing one output tile
      int64_t nTimes = 1;
      for (auto [idxDim, itemDim] : llvm::enumerate(accDims)) {
        // assert(analysisTileSize[itemDim] % tilingSize[itemDim] == 0);
        nTimes *= int64_t(
            ceil(analysisTileSize[itemDim] / double_t(tilingSize[itemDim])));
      }

      double_t operandAccess = 1;
      for (auto dimSize : accTileSize) operandAccess *= dimSize;
      operandAccess *= nTimes;

      // // If value is accessed on one pass, access : one analysize tile
      // if (cluster.checkReuseDistance(outerOrder, accDims, dimMask)) {
      //   for (auto d : accDims) operandAccess *= analysisTileSize[d];
      // }
      // // Else, tile is access for several times
      // else {
      //   int64_t nTimes = 1;
      //   for (auto [idxDim, itemDim] : llvm::enumerate(accDims)) {
      //     assert(analysisTileSize[itemDim] % tilingSize[itemDim] == 0);
      //     nTimes *= analysisTileSize[itemDim] / tilingSize[itemDim];
      //   }
      //   for (auto d : accDims) operandAccess *= tilingSize[d];
      //   operandAccess *= nTimes;
      // }

      // Scale with factor
      operandAccess *= statisticDat[genericOp]["factor"];

      // ECHO("analyze input", "\n")
      // ECHO_LIST(accTileSize, ",")
      // ECHO(statisticDat[genericOp]["factor"], "\n")
      // ECHO(nTimes, "\n")
      // ECHO(operandAccess, "\n")

      llvm::SmallVector<linalg::GenericOp> opSharedInput;

      if (opSharedInput.size() > 1) {
        cluster.getLatterGeneric(itemIns, opSharedInput);
        bool isAllSameTilingOrder = true;
        for (auto [idxSharedInputGeneric, itemSharedInputGeneric] :
             llvm::enumerate(opSharedInput)) {
          if (cluster.getParameter()[itemSharedInputGeneric]["tiling_size"] !=
                  cluster.getParameter()[opSharedInput[0]]["tiling_size"] ||
              cluster.getParameter()[itemSharedInputGeneric]["outer_order"] !=
                  cluster.getParameter()[opSharedInput[0]]["outer_order"]) {
            isAllSameTilingOrder = false;
            break;
          }
        }
        if (isAllSameTilingOrder)
          externalOperandAccess[itemIns] = operandAccess;
        else
          externalOperandAccess[itemIns] += operandAccess;
      } else {
        externalOperandAccess[itemIns] = operandAccess;
      }

      // if (externalOperandAccess.contains(itemIns))
      //   externalOperandAccess[itemIns] =
      //       std::max(externalOperandAccess[itemIns], operandAccess);
      // else
      //   externalOperandAccess[itemIns] = operandAccess;
    }

    // Analyze for each result operand
    if (genericOp.getResults().size() != 1) {
      genericOp.dump();
      assert(0);
    }
    for (auto [idxResult, itemResult] :
         llvm::enumerate(genericOp.getResults())) {
      llvm::SmallVector<linalg::GenericOp> latterGenericOps;
      cluster.getLatterGeneric(itemResult, latterGenericOps);
      // Internal value
      if (!latterGenericOps.empty()) continue;

      auto accDims = getAffineMapAccessDims(
          indexingMaps[idxResult + genericOp.getInputs().size()]);
      llvm::SmallVector<int64_t> accTileSize;
      for (auto [idxDim, itemDim] : llvm::enumerate(accDims)) {
        if (idxDim == accDims.size() - 1) {
          // For those tile less that 64B (AXI burst length), pad them
          int64_t nByte =
              getElementTypeOrSelf(itemResult).getIntOrFloatBitWidth() / 8;
          accTileSize.push_back(std::max(64 / nByte, tilingSize[itemDim]));
        } else
          accTileSize.push_back(tilingSize[itemDim]);
      }

      double_t operandAccess = 1;

      // Outputs are always one pass
      assert(cluster.checkReuseDistance(outerOrder, accDims, dimMask));
      for (auto dimSize : accTileSize) operandAccess *= dimSize;

      // ECHO("analyze output", "\n")
      // ECHO(operandAccess, "\n")

      // ECHO("analyze output", "\n")
      // ECHO_LIST(accTileSize, ",")
      // ECHO(statisticDat[genericOp]["factor"], "\n")
      // ECHO(operandAccess, "\n")

      assert(!externalOperandAccess.contains(itemResult));
      externalOperandAccess[itemResult] =
          operandAccess * statisticDat[genericOp]["factor"];
    }
  }
  for (auto [operand, acc] : externalOperandAccess) {
    // ECHO("adding", "\n")
    // ECHO(acc, "\n")
    externalAccess += acc;
  }

  // Analyze SRAM access
  double_t sramAccess = 0;
  for (auto& op : cluster.getNodeSetTopOrder()) {
    auto genericOp = mlir::dyn_cast<linalg::GenericOp>(op);
    assert(genericOp);

    auto iteratorTypes = genericOp.getIteratorTypesArray();
    auto loopBound = parameter[genericOp]["loop_bound"];
    auto tilingSize = parameter[genericOp]["tiling_size"];
    auto unrollFactor = cluster.getParameter()[genericOp]["unroll_factor"];
    auto innerOrder = cluster.getParameter()[genericOp]["inner_order"];

    auto analysisTileSize = llvm::SmallVector<int64_t>(genericOp.getNumLoops());
    for (auto [idxAnaTileSize, itemAnaTileSize] :
         llvm::enumerate(analysisTileSize)) {
      switch (iteratorTypes[idxAnaTileSize]) {
        case mlir::utils::IteratorType::reduction:
          analysisTileSize[idxAnaTileSize] = loopBound[idxAnaTileSize];
          break;
        case mlir::utils::IteratorType::parallel:
          analysisTileSize[idxAnaTileSize] = tilingSize[idxAnaTileSize];
          break;
        default:
          break;
      }
    }

    llvm::SmallVector<int64_t> mask;
    for (auto [t, u] : llvm::zip(analysisTileSize, unrollFactor)) {
      mask.push_back(t == u);
    }

    for (auto [idxOperand, itemOperand] : llvm::enumerate(op->getOperands())) {
      auto accDims =
          getAffineMapAccessDims(genericOp.getIndexingMapsArray()[idxOperand]);
      int64_t nTile = 1;
      int64_t tileSize = 1;
      for (auto [idxDim, itemDim] : llvm::enumerate(accDims)) {
        if (idxDim == accDims.size() - 1)
          tileSize *=
              std::min(int64_t(cfg.sramWidth / 8), unrollFactor[itemDim]);
        else
          tileSize *= unrollFactor[itemDim];
      }
      if (cluster.checkReuseDistance(innerOrder, accDims, mask)) {
        for (auto [idxDim, itemDim] : llvm::enumerate(accDims)) {
          if (itemDim >= analysisTileSize.size() ||
              itemDim >= unrollFactor.size()) {
            ECHO_LIST(analysisTileSize, ",")
            ECHO_LIST(unrollFactor, ",")
            ECHO(itemDim, "\n")
          }

          assert(unrollFactor[itemDim] != 0);
          // assert(analysisTileSize[itemDim] % unrollFactor[itemDim] == 0);
          nTile *= int64_t(ceil(analysisTileSize[itemDim] /
                                double_t(unrollFactor[itemDim])));
        }
      } else {
        for (auto [t, u] : llvm::zip(analysisTileSize, unrollFactor)) {
          // assert(t % u == 0);
          nTile *= int64_t(ceil(t / double_t(u)));
        }
      }
      sramAccess += nTile * tileSize;
    }
  }

  // Bottleneck cycles & flops
  double_t bottleneckCycles = 0;
  double_t flops = 0;
  for (auto [op, namedMetric] : statisticDat) {
    bottleneckCycles = std::max(bottleneckCycles, namedMetric["cycles"]);
    flops += namedMetric["flops"];
  }

  bottleneckCycles =
      std::max(bottleneckCycles, ceil(externalAccess / cfg.bandwidth));

  double_t throughput = flops / bottleneckCycles;
  double_t computeDensity = flops / externalAccess;

  // llvm::errs() << flops << " " << externalAccess << "\n";

  // metric.throughput = throughput;
  // metric.computeDensity = computeDensity;

  // Determine how many output tiles are needed for whole computation pipeline
  auto lastStageNode =
      mlir::dyn_cast<linalg::GenericOp>(nodeSetTopOrder.back());
  assert(lastStageNode && (lastStageNode.getResults().size() == 1));
  auto lastStageOutputAccDims =
      getAffineMapAccessDims(lastStageNode.getIndexingMapsArray().back());

  int64_t nOutputTiles = 1;
  for (auto [idxDim, itemDim] : llvm::enumerate(lastStageOutputAccDims)) {
    auto dimTileSize =
        cluster.getParameter()[lastStageNode]["tiling_size"][itemDim];
    auto dimLoopBound =
        cluster.getParameter()[lastStageNode]["loop_bound"][itemDim];
    // if (dimLoopBound % dimTileSize != 0) {
    //   ECHO(dimLoopBound, "\n")
    //   ECHO(dimTileSize, "\n")
    //   assert(0);
    // }
    // assert(dimLoopBound % dimTileSize == 0);
    nOutputTiles *= int64_t(ceil(dimLoopBound / double_t(dimTileSize)));
  }

  metric.cycles = bottleneckCycles * nOutputTiles;
  metric.externalAccess = externalAccess * nOutputTiles;
  metric.flops = flops * nOutputTiles;
  metric.sramAccess = sramAccess * nOutputTiles;

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
//                                      PerfModel& model, ArchConfig& archCfg)
//                                      {
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

// unsigned int PruningSolver::solve(GenericOpCluster &cluster, PerfModel
// &model,
//                                   ArchConfig &archCfg) {
//   llvm::errs() << "Start solving, subgraph size : "
//                << cluster.getNodeSetTopOrder().size() <<
//                "\n================\n";
//   for (auto op : cluster.getNodeSetTopOrder())
//     op->dump();
//   llvm::errs() << "=========================\n";
//   std::unordered_map<mlir::Operation *, llvm::SmallVector<int64_t>>
//   curOrder; DimensionRelationNetwork curNetwork =
//   cluster.extractDimRelation();
//   std::vector<std::unordered_map<mlir::Operation *,
//   llvm::SmallVector<int64_t>>>
//       candidateOrder;
//   std::vector<DimensionRelationNetwork> candidateNetworks;

//   auto iter = cluster.getNodeSetTopOrder().begin();
//   auto iterEnd = cluster.getNodeSetTopOrder().end();
//   // llvm::errs() << "start solving\n";
//   generateCandidateNetworks(iter, iterEnd, cluster, curOrder, curNetwork,
//                             candidateOrder, candidateNetworks);

//   // llvm::errs() << "finish networks generating, total : "
//   //              << candidateNetworks.size() << "\n";
//   std::vector<double_t> densityRcd;
//   llvm::errs() << candidateNetworks.size() << " Candidates to search\n";
//   for (auto [order, network] : llvm::zip(candidateOrder,
//   candidateNetworks))
//   {
//     auto parBounds = network.getUndeterminedParsBound();
//     llvm::errs() << parBounds.size() << " Parameters to search\n";
//     auto parGen = ParameterGenerator();
//     for (auto bd : parBounds) {
//       std::vector<int64_t> tiling;
//       for (size_t t = 1; t <= bd; t *= 2)
//         tiling.push_back(t);
//       parGen.addVariable(tiling);
//     }

//     bool flag = false;
//     double_t metric = 0;
//     std::vector<int64_t> bestTilingSize;
//     while (parGen.hasNext()) {

//       auto step0 = std::chrono::high_resolution_clock::now();
//       auto tilingVec = parGen.next();
//       auto step1 = std::chrono::high_resolution_clock::now();

//       // llvm::interleave(tilingVec, llvm::errs(), ",");
//       // llvm::errs() << "\n";
//       network.setUndeterminedPars(tilingVec);
//       if (network.forward())
//         flag = true;
//       else
//         continue;

//       auto step2 = std::chrono::high_resolution_clock::now();

//       cluster.applyOrderTiling(order, network.getDimensionValueMapping());

//       double_t density = model.evaluate(cluster, archCfg).computeDensity;

//       auto step3 = std::chrono::high_resolution_clock::now();

//       if (density > metric) {
//         metric = density;
//         bestTilingSize = tilingVec;
//       }

//       std::chrono::duration<double> d0 = step1 - step0;
//       std::chrono::duration<double> d1 = step2 - step1;
//       std::chrono::duration<double> d2 = step3 - step2;
//       llvm::errs() << "Elapsed time: " << d0.count() << " seconds "
//                    << d1.count() << " seconds " << d2.count() << "
//                    seconds\n";
//     }
//     if (flag) {
//       network.setUndeterminedPars(bestTilingSize);
//       densityRcd.push_back(metric);
//     }
//   }

//   auto maxE = std::max_element(densityRcd.begin(), densityRcd.end());
//   size_t indexMaxDensity = std::distance(densityRcd.begin(), maxE);

//   cluster.applyOrderTiling(
//       candidateOrder[indexMaxDensity],
//       candidateNetworks[indexMaxDensity].getDimensionValueMapping());

//   llvm::errs() << "Finish Solving \n";
//   return *maxE;
// }

void PruningSolver::generateCandidateOrders(
    GenericOpCluster& cluster, std::vector<mlir::Operation*>::iterator curOp,
    std::unordered_map<mlir::Operation*, llvm::SmallVector<int64_t>>& candidate,
    std::vector<std::unordered_map<mlir::Operation*,
                                   llvm::SmallVector<int64_t>>>& orders) {
  if (curOp == cluster.getNodeSetTopOrder().end()) {
    orders.push_back(candidate);
    return;
  }
  auto generic = mlir::dyn_cast<linalg::GenericOp>((*curOp));
  assert(generic);
  auto candidateOrders = generatePermutation(generic.getNumLoops());
  for (auto order : candidateOrders) {
    cluster.applyOrder(
        std::unordered_map<mlir::Operation*, llvm::SmallVector<int64_t>>(
            {{*curOp, order}}));
    // ECHO("CHECKING", "\n")
    // ECHO_LIST(order, ",")
    if (cluster.checkOpOrder(*curOp)) {
      candidate[(*curOp)] = order;
      generateCandidateOrders(cluster, std::next(curOp), candidate, orders);
      candidate.erase(*curOp);
    } else {
      // ECHO("ORDER FAILED", "\n")
    }
  }
}

EvaluationMetric PruningSolver::solve(GenericOpCluster& cluster,
                                      PerfModel& model, ArchConfig& archCfg) {
  llvm::errs() << "Start solving, subgraph size : "
               << cluster.getNodeSetTopOrder().size() << "\n================\n";
  for (auto op : cluster.getNodeSetTopOrder()) op->dump();
  llvm::errs() << "=========================\n";

  DimensionRelationNetwork network = cluster.extractDimRelation();
  auto parBounds = network.getUndeterminedParsBound();

  ParameterGenerator<int64_t> tilingGen = ParameterGenerator<int64_t>();
  ECHO("total pars to search : ", "")
  ECHO(parBounds.size(), "\n")
  for (auto bd : parBounds) {
    // ECHO(bd, "\n")
    std::vector<int64_t> tiling;
    for (size_t t = 1; t <= bd; t *= 2) tiling.push_back(t);
    tilingGen.addVariable(tiling);
  }

  std::vector<std::vector<int64_t>> allTilings;
  while (tilingGen.hasNext()) allTilings.push_back(tilingGen.next());

  // ECHO("finish tiling generating", "\n")
  // ECHO(allTilings.size(), "\n")

  bool existGlobalParameter = false;
  GenericOpCluster globalBestCluster;
  EvaluationMetric globalBest;
  std::vector<int64_t> globalBestTiling;
  std::unordered_map<mlir::Operation*, llvm::SmallVector<int64_t>>
      globalBestOrder;
  ECHO(allTilings.size(), "\n")
  std::mutex resultMutex;
  llvm::parallelForEach
      // llvm::for_each
      (allTilings, [&](const std::vector<int64_t>& tilingVec) {
        bool foundInThread = false;
        EvaluationMetric threadBest;
        std::vector<int64_t> threadBestTiling;
        std::unordered_map<mlir::Operation*, llvm::SmallVector<int64_t>>
            threadBestOrder;
        auto localCluster = cluster;
        DimensionRelationNetwork localNetwork(network);

        localNetwork.setUndeterminedPars(tilingVec);

        // for (auto [dim, par] : localNetwork.dimMapping) {
        //   ECHO("check one dim", "\n")
        //   dim.op->dump();
        //   ECHO(dim.dim, "\n")
        //   for (auto r : par->relations()) {
        //     ECHO(" one rel", "\n")
        //     ECHO(r->deducedParameters().size(), "\n")
        //   }
        // }

        if (!localNetwork.forward())
          return;  // Equivalent to 'continue' in parallelForEach

        // for (auto [xx, yy] : localNetwork.getDimensionValueMapping()) {
        //   if (yy == 0) {
        //     ECHO("notice2 !!!!!!!!!!!!!!,", "\n")
        //     ECHO(localNetwork.getDimensionValueMapping().size(), "\n")
        //     ECHO(tilingVec.size(), "\n")
        //   }
        // }

        // for (auto xx : tilingVec) {
        //   if (xx == 0) {
        //     ECHO("notice1 !!!!!!!!!!!!!!,", "\n")
        //   }
        // }

        // for (auto [xx, yy] : localNetwork.getDimensionValueMapping()) {
        //   if (yy == 0) {
        //     ECHO_LIST(tilingVec, ",")

        //     for (auto [x, y] : localNetwork.getDimensionValueMapping()) {
        //       x.op->dump();
        //       ECHO(x.dim, "\n")
        //       ECHO(y, "\n")
        //       ECHO("check parameter", "\n")
        //       for (auto r : localNetwork.dimMapping[x]->relations()) {
        //         ECHO("one relation", "\n")
        //         ECHO(r->deducedParameters().size(), "\n");
        //       }
        //     }
        //     assert(0);
        //   }
        // }

        localCluster.applyTiling(localNetwork.getDimensionValueMapping());
        if (!localCluster.checkArchConstraint(archCfg)) return;

        std::vector<
            std::unordered_map<mlir::Operation*, llvm::SmallVector<int64_t>>>
            orders;
        std::unordered_map<mlir::Operation*, llvm::SmallVector<int64_t>>
            candidate;

        // ECHO("start genere candidate order", "\n")

        generateCandidateOrders(localCluster,
                                localCluster.getNodeSetTopOrder().begin(),
                                candidate, orders);

        // ECHO(candidate.size(), "\n")

        if (orders.empty()) {
          // ECHO("failed to check order", "\n")
          // for (auto xx : localCluster.getNodeSetTopOrder()) {
          //   ECHO_LIST(localCluster.getParameter()[xx]["tiling_size"], ",")
          // }
          return;
        }

        // ECHO("get here", "\n")

        inferUnrollFactor(localCluster, archCfg);

        // ECHO("get here", "\n")

        for (const auto& orderMap : orders) {
          localCluster.applyOrder(orderMap);

          auto metric = model.evaluate(localCluster, archCfg);

          if (metric.eval() > threadBest.eval()) {
            // ECHO("tiling vec:", "\n")
            // for (auto [xx, yy] : localCluster.getParameter()) {
            //   ECHO_LIST(yy["tiling_size"], ",")
            // }
            // ECHO("unroll vec:", "\n")
            // for (auto [xx, yy] : localCluster.getParameter()) {
            //   ECHO_LIST(yy["unroll_factor"], ",")
            // }
            // ECHO(metric.cycles, "\n")
            // ECHO(metric.flops, "\n")
            // ECHO(metric.externalAccess, "\n")
            // ECHO(metric.sramAccess, "\n")
            // ECHO(metric.eval(), "\n")

            foundInThread = true;
            threadBest = metric;
            threadBestTiling = tilingVec;
            threadBestOrder = orderMap;
          }
        }

        if (foundInThread) {
          std::lock_guard<std::mutex> lock(resultMutex);
          existGlobalParameter = true;
          if (threadBest.eval() > globalBest.eval()) {
            globalBestCluster = localCluster;
            globalBest = threadBest;
            globalBestTiling = std::move(threadBestTiling);
            globalBestOrder = std::move(threadBestOrder);
          }
        }
      });

  ECHO("arrive here!!!", "\n")

  if (cluster.getNodeSetTopOrder().size() == 1) assert(existGlobalParameter);

  if (!existGlobalParameter) {
    ECHO("faile to solve parameter", "\n")
    return EvaluationMetric();
  } else {
    // network.setUndeterminedPars(globalBestTiling);
    // assert(network.forward());
    // cluster.applyTiling(network.getDimensionValueMapping());
    // cluster.applyOrder(globalBestOrder);
    cluster = globalBestCluster;

    ECHO("density : ", "")
    ECHO(globalBest.externalDensity(), "\n")
    // ECHO(globalBest.flops, "\n")
    // ECHO(globalBest.externalAccess, "\n")
    ECHO_LIST(globalBestTiling, ",")
    for (auto [xx, yy] : cluster.getParameter()) {
      ECHO_LIST(yy["tiling_size"], ",")
    }
    for (auto [xx, yy] : cluster.getParameter()) {
      ECHO_LIST(yy["unroll_factor"], ",")
    }
    return globalBest;
  }
}

void PruningSolver::inferUnrollFactor(GenericOpCluster& cluster,
                                      ArchConfig& archCfg) {
  /**
   * Approximate linear programming by assuming best unroll factor allocation
   * by ratio of flops of each pipeline stage
   */

  auto nPipelineOp = cluster.getNodeSetTopOrder().size();

  llvm::SmallVector<bool> isArithStage(nPipelineOp);

  // Statistic of flops for each pipeline operation
  llvm::SmallVector<int64_t> flops(nPipelineOp);

  std::unordered_map<mlir::Operation*, double_t> factors;
  for (auto op : cluster.getNodeSetTopOrder()) {
    factors[op] = 1;
  }

  // Analyze factor and flops for each pipeline stage
  int64_t indexReverseNodes = cluster.getNodeSetTopOrder().size() - 1;
  for (auto riter = cluster.getNodeSetTopOrder().rbegin();
       riter != cluster.getNodeSetTopOrder().rend(); riter++) {
    auto genericOp = mlir::dyn_cast<linalg::GenericOp>(*riter);
    assert(genericOp);
    auto iteratorTypes = genericOp.getIteratorTypesArray();
    auto indexingMaps = genericOp.getIndexingMapsArray();
    auto loopBound = cluster.getParameter()[genericOp]["loop_bound"];
    auto tilingSize = cluster.getParameter()[genericOp]["tiling_size"];

    // Analysis tile : least tile size for producing a output tile
    auto analysisTileSize = llvm::SmallVector<int64_t>(genericOp.getNumLoops());

    for (auto [idxAnaTileSize, itemAnaTileSize] :
         llvm::enumerate(analysisTileSize)) {
      switch (iteratorTypes[idxAnaTileSize]) {
        case mlir::utils::IteratorType::reduction:
          analysisTileSize[idxAnaTileSize] = loopBound[idxAnaTileSize];
          break;
        case mlir::utils::IteratorType::parallel:
          analysisTileSize[idxAnaTileSize] = tilingSize[idxAnaTileSize];
          break;
        default:
          break;
      }
    }

    // ECHO("check ana tile", "\n")
    // ECHO_LIST(analysisTileSize, ",")

    // Analyze analysis tile size with tiling size to get how many tiles are
    // need for each inputs, i.e. factor, forward this factor to producer
    // generic ops
    for (auto [idxIns, itemIns] : llvm::enumerate(genericOp.getInputs())) {
      llvm::SmallVector<linalg::GenericOp> previousGenerics;
      cluster.getPreviousGeneric(itemIns, previousGenerics);

      for (auto [idxPrevGeneric, itemPrevGeneric] :
           llvm::enumerate(previousGenerics)) {
        int64_t factor = 1;
        auto accessDims = getAffineMapAccessDims(indexingMaps[idxIns]);
        for (auto [idxAccDim, itemAccDim] : llvm::enumerate(accessDims)) {
          // assert(analysisTileSize[itemAccDim] % tilingSize[itemAccDim] ==
          // 0);
          factor *= int64_t(analysisTileSize[itemAccDim] /
                            double_t(tilingSize[itemAccDim]));
        }
        if (factor * factors[itemPrevGeneric] > factors[itemPrevGeneric])
          factors[itemPrevGeneric] = factor * factors[itemPrevGeneric];
      }
    }

    int64_t acc = 1;
    for (auto t : analysisTileSize) acc *= t;
    flops[indexReverseNodes--] = acc * factors[genericOp];
  }

  // ECHO("check flops", "\n")
  // ECHO_LIST(flops, ",")

  // Assume unroll factor is perfectly allocated with ratio of flops, as r0,
  // r1, r2 ...
  llvm::SmallVector<double_t> ratioUnrollFactor(nPipelineOp);
  // ECHO_LIST(flops, ",")
  int64_t gcd =
      std::accumulate(flops.begin() + 1, flops.end(), flops[0],
                      [](int64_t a, int64_t b) { return std::gcd(a, b); });
  for (auto [idx, f] : llvm::enumerate(flops))
    ratioUnrollFactor[idx] = flops[idx] / double_t(gcd);

  // Stat use of arith resource with += x_i
  llvm::DenseMap<llvm::StringRef, int64_t> arithResourceUse;
  for (auto& [name, cnt] : archCfg.computeResource) {
    arithResourceUse[name] = 0;
  }
  for (auto [index, op] : llvm::enumerate(cluster.getNodeSetTopOrder())) {
    auto generic = mlir::dyn_cast<linalg::GenericOp>(op);
    assert(generic);
    for (auto& arith : generic.getRegion().front().getOperations()) {
      // Process airth resources
      if (mlir::isa<arith::ArithDialect>(arith.getDialect()) ||
          mlir::isa<math::MathDialect>(arith.getDialect())) {
        std::string resourceName = toString(&arith);
        for (auto operand : arith.getOperands())
          resourceName = resourceName + "_" + toString(operand.getType());
        assert(arith.getResults().size() == 1);
        for (auto result : arith.getResults())
          resourceName = resourceName + "_" + toString(result.getType());
        if (!arithResourceUse.contains(resourceName)) {
          ECHO(resourceName, "\n")
          assert(0);
        }
        arithResourceUse[resourceName] += ratioUnrollFactor[index];
        isArithStage[index] = true;
      }
      // Process sram port resources
    }
  }

  // Look for : when scaling ratio r_i, which resource exhaust first
  double_t bottleneckScale = std::numeric_limits<double_t>::max();
  std::string bottleneckResource = "";
  for (auto [name, opCnt] : arithResourceUse) {
    if (opCnt == 0) continue;
    if (auto c = archCfg.computeResource[name.str()] / double_t(opCnt);
        c < bottleneckScale) {
      bottleneckScale = c;
      bottleneckResource = name;
    }
  }

  // r_i * scale = actual unroll factor
  llvm::SmallVector<int64_t> unrollFactor(nPipelineOp);
  // int64_t bottleneckCycles = 0;
  for (auto [idx, ratio] : llvm::enumerate(ratioUnrollFactor)) {
    // Notice !!!!! : Temporal fix, use LP to make sure larger than 1
    if (isArithStage[idx]) {
      unrollFactor[idx] =
          std::max(1L, int64_t(bottleneckScale * ratioUnrollFactor[idx]));
      // bottleneckCycles =
      //     std::max(int64_t(ceil(flops[idx] / double_t(unrollFactor[idx]))),
      //              bottleneckCycles);
    } else {
      unrollFactor[idx] = 1024;
    }
  }

  // Allocate unroll factor for 2 dimension with lowerest access order, e.g.
  // for (m, n, k) -> (m, k), (m, n, k) -> (k, n), (m, n, k) -> (m, n),
  // dimension k and n appear at last dimension, so unroll n and k for
  // continous access from SRAM

  // Allocate unroll factor for 2 dimension with max tiling size
  for (auto [index, op] : llvm::enumerate(cluster.getNodeSetTopOrder())) {
    auto genericOp = mlir::dyn_cast<linalg::GenericOp>(op);
    auto tiling = cluster.getParameter().at(op)["tiling_size"];

    assert(tiling.size() > 1);
    // if (!isArithStage[index]) continue;

    llvm::SmallVector<int64_t> dimsUnroll;
    // First round
    for (auto [idxAffineMap, itemAffineMap] :
         llvm::enumerate(genericOp.getIndexingMapsArray())) {
      if (dimsUnroll.size() >= 2) break;
      auto accDims = getAffineMapAccessDims(itemAffineMap);
      if (!llvm::is_contained(dimsUnroll, accDims.back()))
        dimsUnroll.push_back(accDims.back());
    }
    // Second round
    for (auto [idxAffineMap, itemAffineMap] :
         llvm::enumerate(genericOp.getIndexingMapsArray())) {
      if (dimsUnroll.size() >= 2) break;
      auto accDims = getAffineMapAccessDims(itemAffineMap);
      if (accDims.size() < 2) continue;
      if (!llvm::is_contained(dimsUnroll, accDims[accDims.size() - 2]))
        dimsUnroll.push_back(accDims[accDims.size() - 2]);
    }
    assert(dimsUnroll.size() == 2);

    // Calculate factor for 2 dimensions
    int64_t lowUnrollFactor = tiling[dimsUnroll[0]];
    int64_t highUnrollFactor = tiling[dimsUnroll[1]];

    if (!(highUnrollFactor > 0 &&
          (highUnrollFactor & (highUnrollFactor - 1)) == 0)) {
      ECHO(lowUnrollFactor, "\n")
      ECHO(highUnrollFactor, "\n")
      ECHO_LIST(tiling, ",")
    }

    assert(lowUnrollFactor > 0 &&
           (lowUnrollFactor & (lowUnrollFactor - 1)) == 0);
    assert(highUnrollFactor > 0 &&
           (highUnrollFactor & (highUnrollFactor - 1)) == 0);

    while (lowUnrollFactor * highUnrollFactor > unrollFactor[index]) {
      if (lowUnrollFactor > highUnrollFactor)
        lowUnrollFactor /= 2;
      else
        highUnrollFactor /= 2;
    }

    // if (highUnrollFactor <= 0 || lowUnrollFactor <= 0) {
    //   for (auto xx : cluster.getNodeSetTopOrder()) {
    //     ECHO_LIST(cluster.getParameter()[xx]["tiling_size"], ",")
    //   }
    //   ECHO(unrollFactor[index], "\n")
    //   ECHO_LIST(ratioUnrollFactor, ",")
    //   ECHO(ratioUnrollFactor[index], "\n")
    //   ECHO(bottleneckScale, "\n")
    // }
    // assert(lowUnrollFactor > 0);
    // assert(highUnrollFactor > 0);

    // std::min(tiling[dimsUnroll[0]], (1L << halfFactorPot));
    // int64_t highUnrollFactor =
    //     std::min(int64_t(unrollFactor[index] / lowUnrollFactor),
    //              cluster.getParameter()[op]["tiling_size"][dimsUnroll[1]]);

    // std::vector<int64_t> idx(tiling.size());
    // std::iota(idx.begin(), idx.end(), 0);
    // std::sort(idx.begin(), idx.end(),
    //           [&](size_t i1, size_t i2) { return tiling[i1] > tiling[i2];
    //           });

    // int64_t halfFactorPot = int64_t(log2(unrollFactor[index])) / 2;
    // int64_t lowUnrollFactor =
    //     std::min(tiling[dimsUnroll[0]], (1L << halfFactorPot));
    // int64_t highUnrollFactor =
    //     std::min(int64_t(unrollFactor[index] / lowUnrollFactor),
    //              cluster.getParameter()[op]["tiling_size"][dimsUnroll[1]]);

    // ECHO("check unro", "\n")
    // ECHO(lowUnrollFactor, "\n")
    // ECHO(highUnrollFactor, "\n")

    cluster.getParameter()[op]["unroll_factor"][dimsUnroll[0]] =
        lowUnrollFactor;
    cluster.getParameter()[op]["unroll_factor"][dimsUnroll[1]] =
        highUnrollFactor;

    // Adjust order, make unroll dimensions stationary
    auto inner_order = cluster.getParameter()[op]["inner_order"];
    auto dimsUnrollSorted = dimsUnroll;
    // Make sure order
    std::sort(dimsUnrollSorted.begin(), dimsUnrollSorted.end());
    inner_order.erase(inner_order.begin() + dimsUnrollSorted[1]);
    inner_order.erase(inner_order.begin() + dimsUnrollSorted[0]);

    inner_order.insert(inner_order.begin(), dimsUnrollSorted[0]);
    inner_order.insert(inner_order.begin(), dimsUnrollSorted[1]);

    cluster.getParameter()[op]["inner_order"] = inner_order;
  }

  // Allocate unroll factor for thoese pure memory stages
  // TODO :
}

void PruningSolver::generateCandidateNetworks(
    std::vector<mlir::Operation*>::iterator curp,
    std::vector<mlir::Operation*>::iterator endp, GenericOpCluster& cluster,
    std::unordered_map<mlir::Operation*, llvm::SmallVector<int64_t>>& curOrder,
    DimensionRelationNetwork& curNetwork,

    std::vector<std::unordered_map<mlir::Operation*,
                                   llvm::SmallVector<int64_t>>>& candidateOrder,
    std::vector<DimensionRelationNetwork>& candidateNetworks) {
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

      if (!checkOrder(cluster, genericOp, order, mask)) continue;

      bool flag = true;
      std::vector<Dimension> newConstDims;
      for (auto [index, maskBit] : llvm::enumerate(mask)) {
        if (maskBit) {
          auto td = Dimension(*curp, index);
          // llvm::errs() << curNetwork.isConst(td) << " "
          //              << curNetwork.getValue(td) << " "
          //              <<
          //              cluster.getParameter()[*curp]["loop_bound"][index]
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

      for (auto& dim : newConstDims) curNetwork.removeConstDimension(dim);
    }
    curOrder.erase(*curp);
  }
}

std::vector<llvm::SmallVector<int64_t>> PruningSolver::generatePermutation(
    size_t n) {
  std::vector<llvm::SmallVector<int64_t>> orders;
  llvm::SmallVector<int64_t> order(n);
  std::iota(order.begin(), order.end(), 0);
  do {
    orders.push_back(order);
  } while (std::next_permutation(order.begin(), order.end()));
  return orders;
}

std::vector<llvm::SmallVector<int64_t>> PruningSolver::generateBitMask(
    size_t n) {
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

bool PruningSolver::checkReuseDistance(llvm::SmallVector<int64_t>& order,
                                       llvm::SmallVector<int64_t>& dims,
                                       llvm::SmallVector<int64_t>& mask) {
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
      if (mask[i]) break;
      for (size_t j = i + 1; j < order.size(); j++) {
        if (!mask[j]) return false;
      }
    } else
      break;
  }
  return true;
}

bool PruningSolver::checkOrder(GenericOpCluster& cluster, linalg::GenericOp& op,
                               llvm::SmallVector<int64_t>& order,
                               llvm::SmallVector<int64_t>& mask) {
  // Check output reuse distance
  {
    assert(op.getOutputs().size() == 1);
    auto outputDims = getAffineMapAccessDims(op.getIndexingMapsArray().back());
    if (!checkReuseDistance(order, outputDims, mask)) return false;
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
      if (!checkReuseDistance(order, inputDims, mask)) return false;
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
  this->solver = llvm::StringSwitch<ParameterSolvingInterface*>(solver)
                     .Case("brute_force", new PruningSolver())
                     .Case("pruning_brute_force", new PruningSolver())
                     .Default(nullptr);
}

ScheduledGenericOpCluster::~ScheduledGenericOpCluster() {
  for (auto cluster : clusters) delete cluster;
  delete solver;
}

std::vector<linalg::GenericOp> ScheduledGenericOpCluster::getTopSortedNodes() {
  std::vector<linalg::GenericOp> topOrderNodes;
  std::unordered_map<mlir::Operation*, unsigned int> inD;
  for (mlir::Operation* op : genericOps) inD.insert({op, 0});
  for (mlir::Operation* op : genericOps) {
    for (auto user : op->getUsers()) {
      if (inD.find(user) != inD.end()) inD[user]++;
    }
  }
  std::queue<mlir::Operation*> nodesWithoutInD;
  for (auto [op, ind] : inD) {
    if (ind == 0) {
      nodesWithoutInD.push(op);
      if (!mlir::dyn_cast<linalg::GenericOp>(op)) op->dump();
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
      if (inD.find(user) == inD.end()) continue;
      inD[user]--;
      if (inD[user] == 0) {
        nodesWithoutInD.push(user);
        if (!mlir::dyn_cast<linalg::GenericOp>(user)) user->dump();
      }
    }
  }
  return topOrderNodes;
}

void ScheduledGenericOpCluster::insertOp(mlir::Operation* op) {
  ops.push_back(op);
}

void ScheduledGenericOpCluster::insertGenericOp(linalg::GenericOp genericOp) {
  genericOps.push_back(genericOp);
}

void ScheduledGenericOpCluster::schedule(mlir::MLIRContext* ctx,
                                         PerfModel& model, ArchConfig& archCfg,
                                         int64_t maxSubgraphOp) {
  // std::vector<mlir::Operation *> opsTopOrder = getTopoOrder(ops);
  // std::vector<mlir::Operation*> opsTopOrder = getTopoOrderALSP(ops);
  std::vector<linalg::GenericOp> genericOps;
  for (auto& op : ops) {
    if (auto generic = mlir::dyn_cast<linalg::GenericOp>(op);
        (!op->hasAttr("accelgen.const_broadcast")) && generic)
      genericOps.push_back(generic);
  }

  auto rootCluster = GenericOpCluster(genericOps.data(),
                                      genericOps.data() + genericOps.size());

  // ECHO("check topo order", "\n")
  std::vector<linalg::GenericOp> genericOpsTopOrder;
  for (auto op : rootCluster.getNodeSetTopOrder()) {
    // op->dump();
    auto generic = mlir::dyn_cast<linalg::GenericOp>(op);
    genericOpsTopOrder.push_back(generic);
    // ECHO("check order", "\n")
    // generic.dump();
  }

  /****test */

  // auto mergedCluster = GenericOpCluster(genericOpsTopOrder.data() + 0,
  //                                       genericOpsTopOrder.data() + 2);
  // ECHO(mergedCluster.checkConnectivity(), "\n")
  // EvaluationMetric mergedMetric = solver->solve(mergedCluster, model,
  // archCfg);

  // return;
  /**** */

  assert(genericOpsTopOrder.size() >= 1);
  EvaluationMetric* dpStatus =
      new EvaluationMetric[genericOpsTopOrder.size() + 1];
  size_t* cutIndex = new size_t[genericOpsTopOrder.size() + 1];
  GenericOpCluster* cutCluster =
      new GenericOpCluster[genericOpsTopOrder.size() + 1];
  // llvm::errs() << genericOpsTopOrder.size() + 1 << "\n";
  dpStatus[0] = EvaluationMetric();
  cutIndex[0] = 0;
  llvm::errs() << "total generic : " << genericOpsTopOrder.size() << "\n";
  for (unsigned int i = 1; i <= genericOpsTopOrder.size(); i++) {
    llvm::errs() << "outer" << i << "\n";
    EvaluationMetric maxMetric;
    GenericOpCluster bestCluster;
    unsigned int index = 1;
    for (unsigned j = 1; j <= i; j++) {
      llvm::errs() << "inner" << j << "\n";
      auto mergedCluster = GenericOpCluster(genericOpsTopOrder.data() + j - 1,
                                            genericOpsTopOrder.data() + i);

      EvaluationMetric mergedMetric;

      // Make sure graph is all connected && size < maxSubGraphOp
      bool isValidGraph =
          mergedCluster.checkConnectivity() && (i - j + 1 <= maxSubgraphOp);
      // ECHO("notice!", "\n")
      // for (auto xx : mergedCluster.getNodeSetTopOrder())
      //   xx->dump();
      // ECHO(mergedCluster.checkConnectivity(), "\n")
      if (isValidGraph) {
        mergedMetric = solver->solve(mergedCluster, model, archCfg);
        mergedCluster.setMetric(mergedMetric);
      } else {
        mergedMetric = model.evaluate(mergedCluster, archCfg);
      }
      // GenericOpCluster(genericOpsTopOrder.data() + j - 1,
      //                  genericOpsTopOrder.data() + i)
      //     .solveBestSchedule(evaluator, solver);

      llvm::errs() << "density : " << mergedMetric.externalDensity()
                   << " throughput : " << mergedMetric.throughput()
                   << " flops: " << mergedMetric.flops
                   << " cycles: " << mergedMetric.cycles << "\n";

      // if (dpStatus[j - 1] + mergedMetric > maxMetric) {
      //   maxMetric = dpStatus[j - 1] + mergedMetric;
      //   index = j;
      // }
      double_t seqDensity = 0;
      double_t seqThroughput = 0;
      double_t target = 0;

      EvaluationMetric updateMetric;
      if (isValidGraph) {
        // seqDensity =
        //     (dpStatus[j - 1].flops + mergedMetric.flops) /
        //     (dpStatus[j - 1].externalAccess + mergedMetric.externalAccess);
        // seqThroughput = (dpStatus[j - 1].flops + mergedMetric.flops) /
        //                 (dpStatus[j - 1].cycles + mergedMetric.cycles);
        // target = seqDensity * 0.8 + seqThroughput * 0.2;
        updateMetric = EvaluationMetric::merge(dpStatus[j - 1], mergedMetric);
      }
      ECHO("check target", "\n")
      ECHO(updateMetric.eval(), "\n")
      // ECHO(updateMetric.flops, "\n")
      ECHO(maxMetric.eval(), "\n")

      ECHO("external access:", "\n")
      ECHO(updateMetric.externalAccess, "\n")
      ECHO("sram access:", "\n")
      ECHO(updateMetric.sramAccess, "\n")
      ECHO("cycles:", "\n")
      ECHO(updateMetric.cycles, "\n")
      ECHO("flops:", "\n")
      ECHO(updateMetric.flops, "\n")

      if (updateMetric.eval() > maxMetric.eval()) {
        // maxMetric.target = target;
        // maxMetric.flops = dpStatus[j - 1].flops + mergedMetric.flops;
        // maxMetric.externalAccess =
        //     dpStatus[j - 1].externalAccess + mergedMetric.externalAccess;
        // maxMetric.cycles = dpStatus[j - 1].cycles + mergedMetric.cycles;
        maxMetric = updateMetric;
        index = j;
        bestCluster = mergedCluster;
      }
    }

    dpStatus[i] = maxMetric;
    // llvm::errs() << index << " " << i << "\n";
    cutIndex[i] = index;
    cutCluster[i] = bestCluster;

    ECHO("now, ", "\n")
    ECHO(cutIndex[i], "\n")
    ECHO(dpStatus[i].externalDensity(), "\n")
  }

  auto p = genericOpsTopOrder.size();
  while (1) {
    // // llvm::errs() << p << " " << cutIndex[p] - 1 << " " << p << "\n";
    // auto cluster =
    //     new GenericOpCluster(genericOpsTopOrder.data() + cutIndex[p] - 1,
    //                          genericOpsTopOrder.data() + p);
    // // cluster->solveBestSchedule(evaluator, solver);
    // solver->solve(*cluster, model, archCfg);
    // clusters.push_back(cluster);
    clusters.push_back(new GenericOpCluster(cutCluster[p]));
    if (cutIndex[p] == 1) break;
    p = cutIndex[p] - 1;
  }
  delete[] dpStatus;
  delete[] cutIndex;
  delete[] cutCluster;
  for (auto c : clusters) c->attachAttribute(ctx);
}

std::vector<mlir::accelgen::GenericOpCluster*>::iterator
ScheduledGenericOpCluster::begin() {
  return clusters.begin();
}

std::vector<mlir::accelgen::GenericOpCluster*>::iterator
ScheduledGenericOpCluster::end() {
  return clusters.end();
}

// ============================
// END OF ScheduledGenericOpCluster
// ============================

}  // namespace mlir::accelgen