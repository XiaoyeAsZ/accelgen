#include "accelgen/Utils/OperationUtils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include <queue>

namespace mlir::accelgen {

std::vector<mlir::Operation*> getTopoOrder(std::vector<mlir::Operation*> ops) {
  std::vector<mlir::Operation*> topOrderNodes;
  std::unordered_map<mlir::Operation*, unsigned int> inD;
  for (mlir::Operation* op : ops) inD[op] = 0;
  for (mlir::Operation* op : ops) {
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

std::vector<linalg::GenericOp> getProducerGeneric(linalg::GenericOp op) {
  std::vector<linalg::GenericOp> producers;
  for (auto operand : op.getInputs()) {
    auto definingOp = operand.getDefiningOp();
    while (definingOp && (!mlir::dyn_cast<linalg::GenericOp>(definingOp))) {
      auto definingOpOperands = definingOp->getOperands();
      assert(definingOpOperands.size() == 1);
      definingOp = definingOpOperands[0].getDefiningOp();
    }
    if (definingOp)
      producers.push_back(mlir::dyn_cast<linalg::GenericOp>(definingOp));
  }
  return producers;
}

}  // namespace mlir::accelgen