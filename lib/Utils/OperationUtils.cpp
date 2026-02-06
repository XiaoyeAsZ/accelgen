#include "accelgen/Utils/OperationUtils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "llvm/ADT/TypeSwitch.h"
#include <queue>

#include "mlir/Dialect/Arith/IR/Arith.h"

namespace mlir::accelgen {

std::vector<mlir::Operation *>
getTopoOrder(const std::vector<mlir::Operation *> &ops) {
  std::vector<mlir::Operation *> topOrderNodes;
  std::unordered_map<mlir::Operation *, unsigned int> inD;
  for (mlir::Operation *op : ops)
    inD[op] = 0;
  for (mlir::Operation *op : ops) {
    for (auto user : op->getUsers())
      inD[user]++;
  }
  std::queue<mlir::Operation *> nodesWithoutInD;
  for (auto [op, ind] : inD) {
    if (ind == 0) {
      nodesWithoutInD.push(op);
      // if (!mlir::dyn_cast<linalg::GenericOp>(op))
      //   op->dump();
    }
  }
  while (!nodesWithoutInD.empty()) {
    auto op = nodesWithoutInD.front();
    nodesWithoutInD.pop();
    topOrderNodes.push_back(op);
    for (auto user : op->getUsers()) {
      // if (inD.find(user) != inD.end()) inD[user]--;
      if (inD.find(user) == inD.end()) {
        op->dump();
        user->dump();
      }

      assert(inD.find(user) != inD.end());
      inD[user]--;
      if (inD[user] == 0) {
        nodesWithoutInD.push(user);
        // if (!mlir::dyn_cast<linalg::GenericOp>(user))
        //   user->dump();
      }
    }
  }
  return topOrderNodes;
}

std::vector<mlir::Operation *>
getTopoOrderALSP(const std::vector<mlir::Operation *> &ops) {
  std::vector<mlir::Operation *> topOrderNodes(ops.size());
  std::unordered_map<mlir::Operation *, unsigned int> inD;
  std::unordered_map<mlir::Operation *, unsigned int> outD;
  for (mlir::Operation *op : ops) {
    inD[op] = 0;
    outD[op] = 0;
  }
  for (mlir::Operation *op : ops) {
    for (auto user : op->getUsers()) {
      if (inD.find(user) != inD.end()) {
        inD[user]++;
        outD[op]++;
      }
    }
  }
  std::queue<mlir::Operation *> nodesWithoutOutD;
  std::vector<mlir::Operation *> tmp;
  for (auto [op, outd] : outD) {
    if (outd == 0) {
      tmp.push_back(op);
    }
  }
  std::sort(
      tmp.begin(), tmp.end(),
      [&](mlir::Operation *a, mlir::Operation *b) { return inD[a] < inD[b]; });
  for (auto op : tmp)
    nodesWithoutOutD.push(op);
  size_t index = topOrderNodes.size() - 1;
  while (!nodesWithoutOutD.empty()) {
    auto op = nodesWithoutOutD.front();
    nodesWithoutOutD.pop();
    topOrderNodes[index--] = op;

    std::vector<mlir::Operation *> nodes;
    for (auto operand : op->getOperands()) {
      // if (inD.find(user) != inD.end()) inD[user]--;
      auto defOp = operand.getDefiningOp();
      if (defOp == nullptr)
        continue;
      if (inD.find(defOp) != inD.end()) {
        outD[defOp]--;
        if (outD[defOp] == 0) {
          nodes.push_back(defOp);
        }
      }
    }
    std::sort(nodes.begin(), nodes.end(),
              [&](mlir::Operation *a, mlir::Operation *b) {
                return inD[a] < inD[b];
              });
    for (auto op : nodes)
      nodesWithoutOutD.push(op);
  }
  return topOrderNodes;
}

std::vector<linalg::GenericOp> getProducerGeneric(mlir::Value operand) {
  std::vector<linalg::GenericOp> producers;
  auto definingOp = operand.getDefiningOp();
  if (definingOp == nullptr) {
    producers.push_back(nullptr);
    return producers;
  }

  mlir::TypeSwitch<mlir::Operation *>(definingOp)
      .Case<linalg::GenericOp>(
          [&](linalg::GenericOp op) { producers.push_back(op); })
      .Case<tensor::CollapseShapeOp>([&](tensor::CollapseShapeOp op) {
        if (op.getSrc().getDefiningOp()) {
          auto prevOp =
              mlir::dyn_cast<linalg::GenericOp>(op.getSrc().getDefiningOp());
          assert(prevOp);
          producers.push_back(prevOp);
        }
      })
      .Case<tensor::ExpandShapeOp>([&](tensor::ExpandShapeOp op) {
        if (op.getSrc().getDefiningOp()) {
          auto prevOp =
              mlir::dyn_cast<linalg::GenericOp>(op.getSrc().getDefiningOp());
          assert(prevOp);
          producers.push_back(prevOp);
        }
      })
      .Case<tensor::ConcatOp>([&](tensor::ConcatOp op) {
        for (auto operand : op.getInputs()) {
          if (operand.getDefiningOp()) {
            if (auto prevOp = mlir::dyn_cast<linalg::GenericOp>(
                    operand.getDefiningOp())) {
              producers.push_back(prevOp);
            } else if (auto prevOp = mlir::dyn_cast<tensor::ExtractSliceOp>(
                           operand.getDefiningOp())) {
              auto prevPrevOp =
                  prevOp.getSource().getDefiningOp<linalg::GenericOp>();
              producers.push_back(prevPrevOp);
            } else
              assert(0);
          }
        }
      })
      .Case<tensor::ExtractSliceOp>([&](tensor::ExtractSliceOp op) {
        if (op.getSource().getDefiningOp()) {
          auto prevOp = op.getSource().getDefiningOp<linalg::GenericOp>();
          assert(prevOp);
          producers.push_back(prevOp);
        }
      })
      .Case<arith::ConstantOp>([&](arith::ConstantOp op) {})
      .Default([&](mlir::Operation *op) {
        op->dump();
        assert(0);
      });

  return producers;
}

llvm::ArrayRef<int64_t> getOperandShape(const mlir::Value &operand) {
  auto shape = mlir::dyn_cast<ShapedType>(operand.getType());
  assert(shape);
  return shape.getShape();
}

} // namespace mlir::accelgen