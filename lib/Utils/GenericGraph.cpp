#include "accelgen/Utils/GenericGraph.h"
#include "mlir/Analysis/TopologicalSortUtils.h"

namespace mlir::accelgen {
GenericGraph::GenericGraph(mlir::func::FuncOp& funcOp) {
  llvm::SetVector<mlir::Operation*> rawOps;
  funcOp->walk([&](mlir::Operation* op) {
    if (auto genericOp = mlir::dyn_cast<linalg::GenericOp>(op)) {
      rawOps.insert(op);
      for (auto ins : genericOp.getInputs()) values.insert(ins);
      assert(genericOp.getResults().size() == 1);
      for (auto res : genericOp.getResults()) values.insert(res);
    } else if (mlir::dyn_cast<tensor::CollapseShapeOp>(op) ||
               mlir::dyn_cast<tensor::ExpandShapeOp>(op) ||
               mlir::dyn_cast<tensor::ConcatOp>(op) ||
               mlir::dyn_cast<tensor::ExtractSliceOp>(op)) {
      rawOps.insert(op);
      for (auto ins : op->getOpResults()) values.insert(ins);
      assert(op->getResults().size() == 1);
      for (auto res : op->getResults()) values.insert(res);
    }
  });
  auto sortedOps = mlir::topologicalSort(rawOps);

  for (auto op : sortedOps) {
    if (auto genericOp = mlir::dyn_cast<linalg::GenericOp>(op)) {
      genericOps.insert(genericOp);

    } else if (mlir::dyn_cast<tensor::CollapseShapeOp>(op) ||
               mlir::dyn_cast<tensor::ExpandShapeOp>(op) ||
               mlir::dyn_cast<tensor::ConcatOp>(op) ||
               mlir::dyn_cast<tensor::ExtractSliceOp>(op)) {
      tensorOps.insert(op);
    }
  }
}

llvm::ArrayRef<mlir::Value> GenericGraph::getValues() {
  return values.getArrayRef();
}

llvm::ArrayRef<mlir::Operation*> GenericGraph::getTensorOps() {
  return tensorOps.getArrayRef();
}

llvm::ArrayRef<linalg::GenericOp> GenericGraph::getGenericOpsInTopoOrder() {
  return genericOps.getArrayRef();
}

mlir::OpOperand* GenericGraph::getPreviousOpOperand(mlir::OpOperand& operand) {
  auto prevOperand = &operand;
  while (true) {
    auto prevOp = prevOperand->get().getDefiningOp();
    // Argument
    if (prevOp == nullptr)
      break;
    else if (mlir::dyn_cast<tensor::ConcatOp>(prevOp)) {
      break;
    } else if (auto collapseOp =
                   mlir::dyn_cast<tensor::CollapseShapeOp>(prevOp)) {
      prevOperand = &collapseOp->getOpOperand(0);
    } else if (auto expandOp = mlir::dyn_cast<tensor::ExpandShapeOp>(prevOp)) {
      prevOperand = &expandOp->getOpOperand(0);
    } else if (auto extractOp =
                   mlir::dyn_cast<tensor::ExtractSliceOp>(prevOp)) {
      prevOperand = &extractOp->getOpOperand(0);
    }
    // Endpoint op
    else if (mlir::dyn_cast<linalg::GenericOp>(prevOp))
      break;
    else {
      prevOp->dump();
      assert(0);
    }
  }
  return prevOperand;
}

}  // namespace mlir::accelgen