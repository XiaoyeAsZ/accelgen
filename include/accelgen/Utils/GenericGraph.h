#ifndef GENERIC_GRAPH_H
#define GENERIC_GRAPH_H

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace accelgen {
class GenericGraph {
 private:
  llvm::SetVector<mlir::linalg::GenericOp> genericOps;
  llvm::SetVector<mlir::Operation*> tensorOps;

  llvm::SetVector<mlir::Value> values;

 public:
  GenericGraph(mlir::func::FuncOp& funcOp);
  ~GenericGraph() = default;

  llvm::ArrayRef<mlir::Value> getValues();
  llvm::ArrayRef<mlir::Operation*> getTensorOps();

  llvm::ArrayRef<linalg::GenericOp> getGenericOpsInTopoOrder();

  mlir::OpOperand* getPreviousOpOperand(mlir::OpOperand& operand);
  mlir::OpOperand* getNextOpOperand(mlir::OpOperand& operand);
};

}  // namespace accelgen
}  // namespace mlir
#endif