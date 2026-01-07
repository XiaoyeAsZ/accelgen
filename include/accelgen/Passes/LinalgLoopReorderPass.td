#ifndef LINALG_LOOP_REORDER_PASS
#define LINALG_LOOP_REORDER_PASS

include "mlir/Pass/PassBase.td"

def LinalgLoopReorder: Pass<"linalg-loop-reorder", "::mlir::ModuleOp"> {
  let summary = "Unify loop order of linalg op for fusing.";
  let description = [{
  }];
}

#endif // LINALG_LOOP_REORDER_PASS
