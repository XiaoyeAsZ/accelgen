#ifndef COLLAPSE_SHAPE_DELAY_PASS
#define COLLAPSE_SHAPE_DELAY_PASS

include "mlir/Pass/PassBase.td"

def CollapseShapeDelayPass: Pass<"collapse-shape-delay", "::mlir::ModuleOp"> {
  let summary = "";
  let description = [{
  }];
}

#endif // COLLAPSE_SHAPE_DELAY_PASS
