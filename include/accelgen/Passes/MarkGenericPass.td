#ifndef MARK_GENERIC_PASS
#define MARK_GENERIC_PASS

include "mlir/Pass/PassBase.td"

def MarkGenericPass: Pass<"mark-generic", "::mlir::ModuleOp"> {
  let summary = "";
  let description = [{
  }];
}

#endif // MARK_GENERIC_PASS
