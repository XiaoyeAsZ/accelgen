#ifndef LINALG_OP_FUSE_PASS
#define LINALG_OP_FUSE_PASS

include "mlir/Pass/PassBase.td"

def LinalgOpFuse: Pass<"linalg-op-fuse", "::mlir::ModuleOp"> {
  let summary = "Fuse op with specific pattern.";
  let description = [{
  }];
}

#endif // LINALG_OP_FUSE_PASS
