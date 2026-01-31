#ifndef FUSE_GENERIC_PASS
#define FUSE_GENERIC_PASS

include "mlir/Pass/PassBase.td"

def FuseGenericPass: Pass<"fuse-generic", "::mlir::ModuleOp"> {
  let summary = "";
  let description = [{
  }];
}

#endif // FUSE_GENERIC_PASS
