#ifndef SPE_FISSION_PASS
#define SPE_FISSION_PASS

include "mlir/Pass/PassBase.td"

def SpeFission: Pass<"spe-fission", "::mlir::ModuleOp"> {
  let summary = "Spe node fission.";
  let description = [{
  }];
}

#endif // SPE_FISSION_PASS
