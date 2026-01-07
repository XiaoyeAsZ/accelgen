#ifndef CONVERT_TO_SPE_PASS
#define CONVERT_TO_SPE_PASS

include "mlir/Pass/PassBase.td"

def ConvertToSpe: Pass<"convert-to-spe", "::mlir::ModuleOp"> {
  let summary = "Convert arith/math etc. to spe.";
  let description = [{
  }];
}

#endif // CONVERT_TO_SPE_PASS
