
include "mlir/Pass/PassBase.td"

def KernelSchedule: Pass<"kernel-schedule", "::mlir::func::FuncOp"> {
  let summary = "";
  let description = [{
  }];
}

def MarkGenericPass: Pass<"mark-generic", "::mlir::ModuleOp"> {
  let summary = "";
  let description = [{
  }];
}

def FuseGenericPass: Pass<"fuse-generic", "::mlir::ModuleOp"> {
  let summary = "";
  let description = [{
  }];
}

def LinalgOpFuse: Pass<"linalg-op-fuse", "::mlir::ModuleOp"> {
  let summary = "Fuse op with specific pattern.";
  let description = [{
  }];
}

def LinalgLoopReorder: Pass<"linalg-loop-reorder", "::mlir::ModuleOp"> {
  let summary = "Unify loop order of linalg op for fusing.";
  let description = [{
  }];
}

def ConvertToSpe: Pass<"convert-to-spe", "::mlir::ModuleOp"> {
  let summary = "Convert arith/math etc. to spe.";
  let description = [{
  }];
}

def ConstructTileGraph: Pass<"construct-tile-graph", "::mlir::func::FuncOp"> {
  let summary = "";
  let description = [{
  }];
}

def ConvertToDap: Pass<"convert-to-dap", "::mlir::ModuleOp"> {
  let summary = "";
  let description = [{
  }];
}

def CollapseShapeDelayPass: Pass<"collapse-shape-delay", "::mlir::ModuleOp"> {
  let summary = "";
  let description = [{
  }];
}


def ModelBaselineAccelerator: Pass<"model-baseline-accelerator", "::mlir::ModuleOp"> {
  let summary = "";
  let description = [{
  }];
  let options = [
    Option<"acceleratorName", "accelerator-name", "std::string","", "">,
    Option<"configPath", "config-path", "std::string","", "">
  ];
}
