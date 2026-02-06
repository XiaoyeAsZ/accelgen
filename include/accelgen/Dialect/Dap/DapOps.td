#ifndef DAP_OPS
#define DAP_OPS

include "accelgen/Dialect/Dap/DapDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Dialect/Linalg/IR/LinalgBase.td"
include "mlir/Dialect/Linalg/IR/LinalgInterfaces.td"


def DataPathOp : Dap_Op<"data_path", []> {
  let summary = "Data Path";
  let description = [{
  }];

  let arguments = (ins
    I32:$input,
    I32Attr:$width
  );

  let results = (outs
    I32:$output 
  );
}

def ConcatOp : Dap_Op<"concat", []> {
  let summary = "Concat";
  let description = [{
  }];

  let arguments = (ins
    I32Attr:$width,
    I32Attr:$fanout
  );

  let results = (outs
    Variadic<I32>:$ports
  );
}

def ReplicateOp : Dap_Op<"replicate", []> {
  let summary = "Replicate";
  let description = [{
  }];

  let arguments = (ins
    I32Attr:$width,
    I32Attr:$fanout
  );

  let results = (outs
    Variadic<I32>:$ports
  );
}

def ReductionOp : Dap_Op<"reduction", []> {
  let summary = "Reduction";
  let description = [{
  }];

  let arguments = (ins
    I32Attr:$fanin,
    StrAttr:$type
  );

  let results = (outs
    Variadic<I32>:$ports
  );
}

#endif // DAP_OPS
