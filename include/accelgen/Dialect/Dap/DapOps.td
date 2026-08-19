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
    I32:$src,
    I32:$dest
  );

  let results = (outs
  );
}


def MulfOp : Dap_Op<"mulf", []> {
  let summary = "Mulf";
  let description = [{
  }];

  let arguments = (ins
  );

  let results = (outs
    AnyType:$Lhs,
    AnyType:$Rhs,
    AnyType:$result
  );
}

def AddfOp : Dap_Op<"add", []> {
  let summary = "Addf";
  let description = [{
  }];

  let arguments = (ins
  );

  let results = (outs
    AnyType:$Lhs,
    AnyType:$Rhs,
    AnyType:$result
  );
}

def SubfOp : Dap_Op<"sub", []> {
  let summary = "Subf";
  let description = [{
  }];

  let arguments = (ins
  );

  let results = (outs
    AnyType:$Lhs,
    AnyType:$Rhs,
    AnyType:$result
  );
}

def DivfOp : Dap_Op<"div", []> {
  let summary = "Divf";
  let description = [{
  }];

  let arguments = (ins
  );

  let results = (outs
    AnyType:$Lhs,
    AnyType:$Rhs,
    AnyType:$result
  );
}

def MaxfOp : Dap_Op<"maximumf", []> {
  let summary = "Maxf";
  let description = [{
  }];

  let arguments = (ins
  );

  let results = (outs
    AnyType:$Lhs,
    AnyType:$Rhs,
    AnyType:$result
  );
}

def NegfOp : Dap_Op<"neg", []> {
  let summary = "Negf";
  let description = [{
  }];

  let arguments = (ins
  );

  let results = (outs
    AnyType:$Src,
    AnyType:$result
  );
}

def TruncfOp : Dap_Op<"truncf", []> {
  let summary = "Negf";
  let description = [{
  }];

  let arguments = (ins
  );

  let results = (outs
    AnyType:$Src,
    AnyType:$result
  );
}

def ExtfOp : Dap_Op<"extf", []> {
  let summary = "Extf";
  let description = [{
  }];

  let arguments = (ins
  );

  let results = (outs
    AnyType:$Src,
    AnyType:$result
  );
}

def ErfOp : Dap_Op<"erf", []> {
  let summary = "Erf";
  let description = [{
  }];

  let arguments = (ins
  );

  let results = (outs
    AnyType:$Src,
    AnyType:$result
  );
}

def SramOp : Dap_Op<"sram", []> {
  let summary = "Sram";
  let description = [{
  }];

  let arguments = (ins
    I32Attr:$width,
    I32Attr:$depth
  );

  let results = (outs
    I32:$write,
    I32:$waddr,
    I32:$wstride,
    I32:$din,
    I32:$read,
    I32:$raddr,
    I32:$rstride,
    I32:$dout
  );
}

def DataNodeOp : Dap_Op<"data_node", []> {
  let summary = "Data Node";
  let description = [{
  }];

  let arguments = (ins
    I32Attr:$depth
  );

  let results = (outs
    AnyType:$src,
    AnyType:$result
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

def DecomposeOp : Dap_Op<"decompose", []> {
  let summary = "Decompose";
  let description = [{
  }];

  let arguments = (ins
    I32Attr:$fanout
  );

  let results = (outs
    I32:$din,
    Variadic<I32>:$douts
  );

  let builders = [
    OpBuilder<(ins "int64_t":$fanout)>
  ];
}

def ComposeOp : Dap_Op<"compose", []> {
  let summary = "Compose";
  let description = [{
  }];

  let arguments = (ins
    I32Attr:$fanin
  );

  let results = (outs
    Variadic<I32>:$dins,
    I32:$dout
  );

  let builders = [
    OpBuilder<(ins "int64_t":$fanin)>
  ];
}

def MuxOp : Dap_Op<"mux", []> {
  let summary = "Mux";
  let description = [{
  }];

  let arguments = (ins
    I32Attr:$fanin
  );

  let results = (outs
    Variadic<I32>:$dins,
    I32:$dout
  );

  let builders = [
    OpBuilder<(ins "int64_t":$fanin)>
  ];
}

def DemuxOp : Dap_Op<"demux", []> {
  let summary = "Demux";
  let description = [{
  }];

  let arguments = (ins
    I32Attr:$fanout
  );

  let results = (outs
    I32:$din,
    Variadic<I32>:$dout
  );

  let builders = [
    OpBuilder<(ins "int64_t":$fanout)>
  ];
}

def BroadcastOp : Dap_Op<"broadcast", []> {
  let summary = "Broadcast";
  let description = [{
  }];

  let arguments = (ins
    I32Attr:$fanout
  );

  let results = (outs
    I32:$din,
    Variadic<I32>:$dout
  );

  let builders = [
    OpBuilder<(ins "int64_t":$fanout)>
  ];
}

#endif // DAP_OPS
