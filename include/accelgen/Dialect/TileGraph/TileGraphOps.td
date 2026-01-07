//===- StandaloneOps.td - Standalone dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TILEGRAPH_OPS
#define TILEGRAPH_OPS

include "accelgen/Dialect/TileGraph/TileGraphDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Dialect/Linalg/IR/LinalgBase.td"
include "mlir/Dialect/Linalg/IR/LinalgInterfaces.td"

def ComputeNodeOp : TileGraph_Op<"compute_node", []> {
  let summary = "Compute Node";
  let description = [{
  }];

  let arguments = (ins
    Variadic<AnyTypeOf<[AnyTensor, AnyFloat, AnyInteger]>>:$operands,
    AffineMapArrayAttr:$indexing_maps,
    IteratorTypeArrayAttr:$iterator_types,
    I64ArrayAttr:$tiling_vector,
    I64ArrayAttr:$unroll_vector,
    I64ArrayAttr:$l2_order,
    I64ArrayAttr:$l1_order
  );

  let results = (outs
    Variadic<AnyTensor>:$results
  );
  
}

def DataPathOp : TileGraph_Op<"data_path", []> {
  let summary = "Data Path";
  let description = [{
  }];

  let arguments = (ins
    AnyTensor:$operand,
    StrAttr:$forward_path
  );

  let results = (outs
    AnyTensor:$result
  );
}

#endif // TILEGRAPH_OPS
