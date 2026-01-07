//===- StandaloneOps.td - Standalone dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SPE_OPS
#define SPE_OPS

include "accelgen/Dialect/Spe/SpeDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def MacOp : Spe_Op<"mac", []> {
  let summary = "Multiplication and accumulation PE";
  let description = [{
  }];

  let arguments = (ins
    AnyTypeOf<[AnyFloat, AnyInteger, AnyMemRef]>:$a,
    AnyTypeOf<[AnyFloat, AnyInteger, AnyMemRef]>:$b,
    AnyTypeOf<[AnyFloat, AnyInteger, AnyMemRef]>:$c
  );

  let results = (outs
    AnyTypeOf<[AnyFloat, AnyInteger]>:$result
  );

  let builders = [
    OpBuilder<(ins
    "ValueRange":$operands,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)
    >
  ];
}

def ExpOp : Spe_Op<"exp", []> {
  let summary = "Exp PE";
  let description = [{
  }];

  let arguments = (ins
    AnyTypeOf<[AnyFloat, AnyInteger]>:$a
  );

  let results = (outs
    AnyTypeOf<[AnyFloat, AnyInteger]>:$result
  );

  let builders = [
    OpBuilder<(ins
    "ValueRange":$operands,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)
    >
  ];
}

def NegOp : Spe_Op<"neg", []> {
  let summary = "Neg PE";
  let description = [{
  }];

  let arguments = (ins
    AnyTypeOf<[AnyFloat, AnyInteger]>:$a
  );

  let results = (outs
    AnyTypeOf<[AnyFloat, AnyInteger]>:$result
  );

  let builders = [
    OpBuilder<(ins
    "ValueRange":$operands,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)
    >
  ];
}

def InvOp : Spe_Op<"inv", []> {
  let summary = "Inv PE";
  let description = [{
  }];

  let arguments = (ins
    AnyTypeOf<[AnyFloat, AnyInteger]>:$a
  );

  let results = (outs
    AnyTypeOf<[AnyFloat, AnyInteger]>:$result
  );

  let builders = [
    OpBuilder<(ins
    "ValueRange":$operands,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)
    >
  ];
}

def AddOp : Spe_Op<"add", []> {
  let summary = "Add PE";
  let description = [{
  }];

  let arguments = (ins
    AnyTypeOf<[AnyFloat, AnyInteger]>:$a,
    AnyTypeOf<[AnyFloat, AnyInteger]>:$b
  );

  let results = (outs
    AnyTypeOf<[AnyFloat, AnyInteger]>:$result
  );

  let builders = [
    OpBuilder<(ins
    "ValueRange":$operands,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)
    >
  ];
}

def MulOp : Spe_Op<"mul", []> {
  let summary = "Mul PE";
  let description = [{
  }];

  let arguments = (ins
    AnyTypeOf<[AnyFloat, AnyInteger]>:$a,
    AnyTypeOf<[AnyFloat, AnyInteger]>:$b
  );

  let results = (outs
    AnyTypeOf<[AnyFloat, AnyInteger]>:$result
  );

  let builders = [
    OpBuilder<(ins
    "ValueRange":$operands,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)
    >
  ];
}

#endif // SPE_OPS
