//===- AccelgenDialect.td - Standalone dialect -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_DIALECT
#define DATAFLOW_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Dataflow dialect definition.
//===----------------------------------------------------------------------===//

def Dataflow_Dialect : Dialect {
    let name = "dataflow";
    let summary = "A standalone out-of-tree MLIR dialect.";
    let description = [{
        This dialect is an example of an out-of-tree MLIR dialect designed to
        illustrate the basic setup required to develop MLIR-based tools without
        working inside of the LLVM source tree.
    }];
    let cppNamespace = "::mlir::dataflow";

    let useDefaultTypePrinterParser = 1;
    let extraClassDeclaration = [{
        void registerTypes();
    }];
}

//===----------------------------------------------------------------------===//
// Base dataflow operation definition.
//===----------------------------------------------------------------------===//

class Dataflow_Op<string mnemonic, list<Trait> traits = []> :
        Op<Accelgen_Dialect, mnemonic, traits>;


//===----------------------------------------------------------------------===//
// Kernel operation definition
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Kernel operation definition.
//===----------------------------------------------------------------------===//

def KernelOp : Dataflow_Op<"kernel", [
    FunctionOpInterface, IsolatedFromAbove
  ]> {
  let summary = "user defined kernel operation";
  let description = [{
  }];

  // Attributes: name and function type, with optional per-arg/per-result attrs
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );

  // One region for the kernel body
  let regions = (region AnyRegion:$body);

  // Builders for convenience in C++
  let builders = [OpBuilder<(ins
    "StringRef":$name, "FunctionType":$type,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)
  >];

  let extraClassDeclaration = [{
  }];

  // We’ll define custom assembly format (like func @name(...) -> ...)
  let hasCustomAssemblyFormat = 1;
  let skipDefaultBuilders = 1;
}

def GemmOp : Dataflow_Op<"gemm", [Pure]> {
  let summary = "gemmop";
  let description = [{
  }];

  let arguments = (ins
    AnyRankedTensor:$lhs,
    AnyRankedTensor:$rhs,
    Optional<AnyRankedTensor>:$bias
  );

  let results = (outs AnyRankedTensor:$output);

  let assemblyFormat = [{
    $lhs `,` $rhs (`bias` `=` $bias^)? attr-dict `:` type($lhs) `,` type($rhs)
      (`,` type($bias)^)? `->` type($output)
  }];

  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

#endif // ACCELGEN_DIALECT
