#ifndef DAP_DIALECT
#define DAP_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Dap dialect definition.
//===----------------------------------------------------------------------===//

def Dap_Dialect : Dialect {
    let name = "dap";
    let summary = "Dataflow Architecture Primitive (DAP) dialect.";
    let description = [{
    }];
    let cppNamespace = "::mlir::dap";

    let useDefaultTypePrinterParser = 1;
    let extraClassDeclaration = [{
        void registerTypes();
    }];
}

//===----------------------------------------------------------------------===//
// Base Dap operation definition.
//===----------------------------------------------------------------------===//

class Dap_Op<string mnemonic, list<Trait> traits = []> :
        Op<Dap_Dialect, mnemonic, traits>;


#endif // DAP_DIALECT
