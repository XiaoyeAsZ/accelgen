#ifndef SPE_DIALECT
#define SPE_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Spe dialect definition.
//===----------------------------------------------------------------------===//

def Spe_Dialect : Dialect {
    let name = "spe";
    let summary = "Standard / Specialized process element dialect.";
    let description = [{
    }];
    let cppNamespace = "::mlir::spe";

    let useDefaultTypePrinterParser = 1;
    let extraClassDeclaration = [{
        void registerTypes();
    }];
}

//===----------------------------------------------------------------------===//
// Base spe operation definition.
//===----------------------------------------------------------------------===//

class Spe_Op<string mnemonic, list<Trait> traits = []> :
        Op<Spe_Dialect, mnemonic, traits>;


#endif // SPE_DIALECT
