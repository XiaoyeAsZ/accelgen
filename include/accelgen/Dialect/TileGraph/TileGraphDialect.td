#ifndef TILEGRAPH_DIALECT
#define TILEGRAPH_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Spe dialect definition.
//===----------------------------------------------------------------------===//

def TileGraph_Dialect : Dialect {
    let name = "tile_graph";
    let summary = "Tile Graph dialect.";
    let description = [{
    }];
    let cppNamespace = "::mlir::tile_graph";

    let useDefaultTypePrinterParser = 1;
    let extraClassDeclaration = [{
        void registerTypes();
    }];
}

//===----------------------------------------------------------------------===//
// Base spe operation definition.
//===----------------------------------------------------------------------===//

class TileGraph_Op<string mnemonic, list<Trait> traits = []> :
        Op<TileGraph_Dialect, mnemonic, traits>;


#endif // TILEGRAPH_DIALECT
