#ifndef CONSTRUCT_TILE_GRAPH_PASS
#define CONSTRUCT_TILE_GRAPH_PASS

include "mlir/Pass/PassBase.td"

def ConstructTileGraph: Pass<"construct-tile-graph", "::mlir::func::FuncOp"> {
  let summary = "";
  let description = [{
  }];
}

#endif // CONSTRUCT_TILE_GRAPH_PASS
