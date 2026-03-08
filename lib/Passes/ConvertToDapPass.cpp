#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include <assert.h>
#include <queue>
#include <vector>
#include "accelgen/Dialect/Dap/DapOps.h"

#include "accelgen/Utils/OperationUtils.h"

#include "accelgen/Passes/AccelgenPasses.h"

namespace mlir::accelgen {
#define GEN_PASS_DEF_CONVERTTODAP
// #include "accelgen/Passes/MarkGenericPass.h.inc"
#include "accelgen/Passes/AccelgenPasses.h.inc"

namespace {

class ConvertToDap : public impl::ConvertToDapBase<ConvertToDap> {
 public:
  using impl::ConvertToDapBase<ConvertToDap>::ConvertToDapBase;

  void runOnOperation() final {
    auto& ctx = getContext();
    auto module = getOperation();
    std::vector<mlir::func::FuncOp> funcs;
    module.walk([&](mlir::func::FuncOp funcOp) { funcs.push_back(funcOp); });

    for (auto funcOp : funcs) {
      OpBuilder builder(funcOp);
      auto funcType = builder.getFunctionType(TypeRange{}, TypeRange{});
      auto dapFunc = builder.create<mlir::func::FuncOp>(
          funcOp.getLoc(), funcOp.getName().str() + "_dap", funcType);
      auto entryBlock = dapFunc.addEntryBlock();

      std::vector<mlir::Operation*> ops;
      auto opsTopOrder = getTopoOrder(ops);

      llvm::DenseMap<mlir::Value, dap::SramOp> sramMap;

      const int64_t sramWidth = 256;  // fixed width : 256 bits

      for (auto op : opsTopOrder) {
        auto generic = mlir::dyn_cast<linalg::GenericOp>(op);
        if (!generic) continue;
        for (auto operand : generic.getOperands()) {
          if (!sramMap.contains(operand)) {
            builder.setInsertionPointToStart(entryBlock);
            // Analyze sram parameter

            // Calculate total sram size needed for a tile
            mlir::ArrayAttr tilingSize = mlir::dyn_cast<mlir::ArrayAttr>(
                generic->getAttr("tiling_size"));
            assert(tilingSize);
            int64_t tileSize = 1;
            for (auto t : tilingSize) {
              tileSize *= mlir::dyn_cast<mlir::IntegerAttr>(t).getInt();
            }

            // with fixed width, Depth = total size / width
            int64_t depth = tileSize / sramWidth;
            assert(tileSize % sramWidth == 0);

            // Calculate maximum parallel data access



            // sramMap[operand] =
            //     builder.create<dap::SramOp>(generic.getLoc(),
            //                                 256,  // fixed sram width
            //     );
          }
        }
      }

      funcOp.walk([&](linalg::GenericOp generic) {

      });
    }
  }
};

}  // namespace
}  // namespace mlir::accelgen
