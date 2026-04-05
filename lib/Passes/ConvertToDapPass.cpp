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
#include "accelgen/Utils/GenericGraph.h"
#include "accelgen/Utils/DebugUtils.h"
#include "accelgen/Utils/AffineMapUtils.h"
#include "accelgen/Utils/ArchUtils.h"

namespace mlir::accelgen {
#define GEN_PASS_DEF_CONVERTTODAP
// #include "accelgen/Passes/MarkGenericPass.h.inc"
#include "accelgen/Passes/AccelgenPasses.h.inc"

namespace {

class ConvertToDap : public impl::ConvertToDapBase<ConvertToDap> {
 public:
  using impl::ConvertToDapBase<ConvertToDap>::ConvertToDapBase;

  void runOnOperation() final {
    mlir::MLIRContext* ctx = &getContext();
    auto module = getOperation();
    std::vector<mlir::func::FuncOp> funcs;
    module.walk([&](mlir::func::FuncOp funcOp) { funcs.push_back(funcOp); });

    mlir::OpBuilder builder(ctx);

    // Build all compute nodes and sram nodes for mapping
    ArchResource archCfg;
    archCfg.load(configPath);

    // Build main function
    builder.setInsertionPointToStart(module.getBody());
    auto mainFunc = builder.create<func::FuncOp>(
        UnknownLoc::get(ctx), "main_syn", FunctionType::get(ctx, {}, {}));
    auto entryBlock = mainFunc.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    builder.create<func::ReturnOp>(UnknownLoc::get(ctx));
    builder.setInsertionPointToStart(entryBlock);

    // SRAM pool
    llvm::SmallVector<mlir::Operation*> sramVec(archCfg.nSramBank);
    // Buid SRAM pool with single bank SRAM
    for (auto [indexSram, itemSram] : llvm::enumerate(sramVec)) {
      itemSram = builder.create<dap::SramOp>(
          UnknownLoc::get(ctx), archCfg.sramWidth, archCfg.sramDepth);
    }

    // Compute PE pool
    llvm::StringMap<llvm::SmallVector<mlir::Operation*>> computeVec;
    for (auto& [name, num] : archCfg.computeResource) {
      computeVec[name] = llvm::SmallVector<mlir::Operation*>(num);
      llvm::SmallVector<llvm::StringRef> tokens;
      name.split(tokens, "_");
      assert(tokens.size() > 2);
      llvm::SmallVector<mlir::Type> types;
      for (int64_t indexType = 1; indexType < tokens.size(); indexType++) {
        auto type = llvm::StringSwitch<mlir::Type>(tokens[indexType])
                        .Case("bf16", mlir::BFloat16Type::get(ctx))
                        .Case("fp32", mlir::Float32Type::get(ctx))
                        .Case("fp64", mlir::Float64Type::get(ctx))
                        .Default(nullptr);
        types.push_back(type);
      }
      auto builderFn =
          llvm::StringSwitch<std::function<mlir::Operation*()>>(tokens[0])
              .Case("mulf",
                    [&]() {
                      return builder.create<dap::MulfOp>(
                          UnknownLoc::get(ctx), types[0], types[1], types[2]);
                    })
              .Case("addf",
                    [&]() {
                      return builder.create<dap::AddfOp>(
                          UnknownLoc::get(ctx), types[0], types[1], types[2]);
                    })
              .Case("subf",
                    [&]() {
                      return builder.create<dap::SubfOp>(
                          UnknownLoc::get(ctx), types[0], types[1], types[2]);
                    })
              .Case("divf",
                    [&]() {
                      return builder.create<dap::DivfOp>(
                          UnknownLoc::get(ctx), types[0], types[1], types[2]);
                    })
              .Case("maximumf",
                    [&]() {
                      return builder.create<dap::MaxfOp>(
                          UnknownLoc::get(ctx), types[0], types[1], types[2]);
                    })
              .Case("negf",
                    [&]() {
                      return builder.create<dap::NegfOp>(UnknownLoc::get(ctx),
                                                         types[0], types[1]);
                    })
              .Case("truncf",
                    [&]() {
                      return builder.create<dap::TruncfOp>(UnknownLoc::get(ctx),
                                                           types[0], types[1]);
                    })
              .Case("extf",
                    [&]() {
                      return builder.create<dap::ExtfOp>(UnknownLoc::get(ctx),
                                                         types[0], types[1]);
                    })
              .Default(nullptr);

      if (!builderFn) {
        llvm::report_fatal_error("unknown op: " + tokens[0]);
      }
      for (auto [indexComp, itemComp] : llvm::enumerate(computeVec[name])) {
        itemComp = builderFn();
      }
    }

    // Build routing network, from SRAM -> PE -> SRAM

    for (auto [indexFuncOp, itemFuncOp] : llvm::enumerate(funcs)) {
      // Build resource pool
      ResourcePool sramPool(sramVec);
      llvm::StringMap<ResourcePool> computePool;
      for (auto& [compName, compVec] : computeVec) {
        computePool[compName] = ResourcePool(compVec);
      }

      auto graph = GenericGraph(itemFuncOp);
      llvm::DenseMap<mlir::OpOperand*, llvm::ArrayRef<mlir::Operation*>>
          valueSramMap;

      // For concat op, sram is shared for 2- input and 1- output, special
      // process
      for (auto tensorOp : graph.getTensorOps()) {
        if (auto concatOp = mlir::dyn_cast<tensor::ConcatOp>(tensorOp)) {
          // TODO : allocate banks
          assert(concatOp.getResult().getNumUses() == 1);
          int64_t nBank;
          llvm::ArrayRef<mlir::Operation*> sramOps;
          for (auto& use : concatOp->getUses()) {
            auto genericOp = mlir::dyn_cast<linalg::GenericOp>(use.getOwner());
            assert(genericOp);

            auto tiling = getTilingOfOperand(genericOp, &use);
            int64_t tileSize = 1;
            for (auto t : tiling) tileSize *= t;

            nBank = std::max(
                1UL, tileSize / (archCfg.sramDepth * archCfg.sramWidth /
                                 getElementTypeOrSelf(use.get().getType())
                                     .getIntOrFloatBitWidth()));

            sramOps = sramPool.get(nBank);
            valueSramMap[&use] = sramOps;
          }

          for (auto& opOperand : concatOp->getOpOperands()) {
            auto prevOpOperand = graph.getPreviousOpOperand(opOperand);
            assert(!valueSramMap.contains(prevOpOperand));
            valueSramMap[prevOpOperand] = sramOps;
          }
        }
      }

      // For other op, standard process
      for (auto genericOp : graph.getGenericOpsInTopoOrder()) {
        // For inputs, check whether is associated with an sram, yes -> direct
        // mapping, no -> create sram (block arguments)
        for (auto [indexIns, itemIns] :
             llvm::enumerate(genericOp.getInputs())) {
          auto& opOperand = genericOp->getOpOperand(indexIns);
          auto prevOpOperand = graph.getPreviousOpOperand(opOperand);
          // Sram has been created by concat op
          if (valueSramMap.contains(&opOperand)) {
            continue;
          }
          // Sram is not created, but has producer
          else if (valueSramMap.contains(prevOpOperand)) {
            valueSramMap[&opOperand] = valueSramMap[prevOpOperand];
          } else {
            auto tiling = getTilingOfOperand(genericOp, &opOperand);
            int64_t tileSize = 1;
            for (auto t : tiling) tileSize *= t;
            int64_t nBank = std::max(
                1UL, tileSize / (archCfg.sramDepth * archCfg.sramWidth /
                                 getElementTypeOrSelf(opOperand.get().getType())
                                     .getIntOrFloatBitWidth()));

            auto sramOps = sramPool.get(nBank);
            valueSramMap[&opOperand] = sramOps;
          }
        }
        // For results, build sram for each use, except for those have been
        // mapped to a sram
        for (auto& use : genericOp->getUses()) {
          // Sram has been created by concat op
          if (valueSramMap.contains(&use)) {
            continue;
          } else {
            auto& opOperand = genericOp->getOpOperands().back();
            auto tiling = getTilingOfOperand(genericOp, &opOperand);
            int64_t tileSize = 1;
            for (auto t : tiling) tileSize *= t;

            int64_t nBank = std::max(
                1UL, tileSize / (archCfg.sramDepth * archCfg.sramWidth /
                                 getElementTypeOrSelf(opOperand.get().getType())
                                     .getIntOrFloatBitWidth()));

            auto sramOps = sramPool.get(nBank);
            valueSramMap[&use] = sramOps;
          }
        }
      }

      // Build routing node and arith node
      for (auto genericOp : graph.getGenericOpsInTopoOrder()) {
        // Check arraysize. e.g. For unroll factor (1, 64, 64, 1) -> An array of
        // 64x64
        auto unrollFactorAttr =
            mlir::dyn_cast<ArrayAttr>(genericOp->getAttr("unroll_factor"));
        assert(unrollFactorAttr);
        llvm::SmallVector<int64_t> arraySize;
        llvm::SmallVector<int64_t> unrollDims;
        int64_t totalUnrollFactor = 1;
        for (auto [indexAttr, itemAttr] : llvm::enumerate(unrollFactorAttr)) {
          auto u =
              mlir::dyn_cast<IntegerAttr>(itemAttr).getValue().getZExtValue();
          if (u != 1) {
            arraySize.push_back(u);
            unrollDims.push_back(indexAttr);
          }
          totalUnrollFactor *= u;
        }

        // if (arraySize.size() != 2) {
        //   continue;
        // }
        assert(arraySize.size() == 2);
        // if (arraySize.size() == 1) arraySize.insert(arraySize.begin(),1);

        // Each operand is corresponding to a block argument
        std::vector<llvm::SmallVector<mlir::Value>> inputPorts(
            genericOp.getOperands().size());
        llvm::SmallVector<mlir::Value> outputPorts;
        for (auto [indexOperand, itemOperand] :
             llvm::enumerate(genericOp->getOpOperands())) {
          auto accDims = getAffineMapAccessDims(
              genericOp.getIndexingMapsArray()[indexOperand]);
          int64_t sramAccessVecLength = 1;
          for (auto dim : accDims) {
            sramAccessVecLength *=
                mlir::dyn_cast<IntegerAttr>(unrollFactorAttr[dim])
                    .getValue()
                    .getZExtValue();
          }

          // SRAM -> Compose -> Mux -> Demux -> Decompose
          auto sramOps = valueSramMap[&itemOperand];
          int64_t totalSramWidth = sramOps.size() * archCfg.sramWidth;
          int64_t nElement =
              totalSramWidth / getElementTypeOrSelf(itemOperand.get().getType())
                                   .getIntOrFloatBitWidth();
          assert(nElement % sramAccessVecLength == 0 ||
                 sramAccessVecLength % nElement == 0);

          int64_t nBankComposeGroup;
          int64_t nDemuxGroup;
          if (nElement < sramAccessVecLength) {
            nBankComposeGroup = sramOps.size();
            nDemuxGroup = sramAccessVecLength / nElement;
          } else {
            nBankComposeGroup =
                sramOps.size() / (nElement / sramAccessVecLength);
            nDemuxGroup = 1;
          }

          // Build Compose op
          llvm::SmallVector<mlir::Value> composePorts;
          for (int64_t indexStartBank = 0; indexStartBank < sramOps.size();
               indexStartBank += nBankComposeGroup) {
            auto composeOp = builder.create<dap::ComposeOp>(
                UnknownLoc::get(ctx), nBankComposeGroup);
            for (auto [indexPort, itemPort] :
                 llvm::enumerate(composeOp.getDins())) {
              builder.create<dap::DataPathOp>(
                  UnknownLoc::get(ctx),
                  mlir::dyn_cast<dap::SramOp>(
                      sramOps[indexStartBank + indexPort])
                      .getDout(),
                  itemPort);
            }
            composePorts.push_back(composeOp.getDout());
          }

          // Build Mux op
          auto muxOp = builder.create<dap::MuxOp>(UnknownLoc::get(ctx),
                                                  composePorts.size());
          for (auto [indexPort, itemPort] : llvm::enumerate(composePorts)) {
            builder.create<dap::DataPathOp>(UnknownLoc::get(ctx), itemPort,
                                            muxOp.getDins()[indexPort]);
          }

          // Build Demux op
          auto demuxOp =
              builder.create<dap::DemuxOp>(UnknownLoc::get(ctx), nDemuxGroup);
          builder.create<dap::DataPathOp>(UnknownLoc::get(ctx), muxOp.getDout(),
                                          demuxOp.getDin());
          int64_t nElementComposeGroup =
              nBankComposeGroup * archCfg.sramWidth /
              getElementTypeOrSelf(itemOperand.get().getType())
                  .getIntOrFloatBitWidth();

          llvm::SmallVector<mlir::Value> ports(sramAccessVecLength);
          for (auto [indexDemuxPort, itemDemuxPort] :
               llvm::enumerate(demuxOp.getDout())) {
            auto decomposeOp = builder.create<dap::DecomposeOp>(
                UnknownLoc::get(ctx), nElementComposeGroup);
            builder.create<dap::DataPathOp>(UnknownLoc::get(ctx), itemDemuxPort,
                                            decomposeOp.getDin());
            for (auto [indexDecomposePort, itemDecomposePort] :
                 llvm::enumerate(decomposeOp.getDouts())) {
              ports[indexDemuxPort * nElementComposeGroup +
                    indexDecomposePort] = itemDecomposePort;
            }
          }

          inputPorts[indexOperand] = ports;
        }

        // Build network for arith op
      }

      // assert(0);

      // for (auto [indexFuncOp, itemFuncOp] : llvm::enumerate(funcs)) {
      //   // Create func op
      //   builder.setInsertionPoint(itemFuncOp);
      //   auto dapFunc = builder.create<func::FuncOp>(
      //       itemFuncOp.getLoc(), itemFuncOp.getName().str() + "_dap",
      //       FunctionType::get(ctx, {}, {}));
      //   auto entryBlock = dapFunc.addEntryBlock();
      //   builder.setInsertionPointToStart(entryBlock);
      //   builder.create<func::ReturnOp>(UnknownLoc::get(ctx));
      //   builder.setInsertionPointToStart(entryBlock);

      //   // Build dap graph
      //   auto graph = GenericGraph(itemFuncOp);

      //   // Build resources pool

      //   llvm::DenseMap<mlir::OpOperand*, dap::SramOp> valueSramMap;

      //   // Build SRAM for all values

      //   // For concat op, sram is shared for 2- input and 1- output,
      //   special
      //   // process
      //   for (auto tensorOp : graph.getTensorOps()) {
      //     if (auto concatOp = mlir::dyn_cast<tensor::ConcatOp>(tensorOp)) {
      //       auto sramOp =
      //           builder.create<dap::SramOp>(concatOp.getLoc(), 512, 128, 1,
      //           2);
      //       for (auto& opOperand : concatOp->getOpOperands()) {
      //         auto prevOpOperand = graph.getPreviousOpOperand(opOperand);
      //         assert(!valueSramMap.contains(prevOpOperand));
      //         valueSramMap[prevOpOperand] = sramOp;
      //       }
      //       assert(concatOp.getResult().getNumUses() == 1);
      //       for (auto& use : concatOp->getUses()) {
      //         assert(mlir::dyn_cast<linalg::GenericOp>(use.getOwner()));
      //         valueSramMap[&use] = sramOp;
      //       }
      //     }
      //   }

      //   // For other op, standard process
      //   for (auto genericOp : graph.getGenericOpsInTopoOrder()) {
      //     // For inputs, check whether is associated with an sram, yes ->
      //     direct
      //     // mapping, no -> create sram (block arguments)
      //     for (auto [indexIns, itemIns] :
      //          llvm::enumerate(genericOp.getInputs())) {
      //       auto& opOperand = genericOp->getOpOperand(indexIns);
      //       auto prevOpOperand = graph.getPreviousOpOperand(opOperand);
      //       // Sram has been created by concat op
      //       if (valueSramMap.contains(&opOperand)) {
      //         continue;
      //       }
      //       // Sram is not created, but has producer
      //       else if (valueSramMap.contains(prevOpOperand)) {
      //         valueSramMap[&opOperand] = valueSramMap[prevOpOperand];
      //       } else {
      //         auto sramOp =
      //         builder.create<dap::SramOp>(opOperand.get().getLoc(),
      //                                                   512, 128, 1, 2);
      //         valueSramMap[&opOperand] = sramOp;
      //       }
      //     }
      //     // For results, build sram for each use, except for those have
      //     been
      //     // mapped to a sram
      //     for (auto& use : genericOp->getUses()) {
      //       // Sram has been created by concat op
      //       if (valueSramMap.contains(&use)) {
      //         continue;
      //       } else {
      //         auto sramOp =
      //             builder.create<dap::SramOp>(use.get().getLoc(), 512, 128,
      //             1, 2);
      //         valueSramMap[&use] = sramOp;
      //       }
      //     }
      //   }

      // // Build routing node and arith node
      // for (auto genericOp : graph.getGenericOpsInTopoOrder()) {
      //   // Check arraysize. e.g. For unroll factor (1, 64, 64, 1) -> An
      //   array of
      //   // 64x64
      //   auto unrollFactorAttr =
      //       mlir::dyn_cast<ArrayAttr>(genericOp->getAttr("unroll_factor"));
      //   assert(unrollFactorAttr);
      //   llvm::SmallVector<int64_t> arraySize;
      //   llvm::SmallVector<int64_t> unrollDims;
      //   int64_t totalUnrollFactor = 1;
      //   for (auto [indexAttr, itemAttr] :
      //   llvm::enumerate(unrollFactorAttr))
      //   {
      //     auto u =
      //         mlir::dyn_cast<IntegerAttr>(itemAttr).getValue().getZExtValue();
      //     if (u != 1) {
      //       arraySize.push_back(u);
      //       unrollDims.push_back(indexAttr);
      //     }
      //     totalUnrollFactor *= u;
      //   }

      //   if (arraySize.size() != 2) {
      //     continue;
      //   }
      //   assert(arraySize.size() == 2);
      //   // if (arraySize.size() == 1) arraySize.insert(arraySize.begin(),
      //   1);

      //   // Each operand is corresponding to a block argument
      //   std::vector<Array2D<mlir::Value>> arrayPorts;
      //   for (auto [indexOperand, itemOperand] :
      //        llvm::enumerate(genericOp->getOpOperands())) {
      //     // auto ports = Array2D<mlir::Value>(arraySize);
      //     auto accDims = getAffineMapAccessDims(
      //         genericOp.getIndexingMapsArray()[indexOperand]);
      //     int64_t sramAccessVecLength = 1;
      //     for (auto dim : accDims) {
      //       sramAccessVecLength *=
      //           mlir::dyn_cast<IntegerAttr>(unrollFactorAttr[dim])
      //               .getValue()
      //               .getZExtValue();
      //     }

      //     // Step 1 : Access all parallel elements with demux as a vectort,
      //     e.g.
      //     // (64x64) is demux with x64 vector acess, resulting in a x4096
      //     vector auto sramOp = valueSramMap[&itemOperand]; int64_t nBank =
      //     mlir::dyn_cast<IntegerAttr>(sramOp->getAttr("nbank"))
      //                         .getValue()
      //                         .getZExtValue();
      //     int64_t width =
      //     mlir::dyn_cast<IntegerAttr>(sramOp->getAttr("width"))
      //                         .getValue()
      //                         .getZExtValue();
      //     auto nElement = nBank * width /
      //                     getElementTypeOrSelf(itemOperand.get().getType())
      //                         .getIntOrFloatBitWidth();
      //     ECHO(sramAccessVecLength, "\n")
      //     ECHO(nElement, "\n")
      //     assert(sramAccessVecLength % nElement == 0);
      //     auto nDemuxPort = sramAccessVecLength / nElement;

      //     ECHO("try allocate 1 : ", "\n")
      //     ECHO(sramAccessVecLength, "\n")
      //     std::vector<mlir::Value> sramAccessVec(sramAccessVecLength);
      //     auto demuxOp = builder.create<dap::DemuxOp>(
      //         itemOperand.get().getLoc(), nDemuxPort);
      //     builder.create<dap::DataNodeOp>(itemOperand.get().getLoc(),
      //                                     sramOp.getDout(),
      //                                     demuxOp.getDin());

      //     for (auto [indexDemux, itemDemux] :
      //          llvm::enumerate(demuxOp.getResults())) {
      //       auto decomposeOp = builder.create<dap::DecomposeOp>(
      //           itemOperand.get().getLoc(), nElement);
      //       builder.create<dap::DataNodeOp>(itemOperand.get().getLoc(),
      //                                       itemDemux,
      //                                       decomposeOp.getDin());
      //       for (auto [indexDecomposeElement, itemDecomposeElement] :
      //            llvm::enumerate(decomposeOp.getResults())) {
      //         sramAccessVec[indexDemux * nElement + indexDecomposeElement]
      //         =
      //             itemDecomposeElement;
      //       }
      //     }

      //     // Step 2 : Feed parallel vector into PE array with broadcast op.
      //     // Vector -> Broadcast -> Vector (NDArray)

      //     ECHO("try allocate 2 : ", "\n")
      //     ECHO(totalUnrollFactor, "\n")
      //     std::vector<mlir::Value> peArrayVec(totalUnrollFactor);
      //     llvm::SmallVector<bool> broadcastDimMask;
      //     for (auto ud : unrollDims) {
      //       if (llvm::is_contained(accDims, ud))
      //         broadcastDimMask.push_back(false);
      //       else
      //         broadcastDimMask.push_back(true);
      //     }

      //     // Temporal. TODO : implement up to 3D support
      //     if (broadcastDimMask[0] && broadcastDimMask[1]) {
      //       assert(0);
      //     } else if (broadcastDimMask[0]) {
      //       for (auto [indexSramVec, itemSramVec] :
      //            llvm::enumerate(sramAccessVec)) {
      //         auto broadcastOp = builder.create<dap::BroadcastOp>(
      //             itemOperand.get().getLoc(), arraySize[1]);
      //         builder.create<dap::DataNodeOp>(itemOperand.get().getLoc(),
      //                                         itemSramVec,
      //                                         broadcastOp.getDin());
      //         for (auto [indexBroadcast, itemBroadcast] :
      //              llvm::enumerate(broadcastOp.getResults())) {
      //           peArrayVec[indexSramVec * arraySize[1] + indexBroadcast] =
      //               itemBroadcast;
      //         }
      //       }
      //     } else if (broadcastDimMask[1]) {
      //       for (auto [indexSramVec, itemSramVec] :
      //            llvm::enumerate(sramAccessVec)) {
      //         auto broadcastOp = builder.create<dap::BroadcastOp>(
      //             itemOperand.get().getLoc(), arraySize[0]);
      //         builder.create<dap::DataNodeOp>(itemOperand.get().getLoc(),
      //                                         itemSramVec,
      //                                         broadcastOp.getDin());
      //         for (auto [indexBroadcast, itemBroadcast] :
      //              llvm::enumerate(broadcastOp.getResults())) {
      //           peArrayVec[indexBroadcast * arraySize[0] + indexSramVec] =
      //               itemBroadcast;
      //         }
      //       }
      //     } else {
      //       for (auto [srcValue, destValue] :
      //            llvm::zip(sramAccessVec, peArrayVec))
      //         destValue = srcValue;
      //     }

      //     arrayPorts.push_back(Array2D<mlir::Value>(arraySize,
      //     peArrayVec));
      //   }
      // }
    }

    // for (auto funcOp : funcs) funcOp.erase();
  }
};

}  // namespace
}  // namespace mlir::accelgen
