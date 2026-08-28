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
#include "accelgen/Passes/AccelgenPasses.h"
#include "accelgen/Utils/AffineMapUtils.h"
#include "accelgen/Utils/ArchUtils.h"
#include "accelgen/Utils/DebugUtils.h"
#include "accelgen/Utils/GenericGraph.h"
#include "accelgen/Utils/OperationUtils.h"

namespace mlir::accelgen {
#define GEN_PASS_DEF_CONVERTTODAP
// #include "accelgen/Passes/MarkGenericPass.h.inc"
#include "accelgen/Passes/AccelgenPasses.h.inc"

namespace {

class ConvertToDap : public impl::ConvertToDapBase<ConvertToDap> {
public:
  using impl::ConvertToDapBase<ConvertToDap>::ConvertToDapBase;

  void runOnOperation() final {
    mlir::MLIRContext *ctx = &getContext();
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
    llvm::SmallVector<mlir::Operation *> sramVec(archCfg.nSramBank);
    // Buid SRAM pool with single bank SRAM
    for (auto [indexSram, itemSram] : llvm::enumerate(sramVec)) {
      itemSram = builder.create<dap::SramOp>(
          UnknownLoc::get(ctx), archCfg.sramWidth, archCfg.sramDepth);
    }

    // DataNode pool
    llvm::SmallVector<mlir::Operation *> dataNodeVec(archCfg.nDataNode);
    for (auto [indexReg, itemReg] : llvm::enumerate(dataNodeVec)) {
      itemReg = builder.create<dap::DataNodeOp>(
          UnknownLoc::get(ctx),
          mlir::TypeRange(
              {mlir::BFloat16Type::get(ctx), mlir::BFloat16Type::get(ctx)}),
          1);
    }

    // Compute PE pool
    llvm::StringMap<llvm::SmallVector<mlir::Operation *>> computeVec;
    for (auto &[name, num] : archCfg.computeResource) {
      computeVec[name] = llvm::SmallVector<mlir::Operation *>(num);
      llvm::SmallVector<llvm::StringRef> tokens;
      name.split(tokens, "_");
      assert(tokens.size() > 2);
      llvm::SmallVector<mlir::Type> types;
      for (int64_t indexType = 1; indexType < tokens.size(); indexType++) {
        auto typeToken = tokens[indexType];
        mlir::Type type;
        if (typeToken.consume_front("i")) {
          unsigned width = 0;
          if (typeToken.getAsInteger(10, width))
            llvm::report_fatal_error("invalid integer PE type: " +
                                     tokens[indexType]);
          type = mlir::IntegerType::get(ctx, width);
        } else {
          type = llvm::StringSwitch<mlir::Type>(tokens[indexType])
                     .Case("bf16", mlir::BFloat16Type::get(ctx))
                     .Case("fp32", mlir::Float32Type::get(ctx))
                     .Case("fp64", mlir::Float64Type::get(ctx))
                     .Default(nullptr);
        }
        if (!type)
          llvm::report_fatal_error("unknown PE type: " + tokens[indexType]);
        types.push_back(type);
      }
      auto builderFn =
          llvm::StringSwitch<std::function<mlir::Operation *()>>(tokens[0])
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
              .Case("erf",
                    [&]() {
                      return builder.create<dap::ErfOp>(UnknownLoc::get(ctx),
                                                        types[0], types[1]);
                    })
              .Case("exp",
                    [&]() {
                      return builder.create<dap::ExpOp>(UnknownLoc::get(ctx),
                                                        types[0], types[1]);
                    })
              .Case("rsqrt",
                    [&]() {
                      return builder.create<dap::RsqrtOp>(UnknownLoc::get(ctx),
                                                          types[0], types[1]);
                    })
              .Case("sqrt",
                    [&]() {
                      return builder.create<dap::SqrtOp>(UnknownLoc::get(ctx),
                                                         types[0], types[1]);
                    })
              .Case("fpowi",
                    [&]() {
                      return builder.create<dap::FPowIOp>(
                          UnknownLoc::get(ctx), types[0], types[1], types[2]);
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

    // Build resource mapping

    llvm::SmallVector<linalg::GenericOp> genericOps;
    llvm::DenseMap<
        mlir::Operation *,
        llvm::DenseMap<mlir::Value, llvm::ArrayRef<mlir::Operation *>>>
        valueSramMap;
    llvm::DenseMap<
        mlir::Operation *,
        llvm::DenseMap<mlir::Value, llvm::ArrayRef<mlir::Operation *>>>
        valueRegMap;
    llvm::DenseMap<mlir::Operation *,
                   llvm::StringMap<llvm::ArrayRef<mlir::Operation *>>>
        valueComputeResourceMap;

    for (auto [indexFuncOp, itemFuncOp] : llvm::enumerate(funcs)) {
      // ECHO("func", "\n")
      // ECHO(indexFuncOp, "\n")

      // Build resource pool
      ResourcePool sramPool(sramVec);
      // ECHO("allocate sram", "\n")
      // ECHO(sramVec.size(), "\n")
      ResourcePool regPool(dataNodeVec);
      llvm::StringMap<ResourcePool> computePool;
      for (auto &[compName, compVec] : computeVec) {
        computePool[compName] = ResourcePool(compVec);
      }

      auto graph = GenericGraph(itemFuncOp);

      // For concat op, sram is shared for 2- input and 1- output, special
      // process
      for (auto tensorOp : graph.getTensorOps()) {
        if (auto concatOp = mlir::dyn_cast<tensor::ConcatOp>(tensorOp)) {
          // TODO : allocate banks
          assert(concatOp.getResult().getNumUses() == 1);
          int64_t nBank;
          llvm::ArrayRef<mlir::Operation *> sramOps;
          for (auto &use : concatOp->getUses()) {
            auto genericOp = mlir::dyn_cast<linalg::GenericOp>(use.getOwner());
            assert(genericOp);

            auto tiling = getTilingOfOperand(genericOp, &use);
            int64_t tileSize = 1;
            for (auto t : tiling)
              tileSize *= t;

            nBank = std::max(1UL, tileSize /
                                      (archCfg.sramDepth * archCfg.sramWidth /
                                       getElementTypeOrSelf(use.get().getType())
                                           .getIntOrFloatBitWidth()));

            // ECHO("concat consume sram", "\n")
            // ECHO(nBank, "\n")
            sramOps = sramPool.get(nBank);

            valueSramMap[genericOp][use.get()] = sramOps;
          }

          for (auto &opOperand : concatOp->getOpOperands()) {
            auto prevOpOperand = graph.getPreviousOpOperand(opOperand);
            auto prevGeneric = prevOpOperand->get().getDefiningOp();
            assert(!valueSramMap.contains(prevGeneric));
            valueSramMap[prevGeneric][prevOpOperand->get()] = sramOps;
          }
        }
      }

      auto genericVec = graph.getGenericOpsInTopoOrder();
      genericOps.append(genericVec.begin(), genericVec.end());

      // For other op, standard process
      for (auto genericOp : genericVec) {
        // For inputs, check whether is associated with an sram, yes -> direct
        // mapping, no -> create sram (block arguments)
        for (auto [indexIns, itemIns] :
             llvm::enumerate(genericOp.getInputs())) {
          auto &opOperand = genericOp->getOpOperand(indexIns);
          auto prevOpOperand = graph.getPreviousOpOperand(opOperand);
          // Sram has been created by concat op
          if (valueSramMap.contains(genericOp) &&
              valueSramMap[genericOp][itemIns].size() != 0) {
            continue;
          }
          // Sram is not created, but has producer
          else if (valueSramMap.contains(
                       prevOpOperand->get().getDefiningOp()) &&
                   valueSramMap[prevOpOperand->get().getDefiningOp()]
                               [prevOpOperand->get()]
                                   .size() != 0) {
            valueSramMap[genericOp][itemIns] =
                valueSramMap[prevOpOperand->get().getDefiningOp()]
                            [prevOpOperand->get()];
          } else {
            auto tiling = getTilingOfOperand(genericOp, &opOperand);
            int64_t tileSize = 1;
            for (auto t : tiling)
              tileSize *= t;
            int64_t nBank = std::max(
                1UL, tileSize / (archCfg.sramDepth * archCfg.sramWidth /
                                 getElementTypeOrSelf(opOperand.get().getType())
                                     .getIntOrFloatBitWidth()));

            // ECHO("input consume sram", "\n")
            // ECHO(nBank, "\n")
            auto sramOps = sramPool.get(nBank);

            valueSramMap[genericOp][itemIns] = sramOps;
          }
        }
        // For results, build sram for each use, except for those have been
        // mapped to a sram
        assert(genericOp.getResults().size() == 1);
        for (auto &use : genericOp->getUses()) {
          // Sram has been created by concat op
          if (valueSramMap.contains(use.getOwner()) &&
              valueSramMap[use.getOwner()][use.get()].size() != 0) {
            continue;
          } else {
            auto &opOperand = genericOp->getOpOperands().back();
            auto tiling = getTilingOfOperand(genericOp, &opOperand);
            int64_t tileSize = 1;
            for (auto t : tiling)
              tileSize *= t;

            int64_t nBank = std::max(
                1UL, tileSize / (archCfg.sramDepth * archCfg.sramWidth /
                                 getElementTypeOrSelf(opOperand.get().getType())
                                     .getIntOrFloatBitWidth()));

            // ECHO("result consume sram", "\n")
            // ECHO(nBank, "\n")
            auto sramOps = sramPool.get(nBank);

            valueSramMap[genericOp][use.get()] = sramOps;
          }
        }

        auto unrollFactorAttr =
            mlir::dyn_cast<ArrayAttr>(genericOp->getAttr("unroll_factor"));
        uint64_t totalUntrollSize = 1;
        for (auto attr : unrollFactorAttr) {
          auto u = mlir::dyn_cast<IntegerAttr>(attr).getValue().getZExtValue();
          totalUntrollSize *= u;
        }

        // Allocate register resource
        for (auto [idxOperand, itemOperand] :
             llvm::enumerate(genericOp.getOperands())) {
          auto dataNodeOpVec = regPool.get(totalUntrollSize);
          valueRegMap[genericOp][itemOperand] = dataNodeOpVec;
        }

        // Allocate compute resources
        genericOp.walk([&](mlir::Operation *op) {
          if (mlir::isa<mlir::arith::ConstantOp>(op)) {
            return;
          }
          if (mlir::isa<mlir::arith::ArithDialect>(op->getDialect()) ||
              mlir::isa<mlir::math::MathDialect>(op->getDialect())) {
            std::string resourceName = getPeResourceName(op);
            valueComputeResourceMap[genericOp][resourceName] =
                computePool[resourceName].get(totalUntrollSize);
          } else if (mlir::isa<linalg::YieldOp>(op))
            ;
          else
            ;
        });
      }

      // for (auto funcOp : funcs) funcOp.erase();
    }

    // Build routing primitives for each generic op
    for (auto &genericOp : genericOps) {
      auto unrollFactorAttr =
          mlir::dyn_cast<ArrayAttr>(genericOp->getAttr("unroll_factor"));
      llvm::SmallVector<uint64_t> unrollFactor;
      llvm::SmallVector<uint64_t> arraySize;
      uint64_t totalUntrollSize = 1;
      for (auto attr : unrollFactorAttr) {
        auto u = mlir::dyn_cast<IntegerAttr>(attr).getValue().getZExtValue();
        unrollFactor.push_back(u);
        if (u > 1)
          arraySize.push_back(u);
        totalUntrollSize *= u;
      }
      assert(arraySize.size() == 1 || arraySize.size() == 2);
      while (arraySize.size() < 2)
        arraySize.push_back(1);

      // Build datanodes array for each operand
      // llvm::SmallVector<llvm::SmallVector<dap::DataNodeOp>> dataNodeVecs;
      // for (auto operand : genericOp.getOperands()) {
      //   llvm::SmallVector<dap::DataNodeOp> nodes;
      //   for (size_t i = 0; i < totalUntrollSize; i++) {
      //     auto dataNodeOp = builder.create<dap::DataNodeOp>(
      //         UnknownLoc::get(ctx),
      //         mlir::TypeRange({operand.getType(), operand.getType()}), 1);
      //     nodes.push_back(dataNodeOp);
      //   }
      //   dataNodeVecs.push_back(nodes);
      // }

      llvm::SmallVector<Array2D<mlir::Operation *>> regArrays;

      // Routing data from SRAM - > DataNode
      for (auto [idxOperand, itemOperand] :
           llvm::enumerate(genericOp.getInputs())) {
        llvm::SmallVector<uint64_t> operandSize;
        auto dims = getAffineMapAccessDims(
            genericOp.getIndexingMapsArray()[idxOperand]);
        uint64_t operandTotalFactor = 1;
        for (auto d : dims) {
          auto u = unrollFactor[d];
          if (u > 1)
            operandSize.push_back(u);
          operandTotalFactor *= u;
        }
        assert(operandSize.size() == 1 || operandSize.size() == 2);
        while (operandSize.size() < 2)
          operandSize.push_back(1);

        auto sramVec = valueSramMap[genericOp][itemOperand];
        auto itemElementType = getElementTypeOrSelf(itemOperand.getType());
        uint64_t nElementSram = sramVec.size() * archCfg.sramWidth /
                                itemElementType.getIntOrFloatBitWidth();
        llvm::SmallVector<mlir::Value> sramPortVec;
        sramPortVec.reserve(nElementSram);
        for (auto &sram : sramVec) {
          uint64_t fanout =
              archCfg.sramWidth / itemElementType.getIntOrFloatBitWidth();
          auto decomposeOp =
              builder.create<dap::DecomposeOp>(UnknownLoc::get(ctx), fanout);
          builder.create<dap::DataPathOp>(
              UnknownLoc::get(ctx), mlir::dyn_cast<dap::SramOp>(sram).getDout(),
              decomposeOp.getDin());
          auto ports = decomposeOp.getDouts();
          sramPortVec.append(ports.begin(), ports.end());
        }

        auto regVec = valueRegMap[genericOp][itemOperand];
        Array2D<mlir::Operation *> regArray(arraySize, regVec);
        regArrays.push_back(regArray);

        llvm::SmallVector<mlir::Operation *> boundryDataNodes;

        if (operandSize[0] < arraySize[0] &&
            operandSize[1] == arraySize[1]) // 1 row -> broadcast -> array
        {
          boundryDataNodes = regArray.row(0);
        } else if (operandSize[0] == arraySize[0] &&
                   operandSize[1] < arraySize[1]) // 1 col -> broadcast -> array

        {
          boundryDataNodes = regArray.col(0);
        } else if (operandSize[0] == arraySize[0] &&
                   operandSize[1] == arraySize[1]) // already array
        {
          boundryDataNodes = regArray.array();
        } else if (operandSize[0] == arraySize[1] &&
                   operandSize[1] == arraySize[0]) // transposed array
        {
          boundryDataNodes = regArray.transpose();
        } else if (operandSize[0] == arraySize[1] &&
                   operandSize[1] == 1) // transposed 1-row broadcast
        {
          boundryDataNodes = regArray.row(0);
        } else if (operandSize[0] == 1 &&
                   operandSize[1] == arraySize[0]) // transposed 1-col broadcast
        {
          boundryDataNodes = regArray.col(0);
        } else if (operandSize[0] == 1 && operandSize[1] == 1) // scalar
        {
          boundryDataNodes = regArray.row(0);
        } else if (operandSize[0] == arraySize[1] &&
                   operandSize[1] == 1) // transpose-expand
        {
          boundryDataNodes = regArray.row(0);
        } else
          assert(0);

        if (nElementSram >= operandTotalFactor) {
          for (unsigned i = 0; i < nElementSram; i++) {
            if (i >= operandTotalFactor)
              break;
            builder.create<dap::DataPathOp>(
                UnknownLoc::get(ctx), sramPortVec[i],
                mlir::dyn_cast<dap::DataNodeOp>(boundryDataNodes[i]).getSrc());
          }
        } else {
          assert(operandTotalFactor % nElementSram == 0);
          auto nDemux = operandTotalFactor / nElementSram;
          llvm::SmallVector<dap::DemuxOp> demuxOps;
          llvm::SmallVector<mlir::Value> demuxPorts(nDemux *
                                                    sramPortVec.size());
          for (auto [idxPort, itemPort] : llvm::enumerate(sramPortVec)) {
            auto demuxOp =
                builder.create<dap::DemuxOp>(UnknownLoc::get(ctx), nDemux);
            builder.create<dap::DataPathOp>(UnknownLoc::get(ctx), itemPort,
                                            demuxOp.getDin());
            for (uint64_t i = 0; i < nDemux; i++) {
              demuxPorts[i * sramPortVec.size() + idxPort] =
                  demuxOp.getDout()[i];
            }
          }
          for (auto [port, op] : llvm::zip(demuxPorts, boundryDataNodes)) {
            auto dataNodeOp = mlir::dyn_cast<dap::DataNodeOp>(op);
            builder.create<dap::DataPathOp>(UnknownLoc::get(ctx), port,
                                            dataNodeOp.getSrc());
          }
        }

        if (idxOperand >= genericOp.getInputs().size())
          continue;

        // Broadcast
        if (operandSize[0] < arraySize[0] &&
            operandSize[1] == arraySize[1]) // 1 row -> broadcast -> array
        {
          for (uint64_t r = 0; r < arraySize[0]; r++) {
            for (uint64_t c = 0; c < arraySize[1]; c++) {
              if (r > 0) {
                builder.create<dap::DataPathOp>(
                    UnknownLoc::get(ctx),
                    mlir::dyn_cast<dap::DataNodeOp>(regArray.at(r - 1, c))
                        .getResult(),
                    mlir::dyn_cast<dap::DataNodeOp>(regArray.at(r, c))
                        .getSrc());
              }
            }
          }
        } else if (operandSize[0] == arraySize[0] &&
                   operandSize[1] < arraySize[1]) // 1 col -> broadcast -> array

        {
          for (uint64_t r = 0; r < arraySize[0]; r++) {
            for (uint64_t c = 0; c < arraySize[1]; c++) {
              if (c > 0) {
                builder.create<dap::DataPathOp>(
                    UnknownLoc::get(ctx),
                    mlir::dyn_cast<dap::DataNodeOp>(regArray.at(r, c - 1))
                        .getResult(),
                    mlir::dyn_cast<dap::DataNodeOp>(regArray.at(r, c))
                        .getSrc());
              }
            }
          }
        } else if (operandSize[0] == arraySize[0] &&
                   operandSize[1] == arraySize[1]) // already array
        {
        } else if (operandSize[0] == arraySize[1] &&
                   operandSize[1] == arraySize[0]) // transposed array

        {
        } else if (operandSize[0] == 1 && operandSize[1] == 1) // scalar
        {
          for (uint64_t r = 0; r < arraySize[0]; r++) {
            for (uint64_t c = 0; c < arraySize[1]; c++) {
              if (r > 0) {
                builder.create<dap::DataPathOp>(
                    UnknownLoc::get(ctx),
                    mlir::dyn_cast<dap::DataNodeOp>(regArray.at(r - 1, c))
                        .getResult(),
                    mlir::dyn_cast<dap::DataNodeOp>(regArray.at(r, c))
                        .getSrc());
              } else if (c > 0) {
                builder.create<dap::DataPathOp>(
                    UnknownLoc::get(ctx),
                    mlir::dyn_cast<dap::DataNodeOp>(regArray.at(r, c - 1))
                        .getResult(),
                    mlir::dyn_cast<dap::DataNodeOp>(regArray.at(r, c))
                        .getSrc());
              }
            }
          }
        } else if (operandSize[0] == arraySize[1] && operandSize[1] == 1) {
          for (uint64_t r = 0; r < arraySize[0]; r++) {
            for (uint64_t c = 0; c < arraySize[1]; c++) {
              if (r > 0) {
                builder.create<dap::DataPathOp>(
                    UnknownLoc::get(ctx),
                    mlir::dyn_cast<dap::DataNodeOp>(regArray.at(r - 1, c))
                        .getResult(),
                    mlir::dyn_cast<dap::DataNodeOp>(regArray.at(r, c))
                        .getSrc());
              }
            }
          }
        } else {
          ECHO_LIST(operandSize, ",")
          ECHO("", "\n")
          ECHO_LIST(arraySize, ",")
          ECHO("", "\n")
          assert(0);
        }
      }

      // The linalg body also exposes the output/init value as a block
      // argument. Keep its register array available for compute routing;
      // output SRAM routing is handled separately below.
      auto outputOperand = genericOp.getOutputs().front();
      regArrays.push_back(Array2D<mlir::Operation *>(
          arraySize, valueRegMap[genericOp][outputOperand]));

      llvm::SmallVector<mlir::Value> arithResults;
      llvm::DenseMap<mlir::Value, llvm::SmallVector<mlir::Value>>
          constSourcePorts;

      // Build routing for arith op
      genericOp.walk([&](mlir::Operation *op) {
        auto getSourcePorts = [&](mlir::Value value) {
          llvm::SmallVector<mlir::Value> sourcePorts;

          if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
            if (mlir::isa<mlir::func::FuncOp>(
                    blockArg.getOwner()->getParentOp()) &&
                !mlir::isa<mlir::ShapedType>(value.getType())) {
              mlir::Attribute constValue;
              if (auto floatType =
                      mlir::dyn_cast<mlir::FloatType>(value.getType())) {
                constValue = mlir::FloatAttr::get(floatType, 1.0);
              } else if (auto integerType = mlir::dyn_cast<mlir::IntegerType>(
                             value.getType())) {
                constValue = mlir::IntegerAttr::get(integerType, 1);
              } else if (mlir::isa<mlir::IndexType>(value.getType())) {
                constValue = builder.getIndexAttr(1);
              } else {
                llvm::report_fatal_error(
                    "unsupported scalar constant argument type");
              }

              auto &ports = constSourcePorts[value];
              if (ports.empty()) {
                ports.reserve(totalUntrollSize);
                for (uint64_t i = 0; i < totalUntrollSize; ++i) {
                  auto dapConstOp = builder.create<dap::ConstOp>(
                      UnknownLoc::get(ctx), constValue);
                  ports.push_back(dapConstOp.getResult());
                }
              }
              sourcePorts.append(ports.begin(), ports.end());
              return sourcePorts;
            }

            auto regArray = regArrays[blockArg.getArgNumber()];
            for (auto reg : regArray.array()) {
              sourcePorts.push_back(
                  mlir::cast<dap::DataNodeOp>(reg).getResult());
            }
            return sourcePorts;
          }

          auto *definingOp = value.getDefiningOp();
          assert(definingOp && "expected a block argument or PE result");
          auto definingPeResources =
              valueComputeResourceMap[genericOp][getPeResourceName(definingOp)];
          Array2D<mlir::Operation *> definingPeArray(arraySize,
                                                     definingPeResources);
          for (auto *definingPe : definingPeArray.array())
            sourcePorts.push_back(definingPe->getResults().back());
          return sourcePorts;
        };

        auto routeComputeOp = [&](mlir::Operation *computeOp) {
          auto peResources =
              valueComputeResourceMap[genericOp][getPeResourceName(computeOp)];
          Array2D<mlir::Operation *> peArray(arraySize, peResources);
          for (auto [idxOperand, operand] :
               llvm::enumerate(computeOp->getOperands())) {
            auto sourcePorts = getSourcePorts(operand);
            for (auto [source, pe] :
                 llvm::zip_equal(sourcePorts, peArray.array())) {
              builder.create<dap::DataPathOp>(
                  UnknownLoc::get(ctx), source,
                  pe->getResult(static_cast<unsigned>(idxOperand)));
            }
          }
        };

        mlir::TypeSwitch<mlir::Operation *>(op)
            .Case<mlir::arith::MulFOp, mlir::arith::AddFOp, mlir::arith::SubFOp,
                  mlir::arith::DivFOp, mlir::arith::MaximumFOp,
                  mlir::arith::NegFOp, mlir::arith::TruncFOp,
                  mlir::arith::ExtFOp, mlir::math::ExpOp, mlir::math::RsqrtOp,
                  mlir::math::SqrtOp, mlir::math::FPowIOp, mlir::math::ErfOp>(
                [&](auto computeOp) { routeComputeOp(computeOp); })
            .Case<mlir::arith::ConstantOp>([&](mlir::arith::ConstantOp) {})
            .Case<mlir::linalg::GenericOp>([&](mlir::linalg::GenericOp) {})
            .Case<mlir::linalg::YieldOp>([&](mlir::linalg::YieldOp yieldOp) {
              auto result = yieldOp.getValues();
              assert(result.size() == 1);
              // for (auto [idxResult, result] :
              //      llvm::enumerate(yieldOp.getValues())) {
              arithResults = getSourcePorts(result[0]);

              // }
            })
            .Default([&](mlir::Operation *) {
              assert(0 && "unsupported generic body op");
            });
      });

      // Build output -> SRAM

      Array2D<mlir::Value> arithResultsArray(arraySize, arithResults);
      auto outputRegArray = regArrays.back();

      llvm::SmallVector<mlir::Operation *> outputRegs;

      assert(genericOp.getResults().size() == 1);
      llvm::SmallVector<uint64_t> resultSize;
      auto dims =
          getAffineMapAccessDims(genericOp.getIndexingMapsArray().back());
      for (auto d : dims) {
        auto u = unrollFactor[d];
        if (u > 1)
          resultSize.push_back(u);
      }
      if (!(resultSize.size() == 0 || resultSize.size() == 1 ||
            resultSize.size() == 2)) {
        ECHO(resultSize.size(), "\n");
        assert(0);
      }
      // assert(resultSize.size() == 1 || resultSize.size() == 2);
      while (resultSize.size() < 2)
        resultSize.push_back(1);

      if (resultSize[0] < arraySize[0] &&
          resultSize[1] == arraySize[1]) // array -> reduction -> 1 row
      {
        outputRegs = outputRegArray.row(outputRegArray.getNumRow() - 1);
        for (uint64_t r = 0; r < arraySize[0]; r++) {
          for (uint64_t c = 0; c < arraySize[1]; c++) {
            if (r > 0)
              builder.create<dap::DataPathOp>(
                  UnknownLoc::get(ctx), arithResultsArray.at(r - 1, c),
                  mlir::dyn_cast<dap::DataNodeOp>(outputRegArray.at(r, c))
                      .getSrc());
          }
        }
      } else if (resultSize[0] == arraySize[0] &&
                 resultSize[1] < arraySize[1]) // array -> reduction -> 1 col
      {
        outputRegs = outputRegArray.col(0);
        for (uint64_t r = 0; r < arraySize[0]; r++) {
          for (uint64_t c = 0; c < arraySize[1]; c++) {
            if (c > 0)
              builder.create<dap::DataPathOp>(
                  UnknownLoc::get(ctx), arithResultsArray.at(r, c - 1),
                  mlir::dyn_cast<dap::DataNodeOp>(outputRegArray.at(r, c))
                      .getSrc());
          }
        }
      } else if (resultSize[0] == arraySize[0] &&
                 resultSize[1] == arraySize[1]) // array
      {
        outputRegs = outputRegArray.array();
      } else
        assert(0);

      llvm::SmallVector<mlir::Value> outputPorts(outputRegs.size());
      for (auto [reg, port] : llvm::zip(outputRegs, outputPorts)) {
        auto dataNodeOp = mlir::dyn_cast<dap::DataNodeOp>(reg);
        port = dataNodeOp.getResult();
      }

      // outputPorts -> SRAM
      auto outputValue = genericOp.getResult(0);
      auto outputSram = valueSramMap[genericOp][outputValue];
      assert(genericOp.getResults().size() == 1);
      uint64_t nOutputSramElement =
          outputSram.size() * archCfg.sramWidth /
          getElementTypeOrSelf(outputValue.getType()).getIntOrFloatBitWidth();

      llvm::SmallVector<mlir::Value> muxPorts(nOutputSramElement);
      if (outputPorts.size() > nOutputSramElement) {
        assert(outputPorts.size() % nOutputSramElement == 0);
        uint64_t fanin = outputPorts.size() / nOutputSramElement;

        for (auto [idxElement, itemElement] : llvm::enumerate(muxPorts)) {
          auto muxOp = builder.create<dap::MuxOp>(UnknownLoc::get(ctx), fanin);
          for (uint64_t i = 0; i < fanin; i++) {
            builder.create<dap::DataPathOp>(
                UnknownLoc::get(ctx),
                outputPorts[i * nOutputSramElement + idxElement],
                muxOp.getDins()[i]);
          }
          itemElement = muxOp.getDout();
        }
      } else if (outputPorts.size() == nOutputSramElement) {
        for (auto [idxElement, itemElement] : llvm::enumerate(muxPorts)) {
          itemElement = outputPorts[idxElement];
        }
      } else {
        assert(nOutputSramElement % outputPorts.size() == 0);
        uint64_t fanout = nOutputSramElement / outputPorts.size();

        for (auto [idxElement, itemElement] : llvm::enumerate(outputPorts)) {
          auto demuxOp =
              builder.create<dap::DemuxOp>(UnknownLoc::get(ctx), fanout);
          builder.create<dap::DataPathOp>(UnknownLoc::get(ctx), itemElement,
                                          demuxOp.getDin());
          for (uint64_t i = 0; i < fanout; i++) {
            muxPorts[idxElement * fanout + i] = demuxOp.getResults()[i];
          }
        }
      }

      // Compose
      assert(muxPorts.size() > outputSram.size());
      uint64_t nElementPerBank =
          archCfg.sramWidth /
          getElementTypeOrSelf(outputValue.getType()).getIntOrFloatBitWidth();
      assert(muxPorts.size() / nElementPerBank == outputSram.size());
      for (auto [idxSram, itemSram] : llvm::enumerate(outputSram)) {
        auto sramOp = mlir::dyn_cast<dap::SramOp>(itemSram);
        auto composeOp = builder.create<dap::ComposeOp>(UnknownLoc::get(ctx),
                                                        nElementPerBank);
        for (uint64_t i = 0; i < nElementPerBank; i++) {
          builder.create<dap::DataPathOp>(
              UnknownLoc::get(ctx), muxPorts[idxSram * nElementPerBank + i],
              composeOp.getDins()[i]);
        }
        builder.create<dap::DataPathOp>(UnknownLoc::get(ctx),
                                        composeOp.getDout(), sramOp.getWaddr());
      }
    }

    // The source functions must remain alive while their linalg operations are
    // used to build the DAP routing network. Remove them once lowering is done;
    // main_syn is not in funcs because it was created after funcs was
    // collected.
    for (auto funcOp : funcs)
      funcOp.erase();
  }

}; // namespace
} // namespace
} // namespace mlir::accelgen
