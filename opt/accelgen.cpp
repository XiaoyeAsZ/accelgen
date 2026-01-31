#include <memory>

#include "mlir/Conversion/TosaToArith/TosaToArith.h"
#include "mlir/Conversion/TosaToLinalg/TosaToLinalg.h"
#include "mlir/Conversion/TosaToTensor/TosaToTensor.h"
#include "mlir/Dialect/Affine/Transforms/Transforms.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

// #include "accelgen/Dialect/Spe/SpeDialect.h"
// #include "accelgen/Dialect/Spe/SpeOps.h"
// #include "accelgen/Passes/LinalgLoopReorderPass.h"
// #include "accelgen/Passes/LinalgOpFusePass.h"
// #include "accelgen/Passes/ConvertToSpePass.h"
#include "accelgen/Dialect/TileGraph/TileGraphDialect.h"
#include "accelgen/Dialect/TileGraph/TileGraphOps.h"
#include "accelgen/Passes/ConstructTileGraphPass.h"
#include "accelgen/Passes/FuseGenericPass.h"
#include "accelgen/Passes/KernelSchedulePass.h"
#include "accelgen/Passes/MarkGenericPass.h"

int main(int argc, char **argv) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFile(argv[1]);
  if (!fileOrErr) {
    llvm::errs() << "Failed to open file: " << fileOrErr.getError().message()
                 << "\n";
    return 1;
  }

  mlir::MLIRContext context;
  mlir::registerAllDialects(context);
  // context.getOrLoadDialect<mlir::spe::SpeDialect>();
  context.getOrLoadDialect<mlir::tile_graph::TileGraphDialect>();

  // mlir::registerAllPasses();
  // mlir::accelgen::registerLinalgLoopReorder();
  // mlir::accelgen::registerLinalgOpFuse();

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);

  if (!module) {
    llvm::errs() << "Failed to parse MLIR file.\n";
    return 1;
  }

  // Set up pass manager
  mlir::PassManager pm(&context);
  // std::string pipeline =
  //     "builtin.module("
  //     "  func.func("
  //     "    tosa-to-arith,"
  //     "    tosa-to-linalg-named,"
  //     "    tosa-to-linalg,"
  //     "    tosa-to-tensor,"
  //     "    tosa-to-arith,"
  //     "    linalg-fuse-elementwise-ops"
  //     "  ),"
  //     "  one-shot-bufferize{bufferize-function-boundaries=true},"
  //     "  func.func("
  //     "    convert-linalg-to-affine-loops,"
  //     "    canonicalize,"
  //     "    cse"
  //     "  )"
  //     ")";
  // mlir::parsePassPipeline(pipeline, pm);

  /****/

  // // Preprocess : lower Tosa -> Affine
  // pm.addNestedPass<mlir::func::FuncOp>(mlir::createTosaToArithPass());
  // pm.addNestedPass<mlir::func::FuncOp>(mlir::tosa::createTosaToLinalgNamed());
  // pm.addNestedPass<mlir::func::FuncOp>(mlir::tosa::createTosaToLinalg());
  // pm.addNestedPass<mlir::func::FuncOp>(mlir::createTosaToTensorPass());
  // pm.addNestedPass<mlir::func::FuncOp>(mlir::createTosaToArithPass());
  // pm.addPass(mlir::accelgen::createLinalgOpFuse());
  // // pm.addNestedPass<mlir::func::FuncOp>(
  // //     mlir::createLinalgElementwiseOpFusionPass());

  // mlir::bufferization::OneShotBufferizePassOptions optionsBufferize;
  // optionsBufferize.bufferizeFunctionBoundaries = true;
  // pm.addPass(mlir::bufferization::createOneShotBufferizePass(optionsBufferize));

  // // pm.addPass(mlir::accelgen::createLinalgLoopReorder());

  // pm.addNestedPass<mlir::func::FuncOp>(
  //     mlir::createConvertLinalgToAffineLoopsPass());
  // pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
  // pm.addNestedPass<mlir::func::FuncOp>(mlir::createCSEPass());

  // pm.addPass(mlir::accelgen::createConvertToSpe());

  // //
  // pm.addNestedPass<mlir::func::FuncOp>(mlir::createConvertLinalgToLoopsPass());

  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::createLinalgGeneralizeNamedOpsPass());

  pm.addPass(mlir::accelgen::createMarkGenericPass());
  pm.addPass(mlir::accelgen::createFuseGenericPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::accelgen::createKernelSchedule());

  // pm.addPass(mlir::createCanonicalizerPass());
  // pm.addPass(mlir::createSymbolDCEPass());

  // pm.addNestedPass<mlir::func::FuncOp>(
  //     mlir::accelgen::createConstructTileGraph());

  /****/

  // loweing to generic

  // Optimization Pass :

  // Run passes
  if (mlir::failed(pm.run(*module))) {
    llvm::errs() << "Failed to run passes on the MLIR module.\n";
    return 1;
  }

  std::error_code ec;
  llvm::raw_fd_ostream outFile(argv[2], ec);
  if (ec) {
    llvm::errs() << "Failed to open output file: " << ec.message() << "\n";
    return 1;
  }

  // Print the transformed module
  module->print(outFile);
  return 0;

  return 0;
}