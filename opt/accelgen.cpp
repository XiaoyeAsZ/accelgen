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
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
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
// #include "accelgen/Dialect/TileGraph/TileGraphDialect.h"
// #include "accelgen/Dialect/TileGraph/TileGraphOps.h"
// #include "accelgen/Passes/ConstructTileGraphPass.h"
// #include "accelgen/Passes/FuseGenericPass.h"
// #include "accelgen/Passes/KernelSchedulePass.h"
// #include "accelgen/Passes/MarkGenericPass.h"
#include "accelgen/Passes/AccelgenPasses.h"
#include "accelgen/Dialect/Dap/DapDialect.h"
#include "accelgen/Dialect/Dap/DapOps.h"

// int main(int argc, char** argv) {
//   mlir::DialectRegistry registry;
//   mlir::MLIRContext ctx(registry);
//   mlir::registerAllDialects(registry);
//   registry.insert<mlir::dap::DapDialect>();
//   mlir::registerAllPasses();
//   mlir::accelgen::registerPasses();

//   return mlir::asMainReturnCode(
//       mlir::MlirOptMain(argc, argv, "Custom MLIR Optimizer\n", registry));
// }

static llvm::cl::opt<std::string> inputFilename(
    llvm::cl::Positional, llvm::cl::desc("<input mlir file>"),
    llvm::cl::Required);

static llvm::cl::opt<std::string> outputFilename(
    "o", llvm::cl::desc("Output filename"), llvm::cl::value_desc("filename"),
    llvm::cl::init("-"));

static llvm::cl::opt<std::string> passPipeline(
    "pass-pipeline", llvm::cl::desc("MLIR pass pipeline"), llvm::cl::init(""));

int main(int argc, char** argv) {
  mlir::registerPassManagerCLOptions();
  mlir::registerDefaultTimingManagerCLOptions();

  llvm::cl::ParseCommandLineOptions(argc, argv, "AccelGen compiler\n");

  auto fileOrErr = llvm::MemoryBuffer::getFile(inputFilename);
  if (!fileOrErr) {
    llvm::errs() << "Failed to open file: " << fileOrErr.getError().message()
                 << "\n";
    return 1;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());

  mlir::MLIRContext context;
  mlir::registerAllDialects(context);
  // context.getOrLoadDialect<mlir::spe::SpeDialect>();
  context.getOrLoadDialect<mlir::dap::DapDialect>();

  mlir::registerAllPasses();
  mlir::accelgen::registerPasses();

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);

  if (!module) {
    llvm::errs() << "Failed to parse MLIR file.\n";
    return 1;
  }

  // Set up pass manager
  mlir::PassManager pm(&context);
  mlir::applyPassManagerCLOptions(pm);

  if (passPipeline.empty()) {
    // pm.addNestedPass<mlir::func::FuncOp>(
    //     mlir::createLinalgGeneralizeNamedOpsPass());
    // pm.addPass(mlir::accelgen::createMarkGenericPass());
    // pm.addPass(mlir::accelgen::createFuseGenericPass());
    // pm.addNestedPass<mlir::func::FuncOp>(
    //     mlir::accelgen::createKernelSchedule());

    assert(0);
  } else {
    if (failed(parsePassPipeline(passPipeline, pm))) {
      llvm::errs() << "Failed to parse pass pipeline\n";
      return 1;
    }
  }

  // pm.dump();

  // Run passes
  if (mlir::failed(pm.run(*module))) {
    llvm::errs() << "Failed to run passes on the MLIR module.\n";
    return 1;
  }

  std::error_code ec;
  llvm::raw_fd_ostream outFile(outputFilename, ec);
  if (ec) {
    llvm::errs() << "Failed to open output file: " << ec.message() << "\n";
    return 1;
  }

  // Print the transformed module
  module->print(outFile);
  return 0;
}