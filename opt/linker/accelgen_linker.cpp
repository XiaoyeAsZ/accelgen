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

#include "accelgen/Passes/AccelgenPasses.h"
#include "accelgen/Dialect/Dap/DapDialect.h"

llvm::cl::list<std::string> inputFiles(llvm::cl::Positional,
                                       llvm::cl::desc("<input MLIR files>"),
                                       llvm::cl::OneOrMore);

llvm::cl::opt<std::string> outputFile("o", llvm::cl::desc("Output file"),
                                      llvm::cl::value_desc("filename"),
                                      llvm::cl::init("-"));

int main(int argc, char** argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv, "MLIR Merge Tool\n");

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);

  mlir::MLIRContext context(registry);

  llvm::SmallVector<mlir::OwningOpRef<mlir::ModuleOp>> modules;

  for (auto& file : inputFiles) {
    auto module = mlir::parseSourceFile<mlir::ModuleOp>(file, &context);

    if (!module) {
      llvm::errs() << "Failed to parse: " << file << "\n";
      return 1;
    }

    modules.push_back(std::move(module));
  }

  llvm::SmallVector<mlir::func::FuncOp> funcs;
  for (auto& m : modules) {
    m.get().walk([&](mlir::func::FuncOp funcOp) { funcs.push_back(funcOp); });
  }

  auto merged = mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  mlir::SymbolTable symbolTable(merged);

  // for (auto [indexFunc, itemFunc] : llvm::enumerate(funcs)) {
  //   auto newFunc = itemFunc.clone();

  //   if (symbolTable.lookup(name)) {
  //     std::string newName =
  //         (name.str() + "_merged_" +
  //          std::to_string(reinterpret_cast<uintptr_t>(func.getOperation())));

  //     newFunc.setName(newName);
  //   }

  //   // Insert into merged module
  //   symbolTable.insert(newFunc);
  // }

  // insert here

  std::error_code ec;
  llvm::raw_fd_ostream os(outputFile, ec);

  if (ec) {
    llvm::errs() << "Cannot open output file\n";
    return 1;
  }

  merged->print(os);
  os << "\n";

  return 0;
}
