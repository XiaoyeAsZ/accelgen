

#include "accelgen/Passes/AccelgenPasses.h"
#include "accelgen/Dialect/Dap/DapOps.h"
#include "mlir/IR/BuiltinOps.h"
#include <algorithm>
#include <cstdint>

namespace mlir::accelgen {
#define GEN_PASS_DEF_MODELAREA
#include "accelgen/Passes/AccelgenPasses.h.inc"

namespace {

class ModelArea : public impl::ModelAreaBase<ModelArea> {
 public:
  using impl::ModelAreaBase<ModelArea>::ModelAreaBase;

  void runOnOperation() final {
    // Unit: um^2. These are the only primitives included in this model.
    const double bf16MacArea = 3243.1;
    const double bf16MuxArea = 13.7894;  // mux 2:1
    const double sramBankArea = 192291;

    auto module = getOperation();
    uint64_t sramBanks = 0;
    uint64_t bf16MulfCount = 0;
    uint64_t bf16AddfCount = 0;
    uint64_t muxOps = 0;
    uint64_t mux2to1Equivalent = 0;

    auto isBf16 = [](mlir::Type type) {
      return mlir::isa<mlir::BFloat16Type>(type);
    };
    auto floorLog2 = [](uint32_t value) {
      uint64_t levels = 0;
      while (value > 1) {
        value >>= 1;
        ++levels;
      }
      return levels;
    };

    module.walk([&](mlir::Operation* operation) {
      if (mlir::isa<dap::SramOp>(operation)) {
        ++sramBanks;
        return;
      }
      if (auto mulf = mlir::dyn_cast<dap::MulfOp>(operation)) {
        if (isBf16(mulf.getResult().getType())) ++bf16MulfCount;
        return;
      }
      if (auto addf = mlir::dyn_cast<dap::AddfOp>(operation)) {
        if (isBf16(addf.getResult().getType())) ++bf16AddfCount;
        return;
      }
      if (auto mux = mlir::dyn_cast<dap::MuxOp>(operation)) {
        ++muxOps;
        mux2to1Equivalent += floorLog2(mux.getFanin());
      }
    });

    // A BF16 MAC is represented by one multiply resource and one add resource.
    // Pair them instead of charging the MAC area twice.
    const uint64_t bf16MacCount = std::max(bf16MulfCount, bf16AddfCount);
    const double sramArea = sramBanks * sramBankArea;
    const double macArea = bf16MacCount * bf16MacArea;
    const double muxArea = mux2to1Equivalent * bf16MuxArea;
    const double totalArea = sramArea + macArea + muxArea;

    llvm::outs() << "===== Model Area =====\n";
    llvm::outs() << "SRAM banks: " << sramBanks << "\n";
    llvm::outs() << "BF16 mulf resources: " << bf16MulfCount << "\n";
    llvm::outs() << "BF16 addf resources: " << bf16AddfCount << "\n";
    llvm::outs() << "BF16 MAC units (paired): " << bf16MacCount << "\n";
    llvm::outs() << "Mux operations: " << muxOps << "\n";
    llvm::outs() << "2:1 mux equivalents: " << mux2to1Equivalent << "\n";
    llvm::outs() << "SRAM area (um^2): " << sramArea << "\n";
    llvm::outs() << "BF16 MAC area (um^2): " << macArea << "\n";
    llvm::outs() << "Mux area (um^2): " << muxArea << "\n";
    llvm::outs() << "Total modeled area (um^2): " << totalArea << "\n";
    llvm::outs() << "Total modeled area (mm^2): " << totalArea / 1e6 << "\n";
    llvm::outs() << "======================\n";
  }
};

}  // namespace
}  // namespace mlir::accelgen
