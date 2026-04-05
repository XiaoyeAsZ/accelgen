#ifndef ARCH_UTILS_H
#define ARCH_UTILS_H

#include <cstddef>
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace accelgen {

class ArchResource {
 public:
  // System
  double_t cycle;

  // DRAM
  double_t bandwidth;  // GB/s
  double_t dramEnergy;

  // SRAM
  size_t sramWidth = 512;  // bit
  size_t sramDepth = 128;
  size_t nSramBank;
  double_t sramEnergy;

  // Compute logic
  llvm::StringMap<size_t> computeResource;  // #
  llvm::StringMap<double_t> computeEnergy;  // #

  ArchResource() = default;
  ~ArchResource() = default;

  void load(llvm::StringRef cfgPath);

  static llvm::StringRef getArithName(llvm::StringRef name);
};

class ResourcePool {
 public:
  ResourcePool() = default;
  ResourcePool(llvm::ArrayRef<mlir::Operation*> resources);

  llvm::ArrayRef<mlir::Operation*> get(size_t nResource);

 private:
  size_t _currect;
  llvm::ArrayRef<mlir::Operation*> _resourcesRef;
};

}  // namespace accelgen
}  // namespace mlir

#endif
