#ifndef ARCH_UTILS_H
#define ARCH_UTILS_H

#include <cstddef>
#include <string>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

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

  // Register
  size_t nDataNode;

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

  void reset();

 private:
  size_t _currect;
  llvm::ArrayRef<mlir::Operation*> _resourcesRef;
};

inline std::string getPeResourceName(mlir::Operation* op) {
  llvm::StringRef operationName =
      llvm::TypeSwitch<mlir::Operation*, llvm::StringRef>(op)
          .Case<mlir::arith::MulFOp>(
              [](auto) -> llvm::StringRef { return "mulf"; })
          .Case<mlir::arith::AddFOp>(
              [](auto) -> llvm::StringRef { return "addf"; })
          .Case<mlir::arith::SubFOp>(
              [](auto) -> llvm::StringRef { return "subf"; })
          .Case<mlir::arith::DivFOp>(
              [](auto) -> llvm::StringRef { return "divf"; })
          .Case<mlir::arith::MaximumFOp>(
              [](auto) -> llvm::StringRef { return "maximumf"; })
          .Case<mlir::arith::NegFOp>(
              [](auto) -> llvm::StringRef { return "negf"; })
          .Case<mlir::arith::TruncFOp>(
              [](auto) -> llvm::StringRef { return "truncf"; })
          .Case<mlir::arith::ExtFOp>(
              [](auto) -> llvm::StringRef { return "extf"; })
          .Case<mlir::math::ExpOp>(
              [](auto) -> llvm::StringRef { return "exp"; })
          .Case<mlir::math::RsqrtOp>(
              [](auto) -> llvm::StringRef { return "rsqrt"; })
          .Case<mlir::math::SqrtOp>(
              [](auto) -> llvm::StringRef { return "sqrt"; })
          .Case<mlir::math::FPowIOp>(
              [](auto) -> llvm::StringRef { return "fpowi"; })
          .Case<mlir::math::ErfOp>(
              [](auto) -> llvm::StringRef { return "erf"; })
          .Default([op](mlir::Operation*) -> llvm::StringRef {
            llvm::report_fatal_error(llvm::Twine("unsupported PE operation: ") +
                                     op->getName().getStringRef());
          });
  std::string resource = operationName.str();

  auto appendType = [&resource](mlir::Type type) {
    if (auto intType = mlir::dyn_cast<mlir::IntegerType>(type)) {
      resource.append("_i");
      resource.append(std::to_string(intType.getWidth()));
      return;
    }

    llvm::StringRef typeName =
        llvm::TypeSwitch<mlir::Type, llvm::StringRef>(type)
            .Case<mlir::BFloat16Type>(
                [](auto) -> llvm::StringRef { return "bf16"; })
            .Case<mlir::Float32Type>(
                [](auto) -> llvm::StringRef { return "fp32"; })
            .Case<mlir::Float64Type>(
                [](auto) -> llvm::StringRef { return "fp64"; })
            .Default([type](mlir::Type) -> llvm::StringRef {
              std::string message;
              llvm::raw_string_ostream os(message);
              os << "unsupported PE type: " << type;
              llvm::report_fatal_error(llvm::StringRef(os.str()));
            });

    resource.push_back('_');
    resource.append(typeName.data(), typeName.size());
  };

  for (mlir::Value operand : op->getOperands()) appendType(operand.getType());
  for (mlir::Value result : op->getResults()) appendType(result.getType());

  return resource;
}

}  // namespace accelgen
}  // namespace mlir

#endif
