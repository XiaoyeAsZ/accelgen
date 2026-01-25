#ifndef KERNEL_SCHEDULE_PASS_H
#define KERNEL_SCHEDULE_PASS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "accelgen/Utils/Types.h"
#include <any>
#include <map>
#include <numeric>
#include <unordered_set>
#include <vector>

namespace mlir {
namespace accelgen {

#define GEN_PASS_DECL
#include "accelgen/Passes/KernelSchedulePass.h.inc"

using ParameterVariant = std::variant<llvm::SmallVector<int64_t>, int64_t>;
using ParameterPointerVariant =
    std::variant<llvm::SmallVector<int64_t>*, int64_t*>;

class ParameterGenerator {
 public:
  void addVariable(std::vector<ParameterVariant> values) {
    std::vector<ParameterVariant> v;
    for (auto& x : values) v.emplace_back(x);
    candidates.push_back(std::move(v));
    indices.push_back(0);
  }

  bool hasNext() const { return _hasNext; }

  void next() {
    // auto result = current();
    for (auto x : indices) llvm::errs() << x << " ";
    llvm::errs() << "\n";
    advance();
    return;
  }

  size_t size() { return candidates.size(); }

  std::vector<std::vector<ParameterVariant>> candidates;
  std::vector<size_t> indices;

 private:
  bool _hasNext = true;

  std::vector<ParameterVariant> current() const {
    std::vector<ParameterVariant> res;
    for (size_t i = 0; i < candidates.size(); i++) {
      // llvm::errs() << i << " " << indices[i] << " " << candidates.size()
      //              << "\n";
      res.push_back(candidates[i][indices[i]]);
    }
    return res;
  }

  void advance() {
    for (int i = indices.size() - 1; i >= 0; i--) {
      indices[i]++;
      if (indices[i] < candidates[i].size()) return;
      indices[i] = 0;
      if (i == 0) {
        _hasNext = false;
        return;
      }
    }
  }
};

}  // namespace accelgen

}  // namespace mlir

#endif  // KERNEL_SCHEDULE_PASS_H