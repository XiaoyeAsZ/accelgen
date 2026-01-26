#include "accelgen/Utils/DSEUtils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include <queue>

namespace mlir::accelgen {

void ParameterGenerator::addVariable(std::vector<int64_t> values) {
  std::vector<int64_t> v;
  for (auto& x : values) v.emplace_back(x);
  candidates.push_back(std::move(v));
  indices.push_back(0);
}

bool ParameterGenerator ::hasNext() const { return _hasNext; }

std::vector<int64_t> ParameterGenerator::next() {
  auto result = current();
  for (auto x : indices) llvm::errs() << x << " ";
  llvm::errs() << "\n";
  advance();
  return result;
}

size_t ParameterGenerator::size() { return candidates.size(); }

std::vector<int64_t> ParameterGenerator::current() const {
  std::vector<int64_t> res;
  for (size_t i = 0; i < candidates.size(); i++) {
    res.push_back(candidates[i][indices[i]]);
  }
  return res;
}

void ParameterGenerator::advance() {
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

}  // namespace mlir::accelgen