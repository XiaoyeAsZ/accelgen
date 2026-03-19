#ifndef DSE_UTILS
#define DSE_UTILS

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Pass/Pass.h"
#include <vector>

namespace mlir {
namespace accelgen {

template <typename _DType>
class ParameterGenerator {
 public:
  void addVariable(const std::vector<_DType>& values) {
    std::vector<_DType> v;
    for (auto& x : values) v.emplace_back(x);
    candidates.push_back(std::move(v));
    indices.push_back(0);
    _hasNext = true;
  }

  bool hasNext() const { return _hasNext; }

  std::vector<_DType> next() {
    auto result = current();
    advance();
    return result;
  }

  size_t size() { return candidates.size(); }

  std::vector<std::vector<_DType>> candidates;
  std::vector<size_t> indices;

  void reset() {
    for (auto& e : indices) {
      e = 0;
      _hasNext = true;
    }
  }

 private:
  bool _hasNext = false;

  std::vector<_DType> current() const {
    std::vector<_DType> res;
    for (size_t i = 0; i < candidates.size(); i++) {
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

template <typename _T>
class DecisionVariable {
 private:
  std::vector<_T> solution;
};

}  // namespace accelgen
}  // namespace mlir

#endif