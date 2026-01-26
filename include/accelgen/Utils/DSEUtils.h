#ifndef DSE_UTILS
#define DSE_UTILS

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include <vector>

namespace mlir {
namespace accelgen {

class ParameterGenerator {
 public:
  void addVariable(std::vector<int64_t> values);
  bool hasNext() const;
  std::vector<int64_t> next();
  size_t size();

  std::vector<std::vector<int64_t>> candidates;
  std::vector<size_t> indices;

 private:
  bool _hasNext = true;

  std::vector<int64_t> current() const;
  void advance();
};

template <typename _T>
class DecisionVariable {
 private:
  std::vector<_T> solution;
};

}  // namespace accelgen
}  // namespace mlir

#endif