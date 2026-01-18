#ifndef DSE_UTILS
#define DSE_UTILS

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include <vector>

namespace mlir {
namespace accelgen {

template <typename _T>
class DecisionVariable {
 private:
  std::vector<_T> solution;
};

}  // namespace accelgen
}  // namespace mlir

#endif