#include "accelgen/Utils/DSEUtils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include <queue>

namespace mlir::accelgen {

// template <typename _DType>
// void ParameterGenerator<_DType>::addVariable(
//     const std::vector<_DType> &values) {
//   std::vector<_DType> v;
//   for (auto &x : values)
//     v.emplace_back(x);
//   candidates.push_back(std::move(v));
//   indices.push_back(0);
// }

// template <typename _DType> bool ParameterGenerator<_DType>::hasNext() const {
//   return _hasNext;
// }

// template <typename _DType>
// std::vector<_DType> ParameterGenerator<_DType>::next() {
//   auto result = current();
//   // for (auto x : indices) llvm::errs() << x << " ";
//   // llvm::errs() << "\n";
//   advance();
//   return result;
// }

// template <typename _DType> size_t ParameterGenerator<_DType>::size() {
//   return candidates.size();
// }

// template <typename _DType>
// std::vector<_DType> ParameterGenerator<_DType>::current() const {
//   std::vector<_DType> res;
//   for (size_t i = 0; i < candidates.size(); i++) {
//     res.push_back(candidates[i][indices[i]]);
//   }
//   return res;
// }

// template <typename _DType> void ParameterGenerator<_DType>::advance() {
//   for (int i = indices.size() - 1; i >= 0; i--) {
//     indices[i]++;
//     if (indices[i] < candidates[i].size())
//       return;
//     indices[i] = 0;
//     if (i == 0) {
//       _hasNext = false;
//       return;
//     }
//   }
// }

} // namespace mlir::accelgen