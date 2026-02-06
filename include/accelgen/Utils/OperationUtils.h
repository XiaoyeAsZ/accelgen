#ifndef OPERATION_UTILS
#define OPERATION_UTILS

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Pass/Pass.h"
#include <vector>

namespace mlir {
namespace accelgen {
std::vector<mlir::Operation*> getTopoOrder(
    const std::vector<mlir::Operation*>& ops);

std::vector<mlir::Operation*> getTopoOrderALSP(
    const std::vector<mlir::Operation*>& ops);

std::vector<linalg::GenericOp> getProducerGeneric(mlir::Value operand);

llvm::ArrayRef<int64_t> getOperandShape(const mlir::Value& operand);

template <typename _Type, typename _ContainerType>
int64_t argwhere(const _ContainerType& container, const _Type& value) {
  auto it = std::find(container.begin(), container.end(), value);
  if (it == container.end()) return -1;
  return std::distance(container.begin(), it);
};

template <typename _Type>
class MergedSet {
 public:
  void insert(_Type item) {
    auto node = new Node;
    node->dat = item;
    node->source = node;
    mapping[item] = node;
    _num++;
  }

  bool check(_Type src, _Type dest) {
    auto s = root(src);
    auto d = root(dest);
    return s == d;
  }

  void merge(_Type src, _Type dest) {
    auto s = root(src);
    auto d = root(dest);
    if (s == d) return;
    _num--;
    s->source = d;
  }

  size_t getNumSet() { return _num; }

 private:
  size_t _num = 0;
  struct Node {
    _Type dat;
    Node* source;
  };

  std::unordered_map<_Type, Node*> mapping;

  Node* root(_Type item) {
    auto r = mapping[item];
    while (r != r->source) {
      r = r->source;
    }
    return r;
  }
};

}  // namespace accelgen
}  // namespace mlir

#endif