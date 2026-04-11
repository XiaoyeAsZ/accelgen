#ifndef OPERATION_UTILS
#define OPERATION_UTILS

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Pass/Pass.h"
#include <vector>

namespace mlir {
namespace accelgen {
std::vector<mlir::Operation *>
getTopoOrder(const std::vector<mlir::Operation *> &ops);

std::vector<mlir::Operation *>
getTopoOrderALSP(const std::vector<mlir::Operation *> &ops);

std::vector<linalg::GenericOp> getProducerGeneric(mlir::Value operand);

llvm::ArrayRef<int64_t> getOperandShape(const mlir::Value &operand);

template <typename _Type, typename _ContainerType>
int64_t argwhere(const _ContainerType &container, const _Type &value) {
  auto it = std::find(container.begin(), container.end(), value);
  if (it == container.end())
    return -1;
  return std::distance(container.begin(), it);
};

template <typename _Type> class MergedSet {
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
    if (s == d)
      return;
    _num--;
    s->source = d;
  }

  size_t getNumSet() { return _num; }

private:
  size_t _num = 0;
  struct Node {
    _Type dat;
    Node *source;
  };

  std::unordered_map<_Type, Node *> mapping;

  Node *root(_Type item) {
    auto r = mapping[item];
    while (r != r->source) {
      r = r->source;
    }
    return r;
  }
};

static std::string toString(mlir::Operation *op) {
  if (mlir::isa<mlir::arith::MulFOp>(op))
    return "mulf";
  if (mlir::isa<mlir::arith::AddFOp>(op))
    return "addf";
  if (mlir::isa<mlir::arith::TruncFOp>(op))
    return "truncf";
  if (mlir::isa<mlir::arith::NegFOp>(op))
    return "negf";
  if (mlir::isa<mlir::arith::SubFOp>(op))
    return "subf";
  if (mlir::isa<mlir::arith::ExtFOp>(op))
    return "extf";
  if (mlir::isa<mlir::arith::MaximumFOp>(op))
    return "maximumf";
  if (mlir::isa<mlir::arith::DivFOp>(op))
    return "divf";
  if (mlir::isa<mlir::math::ExpOp>(op))
    return "exp";
  if (mlir::isa<mlir::math::RsqrtOp>(op))
    return "rsqrt";
  if (mlir::isa<mlir::math::SqrtOp>(op))
    return "rsqrt";
  if (mlir::isa<mlir::math::FPowIOp>(op))
    return "fpowi";
  if (mlir::isa<mlir::math::ErfOp>(op))
    return "erf";
  // if (mlir::isa<mlir::arith::MulIOp>(op)) return "muli";
  // if (mlir::isa<mlir::arith::AddIOp>(op)) return "addi";
  op->dump();
  assert(0);
}

static std::string toString(mlir::Type type) {
  if (type.isBF16())
    return "bf16";
  if (type.isF16())
    return "fp16";
  if (type.isF32())
    return "fp32";
  if (type.isF64())
    return "fp64";

  if (auto intTy = mlir::dyn_cast<mlir::IntegerType>(type))
    return "i" + std::to_string(intTy.getWidth());
  assert(0);
}

llvm::SmallVector<int64_t> getTilingOfOperand(linalg::GenericOp genericOp,
                                              mlir::OpOperand *operand);

template <typename _Type> class Array2D {
private:
  llvm::SmallVector<_Type> _dat;
  llvm::SmallVector<int64_t, 2> _bounds;

public:
  Array2D(llvm::ArrayRef<int64_t> bounds)
      : _bounds(bounds.begin(), bounds.end()) {
    assert(_bounds.size() == 2);
    _dat = llvm::SmallVector<_Type>(_bounds[0] * _bounds[1]);
  }
  Array2D(llvm::ArrayRef<int64_t> bounds, llvm::ArrayRef<_Type> dat)
      : _bounds(bounds.begin(), bounds.end()) {
    assert(_bounds.size() == 2);
    assert(_bounds[0] * _bounds[1] == dat.size());
    _dat = llvm::SmallVector<_Type>(dat.begin(), dat.end());
  }
  ~Array2D() = default;

  _Type &at(int64_t d0, int64_t d1) {
    auto index = d0 * _bounds[0] + d1;
    assert(index < _dat.size());
    return _dat[index];
  }
};

} // namespace accelgen
} // namespace mlir

#endif