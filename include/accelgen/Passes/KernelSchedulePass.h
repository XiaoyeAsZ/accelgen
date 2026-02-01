#ifndef KERNEL_SCHEDULE_PASS_H
#define KERNEL_SCHEDULE_PASS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include <any>
#include <map>
#include <numeric>
#include <unordered_set>
#include <vector>

namespace mlir {
namespace accelgen {

#define GEN_PASS_DECL
#include "accelgen/Passes/KernelSchedulePass.h.inc"

}  // namespace accelgen

}  // namespace mlir

#endif  // KERNEL_SCHEDULE_PASS_H