#ifndef KERNEL_SCHEDULE_PASS
#define KERNEL_SCHEDULE_PASS

include "mlir/Pass/PassBase.td"

def KernelSchedule: Pass<"kernel-schedule", "::mlir::func::FuncOp"> {
  let summary = "";
  let description = [{
  }];
}

#endif // KERNEL_SCHEDULE_PASS
