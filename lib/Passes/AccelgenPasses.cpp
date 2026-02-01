#include "accelgen/Passes/AccelgenPasses.h"

namespace mlir {
namespace accelgen {

// #define GEN_PASS_DECL
// #define GEN_PASS_REGISTRATION
// #include "accelgen/Passes/MarkGenericPass.h.inc"

// #define GEN_PASS_DECL
// #define GEN_PASS_REGISTRATION
// #include "accelgen/Passes/FuseGenericPass.h.inc"

// #define GEN_PASS_DECL
// #define GEN_PASS_REGISTRATION
// #include "accelgen/Passes/KernelSchedulePass.h.inc"

// void registerAccelGenPasses() {
//   registerMarkGenericPass();
//   registerFuseGenericPass();
//   registerKernelSchedulePass();
// }

// #define GEN_PASS_DECL
// #define GEN_PASS_REGISTRATION
// #include "accelgen/Passes/AccelgenPasses.h.inc"

// void registerAccelGenPasses() { registerPasses(); }

}  // namespace accelgen
}  // namespace mlir