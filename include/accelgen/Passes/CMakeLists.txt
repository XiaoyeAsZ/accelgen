set(LLVM_TARGET_DEFINITIONS LinalgLoopReorderPass.td)
mlir_tablegen(LinalgLoopReorderPass.h.inc --gen-pass-decls)
add_public_tablegen_target(MLIRLinalgLoopReorderPassIncGen)

set(LLVM_TARGET_DEFINITIONS LinalgOpFusePass.td)
mlir_tablegen(LinalgOpFusePass.h.inc --gen-pass-decls)
add_public_tablegen_target(MLIRLinalgOpFusePassIncGen)

set(LLVM_TARGET_DEFINITIONS ConvertToSpePass.td)
mlir_tablegen(ConvertToSpePass.h.inc --gen-pass-decls)
add_public_tablegen_target(MLIRConvertToSpePassIncGen)

set(LLVM_TARGET_DEFINITIONS ConstructTileGraphPass.td)
mlir_tablegen(ConstructTileGraphPass.h.inc --gen-pass-decls)
add_public_tablegen_target(MLIRConstructTileGraphPassIncGen)

set(LLVM_TARGET_DEFINITIONS KernelSchedulePass.td)
mlir_tablegen(KernelSchedulePass.h.inc --gen-pass-decls)
add_public_tablegen_target(MLIRKernelSchedulePassIncGen)

set(LLVM_TARGET_DEFINITIONS MarkGenericPass.td)
mlir_tablegen(MarkGenericPass.h.inc --gen-pass-decls)
add_public_tablegen_target(MLIRMarkGenericPassIncGen)

set(LLVM_TARGET_DEFINITIONS FuseGenericPass.td)
mlir_tablegen(FuseGenericPass.h.inc --gen-pass-decls)
add_public_tablegen_target(MLIRFuseGenericPassIncGen)

# set(LLVM_TARGET_DEFINITIONS SpeFissionPass.td)
# mlir_tablegen(SpeFissionPass.h.inc --gen-pass-decls)
# add_public_tablegen_target(MLIRSpeFissionPassIncGen)