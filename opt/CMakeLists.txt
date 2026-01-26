set(LIBS
    MLIRArithDialect
    MLIROptLib
    MLIRRegisterAllDialects
    MLIRRegisterAllPasses
    MLIRAccelgen
    MLIRAccelgenDSE
    MLIRAccelgenUtils
    MLIRSpeDialect
    MLIRTileGraphDialect
  )
add_llvm_executable(accelgen-opt accelgen.cpp)

llvm_update_compile_flags(accelgen-opt)
target_link_libraries(accelgen-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(accelgen-opt)
