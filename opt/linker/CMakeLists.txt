set(LIBS
    MLIRArithDialect
    MLIROptLib
    MLIRRegisterAllDialects
    MLIRRegisterAllPasses
    MLIRDapDialect
    MLIRAccelgen
    MLIRAccelgenDSE
    MLIRAccelgenUtils
    MLIRSpeDialect
    MLIRTileGraphDialect
    
  )

add_llvm_executable(accelgen-linker accelgen_linker.cpp)


llvm_update_compile_flags(accelgen-linker)


target_link_libraries(accelgen-linker PRIVATE ${LIBS})


mlir_check_all_link_libraries(accelgen-linker)
