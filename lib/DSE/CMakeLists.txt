add_mlir_dialect_library(MLIRAccelgenDSE
        KernelScheduleDSE.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/accelgen

        DEPENDS

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRFuncDialect
        )
