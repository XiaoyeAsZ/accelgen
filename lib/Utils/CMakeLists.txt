add_mlir_dialect_library(MLIRAccelgenUtils
        AffineMapUtils.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/accelgen

        DEPENDS

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRFuncDialect
        )
