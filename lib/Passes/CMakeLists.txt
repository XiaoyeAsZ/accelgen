add_mlir_dialect_library(MLIRAccelgen
        LinalgLoopReorderPass.cpp
        LinalgOpFusePass.cpp
        ConvertToSpePass.cpp
        # SpeFissionPass.cpp
        ConstructTileGraphPass.cpp
        KernelSchedulePass.cpp
        MarkGenericPass.cpp
        FuseGenericPass.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/accelgen

        DEPENDS
        MLIRLinalgLoopReorderPassIncGen
        MLIRLinalgOpFusePassIncGen
        MLIRConvertToSpePassIncGen
        # MLIRSpeFissionPassIncGen
        MLIRConstructTileGraphPassIncGen
        MLIRKernelSchedulePassIncGen
        MLIRMarkGenericPassIncGen
        MLIRFuseGenericPassIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRFuncDialect
        )
