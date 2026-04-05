add_mlir_dialect_library(MLIRAccelgen
        AccelgenPasses.cpp
        LinalgLoopReorderPass.cpp
        LinalgOpFusePass.cpp
        ConvertToSpePass.cpp
        # SpeFissionPass.cpp
        ConstructTileGraphPass.cpp
        KernelSchedulePass.cpp
        MarkGenericPass.cpp
        FuseGenericPass.cpp
        ConvertToDapPass.cpp
        CollapseShapeDelayPass.cpp
        ModelBaselineAccelerator.cpp
        EliminateDeadOpPass.cpp
        FoldTensorOpPass.cpp
        ModelPerformance.cpp


        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/accelgen

        DEPENDS
        # MLIRLinalgLoopReorderPassIncGen
        # MLIRLinalgOpFusePassIncGen
        # MLIRConvertToSpePassIncGen
        # # MLIRSpeFissionPassIncGen
        # MLIRConstructTileGraphPassIncGen
        # MLIRKernelSchedulePassIncGen
        # MLIRMarkGenericPassIncGen
        # MLIRFuseGenericPassIncGen
        MLIRAccelgenPassesIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRFuncDialect
        )
