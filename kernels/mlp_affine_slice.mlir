module {
  func.func @main(%arg0: memref<1x1x4096xbf16, strided<[?, ?, ?], offset: ?>>, %arg1: memref<14336x4096xbf16, strided<[?, ?], offset: ?>>) -> memref<1x1x4096xbf16> {
    %cst = arith.constant 0.000000e+00 : bf16
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x14336xbf16>
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<1x1x14336xbf16>
    affine.for %arg5 = 0 to 64 { // parallel
      affine.for %arg6 = 0 to 64 { // parallel
        affine.for %arg4 = 0 to 1 {
          affine.store %cst, %alloc[%arg4, %arg5, %arg6] : memref<1x1x14336xbf16>
          affine.for %arg7 = 0 to 4096 {
            %0 = affine.load %arg0[%arg4, %arg5, %arg7] : memref<1x1x4096xbf16, strided<[?, ?, ?], offset: ?>>
            %1 = affine.load %arg1[%arg6, %arg7] : memref<14336x4096xbf16, strided<[?, ?], offset: ?>>
            %2 = affine.load %alloc[%arg4, %arg5, %arg6] : memref<1x1x14336xbf16>
            %3 = "spe.mac"(%0, %1, %2) : (bf16, bf16, bf16) -> bf16
            affine.store %3, %alloc[%arg4, %arg5, %arg6] : memref<1x1x14336xbf16>
          }
          %0 = affine.load %alloc[%arg4, %arg5, %arg6] : memref<1x1x14336xbf16>
          affine.store %0, %alloc_1[%arg4, %arg5, %arg6] : memref<1x1x14336xbf16>
        }
      }
    }
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<1x1x14336xbf16>
    affine.for %arg5 = 0 to 64 { // parallel
      affine.for %arg4 = 0 to 1 {
        affine.for %arg6 = 0 to 64 {
          %0 = affine.load %alloc_1[%arg4, %arg5, %arg6] : memref<1x1x14336xbf16>
          %1 = "spe.neg"(%0) : (bf16) -> bf16
          affine.store %1, %alloc_2[%arg4, %arg5, %arg6] : memref<1x1x14336xbf16>
        }
      }
    }
    return %alloc_2 : memref<1x1x4096xbf16>
  }
}
