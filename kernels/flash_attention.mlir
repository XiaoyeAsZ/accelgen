module attributes {} {
  func.func @flash_attention(%arg0: memref<?x1x4096x128xf32>, %arg1: memref<?x1x4096x128xf32>, %arg2: memref<?x1x4096x128xf32>, %arg3: memref<?x1x4096x4096xf32>, %arg4: memref<?x1x4096x4096xf32>, %arg5: memref<?x1x4096xf32>, %arg6: memref<?x1x4096xf32>, %arg7: memref<?x1x4096xf32>, %arg8: memref<?x1x4096xf32>, %arg9: memref<?x1x4096x4096xf32>, %arg10: memref<?x1x4096x128xf32>, %arg11: memref<?x1x4096x128xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant -5.242870e+05 : f32
    affine.for %arg12 = 0 to 64 {
      affine.for %arg13 = 0 to 64 {
        affine.for %arg14 = 0 to 128 {
          affine.for %arg15 = 0 to 64 {
            affine.for %arg16 = 0 to 64 {
              %0 = affine.load %arg0[0, 0, %arg15 + %arg12 * 64, %arg14] : memref<?x1x4096x128xf32>
              %1 = affine.load %arg1[0, 0, %arg16 + %arg13 * 64, %arg14] : memref<?x1x4096x128xf32>
              %2 = arith.mulf %0, %1 : f32
              %3 = affine.load %arg3[0, 0, %arg15 + %arg12 * 64, %arg16 + %arg13 * 64] : memref<?x1x4096x4096xf32>
              %4 = arith.addf %3, %2 : f32
              affine.store %4, %arg3[0, 0, %arg15 + %arg12 * 64, %arg16 + %arg13 * 64] : memref<?x1x4096x4096xf32>
            }
          }
        }
        affine.for %arg14 = 0 to 64 {
          %0 = affine.for %arg15 = 0 to 64 iter_args(%arg16 = %cst) -> (f32) {
            %1 = affine.load %arg3[0, 0, %arg14 + %arg12 * 64, %arg15 + %arg13 * 64] : memref<?x1x4096x4096xf32>
            %2 = func.call @f32_max(%arg16, %1) : (f32, f32) -> f32
            affine.yield %2 : f32
          }
          affine.store %0, %arg5[0, 0, %arg14 + %arg12 * 64] : memref<?x1x4096xf32>
        }
        affine.for %arg14 = 0 to 64 {
          %0 = affine.load %arg6[0, 0, %arg14 + %arg12 * 64] : memref<?x1x4096xf32>
          %1 = affine.load %arg5[0, 0, %arg14 + %arg12 * 64] : memref<?x1x4096xf32>
          %2 = func.call @f32_max(%0, %1) : (f32, f32) -> f32
          affine.store %2, %arg5[0, 0, %arg14 + %arg12 * 64] : memref<?x1x4096xf32>
        }
        affine.for %arg14 = 0 to 64 {
          affine.for %arg15 = 0 to 64 {
            %0 = affine.load %arg3[0, 0, %arg14 + %arg12 * 64, %arg15 + %arg13 * 64] : memref<?x1x4096x4096xf32>
            %1 = affine.load %arg5[0, 0, %arg14 + %arg12 * 64] : memref<?x1x4096xf32>
            %2 = arith.subf %0, %1 : f32
            %3 = func.call @f32_exp(%2) : (f32) -> f32
            affine.store %3, %arg4[0, 0, %arg14 + %arg12 * 64, %arg15 + %arg13 * 64] : memref<?x1x4096x4096xf32>
          }
        }
        affine.for %arg14 = 0 to 64 {
          affine.for %arg15 = 0 to 64 {
            %0 = affine.load %arg4[0, 0, %arg14 + %arg12 * 64, %arg15 + %arg13 * 64] : memref<?x1x4096x4096xf32>
            %1 = affine.load %arg7[0, 0, %arg14 + %arg12 * 64] : memref<?x1x4096xf32>
            %2 = arith.addf %1, %0 : f32
            affine.store %2, %arg7[0, 0, %arg14 + %arg12 * 64] : memref<?x1x4096xf32>
          }
        }
        affine.for %arg14 = 0 to 64 {
          affine.for %arg15 = 0 to 64 {
            %0 = affine.load %arg4[0, 0, %arg14 + %arg12 * 64, %arg15 + %arg13 * 64] : memref<?x1x4096x4096xf32>
            %1 = affine.load %arg7[0, 0, %arg14 + %arg12 * 64] : memref<?x1x4096xf32>
            %2 = affine.load %arg8[0, 0, %arg14 + %arg12 * 64] : memref<?x1x4096xf32>
            %3 = affine.load %arg6[0, 0, %arg14 + %arg12 * 64] : memref<?x1x4096xf32>
            %4 = affine.load %arg5[0, 0, %arg14 + %arg12 * 64] : memref<?x1x4096xf32>
            %5 = arith.subf %3, %4 : f32
            %6 = func.call @f32_exp(%5) : (f32) -> f32
            %7 = arith.mulf %2, %6 : f32
            %8 = arith.addf %1, %7 : f32
            %9 = arith.divf %0, %8 : f32
            affine.store %9, %arg9[0, 0, %arg14 + %arg12 * 64, %arg15 + %arg13 * 64] : memref<?x1x4096x4096xf32>
          }
        }
        affine.for %arg14 = 0 to 64 {
          affine.for %arg15 = 0 to 64 {
            affine.for %arg16 = 0 to 128 {
              %0 = affine.load %arg9[0, 0, %arg15 + %arg12 * 64, %arg14 + %arg13 * 64] : memref<?x1x4096x4096xf32>
              %1 = affine.load %arg2[0, 0, %arg14 + %arg13 * 64, %arg16] : memref<?x1x4096x128xf32>
              %2 = arith.mulf %0, %1 : f32
              %3 = affine.load %arg10[0, 0, %arg15 + %arg12 * 64, %arg16] : memref<?x1x4096x128xf32>
              %4 = arith.addf %3, %2 : f32
              affine.store %4, %arg10[0, 0, %arg15 + %arg12 * 64, %arg16] : memref<?x1x4096x128xf32>
            }
          }
        }
        affine.for %arg14 = 0 to 64 {
          affine.for %arg15 = 0 to 128 {
            %0 = affine.load %arg11[0, 0, %arg14 + %arg12 * 64, %arg15] : memref<?x1x4096x128xf32>
            %1 = affine.load %arg6[0, 0, %arg14 + %arg12 * 64] : memref<?x1x4096xf32>
            %2 = affine.load %arg5[0, 0, %arg14 + %arg12 * 64] : memref<?x1x4096xf32>
            %3 = arith.subf %1, %2 : f32
            %4 = func.call @f32_exp(%3) : (f32) -> f32
            %5 = arith.mulf %0, %4 : f32
            %6 = affine.load %arg8[0, 0, %arg14 + %arg12 * 64] : memref<?x1x4096xf32>
            %7 = affine.load %arg7[0, 0, %arg14 + %arg12 * 64] : memref<?x1x4096xf32>
            %8 = arith.addf %6, %7 : f32
            %9 = arith.divf %6, %8 : f32
            %10 = arith.mulf %5, %9 : f32
            %11 = affine.load %arg10[0, 0, %arg14 + %arg12 * 64, %arg15] : memref<?x1x4096x128xf32>
            %12 = arith.addf %10, %11 : f32
            affine.store %12, %arg11[0, 0, %arg14 + %arg12 * 64, %arg15] : memref<?x1x4096x128xf32>
          }
        }
      }
    }
    return
  }
  func.func private @f32_max(f32, f32) -> f32 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @f32_exp(f32) -> f32 attributes {llvm.linkage = #llvm.linkage<external>}
}
