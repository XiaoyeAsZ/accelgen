module {
  memref.global "private" constant @__constant_1x1x1x1xbf16 : memref<1x1x1x1xbf16> = dense<8.837890e-02> {alignment = 64 : i64}
  func.func @main(%arg0: memref<1x1x4096xbf16, strided<[?, ?, ?], offset: ?>>, %arg1: memref<1x1x128xbf16, strided<[?, ?, ?], offset: ?>>, %arg2: memref<1x1x128xbf16, strided<[?, ?, ?], offset: ?>>, %arg3: memref<1x32x1x1xbf16, strided<[?, ?, ?, ?], offset: ?>>, %arg4: memref<4096x4096xbf16, strided<[?, ?], offset: ?>>, %arg5: memref<4096x4096xbf16, strided<[?, ?], offset: ?>>, %arg6: memref<4096x4096xbf16, strided<[?, ?], offset: ?>>, %arg7: memref<4096x4096xbf16, strided<[?, ?], offset: ?>>) -> (memref<1x1x4096xbf16>, memref<1x32x1x1xbf16>) {
    %cst = arith.constant 8.837890e-02 : bf16
    %cst_0 = arith.constant 0.000000e+00 : bf16
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x4096xbf16>
    affine.for %arg8 = 0 to 1 {
      affine.for %arg9 = 0 to 1 {
        affine.for %arg10 = 0 to 4096 {
          affine.store %cst_0, %alloc[%arg8, %arg9, %arg10] : memref<1x1x4096xbf16>
        }
      }
    }
    affine.for %arg8 = 0 to 1 {
      affine.for %arg9 = 0 to 1 {
        affine.for %arg10 = 0 to 4096 {
          affine.for %arg11 = 0 to 4096 {
            %0 = affine.load %arg0[%arg8, %arg9, %arg11] : memref<1x1x4096xbf16, strided<[?, ?, ?], offset: ?>>
            %1 = affine.load %arg4[%arg10, %arg11] : memref<4096x4096xbf16, strided<[?, ?], offset: ?>>
            %2 = affine.load %alloc[%arg8, %arg9, %arg10] : memref<1x1x4096xbf16>
            %3 = "spe.mac"(%0, %1, %2) : (bf16, bf16, bf16) -> bf16
            affine.store %3, %alloc[%arg8, %arg9, %arg10] : memref<1x1x4096xbf16>
          }
        }
      }
    }
    %collapse_shape = memref.collapse_shape %alloc [[0, 1], [2]] : memref<1x1x4096xbf16> into memref<1x4096xbf16>
    %expand_shape = memref.expand_shape %collapse_shape [[0], [1, 2, 3]] output_shape [1, 32, 1, 128] : memref<1x4096xbf16> into memref<1x32x1x128xbf16>
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<1x1x4096xbf16>
    affine.for %arg8 = 0 to 1 {
      affine.for %arg9 = 0 to 1 {
        affine.for %arg10 = 0 to 4096 {
          affine.store %cst_0, %alloc_1[%arg8, %arg9, %arg10] : memref<1x1x4096xbf16>
        }
      }
    }
    affine.for %arg8 = 0 to 1 {
      affine.for %arg9 = 0 to 1 {
        affine.for %arg10 = 0 to 4096 {
          affine.for %arg11 = 0 to 4096 {
            %0 = affine.load %arg0[%arg8, %arg9, %arg11] : memref<1x1x4096xbf16, strided<[?, ?, ?], offset: ?>>
            %1 = affine.load %arg5[%arg10, %arg11] : memref<4096x4096xbf16, strided<[?, ?], offset: ?>>
            %2 = affine.load %alloc_1[%arg8, %arg9, %arg10] : memref<1x1x4096xbf16>
            %3 = "spe.mac"(%0, %1, %2) : (bf16, bf16, bf16) -> bf16
            affine.store %3, %alloc_1[%arg8, %arg9, %arg10] : memref<1x1x4096xbf16>
          }
        }
      }
    }
    %collapse_shape_2 = memref.collapse_shape %alloc_1 [[0, 1], [2]] : memref<1x1x4096xbf16> into memref<1x4096xbf16>
    %expand_shape_3 = memref.expand_shape %collapse_shape_2 [[0], [1, 2, 3]] output_shape [1, 32, 1, 128] : memref<1x4096xbf16> into memref<1x32x1x128xbf16>
    %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<1x1x4096xbf16>
    affine.for %arg8 = 0 to 1 {
      affine.for %arg9 = 0 to 1 {
        affine.for %arg10 = 0 to 4096 {
          affine.store %cst_0, %alloc_4[%arg8, %arg9, %arg10] : memref<1x1x4096xbf16>
        }
      }
    }
    affine.for %arg8 = 0 to 1 {
      affine.for %arg9 = 0 to 1 {
        affine.for %arg10 = 0 to 4096 {
          affine.for %arg11 = 0 to 4096 {
            %0 = affine.load %arg0[%arg8, %arg9, %arg11] : memref<1x1x4096xbf16, strided<[?, ?, ?], offset: ?>>
            %1 = affine.load %arg6[%arg10, %arg11] : memref<4096x4096xbf16, strided<[?, ?], offset: ?>>
            %2 = affine.load %alloc_4[%arg8, %arg9, %arg10] : memref<1x1x4096xbf16>
            %3 = "spe.mac"(%0, %1, %2) : (bf16, bf16, bf16) -> bf16
            affine.store %3, %alloc_4[%arg8, %arg9, %arg10] : memref<1x1x4096xbf16>
          }
        }
      }
    }
    %expand_shape_5 = memref.expand_shape %arg1 [[0], [1, 2], [3]] output_shape [1, 1, 1, 128] : memref<1x1x128xbf16, strided<[?, ?, ?], offset: ?>> into memref<1x1x1x128xbf16, strided<[?, ?, ?, ?], offset: ?>>
    %expand_shape_6 = memref.expand_shape %arg2 [[0], [1, 2], [3]] output_shape [1, 1, 1, 128] : memref<1x1x128xbf16, strided<[?, ?, ?], offset: ?>> into memref<1x1x1x128xbf16, strided<[?, ?, ?, ?], offset: ?>>
    %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<1x32x1x128xbf16>
    affine.for %arg8 = 0 to 1 {
      affine.for %arg9 = 0 to 32 {
        affine.for %arg10 = 0 to 1 {
          affine.for %arg11 = 0 to 128 {
            %0 = affine.load %expand_shape[%arg8, %arg9, %arg10, %arg11] : memref<1x32x1x128xbf16>
            %1 = affine.load %expand_shape_5[%arg8, 0, %arg10, %arg11] : memref<1x1x1x128xbf16, strided<[?, ?, ?, ?], offset: ?>>
            %2 = "spe.add"(%0, %1) : (bf16, bf16) -> bf16
            affine.store %2, %alloc_7[%arg8, %arg9, %arg10, %arg11] : memref<1x32x1x128xbf16>
          }
        }
      }
    }
    %subview = memref.subview %expand_shape[0, 0, 0, 0] [1, 32, 1, 64] [1, 1, 1, 1] : memref<1x32x1x128xbf16> to memref<1x32x1x64xbf16, strided<[4096, 128, 128, 1]>>
    %subview_8 = memref.subview %expand_shape[0, 0, 0, 64] [1, 32, 1, 64] [1, 1, 1, 1] : memref<1x32x1x128xbf16> to memref<1x32x1x64xbf16, strided<[4096, 128, 128, 1], offset: 64>>
    %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<1x32x1x64xbf16>
    affine.for %arg8 = 0 to 1 {
      affine.for %arg9 = 0 to 32 {
        affine.for %arg10 = 0 to 1 {
          affine.for %arg11 = 0 to 64 {
            %0 = affine.load %subview_8[%arg8, %arg9, %arg10, %arg11] : memref<1x32x1x64xbf16, strided<[4096, 128, 128, 1], offset: 64>>
            %1 = "spe.neg"(%0) : (bf16) -> bf16
            affine.store %1, %alloc_9[%arg8, %arg9, %arg10, %arg11] : memref<1x32x1x64xbf16>
          }
        }
      }
    }
    %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<1x32x1x128xbf16>
    %subview_11 = memref.subview %alloc_10[0, 0, 0, 0] [1, 32, 1, 64] [1, 1, 1, 1] : memref<1x32x1x128xbf16> to memref<1x32x1x64xbf16, strided<[4096, 128, 128, 1]>>
    memref.copy %alloc_9, %subview_11 : memref<1x32x1x64xbf16> to memref<1x32x1x64xbf16, strided<[4096, 128, 128, 1]>>
    %subview_12 = memref.subview %alloc_10[0, 0, 0, 64] [1, 32, 1, 64] [1, 1, 1, 1] : memref<1x32x1x128xbf16> to memref<1x32x1x64xbf16, strided<[4096, 128, 128, 1], offset: 64>>
    memref.copy %subview, %subview_12 : memref<1x32x1x64xbf16, strided<[4096, 128, 128, 1]>> to memref<1x32x1x64xbf16, strided<[4096, 128, 128, 1], offset: 64>>
    %alloc_13 = memref.alloc() {alignment = 64 : i64} : memref<1x32x1x128xbf16>
    affine.for %arg8 = 0 to 1 {
      affine.for %arg9 = 0 to 32 {
        affine.for %arg10 = 0 to 1 {
          affine.for %arg11 = 0 to 128 {
            %0 = affine.load %alloc_10[%arg8, %arg9, %arg10, %arg11] : memref<1x32x1x128xbf16>
            %1 = affine.load %expand_shape_6[%arg8, 0, %arg10, %arg11] : memref<1x1x1x128xbf16, strided<[?, ?, ?, ?], offset: ?>>
            %2 = "spe.add"(%0, %1) : (bf16, bf16) -> bf16
            affine.store %2, %alloc_13[%arg8, %arg9, %arg10, %arg11] : memref<1x32x1x128xbf16>
          }
        }
      }
    }
    %alloc_14 = memref.alloc() {alignment = 64 : i64} : memref<1x32x1x128xbf16>
    affine.for %arg8 = 0 to 1 {
      affine.for %arg9 = 0 to 32 {
        affine.for %arg10 = 0 to 1 {
          affine.for %arg11 = 0 to 128 {
            %0 = affine.load %alloc_7[%arg8, %arg9, %arg10, %arg11] : memref<1x32x1x128xbf16>
            %1 = affine.load %alloc_13[%arg8, %arg9, %arg10, %arg11] : memref<1x32x1x128xbf16>
            %2 = "spe.add"(%0, %1) : (bf16, bf16) -> bf16
            affine.store %2, %alloc_14[%arg8, %arg9, %arg10, %arg11] : memref<1x32x1x128xbf16>
          }
        }
      }
    }
    %alloc_15 = memref.alloc() {alignment = 64 : i64} : memref<1x32x1x128xbf16>
    affine.for %arg8 = 0 to 1 {
      affine.for %arg9 = 0 to 32 {
        affine.for %arg10 = 0 to 1 {
          affine.for %arg11 = 0 to 128 {
            %0 = affine.load %expand_shape_3[%arg8, %arg9, %arg10, %arg11] : memref<1x32x1x128xbf16>
            %1 = affine.load %expand_shape_5[%arg8, 0, %arg10, %arg11] : memref<1x1x1x128xbf16, strided<[?, ?, ?, ?], offset: ?>>
            %2 = "spe.add"(%0, %1) : (bf16, bf16) -> bf16
            affine.store %2, %alloc_15[%arg8, %arg9, %arg10, %arg11] : memref<1x32x1x128xbf16>
          }
        }
      }
    }
    %subview_16 = memref.subview %expand_shape_3[0, 0, 0, 0] [1, 32, 1, 64] [1, 1, 1, 1] : memref<1x32x1x128xbf16> to memref<1x32x1x64xbf16, strided<[4096, 128, 128, 1]>>
    %subview_17 = memref.subview %expand_shape_3[0, 0, 0, 64] [1, 32, 1, 64] [1, 1, 1, 1] : memref<1x32x1x128xbf16> to memref<1x32x1x64xbf16, strided<[4096, 128, 128, 1], offset: 64>>
    %alloc_18 = memref.alloc() {alignment = 64 : i64} : memref<1x32x1x64xbf16>
    affine.for %arg8 = 0 to 1 {
      affine.for %arg9 = 0 to 32 {
        affine.for %arg10 = 0 to 1 {
          affine.for %arg11 = 0 to 64 {
            %0 = affine.load %subview_17[%arg8, %arg9, %arg10, %arg11] : memref<1x32x1x64xbf16, strided<[4096, 128, 128, 1], offset: 64>>
            %1 = "spe.neg"(%0) : (bf16) -> bf16
            affine.store %1, %alloc_18[%arg8, %arg9, %arg10, %arg11] : memref<1x32x1x64xbf16>
          }
        }
      }
    }
    %alloc_19 = memref.alloc() {alignment = 64 : i64} : memref<1x32x1x128xbf16>
    %subview_20 = memref.subview %alloc_19[0, 0, 0, 0] [1, 32, 1, 64] [1, 1, 1, 1] : memref<1x32x1x128xbf16> to memref<1x32x1x64xbf16, strided<[4096, 128, 128, 1]>>
    memref.copy %alloc_18, %subview_20 : memref<1x32x1x64xbf16> to memref<1x32x1x64xbf16, strided<[4096, 128, 128, 1]>>
    %subview_21 = memref.subview %alloc_19[0, 0, 0, 64] [1, 32, 1, 64] [1, 1, 1, 1] : memref<1x32x1x128xbf16> to memref<1x32x1x64xbf16, strided<[4096, 128, 128, 1], offset: 64>>
    memref.copy %subview_16, %subview_21 : memref<1x32x1x64xbf16, strided<[4096, 128, 128, 1]>> to memref<1x32x1x64xbf16, strided<[4096, 128, 128, 1], offset: 64>>
    %alloc_22 = memref.alloc() {alignment = 64 : i64} : memref<1x32x1x128xbf16>
    affine.for %arg8 = 0 to 1 {
      affine.for %arg9 = 0 to 32 {
        affine.for %arg10 = 0 to 1 {
          affine.for %arg11 = 0 to 128 {
            %0 = affine.load %alloc_19[%arg8, %arg9, %arg10, %arg11] : memref<1x32x1x128xbf16>
            %1 = affine.load %expand_shape_6[%arg8, 0, %arg10, %arg11] : memref<1x1x1x128xbf16, strided<[?, ?, ?, ?], offset: ?>>
            %2 = "spe.add"(%0, %1) : (bf16, bf16) -> bf16
            affine.store %2, %alloc_22[%arg8, %arg9, %arg10, %arg11] : memref<1x32x1x128xbf16>
          }
        }
      }
    }
    %alloc_23 = memref.alloc() {alignment = 64 : i64} : memref<1x32x1x128xbf16>
    affine.for %arg8 = 0 to 1 {
      affine.for %arg9 = 0 to 32 {
        affine.for %arg10 = 0 to 1 {
          affine.for %arg11 = 0 to 128 {
            %0 = affine.load %alloc_15[%arg8, %arg9, %arg10, %arg11] : memref<1x32x1x128xbf16>
            %1 = affine.load %alloc_22[%arg8, %arg9, %arg10, %arg11] : memref<1x32x1x128xbf16>
            %2 = "spe.add"(%0, %1) : (bf16, bf16) -> bf16
            affine.store %2, %alloc_23[%arg8, %arg9, %arg10, %arg11] : memref<1x32x1x128xbf16>
          }
        }
      }
    }
    %collapse_shape_24 = memref.collapse_shape %alloc_14 [[0, 1], [2], [3]] : memref<1x32x1x128xbf16> into memref<32x1x128xbf16>
    %collapse_shape_25 = memref.collapse_shape %alloc_23 [[0, 1, 2], [3]] : memref<1x32x1x128xbf16> into memref<32x128xbf16>
    %expand_shape_26 = memref.expand_shape %collapse_shape_25 [[0], [1, 2]] output_shape [32, 128, 1] : memref<32x128xbf16> into memref<32x128x1xbf16>
    %alloc_27 = memref.alloc() {alignment = 64 : i64} : memref<32x1x1xbf16>
    affine.for %arg8 = 0 to 32 {
      affine.for %arg9 = 0 to 1 {
        affine.for %arg10 = 0 to 1 {
          affine.store %cst_0, %alloc_27[%arg8, %arg9, %arg10] : memref<32x1x1xbf16>
        }
      }
    }
    affine.for %arg8 = 0 to 32 {
      affine.for %arg9 = 0 to 1 {
        affine.for %arg10 = 0 to 1 {
          affine.for %arg11 = 0 to 128 {
            %0 = affine.load %collapse_shape_24[%arg8, %arg9, %arg11] : memref<32x1x128xbf16>
            %1 = affine.load %expand_shape_26[%arg8, %arg11, %arg10] : memref<32x128x1xbf16>
            %2 = affine.load %alloc_27[%arg8, %arg9, %arg10] : memref<32x1x1xbf16>
            %3 = "spe.mac"(%0, %1, %2) : (bf16, bf16, bf16) -> bf16
            affine.store %3, %alloc_27[%arg8, %arg9, %arg10] : memref<32x1x1xbf16>
          }
        }
      }
    }
    %expand_shape_28 = memref.expand_shape %alloc_27 [[0, 1], [2], [3]] output_shape [1, 32, 1, 1] : memref<32x1x1xbf16> into memref<1x32x1x1xbf16>
    %alloc_29 = memref.alloc() {alignment = 64 : i64} : memref<1x32x1x1xbf16>
    affine.for %arg8 = 0 to 1 {
      affine.for %arg9 = 0 to 32 {
        affine.for %arg10 = 0 to 1 {
          affine.for %arg11 = 0 to 1 {
            %0 = affine.load %expand_shape_28[%arg8, %arg9, %arg10, %arg11] : memref<1x32x1x1xbf16>
            %1 = "spe.add"(%0, %cst) : (bf16, bf16) -> bf16
            affine.store %1, %alloc_29[%arg8, %arg9, %arg10, %arg11] : memref<1x32x1x1xbf16>
          }
        }
      }
    }
    %alloc_30 = memref.alloc() {alignment = 64 : i64} : memref<1x32x1x1xbf16>
    affine.for %arg8 = 0 to 1 {
      affine.for %arg9 = 0 to 32 {
        affine.for %arg10 = 0 to 1 {
          affine.for %arg11 = 0 to 1 {
            %0 = affine.load %alloc_29[%arg8, %arg9, %arg10, %arg11] : memref<1x32x1x1xbf16>
            %1 = affine.load %arg3[%arg8, %arg9, %arg10, %arg11] : memref<1x32x1x1xbf16, strided<[?, ?, ?, ?], offset: ?>>
            %2 = "spe.add"(%0, %1) : (bf16, bf16) -> bf16
            affine.store %2, %alloc_30[%arg8, %arg9, %arg10, %arg11] : memref<1x32x1x1xbf16>
          }
        }
      }
    }
    %alloc_31 = memref.alloc() {alignment = 64 : i64} : memref<1x32x1x1xf32>
    affine.for %arg8 = 0 to 1 {
      affine.for %arg9 = 0 to 32 {
        affine.for %arg10 = 0 to 1 {
          affine.for %arg11 = 0 to 1 {
            %0 = affine.load %alloc_30[%arg8, %arg9, %arg10, %arg11] : memref<1x32x1x1xbf16>
            %1 = arith.extf %0 : bf16 to f32
            affine.store %1, %alloc_31[%arg8, %arg9, %arg10, %arg11] : memref<1x32x1x1xf32>
          }
        }
      }
    }
    %alloc_32 = memref.alloc() {alignment = 64 : i64} : memref<1x32x1x1xf32>
    affine.for %arg8 = 0 to 1 {
      affine.for %arg9 = 0 to 32 {
        affine.for %arg10 = 0 to 1 {
          affine.for %arg11 = 0 to 1 {
            %0 = affine.load %alloc_31[%arg8, %arg9, %arg10, %arg11] : memref<1x32x1x1xf32>
            %1 = arith.subf %0, %0 : f32
            affine.store %1, %alloc_32[%arg8, %arg9, %arg10, %arg11] : memref<1x32x1x1xf32>
          }
        }
      }
    }
    %alloc_33 = memref.alloc() {alignment = 64 : i64} : memref<1x32x1x1xf32>
    affine.for %arg8 = 0 to 1 {
      affine.for %arg9 = 0 to 32 {
        affine.for %arg10 = 0 to 1 {
          affine.for %arg11 = 0 to 1 {
            %0 = affine.load %alloc_32[%arg8, %arg9, %arg10, %arg11] : memref<1x32x1x1xf32>
            %1 = "spe.exp"(%0) : (f32) -> f32
            affine.store %1, %alloc_33[%arg8, %arg9, %arg10, %arg11] : memref<1x32x1x1xf32>
          }
        }
      }
    }
    %alloc_34 = memref.alloc() {alignment = 64 : i64} : memref<1x32x1x1xf32>
    affine.for %arg8 = 0 to 1 {
      affine.for %arg9 = 0 to 32 {
        affine.for %arg10 = 0 to 1 {
          affine.for %arg11 = 0 to 1 {
            %0 = affine.load %alloc_33[%arg8, %arg9, %arg10, %arg11] : memref<1x32x1x1xf32>
            %1 = "spe.inv"(%0) : (f32) -> f32
            affine.store %1, %alloc_34[%arg8, %arg9, %arg10, %arg11] : memref<1x32x1x1xf32>
          }
        }
      }
    }
    %alloc_35 = memref.alloc() {alignment = 64 : i64} : memref<1x32x1x1xf32>
    affine.for %arg8 = 0 to 1 {
      affine.for %arg9 = 0 to 32 {
        affine.for %arg10 = 0 to 1 {
          affine.for %arg11 = 0 to 1 {
            %0 = affine.load %alloc_33[%arg8, %arg9, %arg10, %arg11] : memref<1x32x1x1xf32>
            %1 = affine.load %alloc_34[%arg8, %arg9, %arg10, %arg11] : memref<1x32x1x1xf32>
            %2 = "spe.add"(%0, %1) : (f32, f32) -> f32
            affine.store %2, %alloc_35[%arg8, %arg9, %arg10, %arg11] : memref<1x32x1x1xf32>
          }
        }
      }
    }
    %alloc_36 = memref.alloc() {alignment = 64 : i64} : memref<1x32x1x1xbf16>
    affine.for %arg8 = 0 to 1 {
      affine.for %arg9 = 0 to 32 {
        affine.for %arg10 = 0 to 1 {
          affine.for %arg11 = 0 to 1 {
            %0 = affine.load %alloc_35[%arg8, %arg9, %arg10, %arg11] : memref<1x32x1x1xf32>
            %1 = arith.truncf %0 : f32 to bf16
            affine.store %1, %alloc_36[%arg8, %arg9, %arg10, %arg11] : memref<1x32x1x1xbf16>
          }
        }
      }
    }
    %collapse_shape_37 = memref.collapse_shape %alloc_36 [[0, 1], [2], [3]] : memref<1x32x1x1xbf16> into memref<32x1x1xbf16>
    %collapse_shape_38 = memref.collapse_shape %alloc_4 [[0, 1, 2]] : memref<1x1x4096xbf16> into memref<4096xbf16>
    %expand_shape_39 = memref.expand_shape %collapse_shape_38 [[0, 1, 2]] output_shape [32, 1, 128] : memref<4096xbf16> into memref<32x1x128xbf16>
    %alloc_40 = memref.alloc() {alignment = 64 : i64} : memref<32x1x128xbf16>
    affine.for %arg8 = 0 to 32 {
      affine.for %arg9 = 0 to 1 {
        affine.for %arg10 = 0 to 128 {
          affine.store %cst_0, %alloc_40[%arg8, %arg9, %arg10] : memref<32x1x128xbf16>
        }
      }
    }
    affine.for %arg8 = 0 to 32 {
      affine.for %arg9 = 0 to 1 {
        affine.for %arg10 = 0 to 128 {
          affine.for %arg11 = 0 to 1 {
            %0 = affine.load %collapse_shape_37[%arg8, %arg9, %arg11] : memref<32x1x1xbf16>
            %1 = affine.load %expand_shape_39[%arg8, %arg11, %arg10] : memref<32x1x128xbf16>
            %2 = affine.load %alloc_40[%arg8, %arg9, %arg10] : memref<32x1x128xbf16>
            %3 = "spe.mac"(%0, %1, %2) : (bf16, bf16, bf16) -> bf16
            affine.store %3, %alloc_40[%arg8, %arg9, %arg10] : memref<32x1x128xbf16>
          }
        }
      }
    }
    %collapse_shape_41 = memref.collapse_shape %alloc_40 [[0, 1, 2]] : memref<32x1x128xbf16> into memref<4096xbf16>
    %expand_shape_42 = memref.expand_shape %collapse_shape_41 [[0, 1, 2]] output_shape [1, 1, 4096] : memref<4096xbf16> into memref<1x1x4096xbf16>
    %alloc_43 = memref.alloc() {alignment = 64 : i64} : memref<1x1x4096xbf16>
    affine.for %arg8 = 0 to 1 {
      affine.for %arg9 = 0 to 1 {
        affine.for %arg10 = 0 to 4096 {
          affine.store %cst_0, %alloc_43[%arg8, %arg9, %arg10] : memref<1x1x4096xbf16>
        }
      }
    }
    affine.for %arg8 = 0 to 1 {
      affine.for %arg9 = 0 to 1 {
        affine.for %arg10 = 0 to 4096 {
          affine.for %arg11 = 0 to 4096 {
            %0 = affine.load %expand_shape_42[%arg8, %arg9, %arg11] : memref<1x1x4096xbf16>
            %1 = affine.load %arg7[%arg10, %arg11] : memref<4096x4096xbf16, strided<[?, ?], offset: ?>>
            %2 = affine.load %alloc_43[%arg8, %arg9, %arg10] : memref<1x1x4096xbf16>
            %3 = "spe.mac"(%0, %1, %2) : (bf16, bf16, bf16) -> bf16
            affine.store %3, %alloc_43[%arg8, %arg9, %arg10] : memref<1x1x4096xbf16>
          }
        }
      }
    }
    return %alloc_43, %alloc_36 : memref<1x1x4096xbf16>, memref<1x32x1x1xbf16>
  }
}
