module {
  func.func @main(%arg0: tensor<2x1024x4096xbf16>, %arg1: tensor<1x1x128xbf16>, %arg2: tensor<1x1x128xbf16>, %arg3: !torch.none, %arg4: tensor<4096x4096xbf16>, %arg5: tensor<1024x4096xbf16>, %arg6: tensor<1024x4096xbf16>, %arg7: tensor<4096x4096xbf16>) -> (tensor<2x1024x4096xbf16>, tensor<2x32x1024x1024xbf16>) {
    %0 = tosa.const_shape  {values = dense<[1, 4096, 4096]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %1 = tosa.const_shape  {values = dense<[1, 2048, 4096]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %2 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xbf16>}> : () -> tensor<1xbf16>
    %3 = tosa.const_shape  {values = dense<[2, 1024, 4096]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %4 = tosa.const_shape  {values = dense<[2, 1024, 32, 128]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %5 = tosa.const_shape  {values = dense<[1, 4096, 1024]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %6 = tosa.const_shape  {values = dense<[2, 1024, 8, 128]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %7 = tosa.const_shape  {values = dense<[1, 1, 1, 128]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %8 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %9 = tosa.const_shape  {values = dense<[2, 32, 1024, 64]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %10 = tosa.const_shape  {values = dense<0> : tensor<4xindex>} : () -> !tosa.shape<4>
    %11 = tosa.const_shape  {values = dense<[0, 0, 0, 64]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %12 = tosa.const_shape  {values = dense<[2, 8, 1024, 64]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %13 = tosa.const_shape  {values = dense<[2, 8, 1, 1024, 128]> : tensor<5xindex>} : () -> !tosa.shape<5>
    %14 = tosa.const_shape  {values = dense<[1, 1, 4, 1, 1]> : tensor<5xindex>} : () -> !tosa.shape<5>
    %15 = tosa.const_shape  {values = dense<[2, 32, 1024, 128]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %16 = tosa.const_shape  {values = dense<[64, 1024, 128]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %17 = tosa.const_shape  {values = dense<[64, 128, 1024]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %18 = tosa.const_shape  {values = dense<[2, 32, 1024, 1024]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %19 = tosa.const_shape  {values = dense<[64, 1024, 1024]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %20 = "tosa.const"() <{values = dense<8.837890e-02> : tensor<1x1x1x1xbf16>}> : () -> tensor<1x1x1x1xbf16>
    %21 = tosa.transpose %arg4 {perms = array<i32: 1, 0>} : (tensor<4096x4096xbf16>) -> tensor<4096x4096xbf16>
    %22 = tosa.reshape %21, %0 : (tensor<4096x4096xbf16>, !tosa.shape<3>) -> tensor<1x4096x4096xbf16>
    %23 = tosa.reshape %arg0, %1 : (tensor<2x1024x4096xbf16>, !tosa.shape<3>) -> tensor<1x2048x4096xbf16>
    %24 = tosa.matmul %23, %22, %2, %2 : (tensor<1x2048x4096xbf16>, tensor<1x4096x4096xbf16>, tensor<1xbf16>, tensor<1xbf16>) -> tensor<1x2048x4096xbf16>
    %25 = tosa.reshape %24, %4 : (tensor<1x2048x4096xbf16>, !tosa.shape<4>) -> tensor<2x1024x32x128xbf16>
    %26 = tosa.transpose %25 {perms = array<i32: 0, 2, 1, 3>} : (tensor<2x1024x32x128xbf16>) -> tensor<2x32x1024x128xbf16>
    %27 = tosa.transpose %arg5 {perms = array<i32: 1, 0>} : (tensor<1024x4096xbf16>) -> tensor<4096x1024xbf16>
    %28 = tosa.reshape %27, %5 : (tensor<4096x1024xbf16>, !tosa.shape<3>) -> tensor<1x4096x1024xbf16>
    %29 = tosa.matmul %23, %28, %2, %2 : (tensor<1x2048x4096xbf16>, tensor<1x4096x1024xbf16>, tensor<1xbf16>, tensor<1xbf16>) -> tensor<1x2048x1024xbf16>
    %30 = tosa.reshape %29, %6 : (tensor<1x2048x1024xbf16>, !tosa.shape<4>) -> tensor<2x1024x8x128xbf16>
    %31 = tosa.transpose %30 {perms = array<i32: 0, 2, 1, 3>} : (tensor<2x1024x8x128xbf16>) -> tensor<2x8x1024x128xbf16>
    %32 = tosa.transpose %arg6 {perms = array<i32: 1, 0>} : (tensor<1024x4096xbf16>) -> tensor<4096x1024xbf16>
    %33 = tosa.reshape %32, %5 : (tensor<4096x1024xbf16>, !tosa.shape<3>) -> tensor<1x4096x1024xbf16>
    %34 = tosa.matmul %23, %33, %2, %2 : (tensor<1x2048x4096xbf16>, tensor<1x4096x1024xbf16>, tensor<1xbf16>, tensor<1xbf16>) -> tensor<1x2048x1024xbf16>
    %35 = tosa.reshape %34, %6 : (tensor<1x2048x1024xbf16>, !tosa.shape<4>) -> tensor<2x1024x8x128xbf16>
    %36 = tosa.transpose %35 {perms = array<i32: 0, 2, 1, 3>} : (tensor<2x1024x8x128xbf16>) -> tensor<2x8x1024x128xbf16>
    %37 = tosa.reshape %arg1, %7 : (tensor<1x1x128xbf16>, !tosa.shape<4>) -> tensor<1x1x1x128xbf16>
    %38 = tosa.reshape %arg2, %7 : (tensor<1x1x128xbf16>, !tosa.shape<4>) -> tensor<1x1x1x128xbf16>
    %39 = tosa.mul %26, %37, %8 : (tensor<2x32x1024x128xbf16>, tensor<1x1x1x128xbf16>, tensor<1xi8>) -> tensor<2x32x1024x128xbf16>
    %40 = tosa.slice %26, %10, %9 : (tensor<2x32x1024x128xbf16>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<2x32x1024x64xbf16>
    %41 = tosa.slice %26, %11, %9 : (tensor<2x32x1024x128xbf16>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<2x32x1024x64xbf16>
    %42 = tosa.negate %41, %2, %2 : (tensor<2x32x1024x64xbf16>, tensor<1xbf16>, tensor<1xbf16>) -> tensor<2x32x1024x64xbf16>
    %43 = tosa.concat %42, %40 {axis = 3 : i32} : (tensor<2x32x1024x64xbf16>, tensor<2x32x1024x64xbf16>) -> tensor<2x32x1024x128xbf16>
    %44 = tosa.mul %43, %38, %8 : (tensor<2x32x1024x128xbf16>, tensor<1x1x1x128xbf16>, tensor<1xi8>) -> tensor<2x32x1024x128xbf16>
    %45 = tosa.add %39, %44 : (tensor<2x32x1024x128xbf16>, tensor<2x32x1024x128xbf16>) -> tensor<2x32x1024x128xbf16>
    %46 = tosa.mul %31, %37, %8 : (tensor<2x8x1024x128xbf16>, tensor<1x1x1x128xbf16>, tensor<1xi8>) -> tensor<2x8x1024x128xbf16>
    %47 = tosa.slice %31, %10, %12 : (tensor<2x8x1024x128xbf16>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<2x8x1024x64xbf16>
    %48 = tosa.slice %31, %11, %12 : (tensor<2x8x1024x128xbf16>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<2x8x1024x64xbf16>
    %49 = tosa.negate %48, %2, %2 : (tensor<2x8x1024x64xbf16>, tensor<1xbf16>, tensor<1xbf16>) -> tensor<2x8x1024x64xbf16>
    %50 = tosa.concat %49, %47 {axis = 3 : i32} : (tensor<2x8x1024x64xbf16>, tensor<2x8x1024x64xbf16>) -> tensor<2x8x1024x128xbf16>
    %51 = tosa.mul %50, %38, %8 : (tensor<2x8x1024x128xbf16>, tensor<1x1x1x128xbf16>, tensor<1xi8>) -> tensor<2x8x1024x128xbf16>
    %52 = tosa.add %46, %51 : (tensor<2x8x1024x128xbf16>, tensor<2x8x1024x128xbf16>) -> tensor<2x8x1024x128xbf16>
    %53 = tosa.reshape %52, %13 : (tensor<2x8x1024x128xbf16>, !tosa.shape<5>) -> tensor<2x8x1x1024x128xbf16>
    %54 = tosa.tile %53, %14 : (tensor<2x8x1x1024x128xbf16>, !tosa.shape<5>) -> tensor<2x8x4x1024x128xbf16>
    %55 = tosa.reshape %54, %15 : (tensor<2x8x4x1024x128xbf16>, !tosa.shape<4>) -> tensor<2x32x1024x128xbf16>
    %56 = tosa.reshape %36, %13 : (tensor<2x8x1024x128xbf16>, !tosa.shape<5>) -> tensor<2x8x1x1024x128xbf16>
    %57 = tosa.tile %56, %14 : (tensor<2x8x1x1024x128xbf16>, !tosa.shape<5>) -> tensor<2x8x4x1024x128xbf16>
    %58 = tosa.transpose %55 {perms = array<i32: 0, 1, 3, 2>} : (tensor<2x32x1024x128xbf16>) -> tensor<2x32x128x1024xbf16>
    %59 = tosa.reshape %45, %16 : (tensor<2x32x1024x128xbf16>, !tosa.shape<3>) -> tensor<64x1024x128xbf16>
    %60 = tosa.reshape %58, %17 : (tensor<2x32x128x1024xbf16>, !tosa.shape<3>) -> tensor<64x128x1024xbf16>
    %61 = tosa.matmul %59, %60, %2, %2 : (tensor<64x1024x128xbf16>, tensor<64x128x1024xbf16>, tensor<1xbf16>, tensor<1xbf16>) -> tensor<64x1024x1024xbf16>
    %62 = tosa.reshape %61, %18 : (tensor<64x1024x1024xbf16>, !tosa.shape<4>) -> tensor<2x32x1024x1024xbf16>
    %63 = tosa.mul %62, %20, %8 : (tensor<2x32x1024x1024xbf16>, tensor<1x1x1x1xbf16>, tensor<1xi8>) -> tensor<2x32x1024x1024xbf16>
    %64 = tosa.cast %63 : (tensor<2x32x1024x1024xbf16>) -> tensor<2x32x1024x1024xf32>
    %65 = tosa.reduce_max %64 {axis = 3 : i32} : (tensor<2x32x1024x1024xf32>) -> tensor<2x32x1024x1xf32>
    %66 = tosa.sub %64, %65 : (tensor<2x32x1024x1024xf32>, tensor<2x32x1024x1xf32>) -> tensor<2x32x1024x1024xf32>
    %67 = tosa.exp %66 : (tensor<2x32x1024x1024xf32>) -> tensor<2x32x1024x1024xf32>
    %68 = tosa.reduce_sum %67 {axis = 3 : i32} : (tensor<2x32x1024x1024xf32>) -> tensor<2x32x1024x1xf32>
    %69 = tosa.reciprocal %68 : (tensor<2x32x1024x1xf32>) -> tensor<2x32x1024x1xf32>
    %70 = tosa.mul %67, %69, %8 : (tensor<2x32x1024x1024xf32>, tensor<2x32x1024x1xf32>, tensor<1xi8>) -> tensor<2x32x1024x1024xf32>
    %71 = tosa.cast %70 : (tensor<2x32x1024x1024xf32>) -> tensor<2x32x1024x1024xbf16>
    %72 = tosa.reshape %71, %19 : (tensor<2x32x1024x1024xbf16>, !tosa.shape<3>) -> tensor<64x1024x1024xbf16>
    %73 = tosa.reshape %57, %16 : (tensor<2x8x4x1024x128xbf16>, !tosa.shape<3>) -> tensor<64x1024x128xbf16>
    %74 = tosa.matmul %72, %73, %2, %2 : (tensor<64x1024x1024xbf16>, tensor<64x1024x128xbf16>, tensor<1xbf16>, tensor<1xbf16>) -> tensor<64x1024x128xbf16>
    %75 = tosa.reshape %74, %15 : (tensor<64x1024x128xbf16>, !tosa.shape<4>) -> tensor<2x32x1024x128xbf16>
    %76 = tosa.transpose %75 {perms = array<i32: 0, 2, 1, 3>} : (tensor<2x32x1024x128xbf16>) -> tensor<2x1024x32x128xbf16>
    %77 = tosa.transpose %arg7 {perms = array<i32: 1, 0>} : (tensor<4096x4096xbf16>) -> tensor<4096x4096xbf16>
    %78 = tosa.reshape %77, %0 : (tensor<4096x4096xbf16>, !tosa.shape<3>) -> tensor<1x4096x4096xbf16>
    %79 = tosa.reshape %76, %1 : (tensor<2x1024x32x128xbf16>, !tosa.shape<3>) -> tensor<1x2048x4096xbf16>
    %80 = tosa.matmul %79, %78, %2, %2 : (tensor<1x2048x4096xbf16>, tensor<1x4096x4096xbf16>, tensor<1xbf16>, tensor<1xbf16>) -> tensor<1x2048x4096xbf16>
    %81 = tosa.reshape %80, %3 : (tensor<1x2048x4096xbf16>, !tosa.shape<3>) -> tensor<2x1024x4096xbf16>
    return %81, %71 : tensor<2x1024x4096xbf16>, tensor<2x32x1024x1024xbf16>
  }
}
