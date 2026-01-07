module {
  func.func @main(%arg0: tensor<1x1x4096xbf16>, %arg1: tensor<14336x4096xbf16>, %arg2: tensor<14336x4096xbf16>, %arg3: tensor<4096x14336xbf16>) -> tensor<1x1x4096xbf16> {
    %0 = tosa.const_shape  {values = dense<[1, 4096, 14336]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %1 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xbf16>}> : () -> tensor<1xbf16>
    %2 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %3 = tosa.const_shape  {values = dense<[1, 14336, 4096]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %4 = tosa.transpose %arg1 {perms = array<i32: 1, 0>} : (tensor<14336x4096xbf16>) -> tensor<4096x14336xbf16>
    %5 = tosa.reshape %4, %0 : (tensor<4096x14336xbf16>, !tosa.shape<3>) -> tensor<1x4096x14336xbf16>
    %6 = tosa.matmul %arg0, %5, %1, %1 : (tensor<1x1x4096xbf16>, tensor<1x4096x14336xbf16>, tensor<1xbf16>, tensor<1xbf16>) -> tensor<1x1x14336xbf16>
    %7 = tosa.sigmoid %6 : (tensor<1x1x14336xbf16>) -> tensor<1x1x14336xbf16>
    %8 = tosa.mul %7, %6, %2 : (tensor<1x1x14336xbf16>, tensor<1x1x14336xbf16>, tensor<1xi8>) -> tensor<1x1x14336xbf16>
    %9 = tosa.transpose %arg2 {perms = array<i32: 1, 0>} : (tensor<14336x4096xbf16>) -> tensor<4096x14336xbf16>
    %10 = tosa.reshape %9, %0 : (tensor<4096x14336xbf16>, !tosa.shape<3>) -> tensor<1x4096x14336xbf16>
    %11 = tosa.matmul %arg0, %10, %1, %1 : (tensor<1x1x4096xbf16>, tensor<1x4096x14336xbf16>, tensor<1xbf16>, tensor<1xbf16>) -> tensor<1x1x14336xbf16>
    %12 = tosa.mul %8, %11, %2 : (tensor<1x1x14336xbf16>, tensor<1x1x14336xbf16>, tensor<1xi8>) -> tensor<1x1x14336xbf16>
    %13 = tosa.transpose %arg3 {perms = array<i32: 1, 0>} : (tensor<4096x14336xbf16>) -> tensor<14336x4096xbf16>
    %14 = tosa.reshape %13, %3 : (tensor<14336x4096xbf16>, !tosa.shape<3>) -> tensor<1x14336x4096xbf16>
    %15 = tosa.matmul %12, %14, %1, %1 : (tensor<1x1x14336xbf16>, tensor<1x14336x4096xbf16>, tensor<1xbf16>, tensor<1xbf16>) -> tensor<1x1x4096xbf16>
    return %15 : tensor<1x1x4096xbf16>
  }
}
