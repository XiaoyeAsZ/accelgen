#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  func.func @main(%arg0: tensor<1x1x4096xbf16>, %arg1: tensor<14336x4096xbf16>, %arg2: tensor<14336x4096xbf16>, %arg3: tensor<4096x14336xbf16>) -> tensor<1x1x4096xbf16> {
    %0 = tosa.const_shape  {values = dense<[1, 4096, 14336]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %1 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xbf16>}> : () -> tensor<1xbf16>
    %2 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %3 = tosa.const_shape  {values = dense<[1, 14336, 4096]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %4 = tensor.empty() : tensor<4096x14336xbf16>
    %transposed = linalg.transpose ins(%arg1 : tensor<14336x4096xbf16>) outs(%4 : tensor<4096x14336xbf16>) permutation = [1, 0] 
    %5 = tosa.reshape %transposed, %0 : (tensor<4096x14336xbf16>, !tosa.shape<3>) -> tensor<1x4096x14336xbf16>
    %cst = arith.constant 0.000000e+00 : bf16
    %6 = tensor.empty() : tensor<1x1x14336xbf16>
    %7 = linalg.fill ins(%cst : bf16) outs(%6 : tensor<1x1x14336xbf16>) -> tensor<1x1x14336xbf16>
    %8 = linalg.batch_matmul ins(%arg0, %5 : tensor<1x1x4096xbf16>, tensor<1x4096x14336xbf16>) outs(%7 : tensor<1x1x14336xbf16>) -> tensor<1x1x14336xbf16>
    %9 = tensor.empty() : tensor<1x1x14336xbf16>
    %10 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%8 : tensor<1x1x14336xbf16>) outs(%9 : tensor<1x1x14336xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %cst_4 = arith.constant 1.000000e+00 : bf16
      %25 = arith.negf %in : bf16
      %26 = math.exp %25 : bf16
      %27 = arith.addf %26, %cst_4 : bf16
      %28 = arith.divf %cst_4, %27 : bf16
      linalg.yield %28 : bf16
    } -> tensor<1x1x14336xbf16>
    %11 = tensor.empty() : tensor<1x1x14336xbf16>
    %12 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%10, %8 : tensor<1x1x14336xbf16>, tensor<1x1x14336xbf16>) outs(%11 : tensor<1x1x14336xbf16>) {
    ^bb0(%in: bf16, %in_4: bf16, %out: bf16):
      %25 = arith.mulf %in, %in_4 : bf16
      linalg.yield %25 : bf16
    } -> tensor<1x1x14336xbf16>
    %13 = tensor.empty() : tensor<4096x14336xbf16>
    %transposed_0 = linalg.transpose ins(%arg2 : tensor<14336x4096xbf16>) outs(%13 : tensor<4096x14336xbf16>) permutation = [1, 0] 
    %14 = tosa.reshape %transposed_0, %0 : (tensor<4096x14336xbf16>, !tosa.shape<3>) -> tensor<1x4096x14336xbf16>
    %cst_1 = arith.constant 0.000000e+00 : bf16
    %15 = tensor.empty() : tensor<1x1x14336xbf16>
    %16 = linalg.fill ins(%cst_1 : bf16) outs(%15 : tensor<1x1x14336xbf16>) -> tensor<1x1x14336xbf16>
    %17 = linalg.batch_matmul ins(%arg0, %14 : tensor<1x1x4096xbf16>, tensor<1x4096x14336xbf16>) outs(%16 : tensor<1x1x14336xbf16>) -> tensor<1x1x14336xbf16>
    %18 = tensor.empty() : tensor<1x1x14336xbf16>
    %19 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%12, %17 : tensor<1x1x14336xbf16>, tensor<1x1x14336xbf16>) outs(%18 : tensor<1x1x14336xbf16>) {
    ^bb0(%in: bf16, %in_4: bf16, %out: bf16):
      %25 = arith.mulf %in, %in_4 : bf16
      linalg.yield %25 : bf16
    } -> tensor<1x1x14336xbf16>
    %20 = tensor.empty() : tensor<14336x4096xbf16>
    %transposed_2 = linalg.transpose ins(%arg3 : tensor<4096x14336xbf16>) outs(%20 : tensor<14336x4096xbf16>) permutation = [1, 0] 
    %21 = tosa.reshape %transposed_2, %3 : (tensor<14336x4096xbf16>, !tosa.shape<3>) -> tensor<1x14336x4096xbf16>
    %cst_3 = arith.constant 0.000000e+00 : bf16
    %22 = tensor.empty() : tensor<1x1x4096xbf16>
    %23 = linalg.fill ins(%cst_3 : bf16) outs(%22 : tensor<1x1x4096xbf16>) -> tensor<1x1x4096xbf16>
    %24 = linalg.batch_matmul ins(%19, %21 : tensor<1x1x14336xbf16>, tensor<1x14336x4096xbf16>) outs(%23 : tensor<1x1x4096xbf16>) -> tensor<1x1x4096xbf16>
    return %24 : tensor<1x1x4096xbf16>
  }
}

