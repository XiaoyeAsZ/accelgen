#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  func.func @main(%arg0: tensor<1x1x4096xbf16>, %arg1: tensor<14336x4096xbf16>, %arg2: tensor<14336x4096xbf16>, %arg3: tensor<4096x14336xbf16>) -> tensor<1x1x4096xbf16> {
    %cst = arith.constant 1.000000e+00 : bf16
    %cst_0 = arith.constant 0.000000e+00 : bf16
    %0 = tosa.const_shape  {values = dense<[1, 4096, 14336]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %1 = tosa.const_shape  {values = dense<[1, 14336, 4096]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %2 = tensor.empty() : tensor<4096x14336xbf16>
    %transposed = linalg.transpose ins(%arg1 : tensor<14336x4096xbf16>) outs(%2 : tensor<4096x14336xbf16>) permutation = [1, 0] 
    %3 = tosa.reshape %transposed, %0 : (tensor<4096x14336xbf16>, !tosa.shape<3>) -> tensor<1x4096x14336xbf16>
    %4 = tensor.empty() : tensor<1x1x14336xbf16>
    %5 = linalg.fill ins(%cst_0 : bf16) outs(%4 : tensor<1x1x14336xbf16>) -> tensor<1x1x14336xbf16>
    %6 = linalg.batch_matmul ins(%arg0, %3 : tensor<1x1x4096xbf16>, tensor<1x4096x14336xbf16>) outs(%5 : tensor<1x1x14336xbf16>) -> tensor<1x1x14336xbf16>
    %7 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%6 : tensor<1x1x14336xbf16>) outs(%4 : tensor<1x1x14336xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %17 = arith.negf %in : bf16
      %18 = math.exp %17 : bf16
      %19 = arith.addf %18, %cst : bf16
      %20 = arith.divf %cst, %19 : bf16
      linalg.yield %20 : bf16
    } -> tensor<1x1x14336xbf16>
    %8 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%7, %6 : tensor<1x1x14336xbf16>, tensor<1x1x14336xbf16>) outs(%4 : tensor<1x1x14336xbf16>) {
    ^bb0(%in: bf16, %in_3: bf16, %out: bf16):
      %17 = arith.mulf %in, %in_3 : bf16
      linalg.yield %17 : bf16
    } -> tensor<1x1x14336xbf16>
    %transposed_1 = linalg.transpose ins(%arg2 : tensor<14336x4096xbf16>) outs(%2 : tensor<4096x14336xbf16>) permutation = [1, 0] 
    %9 = tosa.reshape %transposed_1, %0 : (tensor<4096x14336xbf16>, !tosa.shape<3>) -> tensor<1x4096x14336xbf16>
    %10 = linalg.batch_matmul ins(%arg0, %9 : tensor<1x1x4096xbf16>, tensor<1x4096x14336xbf16>) outs(%5 : tensor<1x1x14336xbf16>) -> tensor<1x1x14336xbf16>
    %11 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%8, %10 : tensor<1x1x14336xbf16>, tensor<1x1x14336xbf16>) outs(%4 : tensor<1x1x14336xbf16>) {
    ^bb0(%in: bf16, %in_3: bf16, %out: bf16):
      %17 = arith.mulf %in, %in_3 : bf16
      linalg.yield %17 : bf16
    } -> tensor<1x1x14336xbf16>
    %12 = tensor.empty() : tensor<14336x4096xbf16>
    %transposed_2 = linalg.transpose ins(%arg3 : tensor<4096x14336xbf16>) outs(%12 : tensor<14336x4096xbf16>) permutation = [1, 0] 
    %13 = tosa.reshape %transposed_2, %1 : (tensor<14336x4096xbf16>, !tosa.shape<3>) -> tensor<1x14336x4096xbf16>
    %14 = tensor.empty() : tensor<1x1x4096xbf16>
    %15 = linalg.fill ins(%cst_0 : bf16) outs(%14 : tensor<1x1x4096xbf16>) -> tensor<1x1x4096xbf16>
    %16 = linalg.batch_matmul ins(%11, %13 : tensor<1x1x14336xbf16>, tensor<1x14336x4096xbf16>) outs(%15 : tensor<1x1x4096xbf16>) -> tensor<1x1x4096xbf16>
    return %16 : tensor<1x1x4096xbf16>
  }
}

