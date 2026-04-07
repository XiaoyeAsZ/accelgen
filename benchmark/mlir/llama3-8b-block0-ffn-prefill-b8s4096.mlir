#map = affine_map<(d0, d1, d2) -> (d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  func.func @main(%arg0: tensor<8x4096x4096xbf16>, %arg1: tensor<14336x4096xbf16>, %arg2: tensor<14336x4096xbf16>, %arg3: tensor<4096x14336xbf16>) -> tensor<8x4096x4096xbf16> {
    %cst = arith.constant 0.000000e+00 : bf16
    %cst_0 = arith.constant 1.000000e+00 : bf16
    %0 = tensor.empty() : tensor<4096x14336xbf16>
    %transposed = linalg.transpose ins(%arg1 : tensor<14336x4096xbf16>) outs(%0 : tensor<4096x14336xbf16>) permutation = [1, 0] 
    %1 = tensor.empty() : tensor<8x4096x14336xbf16>
    %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed : tensor<4096x14336xbf16>) outs(%1 : tensor<8x4096x14336xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x4096x14336xbf16>
    %3 = linalg.fill ins(%cst : bf16) outs(%1 : tensor<8x4096x14336xbf16>) -> tensor<8x4096x14336xbf16>
    %4 = linalg.batch_matmul ins(%arg0, %2 : tensor<8x4096x4096xbf16>, tensor<8x4096x14336xbf16>) outs(%3 : tensor<8x4096x14336xbf16>) -> tensor<8x4096x14336xbf16>
    %5 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4 : tensor<8x4096x14336xbf16>) outs(%1 : tensor<8x4096x14336xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %16 = arith.negf %in : bf16
      %17 = math.exp %16 : bf16
      %18 = arith.addf %17, %cst_0 : bf16
      %19 = arith.divf %cst_0, %18 : bf16
      linalg.yield %19 : bf16
    } -> tensor<8x4096x14336xbf16>
    %6 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5, %4 : tensor<8x4096x14336xbf16>, tensor<8x4096x14336xbf16>) outs(%1 : tensor<8x4096x14336xbf16>) {
    ^bb0(%in: bf16, %in_3: bf16, %out: bf16):
      %16 = arith.mulf %in, %in_3 : bf16
      linalg.yield %16 : bf16
    } -> tensor<8x4096x14336xbf16>
    %transposed_1 = linalg.transpose ins(%arg2 : tensor<14336x4096xbf16>) outs(%0 : tensor<4096x14336xbf16>) permutation = [1, 0] 
    %7 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_1 : tensor<4096x14336xbf16>) outs(%1 : tensor<8x4096x14336xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x4096x14336xbf16>
    %8 = linalg.batch_matmul ins(%arg0, %7 : tensor<8x4096x4096xbf16>, tensor<8x4096x14336xbf16>) outs(%3 : tensor<8x4096x14336xbf16>) -> tensor<8x4096x14336xbf16>
    %9 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%6, %8 : tensor<8x4096x14336xbf16>, tensor<8x4096x14336xbf16>) outs(%1 : tensor<8x4096x14336xbf16>) {
    ^bb0(%in: bf16, %in_3: bf16, %out: bf16):
      %16 = arith.mulf %in, %in_3 : bf16
      linalg.yield %16 : bf16
    } -> tensor<8x4096x14336xbf16>
    %10 = tensor.empty() : tensor<14336x4096xbf16>
    %transposed_2 = linalg.transpose ins(%arg3 : tensor<4096x14336xbf16>) outs(%10 : tensor<14336x4096xbf16>) permutation = [1, 0] 
    %11 = tensor.empty() : tensor<8x14336x4096xbf16>
    %12 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_2 : tensor<14336x4096xbf16>) outs(%11 : tensor<8x14336x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x14336x4096xbf16>
    %13 = tensor.empty() : tensor<8x4096x4096xbf16>
    %14 = linalg.fill ins(%cst : bf16) outs(%13 : tensor<8x4096x4096xbf16>) -> tensor<8x4096x4096xbf16>
    %15 = linalg.batch_matmul ins(%9, %12 : tensor<8x4096x14336xbf16>, tensor<8x14336x4096xbf16>) outs(%14 : tensor<8x4096x4096xbf16>) -> tensor<8x4096x4096xbf16>
    return %15 : tensor<8x4096x4096xbf16>
  }
}
