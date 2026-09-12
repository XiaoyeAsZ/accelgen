#map = affine_map<(d0, d1, d2) -> (d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d1, d2)>
module {
  func.func @main(%arg0: tensor<1x1x8192xbf16>, %arg1: tensor<28672x8192xbf16>, %arg2: tensor<28672x8192xbf16>, %arg3: tensor<8192x28672xbf16>) -> tensor<1x1x8192xbf16> {
    %cst = arith.constant 0.000000e+00 : bf16
    %cst_0 = arith.constant 1.000000e+00 : bf16
    %0 = tensor.empty() : tensor<8192x28672xbf16>
    %transposed = linalg.transpose ins(%arg1 : tensor<28672x8192xbf16>) outs(%0 : tensor<8192x28672xbf16>) permutation = [1, 0] 
    %1 = tensor.empty() : tensor<1x1x8192xbf16>
    %collapsed = tensor.collapse_shape %arg0 [[0, 1, 2]] : tensor<1x1x8192xbf16> into tensor<8192xbf16>
    %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapsed : tensor<8192xbf16>) outs(%1 : tensor<1x1x8192xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x1x8192xbf16>
    %3 = tensor.empty() : tensor<1x8192x28672xbf16>
    %4 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed : tensor<8192x28672xbf16>) outs(%3 : tensor<1x8192x28672xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x8192x28672xbf16>
    %5 = tensor.empty() : tensor<1x1x28672xbf16>
    %6 = linalg.fill ins(%cst : bf16) outs(%5 : tensor<1x1x28672xbf16>) -> tensor<1x1x28672xbf16>
    %7 = linalg.batch_matmul ins(%2, %4 : tensor<1x1x8192xbf16>, tensor<1x8192x28672xbf16>) outs(%6 : tensor<1x1x28672xbf16>) -> tensor<1x1x28672xbf16>
    %8 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%7 : tensor<1x1x28672xbf16>) outs(%5 : tensor<1x1x28672xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %19 = arith.negf %in : bf16
      %20 = math.exp %19 : bf16
      %21 = arith.addf %20, %cst_0 : bf16
      %22 = arith.divf %cst_0, %21 : bf16
      linalg.yield %22 : bf16
    } -> tensor<1x1x28672xbf16>
    %9 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%8, %7 : tensor<1x1x28672xbf16>, tensor<1x1x28672xbf16>) outs(%5 : tensor<1x1x28672xbf16>) {
    ^bb0(%in: bf16, %in_4: bf16, %out: bf16):
      %19 = arith.mulf %in, %in_4 : bf16
      linalg.yield %19 : bf16
    } -> tensor<1x1x28672xbf16>
    %transposed_1 = linalg.transpose ins(%arg2 : tensor<28672x8192xbf16>) outs(%0 : tensor<8192x28672xbf16>) permutation = [1, 0] 
    %10 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_1 : tensor<8192x28672xbf16>) outs(%3 : tensor<1x8192x28672xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x8192x28672xbf16>
    %11 = linalg.batch_matmul ins(%2, %10 : tensor<1x1x8192xbf16>, tensor<1x8192x28672xbf16>) outs(%6 : tensor<1x1x28672xbf16>) -> tensor<1x1x28672xbf16>
    %12 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%9, %11 : tensor<1x1x28672xbf16>, tensor<1x1x28672xbf16>) outs(%5 : tensor<1x1x28672xbf16>) {
    ^bb0(%in: bf16, %in_4: bf16, %out: bf16):
      %19 = arith.mulf %in, %in_4 : bf16
      linalg.yield %19 : bf16
    } -> tensor<1x1x28672xbf16>
    %13 = tensor.empty() : tensor<28672x8192xbf16>
    %transposed_2 = linalg.transpose ins(%arg3 : tensor<8192x28672xbf16>) outs(%13 : tensor<28672x8192xbf16>) permutation = [1, 0] 
    %collapsed_3 = tensor.collapse_shape %12 [[0, 1, 2]] : tensor<1x1x28672xbf16> into tensor<28672xbf16>
    %14 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapsed_3 : tensor<28672xbf16>) outs(%5 : tensor<1x1x28672xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x1x28672xbf16>
    %15 = tensor.empty() : tensor<1x28672x8192xbf16>
    %16 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_2 : tensor<28672x8192xbf16>) outs(%15 : tensor<1x28672x8192xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x28672x8192xbf16>
    %17 = linalg.fill ins(%cst : bf16) outs(%1 : tensor<1x1x8192xbf16>) -> tensor<1x1x8192xbf16>
    %18 = linalg.batch_matmul ins(%14, %16 : tensor<1x1x28672xbf16>, tensor<1x28672x8192xbf16>) outs(%17 : tensor<1x1x8192xbf16>) -> tensor<1x1x8192xbf16>
    return %18 : tensor<1x1x8192xbf16>
  }
}
