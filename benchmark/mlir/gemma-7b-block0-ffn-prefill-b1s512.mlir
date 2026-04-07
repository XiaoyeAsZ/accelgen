#map = affine_map<(d0, d1, d2) -> (d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  func.func @main(%arg0: tensor<1x512x3072xbf16>, %arg1: tensor<24576x3072xbf16>, %arg2: tensor<24576x3072xbf16>, %arg3: tensor<3072x24576xbf16>) -> tensor<1x512x3072xbf16> {
    %cst = arith.constant 0.000000e+00 : bf16
    %cst_0 = arith.constant 1.000000e+00 : bf16
    %cst_1 = arith.constant 2.000000e+00 : bf16
    %cst_2 = arith.constant 5.000000e-01 : bf16
    %0 = tensor.empty() : tensor<3072x24576xbf16>
    %transposed = linalg.transpose ins(%arg1 : tensor<24576x3072xbf16>) outs(%0 : tensor<3072x24576xbf16>) permutation = [1, 0] 
    %1 = tensor.empty() : tensor<1x512x3072xbf16>
    %collapsed = tensor.collapse_shape %arg0 [[0, 1], [2]] : tensor<1x512x3072xbf16> into tensor<512x3072xbf16>
    %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapsed : tensor<512x3072xbf16>) outs(%1 : tensor<1x512x3072xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x512x3072xbf16>
    %3 = tensor.empty() : tensor<1x3072x24576xbf16>
    %4 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed : tensor<3072x24576xbf16>) outs(%3 : tensor<1x3072x24576xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x3072x24576xbf16>
    %5 = tensor.empty() : tensor<1x512x24576xbf16>
    %6 = linalg.fill ins(%cst : bf16) outs(%5 : tensor<1x512x24576xbf16>) -> tensor<1x512x24576xbf16>
    %7 = linalg.batch_matmul ins(%2, %4 : tensor<1x512x3072xbf16>, tensor<1x3072x24576xbf16>) outs(%6 : tensor<1x512x24576xbf16>) -> tensor<1x512x24576xbf16>
    %8 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%7 : tensor<1x512x24576xbf16>) outs(%5 : tensor<1x512x24576xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %18 = math.sqrt %cst_1 : bf16
      %19 = arith.divf %in, %18 : bf16
      %20 = math.erf %19 : bf16
      %21 = arith.addf %20, %cst_0 : bf16
      %22 = arith.mulf %21, %cst_2 : bf16
      %23 = arith.mulf %in, %22 : bf16
      linalg.yield %23 : bf16
    } -> tensor<1x512x24576xbf16>
    %transposed_3 = linalg.transpose ins(%arg2 : tensor<24576x3072xbf16>) outs(%0 : tensor<3072x24576xbf16>) permutation = [1, 0] 
    %9 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_3 : tensor<3072x24576xbf16>) outs(%3 : tensor<1x3072x24576xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x3072x24576xbf16>
    %10 = linalg.batch_matmul ins(%2, %9 : tensor<1x512x3072xbf16>, tensor<1x3072x24576xbf16>) outs(%6 : tensor<1x512x24576xbf16>) -> tensor<1x512x24576xbf16>
    %11 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%8, %10 : tensor<1x512x24576xbf16>, tensor<1x512x24576xbf16>) outs(%5 : tensor<1x512x24576xbf16>) {
    ^bb0(%in: bf16, %in_6: bf16, %out: bf16):
      %18 = arith.mulf %in, %in_6 : bf16
      linalg.yield %18 : bf16
    } -> tensor<1x512x24576xbf16>
    %12 = tensor.empty() : tensor<24576x3072xbf16>
    %transposed_4 = linalg.transpose ins(%arg3 : tensor<3072x24576xbf16>) outs(%12 : tensor<24576x3072xbf16>) permutation = [1, 0] 
    %collapsed_5 = tensor.collapse_shape %11 [[0, 1], [2]] : tensor<1x512x24576xbf16> into tensor<512x24576xbf16>
    %13 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapsed_5 : tensor<512x24576xbf16>) outs(%5 : tensor<1x512x24576xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x512x24576xbf16>
    %14 = tensor.empty() : tensor<1x24576x3072xbf16>
    %15 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_4 : tensor<24576x3072xbf16>) outs(%14 : tensor<1x24576x3072xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x24576x3072xbf16>
    %16 = linalg.fill ins(%cst : bf16) outs(%1 : tensor<1x512x3072xbf16>) -> tensor<1x512x3072xbf16>
    %17 = linalg.batch_matmul ins(%13, %15 : tensor<1x512x24576xbf16>, tensor<1x24576x3072xbf16>) outs(%16 : tensor<1x512x3072xbf16>) -> tensor<1x512x3072xbf16>
    return %17 : tensor<1x512x3072xbf16>
  }
}
