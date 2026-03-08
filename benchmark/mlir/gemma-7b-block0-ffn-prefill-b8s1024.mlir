#map = affine_map<(d0, d1, d2) -> (d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  func.func @main(%arg0: tensor<8x1024x3072xbf16>, %arg1: tensor<24576x3072xbf16>, %arg2: tensor<24576x3072xbf16>, %arg3: tensor<3072x24576xbf16>) -> tensor<8x1024x3072xbf16> {
    %cst = arith.constant 0.000000e+00 : bf16
    %cst_0 = arith.constant 1.000000e+00 : bf16
    %cst_1 = arith.constant 2.000000e+00 : bf16
    %cst_2 = arith.constant 5.000000e-01 : bf16
    %0 = tensor.empty() : tensor<3072x24576xbf16>
    %transposed = linalg.transpose ins(%arg1 : tensor<24576x3072xbf16>) outs(%0 : tensor<3072x24576xbf16>) permutation = [1, 0] 
    %1 = tensor.empty() : tensor<8x3072x24576xbf16>
    %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed : tensor<3072x24576xbf16>) outs(%1 : tensor<8x3072x24576xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x3072x24576xbf16>
    %3 = tensor.empty() : tensor<8x1024x24576xbf16>
    %4 = linalg.fill ins(%cst : bf16) outs(%3 : tensor<8x1024x24576xbf16>) -> tensor<8x1024x24576xbf16>
    %5 = linalg.batch_matmul ins(%arg0, %2 : tensor<8x1024x3072xbf16>, tensor<8x3072x24576xbf16>) outs(%4 : tensor<8x1024x24576xbf16>) -> tensor<8x1024x24576xbf16>
    %6 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5 : tensor<8x1024x24576xbf16>) outs(%3 : tensor<8x1024x24576xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %16 = math.sqrt %cst_1 : bf16
      %17 = arith.divf %in, %16 : bf16
      %18 = math.erf %17 : bf16
      %19 = arith.addf %18, %cst_0 : bf16
      %20 = arith.mulf %19, %cst_2 : bf16
      %21 = arith.mulf %in, %20 : bf16
      linalg.yield %21 : bf16
    } -> tensor<8x1024x24576xbf16>
    %transposed_3 = linalg.transpose ins(%arg2 : tensor<24576x3072xbf16>) outs(%0 : tensor<3072x24576xbf16>) permutation = [1, 0] 
    %7 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_3 : tensor<3072x24576xbf16>) outs(%1 : tensor<8x3072x24576xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x3072x24576xbf16>
    %8 = linalg.batch_matmul ins(%arg0, %7 : tensor<8x1024x3072xbf16>, tensor<8x3072x24576xbf16>) outs(%4 : tensor<8x1024x24576xbf16>) -> tensor<8x1024x24576xbf16>
    %9 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%6, %8 : tensor<8x1024x24576xbf16>, tensor<8x1024x24576xbf16>) outs(%3 : tensor<8x1024x24576xbf16>) {
    ^bb0(%in: bf16, %in_5: bf16, %out: bf16):
      %16 = arith.mulf %in, %in_5 : bf16
      linalg.yield %16 : bf16
    } -> tensor<8x1024x24576xbf16>
    %10 = tensor.empty() : tensor<24576x3072xbf16>
    %transposed_4 = linalg.transpose ins(%arg3 : tensor<3072x24576xbf16>) outs(%10 : tensor<24576x3072xbf16>) permutation = [1, 0] 
    %11 = tensor.empty() : tensor<8x24576x3072xbf16>
    %12 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_4 : tensor<24576x3072xbf16>) outs(%11 : tensor<8x24576x3072xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x24576x3072xbf16>
    %13 = tensor.empty() : tensor<8x1024x3072xbf16>
    %14 = linalg.fill ins(%cst : bf16) outs(%13 : tensor<8x1024x3072xbf16>) -> tensor<8x1024x3072xbf16>
    %15 = linalg.batch_matmul ins(%9, %12 : tensor<8x1024x24576xbf16>, tensor<8x24576x3072xbf16>) outs(%14 : tensor<8x1024x3072xbf16>) -> tensor<8x1024x3072xbf16>
    return %15 : tensor<8x1024x3072xbf16>
  }
}
