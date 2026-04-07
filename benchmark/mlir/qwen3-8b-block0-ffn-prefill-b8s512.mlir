#map = affine_map<(d0, d1, d2) -> (d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  func.func @main(%arg0: tensor<8x512x4096xbf16>, %arg1: tensor<12288x4096xbf16>, %arg2: tensor<12288x4096xbf16>, %arg3: tensor<4096x12288xbf16>) -> tensor<8x512x4096xbf16> {
    %cst = arith.constant 0.000000e+00 : bf16
    %cst_0 = arith.constant 1.000000e+00 : bf16
    %0 = tensor.empty() : tensor<4096x12288xbf16>
    %transposed = linalg.transpose ins(%arg1 : tensor<12288x4096xbf16>) outs(%0 : tensor<4096x12288xbf16>) permutation = [1, 0] 
    %1 = tensor.empty() : tensor<8x4096x12288xbf16>
    %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed : tensor<4096x12288xbf16>) outs(%1 : tensor<8x4096x12288xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x4096x12288xbf16>
    %3 = tensor.empty() : tensor<8x512x12288xbf16>
    %4 = linalg.fill ins(%cst : bf16) outs(%3 : tensor<8x512x12288xbf16>) -> tensor<8x512x12288xbf16>
    %5 = linalg.batch_matmul ins(%arg0, %2 : tensor<8x512x4096xbf16>, tensor<8x4096x12288xbf16>) outs(%4 : tensor<8x512x12288xbf16>) -> tensor<8x512x12288xbf16>
    %6 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5 : tensor<8x512x12288xbf16>) outs(%3 : tensor<8x512x12288xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %17 = arith.negf %in : bf16
      %18 = math.exp %17 : bf16
      %19 = arith.addf %18, %cst_0 : bf16
      %20 = arith.divf %cst_0, %19 : bf16
      linalg.yield %20 : bf16
    } -> tensor<8x512x12288xbf16>
    %7 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%6, %5 : tensor<8x512x12288xbf16>, tensor<8x512x12288xbf16>) outs(%3 : tensor<8x512x12288xbf16>) {
    ^bb0(%in: bf16, %in_3: bf16, %out: bf16):
      %17 = arith.mulf %in, %in_3 : bf16
      linalg.yield %17 : bf16
    } -> tensor<8x512x12288xbf16>
    %transposed_1 = linalg.transpose ins(%arg2 : tensor<12288x4096xbf16>) outs(%0 : tensor<4096x12288xbf16>) permutation = [1, 0] 
    %8 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_1 : tensor<4096x12288xbf16>) outs(%1 : tensor<8x4096x12288xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x4096x12288xbf16>
    %9 = linalg.batch_matmul ins(%arg0, %8 : tensor<8x512x4096xbf16>, tensor<8x4096x12288xbf16>) outs(%4 : tensor<8x512x12288xbf16>) -> tensor<8x512x12288xbf16>
    %10 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%7, %9 : tensor<8x512x12288xbf16>, tensor<8x512x12288xbf16>) outs(%3 : tensor<8x512x12288xbf16>) {
    ^bb0(%in: bf16, %in_3: bf16, %out: bf16):
      %17 = arith.mulf %in, %in_3 : bf16
      linalg.yield %17 : bf16
    } -> tensor<8x512x12288xbf16>
    %11 = tensor.empty() : tensor<12288x4096xbf16>
    %transposed_2 = linalg.transpose ins(%arg3 : tensor<4096x12288xbf16>) outs(%11 : tensor<12288x4096xbf16>) permutation = [1, 0] 
    %12 = tensor.empty() : tensor<8x12288x4096xbf16>
    %13 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_2 : tensor<12288x4096xbf16>) outs(%12 : tensor<8x12288x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x12288x4096xbf16>
    %14 = tensor.empty() : tensor<8x512x4096xbf16>
    %15 = linalg.fill ins(%cst : bf16) outs(%14 : tensor<8x512x4096xbf16>) -> tensor<8x512x4096xbf16>
    %16 = linalg.batch_matmul ins(%10, %13 : tensor<8x512x12288xbf16>, tensor<8x12288x4096xbf16>) outs(%15 : tensor<8x512x4096xbf16>) -> tensor<8x512x4096xbf16>
    return %16 : tensor<8x512x4096xbf16>
  }
}
