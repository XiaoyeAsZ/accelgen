#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  func.func @main(%arg0: tensor<1x16x768xbf16>, %arg1: tensor<768x3072xf32>, %arg2: tensor<3072xf32>, %arg3: tensor<3072x768xf32>, %arg4: tensor<768xf32>) -> tensor<1x16x768xf32> {
    %c3_i64 = arith.constant 3 : i64
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 0.79788456080286541 : f64
    %cst_1 = arith.constant 4.471500e-02 : f64
    %cst_2 = arith.constant 5.000000e-01 : f32
    %cst_3 = arith.constant 1.000000e+00 : f32
    %collapsed = tensor.collapse_shape %arg0 [[0, 1], [2]] : tensor<1x16x768xbf16> into tensor<16x768xbf16>
    %0 = tensor.empty() : tensor<16x768xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%collapsed : tensor<16x768xbf16>) outs(%0 : tensor<16x768xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %18 = arith.extf %in : bf16 to f32
      linalg.yield %18 : f32
    } -> tensor<16x768xf32>
    %2 = tensor.empty() : tensor<16x3072xf32>
    %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<16x3072xf32>) -> tensor<16x3072xf32>
    %4 = linalg.matmul ins(%1, %arg1 : tensor<16x768xf32>, tensor<768x3072xf32>) outs(%3 : tensor<16x3072xf32>) -> tensor<16x3072xf32>
    %5 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%4, %arg2 : tensor<16x3072xf32>, tensor<3072xf32>) outs(%2 : tensor<16x3072xf32>) {
    ^bb0(%in: f32, %in_6: f32, %out: f32):
      %18 = arith.addf %in, %in_6 : f32
      linalg.yield %18 : f32
    } -> tensor<16x3072xf32>
    %expanded = tensor.expand_shape %5 [[0, 1], [2]] output_shape [1, 16, 3072] : tensor<16x3072xf32> into tensor<1x16x3072xf32>
    %6 = tensor.empty() : tensor<1x16x3072xf32>
    %7 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded : tensor<1x16x3072xf32>) outs(%6 : tensor<1x16x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %18 = arith.mulf %in, %cst_2 : f32
      linalg.yield %18 : f32
    } -> tensor<1x16x3072xf32>
    %8 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded : tensor<1x16x3072xf32>) outs(%6 : tensor<1x16x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %18 = math.fpowi %in, %c3_i64 : f32, i64
      linalg.yield %18 : f32
    } -> tensor<1x16x3072xf32>
    %9 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%8 : tensor<1x16x3072xf32>) outs(%6 : tensor<1x16x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %18 = arith.truncf %cst_1 : f64 to f32
      %19 = arith.mulf %in, %18 : f32
      linalg.yield %19 : f32
    } -> tensor<1x16x3072xf32>
    %10 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded, %9 : tensor<1x16x3072xf32>, tensor<1x16x3072xf32>) outs(%6 : tensor<1x16x3072xf32>) {
    ^bb0(%in: f32, %in_6: f32, %out: f32):
      %18 = arith.addf %in, %in_6 : f32
      linalg.yield %18 : f32
    } -> tensor<1x16x3072xf32>
    %11 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%10 : tensor<1x16x3072xf32>) outs(%6 : tensor<1x16x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %18 = arith.truncf %cst_0 : f64 to f32
      %19 = arith.mulf %in, %18 : f32
      linalg.yield %19 : f32
    } -> tensor<1x16x3072xf32>
    %12 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%11 : tensor<1x16x3072xf32>) outs(%6 : tensor<1x16x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %18 = math.tanh %in : f32
      linalg.yield %18 : f32
    } -> tensor<1x16x3072xf32>
    %13 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%12 : tensor<1x16x3072xf32>) outs(%6 : tensor<1x16x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %18 = arith.addf %in, %cst_3 : f32
      linalg.yield %18 : f32
    } -> tensor<1x16x3072xf32>
    %14 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%7, %13 : tensor<1x16x3072xf32>, tensor<1x16x3072xf32>) outs(%6 : tensor<1x16x3072xf32>) {
    ^bb0(%in: f32, %in_6: f32, %out: f32):
      %18 = arith.mulf %in, %in_6 : f32
      linalg.yield %18 : f32
    } -> tensor<1x16x3072xf32>
    %collapsed_4 = tensor.collapse_shape %14 [[0, 1], [2]] : tensor<1x16x3072xf32> into tensor<16x3072xf32>
    %15 = linalg.fill ins(%cst : f32) outs(%0 : tensor<16x768xf32>) -> tensor<16x768xf32>
    %16 = linalg.matmul ins(%collapsed_4, %arg3 : tensor<16x3072xf32>, tensor<3072x768xf32>) outs(%15 : tensor<16x768xf32>) -> tensor<16x768xf32>
    %17 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%16, %arg4 : tensor<16x768xf32>, tensor<768xf32>) outs(%0 : tensor<16x768xf32>) {
    ^bb0(%in: f32, %in_6: f32, %out: f32):
      %18 = arith.addf %in, %in_6 : f32
      linalg.yield %18 : f32
    } -> tensor<16x768xf32>
    %expanded_5 = tensor.expand_shape %17 [[0, 1], [2]] output_shape [1, 16, 768] : tensor<16x768xf32> into tensor<1x16x768xf32>
    return %expanded_5 : tensor<1x16x768xf32>
  }
}
