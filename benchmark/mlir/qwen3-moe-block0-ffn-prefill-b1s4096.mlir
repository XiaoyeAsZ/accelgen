#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1, 0)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>
module {
  func.func @main(%arg0: tensor<1x4096x4096xbf16>, %arg1: tensor<16x4096x8xbf16>, %arg2: tensor<16x4096x24576xbf16>, %arg3: tensor<16x12288x4096xbf16>, %arg4: tensor<16x16xbf16>) -> (tensor<1x4096x4096xbf16>, tensor<16x256x8xbf16>) {
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 0xFF800000 : f32
    %cst_1 = arith.constant 1.000000e+00 : bf16
    %collapsed = tensor.collapse_shape %arg0 [[0, 1], [2]] : tensor<1x4096x4096xbf16> into tensor<4096x4096xbf16>
    %expanded = tensor.expand_shape %collapsed [[0, 1], [2]] output_shape [16, 256, 4096] : tensor<4096x4096xbf16> into tensor<16x256x4096xbf16>
    %0 = tensor.empty() : tensor<16x256x8xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<16x256x8xf32>) -> tensor<16x256x8xf32>
    %2 = linalg.batch_matmul ins(%expanded, %arg1 : tensor<16x256x4096xbf16>, tensor<16x4096x8xbf16>) outs(%1 : tensor<16x256x8xf32>) -> tensor<16x256x8xf32>
    %3 = tensor.empty() : tensor<16x256x8xbf16>
    %4 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<16x256x8xf32>) outs(%3 : tensor<16x256x8xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %33 = arith.truncf %in : f32 to bf16
      linalg.yield %33 : bf16
    } -> tensor<16x256x8xbf16>
    %5 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4 : tensor<16x256x8xbf16>) outs(%0 : tensor<16x256x8xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %33 = arith.extf %in : bf16 to f32
      linalg.yield %33 : f32
    } -> tensor<16x256x8xf32>
    %6 = tensor.empty() : tensor<16x256xi64>
    %7 = linalg.fill ins(%c0_i64 : i64) outs(%6 : tensor<16x256xi64>) -> tensor<16x256xi64>
    %8 = tensor.empty() : tensor<16x256xf32>
    %9 = linalg.fill ins(%cst_0 : f32) outs(%8 : tensor<16x256xf32>) -> tensor<16x256xf32>
    %10:2 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%5 : tensor<16x256x8xf32>) outs(%9, %7 : tensor<16x256xf32>, tensor<16x256xi64>) {
    ^bb0(%in: f32, %out: f32, %out_9: i64):
      %33 = linalg.index 2 : index
      %34 = arith.index_cast %33 : index to i64
      %35 = arith.maximumf %in, %out : f32
      %36 = arith.cmpf ogt, %in, %out : f32
      %37 = arith.select %36, %34, %out_9 : i64
      linalg.yield %35, %37 : f32, i64
    } -> (tensor<16x256xf32>, tensor<16x256xi64>)
    %expanded_2 = tensor.expand_shape %10#0 [[0], [1, 2]] output_shape [16, 256, 1] : tensor<16x256xf32> into tensor<16x256x1xf32>
    %11 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5, %expanded_2 : tensor<16x256x8xf32>, tensor<16x256x1xf32>) outs(%0 : tensor<16x256x8xf32>) {
    ^bb0(%in: f32, %in_9: f32, %out: f32):
      %33 = arith.subf %in, %in_9 : f32
      linalg.yield %33 : f32
    } -> tensor<16x256x8xf32>
    %12 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%11 : tensor<16x256x8xf32>) outs(%0 : tensor<16x256x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %33 = math.exp %in : f32
      linalg.yield %33 : f32
    } -> tensor<16x256x8xf32>
    %13 = tensor.empty() : tensor<16x256x1xf32>
    %14 = linalg.fill ins(%cst : f32) outs(%13 : tensor<16x256x1xf32>) -> tensor<16x256x1xf32>
    %15 = linalg.generic {indexing_maps = [#map, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%12 : tensor<16x256x8xf32>) outs(%14 : tensor<16x256x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %33 = arith.addf %in, %out : f32
      linalg.yield %33 : f32
    } -> tensor<16x256x1xf32>
    %16 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%12, %15 : tensor<16x256x8xf32>, tensor<16x256x1xf32>) outs(%0 : tensor<16x256x8xf32>) {
    ^bb0(%in: f32, %in_9: f32, %out: f32):
      %33 = arith.divf %in, %in_9 : f32
      linalg.yield %33 : f32
    } -> tensor<16x256x8xf32>
    %17 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%16 : tensor<16x256x8xf32>) outs(%3 : tensor<16x256x8xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %33 = arith.truncf %in : f32 to bf16
      linalg.yield %33 : bf16
    } -> tensor<16x256x8xbf16>
    %18 = tensor.empty() : tensor<16x256x24576xf32>
    %19 = linalg.fill ins(%cst : f32) outs(%18 : tensor<16x256x24576xf32>) -> tensor<16x256x24576xf32>
    %20 = linalg.batch_matmul ins(%expanded, %arg2 : tensor<16x256x4096xbf16>, tensor<16x4096x24576xbf16>) outs(%19 : tensor<16x256x24576xf32>) -> tensor<16x256x24576xf32>
    %21 = tensor.empty() : tensor<16x256x24576xbf16>
    %22 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%20 : tensor<16x256x24576xf32>) outs(%21 : tensor<16x256x24576xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %33 = arith.truncf %in : f32 to bf16
      linalg.yield %33 : bf16
    } -> tensor<16x256x24576xbf16>
    %expanded_3 = tensor.expand_shape %22 [[0], [1], [2, 3]] output_shape [16, 256, 8, 3072] : tensor<16x256x24576xbf16> into tensor<16x256x8x3072xbf16>
    %extracted_slice = tensor.extract_slice %expanded_3[0, 0, 0, 0] [16, 256, 8, 1536] [1, 1, 1, 1] : tensor<16x256x8x3072xbf16> to tensor<16x256x8x1536xbf16>
    %extracted_slice_4 = tensor.extract_slice %expanded_3[0, 0, 0, 1536] [16, 256, 8, 1536] [1, 1, 1, 1] : tensor<16x256x8x3072xbf16> to tensor<16x256x8x1536xbf16>
    %23 = tensor.empty() : tensor<16x256x8x1536xbf16>
    %24 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice : tensor<16x256x8x1536xbf16>) outs(%23 : tensor<16x256x8x1536xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %33 = arith.negf %in : bf16
      %34 = math.exp %33 : bf16
      %35 = arith.addf %34, %cst_1 : bf16
      %36 = arith.divf %cst_1, %35 : bf16
      linalg.yield %36 : bf16
    } -> tensor<16x256x8x1536xbf16>
    %25 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%24, %extracted_slice : tensor<16x256x8x1536xbf16>, tensor<16x256x8x1536xbf16>) outs(%23 : tensor<16x256x8x1536xbf16>) {
    ^bb0(%in: bf16, %in_9: bf16, %out: bf16):
      %33 = arith.mulf %in, %in_9 : bf16
      linalg.yield %33 : bf16
    } -> tensor<16x256x8x1536xbf16>
    %26 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%25, %extracted_slice_4 : tensor<16x256x8x1536xbf16>, tensor<16x256x8x1536xbf16>) outs(%23 : tensor<16x256x8x1536xbf16>) {
    ^bb0(%in: bf16, %in_9: bf16, %out: bf16):
      %33 = arith.mulf %in, %in_9 : bf16
      linalg.yield %33 : bf16
    } -> tensor<16x256x8x1536xbf16>
    %expanded_5 = tensor.expand_shape %17 [[0], [1], [2, 3]] output_shape [16, 256, 8, 1] : tensor<16x256x8xbf16> into tensor<16x256x8x1xbf16>
    %27 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%26, %expanded_5 : tensor<16x256x8x1536xbf16>, tensor<16x256x8x1xbf16>) outs(%23 : tensor<16x256x8x1536xbf16>) {
    ^bb0(%in: bf16, %in_9: bf16, %out: bf16):
      %33 = arith.mulf %in, %in_9 : bf16
      linalg.yield %33 : bf16
    } -> tensor<16x256x8x1536xbf16>
    %collapsed_6 = tensor.collapse_shape %27 [[0], [1], [2, 3]] : tensor<16x256x8x1536xbf16> into tensor<16x256x12288xbf16>
    %28 = tensor.empty() : tensor<16x256x4096xf32>
    %29 = linalg.fill ins(%cst : f32) outs(%28 : tensor<16x256x4096xf32>) -> tensor<16x256x4096xf32>
    %30 = linalg.batch_matmul ins(%collapsed_6, %arg3 : tensor<16x256x12288xbf16>, tensor<16x12288x4096xbf16>) outs(%29 : tensor<16x256x4096xf32>) -> tensor<16x256x4096xf32>
    %31 = tensor.empty() : tensor<16x256x4096xbf16>
    %32 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%30 : tensor<16x256x4096xf32>) outs(%31 : tensor<16x256x4096xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %33 = arith.truncf %in : f32 to bf16
      linalg.yield %33 : bf16
    } -> tensor<16x256x4096xbf16>
    %collapsed_7 = tensor.collapse_shape %32 [[0, 1], [2]] : tensor<16x256x4096xbf16> into tensor<4096x4096xbf16>
    %expanded_8 = tensor.expand_shape %collapsed_7 [[0, 1], [2]] output_shape [1, 4096, 4096] : tensor<4096x4096xbf16> into tensor<1x4096x4096xbf16>
    return %expanded_8, %4 : tensor<1x4096x4096xbf16>, tensor<16x256x8xbf16>
  }
}
