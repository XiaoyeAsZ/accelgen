#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map6 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map7 = affine_map<(d0, d1, d2) -> (d0, d1, 0)>
module {
  func.func @main(%arg0: tensor<8x1x4096xbf16>, %arg1: tensor<8x4096x24576xbf16>, %arg2: tensor<8x12288x4096xbf16>, %arg3: tensor<16x16xbf16>, %arg4: tensor<128x4096xbf16>) -> (tensor<8x1x4096xbf16>, tensor<8x1x128xbf16>) {
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant 0.000000e+00 : bf16
    %cst_0 = arith.constant 0xFF800000 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %cst_2 = arith.constant 1.000000e+00 : bf16
    %0 = tensor.empty() : tensor<4096x128xbf16>
    %transposed = linalg.transpose ins(%arg4 : tensor<128x4096xbf16>) outs(%0 : tensor<4096x128xbf16>) permutation = [1, 0] 
    %1 = tensor.empty() : tensor<8x1x4096xbf16>
    %collapsed = tensor.collapse_shape %arg0 [[0, 1], [2]] : tensor<8x1x4096xbf16> into tensor<8x4096xbf16>
    %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapsed : tensor<8x4096xbf16>) outs(%1 : tensor<8x1x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x1x4096xbf16>
    %3 = tensor.empty() : tensor<8x4096x128xbf16>
    %4 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed : tensor<4096x128xbf16>) outs(%3 : tensor<8x4096x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x4096x128xbf16>
    %5 = tensor.empty() : tensor<8x1x128xbf16>
    %6 = linalg.fill ins(%cst : bf16) outs(%5 : tensor<8x1x128xbf16>) -> tensor<8x1x128xbf16>
    %7 = linalg.batch_matmul ins(%2, %4 : tensor<8x1x4096xbf16>, tensor<8x4096x128xbf16>) outs(%6 : tensor<8x1x128xbf16>) -> tensor<8x1x128xbf16>
    %expanded = tensor.expand_shape %7 [[0], [1], [2, 3]] output_shape [8, 1, 16, 8] : tensor<8x1x128xbf16> into tensor<8x1x16x8xbf16>
    %extracted_slice = tensor.extract_slice %arg3[0, 0] [8, 16] [1, 1] : tensor<16x16xbf16> to tensor<8x16xbf16>
    %extracted_slice_3 = tensor.extract_slice %extracted_slice[0, 0] [8, 8] [1, 1] : tensor<8x16xbf16> to tensor<8x8xbf16>
    %expanded_4 = tensor.expand_shape %extracted_slice_3 [[0], [1, 2, 3]] output_shape [8, 1, 8, 1] : tensor<8x8xbf16> into tensor<8x1x8x1xbf16>
    %extracted_slice_5 = tensor.extract_slice %expanded[0, 0, 0, 0] [8, 1, 8, 8] [1, 1, 1, 1] : tensor<8x1x16x8xbf16> to tensor<8x1x8x8xbf16>
    %8 = tensor.empty() : tensor<8x1x8x8xbf16>
    %9 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_5, %expanded_4 : tensor<8x1x8x8xbf16>, tensor<8x1x8x1xbf16>) outs(%8 : tensor<8x1x8x8xbf16>) {
    ^bb0(%in: bf16, %in_12: bf16, %out: bf16):
      %41 = arith.mulf %in, %in_12 : bf16
      linalg.yield %41 : bf16
    } -> tensor<8x1x8x8xbf16>
    %10 = tensor.empty() : tensor<8x1x8xbf16>
    %11 = linalg.fill ins(%cst : bf16) outs(%10 : tensor<8x1x8xbf16>) -> tensor<8x1x8xbf16>
    %12 = linalg.generic {indexing_maps = [#map3, #map5], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%9 : tensor<8x1x8x8xbf16>) outs(%11 : tensor<8x1x8xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %41 = arith.addf %in, %out : bf16
      linalg.yield %41 : bf16
    } -> tensor<8x1x8xbf16>
    %13 = tensor.empty() : tensor<8x1x8xf32>
    %14 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%12 : tensor<8x1x8xbf16>) outs(%13 : tensor<8x1x8xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %41 = arith.extf %in : bf16 to f32
      linalg.yield %41 : f32
    } -> tensor<8x1x8xf32>
    %15 = tensor.empty() : tensor<8x1xi64>
    %16 = linalg.fill ins(%c0_i64 : i64) outs(%15 : tensor<8x1xi64>) -> tensor<8x1xi64>
    %17 = tensor.empty() : tensor<8x1xf32>
    %18 = linalg.fill ins(%cst_0 : f32) outs(%17 : tensor<8x1xf32>) -> tensor<8x1xf32>
    %19:2 = linalg.generic {indexing_maps = [#map1, #map6, #map6], iterator_types = ["parallel", "parallel", "reduction"]} ins(%14 : tensor<8x1x8xf32>) outs(%18, %16 : tensor<8x1xf32>, tensor<8x1xi64>) {
    ^bb0(%in: f32, %out: f32, %out_12: i64):
      %41 = linalg.index 2 : index
      %42 = arith.index_cast %41 : index to i64
      %43 = arith.maximumf %in, %out : f32
      %44 = arith.cmpf ogt, %in, %out : f32
      %45 = arith.select %44, %42, %out_12 : i64
      linalg.yield %43, %45 : f32, i64
    } -> (tensor<8x1xf32>, tensor<8x1xi64>)
    %expanded_6 = tensor.expand_shape %19#0 [[0], [1, 2]] output_shape [8, 1, 1] : tensor<8x1xf32> into tensor<8x1x1xf32>
    %20 = linalg.generic {indexing_maps = [#map1, #map7, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%14, %expanded_6 : tensor<8x1x8xf32>, tensor<8x1x1xf32>) outs(%13 : tensor<8x1x8xf32>) {
    ^bb0(%in: f32, %in_12: f32, %out: f32):
      %41 = arith.subf %in, %in_12 : f32
      linalg.yield %41 : f32
    } -> tensor<8x1x8xf32>
    %21 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%20 : tensor<8x1x8xf32>) outs(%13 : tensor<8x1x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %41 = math.exp %in : f32
      linalg.yield %41 : f32
    } -> tensor<8x1x8xf32>
    %22 = tensor.empty() : tensor<8x1x1xf32>
    %23 = linalg.fill ins(%cst_1 : f32) outs(%22 : tensor<8x1x1xf32>) -> tensor<8x1x1xf32>
    %24 = linalg.generic {indexing_maps = [#map1, #map7], iterator_types = ["parallel", "parallel", "reduction"]} ins(%21 : tensor<8x1x8xf32>) outs(%23 : tensor<8x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %41 = arith.addf %in, %out : f32
      linalg.yield %41 : f32
    } -> tensor<8x1x1xf32>
    %25 = linalg.generic {indexing_maps = [#map1, #map7, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%21, %24 : tensor<8x1x8xf32>, tensor<8x1x1xf32>) outs(%13 : tensor<8x1x8xf32>) {
    ^bb0(%in: f32, %in_12: f32, %out: f32):
      %41 = arith.divf %in, %in_12 : f32
      linalg.yield %41 : f32
    } -> tensor<8x1x8xf32>
    %26 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%25 : tensor<8x1x8xf32>) outs(%10 : tensor<8x1x8xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %41 = arith.truncf %in : f32 to bf16
      linalg.yield %41 : bf16
    } -> tensor<8x1x8xbf16>
    %27 = tensor.empty() : tensor<8x1x24576xf32>
    %28 = linalg.fill ins(%cst_1 : f32) outs(%27 : tensor<8x1x24576xf32>) -> tensor<8x1x24576xf32>
    %29 = linalg.batch_matmul ins(%arg0, %arg1 : tensor<8x1x4096xbf16>, tensor<8x4096x24576xbf16>) outs(%28 : tensor<8x1x24576xf32>) -> tensor<8x1x24576xf32>
    %30 = tensor.empty() : tensor<8x1x24576xbf16>
    %31 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%29 : tensor<8x1x24576xf32>) outs(%30 : tensor<8x1x24576xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %41 = arith.truncf %in : f32 to bf16
      linalg.yield %41 : bf16
    } -> tensor<8x1x24576xbf16>
    %expanded_7 = tensor.expand_shape %31 [[0], [1], [2, 3]] output_shape [8, 1, 8, 3072] : tensor<8x1x24576xbf16> into tensor<8x1x8x3072xbf16>
    %extracted_slice_8 = tensor.extract_slice %expanded_7[0, 0, 0, 0] [8, 1, 8, 1536] [1, 1, 1, 1] : tensor<8x1x8x3072xbf16> to tensor<8x1x8x1536xbf16>
    %extracted_slice_9 = tensor.extract_slice %expanded_7[0, 0, 0, 1536] [8, 1, 8, 1536] [1, 1, 1, 1] : tensor<8x1x8x3072xbf16> to tensor<8x1x8x1536xbf16>
    %32 = tensor.empty() : tensor<8x1x8x1536xbf16>
    %33 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_8 : tensor<8x1x8x1536xbf16>) outs(%32 : tensor<8x1x8x1536xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %41 = arith.negf %in : bf16
      %42 = math.exp %41 : bf16
      %43 = arith.addf %42, %cst_2 : bf16
      %44 = arith.divf %cst_2, %43 : bf16
      linalg.yield %44 : bf16
    } -> tensor<8x1x8x1536xbf16>
    %34 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%33, %extracted_slice_8 : tensor<8x1x8x1536xbf16>, tensor<8x1x8x1536xbf16>) outs(%32 : tensor<8x1x8x1536xbf16>) {
    ^bb0(%in: bf16, %in_12: bf16, %out: bf16):
      %41 = arith.mulf %in, %in_12 : bf16
      linalg.yield %41 : bf16
    } -> tensor<8x1x8x1536xbf16>
    %35 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%34, %extracted_slice_9 : tensor<8x1x8x1536xbf16>, tensor<8x1x8x1536xbf16>) outs(%32 : tensor<8x1x8x1536xbf16>) {
    ^bb0(%in: bf16, %in_12: bf16, %out: bf16):
      %41 = arith.mulf %in, %in_12 : bf16
      linalg.yield %41 : bf16
    } -> tensor<8x1x8x1536xbf16>
    %expanded_10 = tensor.expand_shape %26 [[0], [1], [2, 3]] output_shape [8, 1, 8, 1] : tensor<8x1x8xbf16> into tensor<8x1x8x1xbf16>
    %36 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%35, %expanded_10 : tensor<8x1x8x1536xbf16>, tensor<8x1x8x1xbf16>) outs(%32 : tensor<8x1x8x1536xbf16>) {
    ^bb0(%in: bf16, %in_12: bf16, %out: bf16):
      %41 = arith.mulf %in, %in_12 : bf16
      linalg.yield %41 : bf16
    } -> tensor<8x1x8x1536xbf16>
    %collapsed_11 = tensor.collapse_shape %36 [[0], [1], [2, 3]] : tensor<8x1x8x1536xbf16> into tensor<8x1x12288xbf16>
    %37 = tensor.empty() : tensor<8x1x4096xf32>
    %38 = linalg.fill ins(%cst_1 : f32) outs(%37 : tensor<8x1x4096xf32>) -> tensor<8x1x4096xf32>
    %39 = linalg.batch_matmul ins(%collapsed_11, %arg2 : tensor<8x1x12288xbf16>, tensor<8x12288x4096xbf16>) outs(%38 : tensor<8x1x4096xf32>) -> tensor<8x1x4096xf32>
    %40 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%39 : tensor<8x1x4096xf32>) outs(%1 : tensor<8x1x4096xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %41 = arith.truncf %in : f32 to bf16
      linalg.yield %41 : bf16
    } -> tensor<8x1x4096xbf16>
    return %40, %7 : tensor<8x1x4096xbf16>, tensor<8x1x128xbf16>
  }
}
