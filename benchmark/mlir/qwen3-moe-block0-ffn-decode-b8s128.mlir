#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map5 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map6 = affine_map<(d0, d1, d2) -> (d0, d1, 0)>
module {
  func.func @main(%arg0: tensor<8x1x4096xbf16>, %arg1: tensor<16x4096x24576xbf16>, %arg2: tensor<16x12288x4096xbf16>, %arg3: tensor<16x16xbf16>, %arg4: tensor<128x4096xbf16>) -> (tensor<8x1x4096xbf16>, tensor<8x128xbf16>) {
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : bf16
    %cst_1 = arith.constant 0xFF800000 : f32
    %cst_2 = arith.constant 1.000000e+00 : bf16
    %collapsed = tensor.collapse_shape %arg0 [[0], [1, 2]] : tensor<8x1x4096xbf16> into tensor<8x4096xbf16>
    %0 = tensor.empty() : tensor<4096x128xbf16>
    %transposed = linalg.transpose ins(%arg4 : tensor<128x4096xbf16>) outs(%0 : tensor<4096x128xbf16>) permutation = [1, 0] 
    %1 = tensor.empty() : tensor<8x128xf32>
    %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<8x128xf32>) -> tensor<8x128xf32>
    %3 = linalg.matmul ins(%collapsed, %transposed : tensor<8x4096xbf16>, tensor<4096x128xbf16>) outs(%2 : tensor<8x128xf32>) -> tensor<8x128xf32>
    %4 = tensor.empty() : tensor<8x128xbf16>
    %5 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%3 : tensor<8x128xf32>) outs(%4 : tensor<8x128xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %40 = arith.truncf %in : f32 to bf16
      linalg.yield %40 : bf16
    } -> tensor<8x128xbf16>
    %expanded = tensor.expand_shape %5 [[0], [1, 2, 3]] output_shape [8, 1, 16, 8] : tensor<8x128xbf16> into tensor<8x1x16x8xbf16>
    %extracted_slice = tensor.extract_slice %arg3[0, 0] [8, 16] [1, 1] : tensor<16x16xbf16> to tensor<8x16xbf16>
    %extracted_slice_3 = tensor.extract_slice %extracted_slice[0, 0] [8, 8] [1, 1] : tensor<8x16xbf16> to tensor<8x8xbf16>
    %expanded_4 = tensor.expand_shape %extracted_slice_3 [[0], [1, 2, 3]] output_shape [8, 1, 8, 1] : tensor<8x8xbf16> into tensor<8x1x8x1xbf16>
    %extracted_slice_5 = tensor.extract_slice %expanded[0, 0, 0, 0] [8, 1, 8, 8] [1, 1, 1, 1] : tensor<8x1x16x8xbf16> to tensor<8x1x8x8xbf16>
    %6 = tensor.empty() : tensor<8x1x8x8xbf16>
    %7 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_5, %expanded_4 : tensor<8x1x8x8xbf16>, tensor<8x1x8x1xbf16>) outs(%6 : tensor<8x1x8x8xbf16>) {
    ^bb0(%in: bf16, %in_14: bf16, %out: bf16):
      %40 = arith.mulf %in, %in_14 : bf16
      linalg.yield %40 : bf16
    } -> tensor<8x1x8x8xbf16>
    %8 = tensor.empty() : tensor<8x1x8xbf16>
    %9 = linalg.fill ins(%cst_0 : bf16) outs(%8 : tensor<8x1x8xbf16>) -> tensor<8x1x8xbf16>
    %10 = linalg.generic {indexing_maps = [#map1, #map3], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%7 : tensor<8x1x8x8xbf16>) outs(%9 : tensor<8x1x8xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %40 = arith.addf %in, %out : bf16
      linalg.yield %40 : bf16
    } -> tensor<8x1x8xbf16>
    %11 = tensor.empty() : tensor<8x1x8xf32>
    %12 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%10 : tensor<8x1x8xbf16>) outs(%11 : tensor<8x1x8xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %40 = arith.extf %in : bf16 to f32
      linalg.yield %40 : f32
    } -> tensor<8x1x8xf32>
    %13 = tensor.empty() : tensor<8x1xi64>
    %14 = linalg.fill ins(%c0_i64 : i64) outs(%13 : tensor<8x1xi64>) -> tensor<8x1xi64>
    %15 = tensor.empty() : tensor<8x1xf32>
    %16 = linalg.fill ins(%cst_1 : f32) outs(%15 : tensor<8x1xf32>) -> tensor<8x1xf32>
    %17:2 = linalg.generic {indexing_maps = [#map4, #map5, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%12 : tensor<8x1x8xf32>) outs(%16, %14 : tensor<8x1xf32>, tensor<8x1xi64>) {
    ^bb0(%in: f32, %out: f32, %out_14: i64):
      %40 = linalg.index 2 : index
      %41 = arith.index_cast %40 : index to i64
      %42 = arith.maximumf %in, %out : f32
      %43 = arith.cmpf ogt, %in, %out : f32
      %44 = arith.select %43, %41, %out_14 : i64
      linalg.yield %42, %44 : f32, i64
    } -> (tensor<8x1xf32>, tensor<8x1xi64>)
    %expanded_6 = tensor.expand_shape %17#0 [[0], [1, 2]] output_shape [8, 1, 1] : tensor<8x1xf32> into tensor<8x1x1xf32>
    %18 = linalg.generic {indexing_maps = [#map4, #map6, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%12, %expanded_6 : tensor<8x1x8xf32>, tensor<8x1x1xf32>) outs(%11 : tensor<8x1x8xf32>) {
    ^bb0(%in: f32, %in_14: f32, %out: f32):
      %40 = arith.subf %in, %in_14 : f32
      linalg.yield %40 : f32
    } -> tensor<8x1x8xf32>
    %19 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%18 : tensor<8x1x8xf32>) outs(%11 : tensor<8x1x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %40 = math.exp %in : f32
      linalg.yield %40 : f32
    } -> tensor<8x1x8xf32>
    %20 = tensor.empty() : tensor<8x1x1xf32>
    %21 = linalg.fill ins(%cst : f32) outs(%20 : tensor<8x1x1xf32>) -> tensor<8x1x1xf32>
    %22 = linalg.generic {indexing_maps = [#map4, #map6], iterator_types = ["parallel", "parallel", "reduction"]} ins(%19 : tensor<8x1x8xf32>) outs(%21 : tensor<8x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %40 = arith.addf %in, %out : f32
      linalg.yield %40 : f32
    } -> tensor<8x1x1xf32>
    %23 = linalg.generic {indexing_maps = [#map4, #map6, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%19, %22 : tensor<8x1x8xf32>, tensor<8x1x1xf32>) outs(%11 : tensor<8x1x8xf32>) {
    ^bb0(%in: f32, %in_14: f32, %out: f32):
      %40 = arith.divf %in, %in_14 : f32
      linalg.yield %40 : f32
    } -> tensor<8x1x8xf32>
    %24 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23 : tensor<8x1x8xf32>) outs(%8 : tensor<8x1x8xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %40 = arith.truncf %in : f32 to bf16
      linalg.yield %40 : bf16
    } -> tensor<8x1x8xbf16>
    %extracted_slice_7 = tensor.extract_slice %arg1[0, 0, 0] [8, 4096, 24576] [1, 1, 1] : tensor<16x4096x24576xbf16> to tensor<8x4096x24576xbf16>
    %25 = tensor.empty() : tensor<8x1x24576xf32>
    %26 = linalg.fill ins(%cst : f32) outs(%25 : tensor<8x1x24576xf32>) -> tensor<8x1x24576xf32>
    %27 = linalg.batch_matmul ins(%arg0, %extracted_slice_7 : tensor<8x1x4096xbf16>, tensor<8x4096x24576xbf16>) outs(%26 : tensor<8x1x24576xf32>) -> tensor<8x1x24576xf32>
    %28 = tensor.empty() : tensor<8x1x24576xbf16>
    %29 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%27 : tensor<8x1x24576xf32>) outs(%28 : tensor<8x1x24576xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %40 = arith.truncf %in : f32 to bf16
      linalg.yield %40 : bf16
    } -> tensor<8x1x24576xbf16>
    %expanded_8 = tensor.expand_shape %29 [[0], [1], [2, 3]] output_shape [8, 1, 8, 3072] : tensor<8x1x24576xbf16> into tensor<8x1x8x3072xbf16>
    %extracted_slice_9 = tensor.extract_slice %expanded_8[0, 0, 0, 0] [8, 1, 8, 1536] [1, 1, 1, 1] : tensor<8x1x8x3072xbf16> to tensor<8x1x8x1536xbf16>
    %extracted_slice_10 = tensor.extract_slice %expanded_8[0, 0, 0, 1536] [8, 1, 8, 1536] [1, 1, 1, 1] : tensor<8x1x8x3072xbf16> to tensor<8x1x8x1536xbf16>
    %30 = tensor.empty() : tensor<8x1x8x1536xbf16>
    %31 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_9 : tensor<8x1x8x1536xbf16>) outs(%30 : tensor<8x1x8x1536xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %40 = arith.negf %in : bf16
      %41 = math.exp %40 : bf16
      %42 = arith.addf %41, %cst_2 : bf16
      %43 = arith.divf %cst_2, %42 : bf16
      linalg.yield %43 : bf16
    } -> tensor<8x1x8x1536xbf16>
    %32 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%31, %extracted_slice_9 : tensor<8x1x8x1536xbf16>, tensor<8x1x8x1536xbf16>) outs(%30 : tensor<8x1x8x1536xbf16>) {
    ^bb0(%in: bf16, %in_14: bf16, %out: bf16):
      %40 = arith.mulf %in, %in_14 : bf16
      linalg.yield %40 : bf16
    } -> tensor<8x1x8x1536xbf16>
    %33 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%32, %extracted_slice_10 : tensor<8x1x8x1536xbf16>, tensor<8x1x8x1536xbf16>) outs(%30 : tensor<8x1x8x1536xbf16>) {
    ^bb0(%in: bf16, %in_14: bf16, %out: bf16):
      %40 = arith.mulf %in, %in_14 : bf16
      linalg.yield %40 : bf16
    } -> tensor<8x1x8x1536xbf16>
    %expanded_11 = tensor.expand_shape %24 [[0], [1], [2, 3]] output_shape [8, 1, 8, 1] : tensor<8x1x8xbf16> into tensor<8x1x8x1xbf16>
    %34 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%33, %expanded_11 : tensor<8x1x8x1536xbf16>, tensor<8x1x8x1xbf16>) outs(%30 : tensor<8x1x8x1536xbf16>) {
    ^bb0(%in: bf16, %in_14: bf16, %out: bf16):
      %40 = arith.mulf %in, %in_14 : bf16
      linalg.yield %40 : bf16
    } -> tensor<8x1x8x1536xbf16>
    %collapsed_12 = tensor.collapse_shape %34 [[0], [1], [2, 3]] : tensor<8x1x8x1536xbf16> into tensor<8x1x12288xbf16>
    %extracted_slice_13 = tensor.extract_slice %arg2[0, 0, 0] [8, 12288, 4096] [1, 1, 1] : tensor<16x12288x4096xbf16> to tensor<8x12288x4096xbf16>
    %35 = tensor.empty() : tensor<8x1x4096xf32>
    %36 = linalg.fill ins(%cst : f32) outs(%35 : tensor<8x1x4096xf32>) -> tensor<8x1x4096xf32>
    %37 = linalg.batch_matmul ins(%collapsed_12, %extracted_slice_13 : tensor<8x1x12288xbf16>, tensor<8x12288x4096xbf16>) outs(%36 : tensor<8x1x4096xf32>) -> tensor<8x1x4096xf32>
    %38 = tensor.empty() : tensor<8x1x4096xbf16>
    %39 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%37 : tensor<8x1x4096xf32>) outs(%38 : tensor<8x1x4096xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %40 = arith.truncf %in : f32 to bf16
      linalg.yield %40 : bf16
    } -> tensor<8x1x4096xbf16>
    return %39, %5 : tensor<8x1x4096xbf16>, tensor<8x128xbf16>
  }
}
