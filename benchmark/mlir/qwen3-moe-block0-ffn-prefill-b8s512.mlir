#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, 0)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map5 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map6 = affine_map<(d0, d1, d2) -> (d0, d1, 0)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>
module {
  func.func @main(%arg0: tensor<8x512x4096xbf16>, %arg1: tensor<16x4096x24576xbf16>, %arg2: tensor<16x12288x4096xbf16>, %arg3: tensor<16x16xbf16>, %arg4: tensor<128x4096xbf16>) -> (tensor<8x512x4096xbf16>, tensor<4096x128xbf16>) {
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : bf16
    %cst_1 = arith.constant 0xFF800000 : f32
    %cst_2 = arith.constant 1.000000e+00 : bf16
    %collapsed = tensor.collapse_shape %arg0 [[0, 1], [2]] : tensor<8x512x4096xbf16> into tensor<4096x4096xbf16>
    %expanded = tensor.expand_shape %collapsed [[0, 1], [2]] output_shape [16, 256, 4096] : tensor<4096x4096xbf16> into tensor<16x256x4096xbf16>
    %0 = tensor.empty() : tensor<4096x128xbf16>
    %transposed = linalg.transpose ins(%arg4 : tensor<128x4096xbf16>) outs(%0 : tensor<4096x128xbf16>) permutation = [1, 0] 
    %1 = tensor.empty() : tensor<4096x128xf32>
    %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<4096x128xf32>) -> tensor<4096x128xf32>
    %3 = linalg.matmul ins(%collapsed, %transposed : tensor<4096x4096xbf16>, tensor<4096x128xbf16>) outs(%2 : tensor<4096x128xf32>) -> tensor<4096x128xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%3 : tensor<4096x128xf32>) outs(%0 : tensor<4096x128xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %39 = arith.truncf %in : f32 to bf16
      linalg.yield %39 : bf16
    } -> tensor<4096x128xbf16>
    %expanded_3 = tensor.expand_shape %4 [[0, 1], [2, 3]] output_shape [16, 256, 16, 8] : tensor<4096x128xbf16> into tensor<16x256x16x8xbf16>
    %expanded_4 = tensor.expand_shape %arg3 [[0], [1, 2, 3]] output_shape [16, 1, 16, 1] : tensor<16x16xbf16> into tensor<16x1x16x1xbf16>
    %5 = tensor.empty() : tensor<16x256x16x8xbf16>
    %6 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_3, %expanded_4 : tensor<16x256x16x8xbf16>, tensor<16x1x16x1xbf16>) outs(%5 : tensor<16x256x16x8xbf16>) {
    ^bb0(%in: bf16, %in_12: bf16, %out: bf16):
      %39 = arith.mulf %in, %in_12 : bf16
      linalg.yield %39 : bf16
    } -> tensor<16x256x16x8xbf16>
    %7 = tensor.empty() : tensor<16x256x8xbf16>
    %8 = linalg.fill ins(%cst_0 : bf16) outs(%7 : tensor<16x256x8xbf16>) -> tensor<16x256x8xbf16>
    %9 = linalg.generic {indexing_maps = [#map1, #map3], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%6 : tensor<16x256x16x8xbf16>) outs(%8 : tensor<16x256x8xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %39 = arith.addf %in, %out : bf16
      linalg.yield %39 : bf16
    } -> tensor<16x256x8xbf16>
    %10 = tensor.empty() : tensor<16x256x8xf32>
    %11 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%9 : tensor<16x256x8xbf16>) outs(%10 : tensor<16x256x8xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %39 = arith.extf %in : bf16 to f32
      linalg.yield %39 : f32
    } -> tensor<16x256x8xf32>
    %12 = tensor.empty() : tensor<16x256xi64>
    %13 = linalg.fill ins(%c0_i64 : i64) outs(%12 : tensor<16x256xi64>) -> tensor<16x256xi64>
    %14 = tensor.empty() : tensor<16x256xf32>
    %15 = linalg.fill ins(%cst_1 : f32) outs(%14 : tensor<16x256xf32>) -> tensor<16x256xf32>
    %16:2 = linalg.generic {indexing_maps = [#map4, #map5, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%11 : tensor<16x256x8xf32>) outs(%15, %13 : tensor<16x256xf32>, tensor<16x256xi64>) {
    ^bb0(%in: f32, %out: f32, %out_12: i64):
      %39 = linalg.index 2 : index
      %40 = arith.index_cast %39 : index to i64
      %41 = arith.maximumf %in, %out : f32
      %42 = arith.cmpf ogt, %in, %out : f32
      %43 = arith.select %42, %40, %out_12 : i64
      linalg.yield %41, %43 : f32, i64
    } -> (tensor<16x256xf32>, tensor<16x256xi64>)
    %expanded_5 = tensor.expand_shape %16#0 [[0], [1, 2]] output_shape [16, 256, 1] : tensor<16x256xf32> into tensor<16x256x1xf32>
    %17 = linalg.generic {indexing_maps = [#map4, #map6, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%11, %expanded_5 : tensor<16x256x8xf32>, tensor<16x256x1xf32>) outs(%10 : tensor<16x256x8xf32>) {
    ^bb0(%in: f32, %in_12: f32, %out: f32):
      %39 = arith.subf %in, %in_12 : f32
      linalg.yield %39 : f32
    } -> tensor<16x256x8xf32>
    %18 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17 : tensor<16x256x8xf32>) outs(%10 : tensor<16x256x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %39 = math.exp %in : f32
      linalg.yield %39 : f32
    } -> tensor<16x256x8xf32>
    %19 = tensor.empty() : tensor<16x256x1xf32>
    %20 = linalg.fill ins(%cst : f32) outs(%19 : tensor<16x256x1xf32>) -> tensor<16x256x1xf32>
    %21 = linalg.generic {indexing_maps = [#map4, #map6], iterator_types = ["parallel", "parallel", "reduction"]} ins(%18 : tensor<16x256x8xf32>) outs(%20 : tensor<16x256x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %39 = arith.addf %in, %out : f32
      linalg.yield %39 : f32
    } -> tensor<16x256x1xf32>
    %22 = linalg.generic {indexing_maps = [#map4, #map6, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%18, %21 : tensor<16x256x8xf32>, tensor<16x256x1xf32>) outs(%10 : tensor<16x256x8xf32>) {
    ^bb0(%in: f32, %in_12: f32, %out: f32):
      %39 = arith.divf %in, %in_12 : f32
      linalg.yield %39 : f32
    } -> tensor<16x256x8xf32>
    %23 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%22 : tensor<16x256x8xf32>) outs(%7 : tensor<16x256x8xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %39 = arith.truncf %in : f32 to bf16
      linalg.yield %39 : bf16
    } -> tensor<16x256x8xbf16>
    %24 = tensor.empty() : tensor<16x256x24576xf32>
    %25 = linalg.fill ins(%cst : f32) outs(%24 : tensor<16x256x24576xf32>) -> tensor<16x256x24576xf32>
    %26 = linalg.batch_matmul ins(%expanded, %arg1 : tensor<16x256x4096xbf16>, tensor<16x4096x24576xbf16>) outs(%25 : tensor<16x256x24576xf32>) -> tensor<16x256x24576xf32>
    %27 = tensor.empty() : tensor<16x256x24576xbf16>
    %28 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%26 : tensor<16x256x24576xf32>) outs(%27 : tensor<16x256x24576xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %39 = arith.truncf %in : f32 to bf16
      linalg.yield %39 : bf16
    } -> tensor<16x256x24576xbf16>
    %expanded_6 = tensor.expand_shape %28 [[0], [1], [2, 3]] output_shape [16, 256, 8, 3072] : tensor<16x256x24576xbf16> into tensor<16x256x8x3072xbf16>
    %extracted_slice = tensor.extract_slice %expanded_6[0, 0, 0, 0] [16, 256, 8, 1536] [1, 1, 1, 1] : tensor<16x256x8x3072xbf16> to tensor<16x256x8x1536xbf16>
    %extracted_slice_7 = tensor.extract_slice %expanded_6[0, 0, 0, 1536] [16, 256, 8, 1536] [1, 1, 1, 1] : tensor<16x256x8x3072xbf16> to tensor<16x256x8x1536xbf16>
    %29 = tensor.empty() : tensor<16x256x8x1536xbf16>
    %30 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice : tensor<16x256x8x1536xbf16>) outs(%29 : tensor<16x256x8x1536xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %39 = arith.negf %in : bf16
      %40 = math.exp %39 : bf16
      %41 = arith.addf %40, %cst_2 : bf16
      %42 = arith.divf %cst_2, %41 : bf16
      linalg.yield %42 : bf16
    } -> tensor<16x256x8x1536xbf16>
    %31 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%30, %extracted_slice : tensor<16x256x8x1536xbf16>, tensor<16x256x8x1536xbf16>) outs(%29 : tensor<16x256x8x1536xbf16>) {
    ^bb0(%in: bf16, %in_12: bf16, %out: bf16):
      %39 = arith.mulf %in, %in_12 : bf16
      linalg.yield %39 : bf16
    } -> tensor<16x256x8x1536xbf16>
    %32 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%31, %extracted_slice_7 : tensor<16x256x8x1536xbf16>, tensor<16x256x8x1536xbf16>) outs(%29 : tensor<16x256x8x1536xbf16>) {
    ^bb0(%in: bf16, %in_12: bf16, %out: bf16):
      %39 = arith.mulf %in, %in_12 : bf16
      linalg.yield %39 : bf16
    } -> tensor<16x256x8x1536xbf16>
    %expanded_8 = tensor.expand_shape %23 [[0], [1], [2, 3]] output_shape [16, 256, 8, 1] : tensor<16x256x8xbf16> into tensor<16x256x8x1xbf16>
    %33 = linalg.generic {indexing_maps = [#map1, #map7, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%32, %expanded_8 : tensor<16x256x8x1536xbf16>, tensor<16x256x8x1xbf16>) outs(%29 : tensor<16x256x8x1536xbf16>) {
    ^bb0(%in: bf16, %in_12: bf16, %out: bf16):
      %39 = arith.mulf %in, %in_12 : bf16
      linalg.yield %39 : bf16
    } -> tensor<16x256x8x1536xbf16>
    %collapsed_9 = tensor.collapse_shape %33 [[0], [1], [2, 3]] : tensor<16x256x8x1536xbf16> into tensor<16x256x12288xbf16>
    %34 = tensor.empty() : tensor<16x256x4096xf32>
    %35 = linalg.fill ins(%cst : f32) outs(%34 : tensor<16x256x4096xf32>) -> tensor<16x256x4096xf32>
    %36 = linalg.batch_matmul ins(%collapsed_9, %arg2 : tensor<16x256x12288xbf16>, tensor<16x12288x4096xbf16>) outs(%35 : tensor<16x256x4096xf32>) -> tensor<16x256x4096xf32>
    %37 = tensor.empty() : tensor<16x256x4096xbf16>
    %38 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36 : tensor<16x256x4096xf32>) outs(%37 : tensor<16x256x4096xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %39 = arith.truncf %in : f32 to bf16
      linalg.yield %39 : bf16
    } -> tensor<16x256x4096xbf16>
    %collapsed_10 = tensor.collapse_shape %38 [[0, 1], [2]] : tensor<16x256x4096xbf16> into tensor<4096x4096xbf16>
    %expanded_11 = tensor.expand_shape %collapsed_10 [[0, 1], [2]] output_shape [8, 512, 4096] : tensor<4096x4096xbf16> into tensor<8x512x4096xbf16>
    return %expanded_11, %4 : tensor<8x512x4096xbf16>, tensor<4096x128xbf16>
  }
}
