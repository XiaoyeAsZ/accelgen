#map = affine_map<(d0, d1, d2) -> (d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> (0, 0, d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (0, d1, 0, 0)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>
module {
  func.func @main(%arg0: tensor<8x4096x4096xbf16>, %arg1: tensor<1x4096x128xbf16>, %arg2: tensor<1x4096x128xbf16>, %arg3: tensor<1x32x1x1xbf16>, %arg4: tensor<4096x4096xbf16>, %arg5: tensor<4096x4096xbf16>, %arg6: tensor<4096x4096xbf16>, %arg7: tensor<4096x4096xbf16>) -> (tensor<8x4096x4096xbf16>, tensor<8x32x4096x4096xbf16>) {
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant 0.000000e+00 : bf16
    %cst_0 = arith.constant 0xFF800000 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %cst_2 = arith.constant 0.088388347648318447 : f64
    %0 = tensor.empty() : tensor<4096x4096xbf16>
    %transposed = linalg.transpose ins(%arg4 : tensor<4096x4096xbf16>) outs(%0 : tensor<4096x4096xbf16>) permutation = [1, 0] 
    %1 = tensor.empty() : tensor<8x4096x4096xbf16>
    %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed : tensor<4096x4096xbf16>) outs(%1 : tensor<8x4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x4096x4096xbf16>
    %3 = linalg.fill ins(%cst : bf16) outs(%1 : tensor<8x4096x4096xbf16>) -> tensor<8x4096x4096xbf16>
    %4 = linalg.batch_matmul ins(%arg0, %2 : tensor<8x4096x4096xbf16>, tensor<8x4096x4096xbf16>) outs(%3 : tensor<8x4096x4096xbf16>) -> tensor<8x4096x4096xbf16>
    %expanded = tensor.expand_shape %4 [[0], [1], [2, 3]] output_shape [8, 4096, 32, 128] : tensor<8x4096x4096xbf16> into tensor<8x4096x32x128xbf16>
    %5 = tensor.empty() : tensor<8x32x4096x128xbf16>
    %transposed_3 = linalg.transpose ins(%expanded : tensor<8x4096x32x128xbf16>) outs(%5 : tensor<8x32x4096x128xbf16>) permutation = [0, 2, 1, 3] 
    %transposed_4 = linalg.transpose ins(%arg5 : tensor<4096x4096xbf16>) outs(%0 : tensor<4096x4096xbf16>) permutation = [1, 0] 
    %6 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_4 : tensor<4096x4096xbf16>) outs(%1 : tensor<8x4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x4096x4096xbf16>
    %7 = linalg.batch_matmul ins(%arg0, %6 : tensor<8x4096x4096xbf16>, tensor<8x4096x4096xbf16>) outs(%3 : tensor<8x4096x4096xbf16>) -> tensor<8x4096x4096xbf16>
    %expanded_5 = tensor.expand_shape %7 [[0], [1], [2, 3]] output_shape [8, 4096, 32, 128] : tensor<8x4096x4096xbf16> into tensor<8x4096x32x128xbf16>
    %transposed_6 = linalg.transpose ins(%expanded_5 : tensor<8x4096x32x128xbf16>) outs(%5 : tensor<8x32x4096x128xbf16>) permutation = [0, 2, 1, 3] 
    %transposed_7 = linalg.transpose ins(%arg6 : tensor<4096x4096xbf16>) outs(%0 : tensor<4096x4096xbf16>) permutation = [1, 0] 
    %8 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_7 : tensor<4096x4096xbf16>) outs(%1 : tensor<8x4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x4096x4096xbf16>
    %9 = linalg.batch_matmul ins(%arg0, %8 : tensor<8x4096x4096xbf16>, tensor<8x4096x4096xbf16>) outs(%3 : tensor<8x4096x4096xbf16>) -> tensor<8x4096x4096xbf16>
    %expanded_8 = tensor.expand_shape %9 [[0], [1], [2, 3]] output_shape [8, 4096, 32, 128] : tensor<8x4096x4096xbf16> into tensor<8x4096x32x128xbf16>
    %transposed_9 = linalg.transpose ins(%expanded_8 : tensor<8x4096x32x128xbf16>) outs(%5 : tensor<8x32x4096x128xbf16>) permutation = [0, 2, 1, 3] 
    %expanded_10 = tensor.expand_shape %arg1 [[0], [1, 2], [3]] output_shape [1, 1, 4096, 128] : tensor<1x4096x128xbf16> into tensor<1x1x4096x128xbf16>
    %expanded_11 = tensor.expand_shape %arg2 [[0], [1, 2], [3]] output_shape [1, 1, 4096, 128] : tensor<1x4096x128xbf16> into tensor<1x1x4096x128xbf16>
    %10 = linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_3, %expanded_10 : tensor<8x32x4096x128xbf16>, tensor<1x1x4096x128xbf16>) outs(%5 : tensor<8x32x4096x128xbf16>) {
    ^bb0(%in: bf16, %in_26: bf16, %out: bf16):
      %46 = arith.mulf %in, %in_26 : bf16
      linalg.yield %46 : bf16
    } -> tensor<8x32x4096x128xbf16>
    %extracted_slice = tensor.extract_slice %transposed_3[0, 0, 0, 0] [8, 32, 4096, 64] [1, 1, 1, 1] : tensor<8x32x4096x128xbf16> to tensor<8x32x4096x64xbf16>
    %extracted_slice_12 = tensor.extract_slice %transposed_3[0, 0, 0, 64] [8, 32, 4096, 64] [1, 1, 1, 1] : tensor<8x32x4096x128xbf16> to tensor<8x32x4096x64xbf16>
    %11 = tensor.empty() : tensor<8x32x4096x64xbf16>
    %12 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_12 : tensor<8x32x4096x64xbf16>) outs(%11 : tensor<8x32x4096x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %46 = arith.negf %in : bf16
      linalg.yield %46 : bf16
    } -> tensor<8x32x4096x64xbf16>
    %concat = tensor.concat dim(3) %12, %extracted_slice : (tensor<8x32x4096x64xbf16>, tensor<8x32x4096x64xbf16>) -> tensor<8x32x4096x128xbf16>
    %13 = linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat, %expanded_11 : tensor<8x32x4096x128xbf16>, tensor<1x1x4096x128xbf16>) outs(%5 : tensor<8x32x4096x128xbf16>) {
    ^bb0(%in: bf16, %in_26: bf16, %out: bf16):
      %46 = arith.mulf %in, %in_26 : bf16
      linalg.yield %46 : bf16
    } -> tensor<8x32x4096x128xbf16>
    %14 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%10, %13 : tensor<8x32x4096x128xbf16>, tensor<8x32x4096x128xbf16>) outs(%5 : tensor<8x32x4096x128xbf16>) {
    ^bb0(%in: bf16, %in_26: bf16, %out: bf16):
      %46 = arith.addf %in, %in_26 : bf16
      linalg.yield %46 : bf16
    } -> tensor<8x32x4096x128xbf16>
    %15 = linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_6, %expanded_10 : tensor<8x32x4096x128xbf16>, tensor<1x1x4096x128xbf16>) outs(%5 : tensor<8x32x4096x128xbf16>) {
    ^bb0(%in: bf16, %in_26: bf16, %out: bf16):
      %46 = arith.mulf %in, %in_26 : bf16
      linalg.yield %46 : bf16
    } -> tensor<8x32x4096x128xbf16>
    %extracted_slice_13 = tensor.extract_slice %transposed_6[0, 0, 0, 0] [8, 32, 4096, 64] [1, 1, 1, 1] : tensor<8x32x4096x128xbf16> to tensor<8x32x4096x64xbf16>
    %extracted_slice_14 = tensor.extract_slice %transposed_6[0, 0, 0, 64] [8, 32, 4096, 64] [1, 1, 1, 1] : tensor<8x32x4096x128xbf16> to tensor<8x32x4096x64xbf16>
    %16 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_14 : tensor<8x32x4096x64xbf16>) outs(%11 : tensor<8x32x4096x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %46 = arith.negf %in : bf16
      linalg.yield %46 : bf16
    } -> tensor<8x32x4096x64xbf16>
    %concat_15 = tensor.concat dim(3) %16, %extracted_slice_13 : (tensor<8x32x4096x64xbf16>, tensor<8x32x4096x64xbf16>) -> tensor<8x32x4096x128xbf16>
    %17 = linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_15, %expanded_11 : tensor<8x32x4096x128xbf16>, tensor<1x1x4096x128xbf16>) outs(%5 : tensor<8x32x4096x128xbf16>) {
    ^bb0(%in: bf16, %in_26: bf16, %out: bf16):
      %46 = arith.mulf %in, %in_26 : bf16
      linalg.yield %46 : bf16
    } -> tensor<8x32x4096x128xbf16>
    %18 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%15, %17 : tensor<8x32x4096x128xbf16>, tensor<8x32x4096x128xbf16>) outs(%5 : tensor<8x32x4096x128xbf16>) {
    ^bb0(%in: bf16, %in_26: bf16, %out: bf16):
      %46 = arith.addf %in, %in_26 : bf16
      linalg.yield %46 : bf16
    } -> tensor<8x32x4096x128xbf16>
    %19 = tensor.empty() : tensor<8x32x128x4096xbf16>
    %transposed_16 = linalg.transpose ins(%18 : tensor<8x32x4096x128xbf16>) outs(%19 : tensor<8x32x128x4096xbf16>) permutation = [0, 1, 3, 2] 
    %collapsed = tensor.collapse_shape %14 [[0, 1], [2], [3]] : tensor<8x32x4096x128xbf16> into tensor<256x4096x128xbf16>
    %collapsed_17 = tensor.collapse_shape %transposed_16 [[0, 1], [2], [3]] : tensor<8x32x128x4096xbf16> into tensor<256x128x4096xbf16>
    %20 = tensor.empty() : tensor<256x4096x4096xbf16>
    %21 = linalg.fill ins(%cst : bf16) outs(%20 : tensor<256x4096x4096xbf16>) -> tensor<256x4096x4096xbf16>
    %22 = linalg.batch_matmul ins(%collapsed, %collapsed_17 : tensor<256x4096x128xbf16>, tensor<256x128x4096xbf16>) outs(%21 : tensor<256x4096x4096xbf16>) -> tensor<256x4096x4096xbf16>
    %expanded_18 = tensor.expand_shape %22 [[0, 1], [2], [3]] output_shape [8, 32, 4096, 4096] : tensor<256x4096x4096xbf16> into tensor<8x32x4096x4096xbf16>
    %23 = tensor.empty() : tensor<8x32x4096x4096xbf16>
    %24 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_18 : tensor<8x32x4096x4096xbf16>) outs(%23 : tensor<8x32x4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %46 = arith.truncf %cst_2 : f64 to bf16
      %47 = arith.mulf %in, %46 : bf16
      linalg.yield %47 : bf16
    } -> tensor<8x32x4096x4096xbf16>
    %25 = linalg.generic {indexing_maps = [#map2, #map4, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%24, %arg3 : tensor<8x32x4096x4096xbf16>, tensor<1x32x1x1xbf16>) outs(%23 : tensor<8x32x4096x4096xbf16>) {
    ^bb0(%in: bf16, %in_26: bf16, %out: bf16):
      %46 = arith.addf %in, %in_26 : bf16
      linalg.yield %46 : bf16
    } -> tensor<8x32x4096x4096xbf16>
    %26 = tensor.empty() : tensor<8x32x4096x4096xf32>
    %27 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%25 : tensor<8x32x4096x4096xbf16>) outs(%26 : tensor<8x32x4096x4096xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %46 = arith.extf %in : bf16 to f32
      linalg.yield %46 : f32
    } -> tensor<8x32x4096x4096xf32>
    %28 = tensor.empty() : tensor<8x32x4096xi64>
    %29 = linalg.fill ins(%c0_i64 : i64) outs(%28 : tensor<8x32x4096xi64>) -> tensor<8x32x4096xi64>
    %30 = tensor.empty() : tensor<8x32x4096xf32>
    %31 = linalg.fill ins(%cst_0 : f32) outs(%30 : tensor<8x32x4096xf32>) -> tensor<8x32x4096xf32>
    %32:2 = linalg.generic {indexing_maps = [#map2, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%27 : tensor<8x32x4096x4096xf32>) outs(%31, %29 : tensor<8x32x4096xf32>, tensor<8x32x4096xi64>) {
    ^bb0(%in: f32, %out: f32, %out_26: i64):
      %46 = linalg.index 3 : index
      %47 = arith.index_cast %46 : index to i64
      %48 = arith.maximumf %in, %out : f32
      %49 = arith.cmpf ogt, %in, %out : f32
      %50 = arith.select %49, %47, %out_26 : i64
      linalg.yield %48, %50 : f32, i64
    } -> (tensor<8x32x4096xf32>, tensor<8x32x4096xi64>)
    %expanded_19 = tensor.expand_shape %32#0 [[0], [1], [2, 3]] output_shape [8, 32, 4096, 1] : tensor<8x32x4096xf32> into tensor<8x32x4096x1xf32>
    %33 = linalg.generic {indexing_maps = [#map2, #map6, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%27, %expanded_19 : tensor<8x32x4096x4096xf32>, tensor<8x32x4096x1xf32>) outs(%26 : tensor<8x32x4096x4096xf32>) {
    ^bb0(%in: f32, %in_26: f32, %out: f32):
      %46 = arith.subf %in, %in_26 : f32
      linalg.yield %46 : f32
    } -> tensor<8x32x4096x4096xf32>
    %34 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%33 : tensor<8x32x4096x4096xf32>) outs(%26 : tensor<8x32x4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %46 = math.exp %in : f32
      linalg.yield %46 : f32
    } -> tensor<8x32x4096x4096xf32>
    %35 = tensor.empty() : tensor<8x32x4096x1xf32>
    %36 = linalg.fill ins(%cst_1 : f32) outs(%35 : tensor<8x32x4096x1xf32>) -> tensor<8x32x4096x1xf32>
    %37 = linalg.generic {indexing_maps = [#map2, #map6], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%34 : tensor<8x32x4096x4096xf32>) outs(%36 : tensor<8x32x4096x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %46 = arith.addf %in, %out : f32
      linalg.yield %46 : f32
    } -> tensor<8x32x4096x1xf32>
    %38 = linalg.generic {indexing_maps = [#map2, #map6, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%34, %37 : tensor<8x32x4096x4096xf32>, tensor<8x32x4096x1xf32>) outs(%26 : tensor<8x32x4096x4096xf32>) {
    ^bb0(%in: f32, %in_26: f32, %out: f32):
      %46 = arith.divf %in, %in_26 : f32
      linalg.yield %46 : f32
    } -> tensor<8x32x4096x4096xf32>
    %39 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%38 : tensor<8x32x4096x4096xf32>) outs(%23 : tensor<8x32x4096x4096xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %46 = arith.truncf %in : f32 to bf16
      linalg.yield %46 : bf16
    } -> tensor<8x32x4096x4096xbf16>
    %collapsed_20 = tensor.collapse_shape %39 [[0, 1], [2], [3]] : tensor<8x32x4096x4096xbf16> into tensor<256x4096x4096xbf16>
    %collapsed_21 = tensor.collapse_shape %transposed_9 [[0, 1], [2], [3]] : tensor<8x32x4096x128xbf16> into tensor<256x4096x128xbf16>
    %40 = tensor.empty() : tensor<256x4096x128xbf16>
    %41 = linalg.fill ins(%cst : bf16) outs(%40 : tensor<256x4096x128xbf16>) -> tensor<256x4096x128xbf16>
    %42 = linalg.batch_matmul ins(%collapsed_20, %collapsed_21 : tensor<256x4096x4096xbf16>, tensor<256x4096x128xbf16>) outs(%41 : tensor<256x4096x128xbf16>) -> tensor<256x4096x128xbf16>
    %expanded_22 = tensor.expand_shape %42 [[0, 1], [2], [3]] output_shape [8, 32, 4096, 128] : tensor<256x4096x128xbf16> into tensor<8x32x4096x128xbf16>
    %43 = tensor.empty() : tensor<8x4096x32x128xbf16>
    %transposed_23 = linalg.transpose ins(%expanded_22 : tensor<8x32x4096x128xbf16>) outs(%43 : tensor<8x4096x32x128xbf16>) permutation = [0, 2, 1, 3] 
    %collapsed_24 = tensor.collapse_shape %transposed_23 [[0], [1], [2, 3]] : tensor<8x4096x32x128xbf16> into tensor<8x4096x4096xbf16>
    %transposed_25 = linalg.transpose ins(%arg7 : tensor<4096x4096xbf16>) outs(%0 : tensor<4096x4096xbf16>) permutation = [1, 0] 
    %44 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_25 : tensor<4096x4096xbf16>) outs(%1 : tensor<8x4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x4096x4096xbf16>
    %45 = linalg.batch_matmul ins(%collapsed_24, %44 : tensor<8x4096x4096xbf16>, tensor<8x4096x4096xbf16>) outs(%3 : tensor<8x4096x4096xbf16>) -> tensor<8x4096x4096xbf16>
    return %45, %39 : tensor<8x4096x4096xbf16>, tensor<8x32x4096x4096xbf16>
  }
}
