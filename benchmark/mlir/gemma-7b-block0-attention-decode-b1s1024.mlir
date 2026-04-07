#map = affine_map<(d0, d1, d2) -> (d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d1, d3)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>
#map8 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
module {
  func.func @main(%arg0: tensor<1x1x3072xbf16>, %arg1: tensor<1x1x256xbf16>, %arg2: tensor<1x1x256xbf16>, %arg3: tensor<1x16x1x1xbf16>, %arg4: tensor<1x16x1023x256xbf16>, %arg5: tensor<1x16x1023x256xbf16>, %arg6: tensor<4096x3072xbf16>, %arg7: tensor<4096x3072xbf16>, %arg8: tensor<4096x3072xbf16>, %arg9: tensor<3072x4096xbf16>) -> tensor<1x1x3072xbf16> {
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant 0.000000e+00 : bf16
    %cst_0 = arith.constant 0xFF800000 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %cst_2 = arith.constant 6.250000e-02 : bf16
    %0 = tensor.empty() : tensor<3072x4096xbf16>
    %transposed = linalg.transpose ins(%arg6 : tensor<4096x3072xbf16>) outs(%0 : tensor<3072x4096xbf16>) permutation = [1, 0] 
    %1 = tensor.empty() : tensor<1x1x3072xbf16>
    %collapsed = tensor.collapse_shape %arg0 [[0, 1, 2]] : tensor<1x1x3072xbf16> into tensor<3072xbf16>
    %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapsed : tensor<3072xbf16>) outs(%1 : tensor<1x1x3072xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x1x3072xbf16>
    %3 = tensor.empty() : tensor<1x3072x4096xbf16>
    %4 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed : tensor<3072x4096xbf16>) outs(%3 : tensor<1x3072x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x3072x4096xbf16>
    %5 = tensor.empty() : tensor<1x1x4096xbf16>
    %6 = linalg.fill ins(%cst : bf16) outs(%5 : tensor<1x1x4096xbf16>) -> tensor<1x1x4096xbf16>
    %7 = linalg.batch_matmul ins(%2, %4 : tensor<1x1x3072xbf16>, tensor<1x3072x4096xbf16>) outs(%6 : tensor<1x1x4096xbf16>) -> tensor<1x1x4096xbf16>
    %expanded = tensor.expand_shape %7 [[0], [1], [2, 3]] output_shape [1, 1, 16, 256] : tensor<1x1x4096xbf16> into tensor<1x1x16x256xbf16>
    %8 = tensor.empty() : tensor<1x16x1x256xbf16>
    %transposed_3 = linalg.transpose ins(%expanded : tensor<1x1x16x256xbf16>) outs(%8 : tensor<1x16x1x256xbf16>) permutation = [0, 2, 1, 3] 
    %transposed_4 = linalg.transpose ins(%arg7 : tensor<4096x3072xbf16>) outs(%0 : tensor<3072x4096xbf16>) permutation = [1, 0] 
    %9 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_4 : tensor<3072x4096xbf16>) outs(%3 : tensor<1x3072x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x3072x4096xbf16>
    %10 = linalg.batch_matmul ins(%2, %9 : tensor<1x1x3072xbf16>, tensor<1x3072x4096xbf16>) outs(%6 : tensor<1x1x4096xbf16>) -> tensor<1x1x4096xbf16>
    %expanded_5 = tensor.expand_shape %10 [[0], [1], [2, 3]] output_shape [1, 1, 16, 256] : tensor<1x1x4096xbf16> into tensor<1x1x16x256xbf16>
    %transposed_6 = linalg.transpose ins(%expanded_5 : tensor<1x1x16x256xbf16>) outs(%8 : tensor<1x16x1x256xbf16>) permutation = [0, 2, 1, 3] 
    %transposed_7 = linalg.transpose ins(%arg8 : tensor<4096x3072xbf16>) outs(%0 : tensor<3072x4096xbf16>) permutation = [1, 0] 
    %11 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_7 : tensor<3072x4096xbf16>) outs(%3 : tensor<1x3072x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x3072x4096xbf16>
    %12 = linalg.batch_matmul ins(%2, %11 : tensor<1x1x3072xbf16>, tensor<1x3072x4096xbf16>) outs(%6 : tensor<1x1x4096xbf16>) -> tensor<1x1x4096xbf16>
    %expanded_8 = tensor.expand_shape %12 [[0], [1], [2, 3]] output_shape [1, 1, 16, 256] : tensor<1x1x4096xbf16> into tensor<1x1x16x256xbf16>
    %transposed_9 = linalg.transpose ins(%expanded_8 : tensor<1x1x16x256xbf16>) outs(%8 : tensor<1x16x1x256xbf16>) permutation = [0, 2, 1, 3] 
    %expanded_10 = tensor.expand_shape %arg1 [[0], [1, 2], [3]] output_shape [1, 1, 1, 256] : tensor<1x1x256xbf16> into tensor<1x1x1x256xbf16>
    %expanded_11 = tensor.expand_shape %arg2 [[0], [1, 2], [3]] output_shape [1, 1, 1, 256] : tensor<1x1x256xbf16> into tensor<1x1x1x256xbf16>
    %13 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_3, %expanded_10 : tensor<1x16x1x256xbf16>, tensor<1x1x1x256xbf16>) outs(%8 : tensor<1x16x1x256xbf16>) {
    ^bb0(%in: bf16, %in_33: bf16, %out: bf16):
      %58 = arith.mulf %in, %in_33 : bf16
      linalg.yield %58 : bf16
    } -> tensor<1x16x1x256xbf16>
    %extracted_slice = tensor.extract_slice %transposed_3[0, 0, 0, 0] [1, 16, 1, 128] [1, 1, 1, 1] : tensor<1x16x1x256xbf16> to tensor<1x16x1x128xbf16>
    %extracted_slice_12 = tensor.extract_slice %transposed_3[0, 0, 0, 128] [1, 16, 1, 128] [1, 1, 1, 1] : tensor<1x16x1x256xbf16> to tensor<1x16x1x128xbf16>
    %14 = tensor.empty() : tensor<1x16x1x128xbf16>
    %15 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_12 : tensor<1x16x1x128xbf16>) outs(%14 : tensor<1x16x1x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %58 = arith.negf %in : bf16
      linalg.yield %58 : bf16
    } -> tensor<1x16x1x128xbf16>
    %concat = tensor.concat dim(3) %15, %extracted_slice : (tensor<1x16x1x128xbf16>, tensor<1x16x1x128xbf16>) -> tensor<1x16x1x256xbf16>
    %16 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat, %expanded_11 : tensor<1x16x1x256xbf16>, tensor<1x1x1x256xbf16>) outs(%8 : tensor<1x16x1x256xbf16>) {
    ^bb0(%in: bf16, %in_33: bf16, %out: bf16):
      %58 = arith.mulf %in, %in_33 : bf16
      linalg.yield %58 : bf16
    } -> tensor<1x16x1x256xbf16>
    %17 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%13, %16 : tensor<1x16x1x256xbf16>, tensor<1x16x1x256xbf16>) outs(%8 : tensor<1x16x1x256xbf16>) {
    ^bb0(%in: bf16, %in_33: bf16, %out: bf16):
      %58 = arith.addf %in, %in_33 : bf16
      linalg.yield %58 : bf16
    } -> tensor<1x16x1x256xbf16>
    %18 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_6, %expanded_10 : tensor<1x16x1x256xbf16>, tensor<1x1x1x256xbf16>) outs(%8 : tensor<1x16x1x256xbf16>) {
    ^bb0(%in: bf16, %in_33: bf16, %out: bf16):
      %58 = arith.mulf %in, %in_33 : bf16
      linalg.yield %58 : bf16
    } -> tensor<1x16x1x256xbf16>
    %extracted_slice_13 = tensor.extract_slice %transposed_6[0, 0, 0, 0] [1, 16, 1, 128] [1, 1, 1, 1] : tensor<1x16x1x256xbf16> to tensor<1x16x1x128xbf16>
    %extracted_slice_14 = tensor.extract_slice %transposed_6[0, 0, 0, 128] [1, 16, 1, 128] [1, 1, 1, 1] : tensor<1x16x1x256xbf16> to tensor<1x16x1x128xbf16>
    %19 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_14 : tensor<1x16x1x128xbf16>) outs(%14 : tensor<1x16x1x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %58 = arith.negf %in : bf16
      linalg.yield %58 : bf16
    } -> tensor<1x16x1x128xbf16>
    %concat_15 = tensor.concat dim(3) %19, %extracted_slice_13 : (tensor<1x16x1x128xbf16>, tensor<1x16x1x128xbf16>) -> tensor<1x16x1x256xbf16>
    %20 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_15, %expanded_11 : tensor<1x16x1x256xbf16>, tensor<1x1x1x256xbf16>) outs(%8 : tensor<1x16x1x256xbf16>) {
    ^bb0(%in: bf16, %in_33: bf16, %out: bf16):
      %58 = arith.mulf %in, %in_33 : bf16
      linalg.yield %58 : bf16
    } -> tensor<1x16x1x256xbf16>
    %21 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%18, %20 : tensor<1x16x1x256xbf16>, tensor<1x16x1x256xbf16>) outs(%8 : tensor<1x16x1x256xbf16>) {
    ^bb0(%in: bf16, %in_33: bf16, %out: bf16):
      %58 = arith.addf %in, %in_33 : bf16
      linalg.yield %58 : bf16
    } -> tensor<1x16x1x256xbf16>
    %concat_16 = tensor.concat dim(2) %arg4, %21 : (tensor<1x16x1023x256xbf16>, tensor<1x16x1x256xbf16>) -> tensor<1x16x1024x256xbf16>
    %concat_17 = tensor.concat dim(2) %arg5, %transposed_9 : (tensor<1x16x1023x256xbf16>, tensor<1x16x1x256xbf16>) -> tensor<1x16x1024x256xbf16>
    %22 = tensor.empty() : tensor<1x16x256x1024xbf16>
    %transposed_18 = linalg.transpose ins(%concat_16 : tensor<1x16x1024x256xbf16>) outs(%22 : tensor<1x16x256x1024xbf16>) permutation = [0, 1, 3, 2] 
    %collapsed_19 = tensor.collapse_shape %17 [[0, 1, 2], [3]] : tensor<1x16x1x256xbf16> into tensor<16x256xbf16>
    %23 = linalg.generic {indexing_maps = [#map5, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%collapsed_19 : tensor<16x256xbf16>) outs(%8 : tensor<1x16x1x256xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x16x1x256xbf16>
    %collapsed_20 = tensor.collapse_shape %transposed_18 [[0, 1], [2], [3]] : tensor<1x16x256x1024xbf16> into tensor<16x256x1024xbf16>
    %24 = linalg.generic {indexing_maps = [#map6, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%collapsed_20 : tensor<16x256x1024xbf16>) outs(%22 : tensor<1x16x256x1024xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x16x256x1024xbf16>
    %collapsed_21 = tensor.collapse_shape %23 [[0, 1], [2], [3]] : tensor<1x16x1x256xbf16> into tensor<16x1x256xbf16>
    %collapsed_22 = tensor.collapse_shape %24 [[0, 1], [2], [3]] : tensor<1x16x256x1024xbf16> into tensor<16x256x1024xbf16>
    %25 = tensor.empty() : tensor<16x1x1024xbf16>
    %26 = linalg.fill ins(%cst : bf16) outs(%25 : tensor<16x1x1024xbf16>) -> tensor<16x1x1024xbf16>
    %27 = linalg.batch_matmul ins(%collapsed_21, %collapsed_22 : tensor<16x1x256xbf16>, tensor<16x256x1024xbf16>) outs(%26 : tensor<16x1x1024xbf16>) -> tensor<16x1x1024xbf16>
    %expanded_23 = tensor.expand_shape %27 [[0, 1], [2], [3]] output_shape [1, 16, 1, 1024] : tensor<16x1x1024xbf16> into tensor<1x16x1x1024xbf16>
    %28 = tensor.empty() : tensor<1x16x1x1024xbf16>
    %29 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_23 : tensor<1x16x1x1024xbf16>) outs(%28 : tensor<1x16x1x1024xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %58 = arith.mulf %in, %cst_2 : bf16
      linalg.yield %58 : bf16
    } -> tensor<1x16x1x1024xbf16>
    %30 = linalg.generic {indexing_maps = [#map3, #map7, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%29, %arg3 : tensor<1x16x1x1024xbf16>, tensor<1x16x1x1xbf16>) outs(%28 : tensor<1x16x1x1024xbf16>) {
    ^bb0(%in: bf16, %in_33: bf16, %out: bf16):
      %58 = arith.addf %in, %in_33 : bf16
      linalg.yield %58 : bf16
    } -> tensor<1x16x1x1024xbf16>
    %31 = tensor.empty() : tensor<1x16x1x1024xf32>
    %32 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%30 : tensor<1x16x1x1024xbf16>) outs(%31 : tensor<1x16x1x1024xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %58 = arith.extf %in : bf16 to f32
      linalg.yield %58 : f32
    } -> tensor<1x16x1x1024xf32>
    %33 = tensor.empty() : tensor<1x16x1xi64>
    %34 = linalg.fill ins(%c0_i64 : i64) outs(%33 : tensor<1x16x1xi64>) -> tensor<1x16x1xi64>
    %35 = tensor.empty() : tensor<1x16x1xf32>
    %36 = linalg.fill ins(%cst_0 : f32) outs(%35 : tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
    %37:2 = linalg.generic {indexing_maps = [#map3, #map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%32 : tensor<1x16x1x1024xf32>) outs(%36, %34 : tensor<1x16x1xf32>, tensor<1x16x1xi64>) {
    ^bb0(%in: f32, %out: f32, %out_33: i64):
      %58 = linalg.index 3 : index
      %59 = arith.index_cast %58 : index to i64
      %60 = arith.maximumf %in, %out : f32
      %61 = arith.cmpf ogt, %in, %out : f32
      %62 = arith.select %61, %59, %out_33 : i64
      linalg.yield %60, %62 : f32, i64
    } -> (tensor<1x16x1xf32>, tensor<1x16x1xi64>)
    %expanded_24 = tensor.expand_shape %37#0 [[0], [1], [2, 3]] output_shape [1, 16, 1, 1] : tensor<1x16x1xf32> into tensor<1x16x1x1xf32>
    %38 = linalg.generic {indexing_maps = [#map3, #map7, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%32, %expanded_24 : tensor<1x16x1x1024xf32>, tensor<1x16x1x1xf32>) outs(%31 : tensor<1x16x1x1024xf32>) {
    ^bb0(%in: f32, %in_33: f32, %out: f32):
      %58 = arith.subf %in, %in_33 : f32
      linalg.yield %58 : f32
    } -> tensor<1x16x1x1024xf32>
    %39 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%38 : tensor<1x16x1x1024xf32>) outs(%31 : tensor<1x16x1x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %58 = math.exp %in : f32
      linalg.yield %58 : f32
    } -> tensor<1x16x1x1024xf32>
    %40 = tensor.empty() : tensor<1x16x1x1xf32>
    %41 = linalg.fill ins(%cst_1 : f32) outs(%40 : tensor<1x16x1x1xf32>) -> tensor<1x16x1x1xf32>
    %42 = linalg.generic {indexing_maps = [#map3, #map7], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%39 : tensor<1x16x1x1024xf32>) outs(%41 : tensor<1x16x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %58 = arith.addf %in, %out : f32
      linalg.yield %58 : f32
    } -> tensor<1x16x1x1xf32>
    %43 = linalg.generic {indexing_maps = [#map3, #map7, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%39, %42 : tensor<1x16x1x1024xf32>, tensor<1x16x1x1xf32>) outs(%31 : tensor<1x16x1x1024xf32>) {
    ^bb0(%in: f32, %in_33: f32, %out: f32):
      %58 = arith.divf %in, %in_33 : f32
      linalg.yield %58 : f32
    } -> tensor<1x16x1x1024xf32>
    %44 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%43 : tensor<1x16x1x1024xf32>) outs(%28 : tensor<1x16x1x1024xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %58 = arith.truncf %in : f32 to bf16
      linalg.yield %58 : bf16
    } -> tensor<1x16x1x1024xbf16>
    %collapsed_25 = tensor.collapse_shape %44 [[0, 1, 2], [3]] : tensor<1x16x1x1024xbf16> into tensor<16x1024xbf16>
    %45 = linalg.generic {indexing_maps = [#map5, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%collapsed_25 : tensor<16x1024xbf16>) outs(%28 : tensor<1x16x1x1024xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x16x1x1024xbf16>
    %46 = tensor.empty() : tensor<1x16x1024x256xbf16>
    %collapsed_26 = tensor.collapse_shape %concat_17 [[0, 1], [2], [3]] : tensor<1x16x1024x256xbf16> into tensor<16x1024x256xbf16>
    %47 = linalg.generic {indexing_maps = [#map6, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%collapsed_26 : tensor<16x1024x256xbf16>) outs(%46 : tensor<1x16x1024x256xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x16x1024x256xbf16>
    %collapsed_27 = tensor.collapse_shape %45 [[0, 1], [2], [3]] : tensor<1x16x1x1024xbf16> into tensor<16x1x1024xbf16>
    %collapsed_28 = tensor.collapse_shape %47 [[0, 1], [2], [3]] : tensor<1x16x1024x256xbf16> into tensor<16x1024x256xbf16>
    %48 = tensor.empty() : tensor<16x1x256xbf16>
    %49 = linalg.fill ins(%cst : bf16) outs(%48 : tensor<16x1x256xbf16>) -> tensor<16x1x256xbf16>
    %50 = linalg.batch_matmul ins(%collapsed_27, %collapsed_28 : tensor<16x1x1024xbf16>, tensor<16x1024x256xbf16>) outs(%49 : tensor<16x1x256xbf16>) -> tensor<16x1x256xbf16>
    %expanded_29 = tensor.expand_shape %50 [[0, 1], [2], [3]] output_shape [1, 16, 1, 256] : tensor<16x1x256xbf16> into tensor<1x16x1x256xbf16>
    %51 = tensor.empty() : tensor<1x1x16x256xbf16>
    %transposed_30 = linalg.transpose ins(%expanded_29 : tensor<1x16x1x256xbf16>) outs(%51 : tensor<1x1x16x256xbf16>) permutation = [0, 2, 1, 3] 
    %52 = tensor.empty() : tensor<4096x3072xbf16>
    %transposed_31 = linalg.transpose ins(%arg9 : tensor<3072x4096xbf16>) outs(%52 : tensor<4096x3072xbf16>) permutation = [1, 0] 
    %collapsed_32 = tensor.collapse_shape %transposed_30 [[0, 1, 2, 3]] : tensor<1x1x16x256xbf16> into tensor<4096xbf16>
    %53 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapsed_32 : tensor<4096xbf16>) outs(%5 : tensor<1x1x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x1x4096xbf16>
    %54 = tensor.empty() : tensor<1x4096x3072xbf16>
    %55 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_31 : tensor<4096x3072xbf16>) outs(%54 : tensor<1x4096x3072xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x4096x3072xbf16>
    %56 = linalg.fill ins(%cst : bf16) outs(%1 : tensor<1x1x3072xbf16>) -> tensor<1x1x3072xbf16>
    %57 = linalg.batch_matmul ins(%53, %55 : tensor<1x1x4096xbf16>, tensor<1x4096x3072xbf16>) outs(%56 : tensor<1x1x3072xbf16>) -> tensor<1x1x3072xbf16>
    return %57 : tensor<1x1x3072xbf16>
  }
}
