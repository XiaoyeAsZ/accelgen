#map = affine_map<(d0, d1, d2) -> (d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, 0)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>
module {
  func.func @main(%arg0: tensor<1x256x3072xbf16>, %arg1: tensor<1x256x256xbf16>, %arg2: tensor<1x256x256xbf16>, %arg3: tensor<1x16x1x1xbf16>, %arg4: tensor<4096x3072xbf16>, %arg5: tensor<4096x3072xbf16>, %arg6: tensor<4096x3072xbf16>, %arg7: tensor<3072x4096xbf16>) -> tensor<1x256x3072xbf16> {
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant 0.000000e+00 : bf16
    %cst_0 = arith.constant 0xFF800000 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %cst_2 = arith.constant 6.250000e-02 : bf16
    %0 = tensor.empty() : tensor<3072x4096xbf16>
    %transposed = linalg.transpose ins(%arg4 : tensor<4096x3072xbf16>) outs(%0 : tensor<3072x4096xbf16>) permutation = [1, 0] 
    %1 = tensor.empty() : tensor<1x256x3072xbf16>
    %collapsed = tensor.collapse_shape %arg0 [[0, 1], [2]] : tensor<1x256x3072xbf16> into tensor<256x3072xbf16>
    %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapsed : tensor<256x3072xbf16>) outs(%1 : tensor<1x256x3072xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x256x3072xbf16>
    %3 = tensor.empty() : tensor<1x3072x4096xbf16>
    %4 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed : tensor<3072x4096xbf16>) outs(%3 : tensor<1x3072x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x3072x4096xbf16>
    %5 = tensor.empty() : tensor<1x256x4096xbf16>
    %6 = linalg.fill ins(%cst : bf16) outs(%5 : tensor<1x256x4096xbf16>) -> tensor<1x256x4096xbf16>
    %7 = linalg.batch_matmul ins(%2, %4 : tensor<1x256x3072xbf16>, tensor<1x3072x4096xbf16>) outs(%6 : tensor<1x256x4096xbf16>) -> tensor<1x256x4096xbf16>
    %expanded = tensor.expand_shape %7 [[0], [1], [2, 3]] output_shape [1, 256, 16, 256] : tensor<1x256x4096xbf16> into tensor<1x256x16x256xbf16>
    %8 = tensor.empty() : tensor<1x16x256x256xbf16>
    %transposed_3 = linalg.transpose ins(%expanded : tensor<1x256x16x256xbf16>) outs(%8 : tensor<1x16x256x256xbf16>) permutation = [0, 2, 1, 3] 
    %transposed_4 = linalg.transpose ins(%arg5 : tensor<4096x3072xbf16>) outs(%0 : tensor<3072x4096xbf16>) permutation = [1, 0] 
    %9 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_4 : tensor<3072x4096xbf16>) outs(%3 : tensor<1x3072x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x3072x4096xbf16>
    %10 = linalg.batch_matmul ins(%2, %9 : tensor<1x256x3072xbf16>, tensor<1x3072x4096xbf16>) outs(%6 : tensor<1x256x4096xbf16>) -> tensor<1x256x4096xbf16>
    %expanded_5 = tensor.expand_shape %10 [[0], [1], [2, 3]] output_shape [1, 256, 16, 256] : tensor<1x256x4096xbf16> into tensor<1x256x16x256xbf16>
    %transposed_6 = linalg.transpose ins(%expanded_5 : tensor<1x256x16x256xbf16>) outs(%8 : tensor<1x16x256x256xbf16>) permutation = [0, 2, 1, 3] 
    %transposed_7 = linalg.transpose ins(%arg6 : tensor<4096x3072xbf16>) outs(%0 : tensor<3072x4096xbf16>) permutation = [1, 0] 
    %11 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_7 : tensor<3072x4096xbf16>) outs(%3 : tensor<1x3072x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x3072x4096xbf16>
    %12 = linalg.batch_matmul ins(%2, %11 : tensor<1x256x3072xbf16>, tensor<1x3072x4096xbf16>) outs(%6 : tensor<1x256x4096xbf16>) -> tensor<1x256x4096xbf16>
    %expanded_8 = tensor.expand_shape %12 [[0], [1], [2, 3]] output_shape [1, 256, 16, 256] : tensor<1x256x4096xbf16> into tensor<1x256x16x256xbf16>
    %transposed_9 = linalg.transpose ins(%expanded_8 : tensor<1x256x16x256xbf16>) outs(%8 : tensor<1x16x256x256xbf16>) permutation = [0, 2, 1, 3] 
    %expanded_10 = tensor.expand_shape %arg1 [[0], [1, 2], [3]] output_shape [1, 1, 256, 256] : tensor<1x256x256xbf16> into tensor<1x1x256x256xbf16>
    %expanded_11 = tensor.expand_shape %arg2 [[0], [1, 2], [3]] output_shape [1, 1, 256, 256] : tensor<1x256x256xbf16> into tensor<1x1x256x256xbf16>
    %13 = linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_3, %expanded_10 : tensor<1x16x256x256xbf16>, tensor<1x1x256x256xbf16>) outs(%8 : tensor<1x16x256x256xbf16>) {
    ^bb0(%in: bf16, %in_31: bf16, %out: bf16):
      %53 = arith.mulf %in, %in_31 : bf16
      linalg.yield %53 : bf16
    } -> tensor<1x16x256x256xbf16>
    %extracted_slice = tensor.extract_slice %transposed_3[0, 0, 0, 0] [1, 16, 256, 128] [1, 1, 1, 1] : tensor<1x16x256x256xbf16> to tensor<1x16x256x128xbf16>
    %extracted_slice_12 = tensor.extract_slice %transposed_3[0, 0, 0, 128] [1, 16, 256, 128] [1, 1, 1, 1] : tensor<1x16x256x256xbf16> to tensor<1x16x256x128xbf16>
    %14 = tensor.empty() : tensor<1x16x256x128xbf16>
    %15 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_12 : tensor<1x16x256x128xbf16>) outs(%14 : tensor<1x16x256x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %53 = arith.negf %in : bf16
      linalg.yield %53 : bf16
    } -> tensor<1x16x256x128xbf16>
    %concat = tensor.concat dim(3) %15, %extracted_slice : (tensor<1x16x256x128xbf16>, tensor<1x16x256x128xbf16>) -> tensor<1x16x256x256xbf16>
    %16 = linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat, %expanded_11 : tensor<1x16x256x256xbf16>, tensor<1x1x256x256xbf16>) outs(%8 : tensor<1x16x256x256xbf16>) {
    ^bb0(%in: bf16, %in_31: bf16, %out: bf16):
      %53 = arith.mulf %in, %in_31 : bf16
      linalg.yield %53 : bf16
    } -> tensor<1x16x256x256xbf16>
    %17 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%13, %16 : tensor<1x16x256x256xbf16>, tensor<1x16x256x256xbf16>) outs(%8 : tensor<1x16x256x256xbf16>) {
    ^bb0(%in: bf16, %in_31: bf16, %out: bf16):
      %53 = arith.addf %in, %in_31 : bf16
      linalg.yield %53 : bf16
    } -> tensor<1x16x256x256xbf16>
    %18 = linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_6, %expanded_10 : tensor<1x16x256x256xbf16>, tensor<1x1x256x256xbf16>) outs(%8 : tensor<1x16x256x256xbf16>) {
    ^bb0(%in: bf16, %in_31: bf16, %out: bf16):
      %53 = arith.mulf %in, %in_31 : bf16
      linalg.yield %53 : bf16
    } -> tensor<1x16x256x256xbf16>
    %extracted_slice_13 = tensor.extract_slice %transposed_6[0, 0, 0, 0] [1, 16, 256, 128] [1, 1, 1, 1] : tensor<1x16x256x256xbf16> to tensor<1x16x256x128xbf16>
    %extracted_slice_14 = tensor.extract_slice %transposed_6[0, 0, 0, 128] [1, 16, 256, 128] [1, 1, 1, 1] : tensor<1x16x256x256xbf16> to tensor<1x16x256x128xbf16>
    %19 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_14 : tensor<1x16x256x128xbf16>) outs(%14 : tensor<1x16x256x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %53 = arith.negf %in : bf16
      linalg.yield %53 : bf16
    } -> tensor<1x16x256x128xbf16>
    %concat_15 = tensor.concat dim(3) %19, %extracted_slice_13 : (tensor<1x16x256x128xbf16>, tensor<1x16x256x128xbf16>) -> tensor<1x16x256x256xbf16>
    %20 = linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_15, %expanded_11 : tensor<1x16x256x256xbf16>, tensor<1x1x256x256xbf16>) outs(%8 : tensor<1x16x256x256xbf16>) {
    ^bb0(%in: bf16, %in_31: bf16, %out: bf16):
      %53 = arith.mulf %in, %in_31 : bf16
      linalg.yield %53 : bf16
    } -> tensor<1x16x256x256xbf16>
    %21 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%18, %20 : tensor<1x16x256x256xbf16>, tensor<1x16x256x256xbf16>) outs(%8 : tensor<1x16x256x256xbf16>) {
    ^bb0(%in: bf16, %in_31: bf16, %out: bf16):
      %53 = arith.addf %in, %in_31 : bf16
      linalg.yield %53 : bf16
    } -> tensor<1x16x256x256xbf16>
    %transposed_16 = linalg.transpose ins(%21 : tensor<1x16x256x256xbf16>) outs(%8 : tensor<1x16x256x256xbf16>) permutation = [0, 1, 3, 2] 
    %collapsed_17 = tensor.collapse_shape %17 [[0, 1], [2], [3]] : tensor<1x16x256x256xbf16> into tensor<16x256x256xbf16>
    %22 = linalg.generic {indexing_maps = [#map4, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%collapsed_17 : tensor<16x256x256xbf16>) outs(%8 : tensor<1x16x256x256xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x16x256x256xbf16>
    %collapsed_18 = tensor.collapse_shape %transposed_16 [[0, 1], [2], [3]] : tensor<1x16x256x256xbf16> into tensor<16x256x256xbf16>
    %23 = linalg.generic {indexing_maps = [#map4, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%collapsed_18 : tensor<16x256x256xbf16>) outs(%8 : tensor<1x16x256x256xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x16x256x256xbf16>
    %collapsed_19 = tensor.collapse_shape %22 [[0, 1], [2], [3]] : tensor<1x16x256x256xbf16> into tensor<16x256x256xbf16>
    %collapsed_20 = tensor.collapse_shape %23 [[0, 1], [2], [3]] : tensor<1x16x256x256xbf16> into tensor<16x256x256xbf16>
    %24 = tensor.empty() : tensor<16x256x256xbf16>
    %25 = linalg.fill ins(%cst : bf16) outs(%24 : tensor<16x256x256xbf16>) -> tensor<16x256x256xbf16>
    %26 = linalg.batch_matmul ins(%collapsed_19, %collapsed_20 : tensor<16x256x256xbf16>, tensor<16x256x256xbf16>) outs(%25 : tensor<16x256x256xbf16>) -> tensor<16x256x256xbf16>
    %expanded_21 = tensor.expand_shape %26 [[0, 1], [2], [3]] output_shape [1, 16, 256, 256] : tensor<16x256x256xbf16> into tensor<1x16x256x256xbf16>
    %27 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_21 : tensor<1x16x256x256xbf16>) outs(%8 : tensor<1x16x256x256xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %53 = arith.mulf %in, %cst_2 : bf16
      linalg.yield %53 : bf16
    } -> tensor<1x16x256x256xbf16>
    %28 = linalg.generic {indexing_maps = [#map2, #map5, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%27, %arg3 : tensor<1x16x256x256xbf16>, tensor<1x16x1x1xbf16>) outs(%8 : tensor<1x16x256x256xbf16>) {
    ^bb0(%in: bf16, %in_31: bf16, %out: bf16):
      %53 = arith.addf %in, %in_31 : bf16
      linalg.yield %53 : bf16
    } -> tensor<1x16x256x256xbf16>
    %29 = tensor.empty() : tensor<1x16x256x256xf32>
    %30 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%28 : tensor<1x16x256x256xbf16>) outs(%29 : tensor<1x16x256x256xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %53 = arith.extf %in : bf16 to f32
      linalg.yield %53 : f32
    } -> tensor<1x16x256x256xf32>
    %31 = tensor.empty() : tensor<1x16x256xi64>
    %32 = linalg.fill ins(%c0_i64 : i64) outs(%31 : tensor<1x16x256xi64>) -> tensor<1x16x256xi64>
    %33 = tensor.empty() : tensor<1x16x256xf32>
    %34 = linalg.fill ins(%cst_0 : f32) outs(%33 : tensor<1x16x256xf32>) -> tensor<1x16x256xf32>
    %35:2 = linalg.generic {indexing_maps = [#map2, #map6, #map6], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%30 : tensor<1x16x256x256xf32>) outs(%34, %32 : tensor<1x16x256xf32>, tensor<1x16x256xi64>) {
    ^bb0(%in: f32, %out: f32, %out_31: i64):
      %53 = linalg.index 3 : index
      %54 = arith.index_cast %53 : index to i64
      %55 = arith.maximumf %in, %out : f32
      %56 = arith.cmpf ogt, %in, %out : f32
      %57 = arith.select %56, %54, %out_31 : i64
      linalg.yield %55, %57 : f32, i64
    } -> (tensor<1x16x256xf32>, tensor<1x16x256xi64>)
    %expanded_22 = tensor.expand_shape %35#0 [[0], [1], [2, 3]] output_shape [1, 16, 256, 1] : tensor<1x16x256xf32> into tensor<1x16x256x1xf32>
    %36 = linalg.generic {indexing_maps = [#map2, #map7, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%30, %expanded_22 : tensor<1x16x256x256xf32>, tensor<1x16x256x1xf32>) outs(%29 : tensor<1x16x256x256xf32>) {
    ^bb0(%in: f32, %in_31: f32, %out: f32):
      %53 = arith.subf %in, %in_31 : f32
      linalg.yield %53 : f32
    } -> tensor<1x16x256x256xf32>
    %37 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%36 : tensor<1x16x256x256xf32>) outs(%29 : tensor<1x16x256x256xf32>) {
    ^bb0(%in: f32, %out: f32):
      %53 = math.exp %in : f32
      linalg.yield %53 : f32
    } -> tensor<1x16x256x256xf32>
    %38 = tensor.empty() : tensor<1x16x256x1xf32>
    %39 = linalg.fill ins(%cst_1 : f32) outs(%38 : tensor<1x16x256x1xf32>) -> tensor<1x16x256x1xf32>
    %40 = linalg.generic {indexing_maps = [#map2, #map7], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%37 : tensor<1x16x256x256xf32>) outs(%39 : tensor<1x16x256x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %53 = arith.addf %in, %out : f32
      linalg.yield %53 : f32
    } -> tensor<1x16x256x1xf32>
    %41 = linalg.generic {indexing_maps = [#map2, #map7, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%37, %40 : tensor<1x16x256x256xf32>, tensor<1x16x256x1xf32>) outs(%29 : tensor<1x16x256x256xf32>) {
    ^bb0(%in: f32, %in_31: f32, %out: f32):
      %53 = arith.divf %in, %in_31 : f32
      linalg.yield %53 : f32
    } -> tensor<1x16x256x256xf32>
    %42 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%41 : tensor<1x16x256x256xf32>) outs(%8 : tensor<1x16x256x256xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %53 = arith.truncf %in : f32 to bf16
      linalg.yield %53 : bf16
    } -> tensor<1x16x256x256xbf16>
    %collapsed_23 = tensor.collapse_shape %42 [[0, 1], [2], [3]] : tensor<1x16x256x256xbf16> into tensor<16x256x256xbf16>
    %43 = linalg.generic {indexing_maps = [#map4, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%collapsed_23 : tensor<16x256x256xbf16>) outs(%8 : tensor<1x16x256x256xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x16x256x256xbf16>
    %collapsed_24 = tensor.collapse_shape %transposed_9 [[0, 1], [2], [3]] : tensor<1x16x256x256xbf16> into tensor<16x256x256xbf16>
    %44 = linalg.generic {indexing_maps = [#map4, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%collapsed_24 : tensor<16x256x256xbf16>) outs(%8 : tensor<1x16x256x256xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x16x256x256xbf16>
    %collapsed_25 = tensor.collapse_shape %43 [[0, 1], [2], [3]] : tensor<1x16x256x256xbf16> into tensor<16x256x256xbf16>
    %collapsed_26 = tensor.collapse_shape %44 [[0, 1], [2], [3]] : tensor<1x16x256x256xbf16> into tensor<16x256x256xbf16>
    %45 = linalg.batch_matmul ins(%collapsed_25, %collapsed_26 : tensor<16x256x256xbf16>, tensor<16x256x256xbf16>) outs(%25 : tensor<16x256x256xbf16>) -> tensor<16x256x256xbf16>
    %expanded_27 = tensor.expand_shape %45 [[0, 1], [2], [3]] output_shape [1, 16, 256, 256] : tensor<16x256x256xbf16> into tensor<1x16x256x256xbf16>
    %46 = tensor.empty() : tensor<1x256x16x256xbf16>
    %transposed_28 = linalg.transpose ins(%expanded_27 : tensor<1x16x256x256xbf16>) outs(%46 : tensor<1x256x16x256xbf16>) permutation = [0, 2, 1, 3] 
    %47 = tensor.empty() : tensor<4096x3072xbf16>
    %transposed_29 = linalg.transpose ins(%arg7 : tensor<3072x4096xbf16>) outs(%47 : tensor<4096x3072xbf16>) permutation = [1, 0] 
    %collapsed_30 = tensor.collapse_shape %transposed_28 [[0, 1], [2, 3]] : tensor<1x256x16x256xbf16> into tensor<256x4096xbf16>
    %48 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapsed_30 : tensor<256x4096xbf16>) outs(%5 : tensor<1x256x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x256x4096xbf16>
    %49 = tensor.empty() : tensor<1x4096x3072xbf16>
    %50 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_29 : tensor<4096x3072xbf16>) outs(%49 : tensor<1x4096x3072xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x4096x3072xbf16>
    %51 = linalg.fill ins(%cst : bf16) outs(%1 : tensor<1x256x3072xbf16>) -> tensor<1x256x3072xbf16>
    %52 = linalg.batch_matmul ins(%48, %50 : tensor<1x256x4096xbf16>, tensor<1x4096x3072xbf16>) outs(%51 : tensor<1x256x3072xbf16>) -> tensor<1x256x3072xbf16>
    return %52 : tensor<1x256x3072xbf16>
  }
}
