#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (0, 0, d2, d3)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map6 = affine_map<(d0, d1, d2, d3) -> (0, d1, d2, 0)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map8 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>
module {
  func.func @main(%arg0: tensor<8x1x4096xbf16>, %arg1: tensor<1x1x128xbf16>, %arg2: tensor<1x1x128xbf16>, %arg3: tensor<1x32x1x1xbf16>, %arg4: tensor<8x32x1023x128xbf16>, %arg5: tensor<8x32x1023x128xbf16>, %arg6: tensor<4096x4096xbf16>, %arg7: tensor<4096x4096xbf16>, %arg8: tensor<4096x4096xbf16>, %arg9: tensor<4096x4096xbf16>) -> (tensor<8x1x4096xbf16>, tensor<8x32x1x1024xbf16>) {
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant 0.000000e+00 : bf16
    %cst_0 = arith.constant 0xFF800000 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %cst_2 = arith.constant 0.088388347648318447 : f64
    %0 = tensor.empty() : tensor<4096x4096xbf16>
    %transposed = linalg.transpose ins(%arg6 : tensor<4096x4096xbf16>) outs(%0 : tensor<4096x4096xbf16>) permutation = [1, 0] 
    %1 = tensor.empty() : tensor<8x1x4096xbf16>
    %collapsed = tensor.collapse_shape %arg0 [[0, 1], [2]] : tensor<8x1x4096xbf16> into tensor<8x4096xbf16>
    %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapsed : tensor<8x4096xbf16>) outs(%1 : tensor<8x1x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x1x4096xbf16>
    %3 = tensor.empty() : tensor<8x4096x4096xbf16>
    %4 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed : tensor<4096x4096xbf16>) outs(%3 : tensor<8x4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x4096x4096xbf16>
    %5 = linalg.fill ins(%cst : bf16) outs(%1 : tensor<8x1x4096xbf16>) -> tensor<8x1x4096xbf16>
    %6 = linalg.batch_matmul ins(%2, %4 : tensor<8x1x4096xbf16>, tensor<8x4096x4096xbf16>) outs(%5 : tensor<8x1x4096xbf16>) -> tensor<8x1x4096xbf16>
    %expanded = tensor.expand_shape %6 [[0], [1], [2, 3]] output_shape [8, 1, 32, 128] : tensor<8x1x4096xbf16> into tensor<8x1x32x128xbf16>
    %7 = tensor.empty() : tensor<8x32x1x128xbf16>
    %transposed_3 = linalg.transpose ins(%expanded : tensor<8x1x32x128xbf16>) outs(%7 : tensor<8x32x1x128xbf16>) permutation = [0, 2, 1, 3] 
    %transposed_4 = linalg.transpose ins(%arg7 : tensor<4096x4096xbf16>) outs(%0 : tensor<4096x4096xbf16>) permutation = [1, 0] 
    %8 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_4 : tensor<4096x4096xbf16>) outs(%3 : tensor<8x4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x4096x4096xbf16>
    %9 = linalg.batch_matmul ins(%2, %8 : tensor<8x1x4096xbf16>, tensor<8x4096x4096xbf16>) outs(%5 : tensor<8x1x4096xbf16>) -> tensor<8x1x4096xbf16>
    %expanded_5 = tensor.expand_shape %9 [[0], [1], [2, 3]] output_shape [8, 1, 32, 128] : tensor<8x1x4096xbf16> into tensor<8x1x32x128xbf16>
    %transposed_6 = linalg.transpose ins(%expanded_5 : tensor<8x1x32x128xbf16>) outs(%7 : tensor<8x32x1x128xbf16>) permutation = [0, 2, 1, 3] 
    %transposed_7 = linalg.transpose ins(%arg8 : tensor<4096x4096xbf16>) outs(%0 : tensor<4096x4096xbf16>) permutation = [1, 0] 
    %10 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_7 : tensor<4096x4096xbf16>) outs(%3 : tensor<8x4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x4096x4096xbf16>
    %11 = linalg.batch_matmul ins(%2, %10 : tensor<8x1x4096xbf16>, tensor<8x4096x4096xbf16>) outs(%5 : tensor<8x1x4096xbf16>) -> tensor<8x1x4096xbf16>
    %expanded_8 = tensor.expand_shape %11 [[0], [1], [2, 3]] output_shape [8, 1, 32, 128] : tensor<8x1x4096xbf16> into tensor<8x1x32x128xbf16>
    %transposed_9 = linalg.transpose ins(%expanded_8 : tensor<8x1x32x128xbf16>) outs(%7 : tensor<8x32x1x128xbf16>) permutation = [0, 2, 1, 3] 
    %expanded_10 = tensor.expand_shape %arg1 [[0], [1, 2], [3]] output_shape [1, 1, 1, 128] : tensor<1x1x128xbf16> into tensor<1x1x1x128xbf16>
    %expanded_11 = tensor.expand_shape %arg2 [[0], [1, 2], [3]] output_shape [1, 1, 1, 128] : tensor<1x1x128xbf16> into tensor<1x1x1x128xbf16>
    %12 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_3, %expanded_10 : tensor<8x32x1x128xbf16>, tensor<1x1x1x128xbf16>) outs(%7 : tensor<8x32x1x128xbf16>) {
    ^bb0(%in: bf16, %in_31: bf16, %out: bf16):
      %51 = arith.mulf %in, %in_31 : bf16
      linalg.yield %51 : bf16
    } -> tensor<8x32x1x128xbf16>
    %extracted_slice = tensor.extract_slice %transposed_3[0, 0, 0, 0] [8, 32, 1, 64] [1, 1, 1, 1] : tensor<8x32x1x128xbf16> to tensor<8x32x1x64xbf16>
    %extracted_slice_12 = tensor.extract_slice %transposed_3[0, 0, 0, 64] [8, 32, 1, 64] [1, 1, 1, 1] : tensor<8x32x1x128xbf16> to tensor<8x32x1x64xbf16>
    %13 = tensor.empty() : tensor<8x32x1x64xbf16>
    %14 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_12 : tensor<8x32x1x64xbf16>) outs(%13 : tensor<8x32x1x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %51 = arith.negf %in : bf16
      linalg.yield %51 : bf16
    } -> tensor<8x32x1x64xbf16>
    %concat = tensor.concat dim(3) %14, %extracted_slice : (tensor<8x32x1x64xbf16>, tensor<8x32x1x64xbf16>) -> tensor<8x32x1x128xbf16>
    %15 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat, %expanded_11 : tensor<8x32x1x128xbf16>, tensor<1x1x1x128xbf16>) outs(%7 : tensor<8x32x1x128xbf16>) {
    ^bb0(%in: bf16, %in_31: bf16, %out: bf16):
      %51 = arith.mulf %in, %in_31 : bf16
      linalg.yield %51 : bf16
    } -> tensor<8x32x1x128xbf16>
    %16 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%12, %15 : tensor<8x32x1x128xbf16>, tensor<8x32x1x128xbf16>) outs(%7 : tensor<8x32x1x128xbf16>) {
    ^bb0(%in: bf16, %in_31: bf16, %out: bf16):
      %51 = arith.addf %in, %in_31 : bf16
      linalg.yield %51 : bf16
    } -> tensor<8x32x1x128xbf16>
    %17 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_6, %expanded_10 : tensor<8x32x1x128xbf16>, tensor<1x1x1x128xbf16>) outs(%7 : tensor<8x32x1x128xbf16>) {
    ^bb0(%in: bf16, %in_31: bf16, %out: bf16):
      %51 = arith.mulf %in, %in_31 : bf16
      linalg.yield %51 : bf16
    } -> tensor<8x32x1x128xbf16>
    %extracted_slice_13 = tensor.extract_slice %transposed_6[0, 0, 0, 0] [8, 32, 1, 64] [1, 1, 1, 1] : tensor<8x32x1x128xbf16> to tensor<8x32x1x64xbf16>
    %extracted_slice_14 = tensor.extract_slice %transposed_6[0, 0, 0, 64] [8, 32, 1, 64] [1, 1, 1, 1] : tensor<8x32x1x128xbf16> to tensor<8x32x1x64xbf16>
    %18 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_14 : tensor<8x32x1x64xbf16>) outs(%13 : tensor<8x32x1x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %51 = arith.negf %in : bf16
      linalg.yield %51 : bf16
    } -> tensor<8x32x1x64xbf16>
    %concat_15 = tensor.concat dim(3) %18, %extracted_slice_13 : (tensor<8x32x1x64xbf16>, tensor<8x32x1x64xbf16>) -> tensor<8x32x1x128xbf16>
    %19 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_15, %expanded_11 : tensor<8x32x1x128xbf16>, tensor<1x1x1x128xbf16>) outs(%7 : tensor<8x32x1x128xbf16>) {
    ^bb0(%in: bf16, %in_31: bf16, %out: bf16):
      %51 = arith.mulf %in, %in_31 : bf16
      linalg.yield %51 : bf16
    } -> tensor<8x32x1x128xbf16>
    %20 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%17, %19 : tensor<8x32x1x128xbf16>, tensor<8x32x1x128xbf16>) outs(%7 : tensor<8x32x1x128xbf16>) {
    ^bb0(%in: bf16, %in_31: bf16, %out: bf16):
      %51 = arith.addf %in, %in_31 : bf16
      linalg.yield %51 : bf16
    } -> tensor<8x32x1x128xbf16>
    %concat_16 = tensor.concat dim(2) %arg4, %20 : (tensor<8x32x1023x128xbf16>, tensor<8x32x1x128xbf16>) -> tensor<8x32x1024x128xbf16>
    %concat_17 = tensor.concat dim(2) %arg5, %transposed_9 : (tensor<8x32x1023x128xbf16>, tensor<8x32x1x128xbf16>) -> tensor<8x32x1024x128xbf16>
    %21 = tensor.empty() : tensor<8x32x128x1024xbf16>
    %transposed_18 = linalg.transpose ins(%concat_16 : tensor<8x32x1024x128xbf16>) outs(%21 : tensor<8x32x128x1024xbf16>) permutation = [0, 1, 3, 2] 
    %collapsed_19 = tensor.collapse_shape %16 [[0], [1, 2], [3]] : tensor<8x32x1x128xbf16> into tensor<8x32x128xbf16>
    %22 = linalg.generic {indexing_maps = [#map5, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%collapsed_19 : tensor<8x32x128xbf16>) outs(%7 : tensor<8x32x1x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x32x1x128xbf16>
    %collapsed_20 = tensor.collapse_shape %22 [[0, 1], [2], [3]] : tensor<8x32x1x128xbf16> into tensor<256x1x128xbf16>
    %collapsed_21 = tensor.collapse_shape %transposed_18 [[0, 1], [2], [3]] : tensor<8x32x128x1024xbf16> into tensor<256x128x1024xbf16>
    %23 = tensor.empty() : tensor<256x1x1024xbf16>
    %24 = linalg.fill ins(%cst : bf16) outs(%23 : tensor<256x1x1024xbf16>) -> tensor<256x1x1024xbf16>
    %25 = linalg.batch_matmul ins(%collapsed_20, %collapsed_21 : tensor<256x1x128xbf16>, tensor<256x128x1024xbf16>) outs(%24 : tensor<256x1x1024xbf16>) -> tensor<256x1x1024xbf16>
    %expanded_22 = tensor.expand_shape %25 [[0, 1], [2], [3]] output_shape [8, 32, 1, 1024] : tensor<256x1x1024xbf16> into tensor<8x32x1x1024xbf16>
    %26 = tensor.empty() : tensor<8x32x1x1024xbf16>
    %27 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_22 : tensor<8x32x1x1024xbf16>) outs(%26 : tensor<8x32x1x1024xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %51 = arith.truncf %cst_2 : f64 to bf16
      %52 = arith.mulf %in, %51 : bf16
      linalg.yield %52 : bf16
    } -> tensor<8x32x1x1024xbf16>
    %28 = linalg.generic {indexing_maps = [#map3, #map6, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%27, %arg3 : tensor<8x32x1x1024xbf16>, tensor<1x32x1x1xbf16>) outs(%26 : tensor<8x32x1x1024xbf16>) {
    ^bb0(%in: bf16, %in_31: bf16, %out: bf16):
      %51 = arith.addf %in, %in_31 : bf16
      linalg.yield %51 : bf16
    } -> tensor<8x32x1x1024xbf16>
    %29 = tensor.empty() : tensor<8x32x1x1024xf32>
    %30 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%28 : tensor<8x32x1x1024xbf16>) outs(%29 : tensor<8x32x1x1024xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %51 = arith.extf %in : bf16 to f32
      linalg.yield %51 : f32
    } -> tensor<8x32x1x1024xf32>
    %31 = tensor.empty() : tensor<8x32x1xi64>
    %32 = linalg.fill ins(%c0_i64 : i64) outs(%31 : tensor<8x32x1xi64>) -> tensor<8x32x1xi64>
    %33 = tensor.empty() : tensor<8x32x1xf32>
    %34 = linalg.fill ins(%cst_0 : f32) outs(%33 : tensor<8x32x1xf32>) -> tensor<8x32x1xf32>
    %35:2 = linalg.generic {indexing_maps = [#map3, #map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%30 : tensor<8x32x1x1024xf32>) outs(%34, %32 : tensor<8x32x1xf32>, tensor<8x32x1xi64>) {
    ^bb0(%in: f32, %out: f32, %out_31: i64):
      %51 = linalg.index 3 : index
      %52 = arith.index_cast %51 : index to i64
      %53 = arith.maximumf %in, %out : f32
      %54 = arith.cmpf ogt, %in, %out : f32
      %55 = arith.select %54, %52, %out_31 : i64
      linalg.yield %53, %55 : f32, i64
    } -> (tensor<8x32x1xf32>, tensor<8x32x1xi64>)
    %expanded_23 = tensor.expand_shape %35#0 [[0], [1], [2, 3]] output_shape [8, 32, 1, 1] : tensor<8x32x1xf32> into tensor<8x32x1x1xf32>
    %36 = linalg.generic {indexing_maps = [#map3, #map8, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%30, %expanded_23 : tensor<8x32x1x1024xf32>, tensor<8x32x1x1xf32>) outs(%29 : tensor<8x32x1x1024xf32>) {
    ^bb0(%in: f32, %in_31: f32, %out: f32):
      %51 = arith.subf %in, %in_31 : f32
      linalg.yield %51 : f32
    } -> tensor<8x32x1x1024xf32>
    %37 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%36 : tensor<8x32x1x1024xf32>) outs(%29 : tensor<8x32x1x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %51 = math.exp %in : f32
      linalg.yield %51 : f32
    } -> tensor<8x32x1x1024xf32>
    %38 = tensor.empty() : tensor<8x32x1x1xf32>
    %39 = linalg.fill ins(%cst_1 : f32) outs(%38 : tensor<8x32x1x1xf32>) -> tensor<8x32x1x1xf32>
    %40 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%37 : tensor<8x32x1x1024xf32>) outs(%39 : tensor<8x32x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %51 = arith.addf %in, %out : f32
      linalg.yield %51 : f32
    } -> tensor<8x32x1x1xf32>
    %41 = linalg.generic {indexing_maps = [#map3, #map8, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%37, %40 : tensor<8x32x1x1024xf32>, tensor<8x32x1x1xf32>) outs(%29 : tensor<8x32x1x1024xf32>) {
    ^bb0(%in: f32, %in_31: f32, %out: f32):
      %51 = arith.divf %in, %in_31 : f32
      linalg.yield %51 : f32
    } -> tensor<8x32x1x1024xf32>
    %42 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%41 : tensor<8x32x1x1024xf32>) outs(%26 : tensor<8x32x1x1024xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %51 = arith.truncf %in : f32 to bf16
      linalg.yield %51 : bf16
    } -> tensor<8x32x1x1024xbf16>
    %collapsed_24 = tensor.collapse_shape %42 [[0], [1, 2], [3]] : tensor<8x32x1x1024xbf16> into tensor<8x32x1024xbf16>
    %43 = linalg.generic {indexing_maps = [#map5, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%collapsed_24 : tensor<8x32x1024xbf16>) outs(%26 : tensor<8x32x1x1024xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x32x1x1024xbf16>
    %collapsed_25 = tensor.collapse_shape %43 [[0, 1], [2], [3]] : tensor<8x32x1x1024xbf16> into tensor<256x1x1024xbf16>
    %collapsed_26 = tensor.collapse_shape %concat_17 [[0, 1], [2], [3]] : tensor<8x32x1024x128xbf16> into tensor<256x1024x128xbf16>
    %44 = tensor.empty() : tensor<256x1x128xbf16>
    %45 = linalg.fill ins(%cst : bf16) outs(%44 : tensor<256x1x128xbf16>) -> tensor<256x1x128xbf16>
    %46 = linalg.batch_matmul ins(%collapsed_25, %collapsed_26 : tensor<256x1x1024xbf16>, tensor<256x1024x128xbf16>) outs(%45 : tensor<256x1x128xbf16>) -> tensor<256x1x128xbf16>
    %expanded_27 = tensor.expand_shape %46 [[0, 1], [2], [3]] output_shape [8, 32, 1, 128] : tensor<256x1x128xbf16> into tensor<8x32x1x128xbf16>
    %47 = tensor.empty() : tensor<8x1x32x128xbf16>
    %transposed_28 = linalg.transpose ins(%expanded_27 : tensor<8x32x1x128xbf16>) outs(%47 : tensor<8x1x32x128xbf16>) permutation = [0, 2, 1, 3] 
    %transposed_29 = linalg.transpose ins(%arg9 : tensor<4096x4096xbf16>) outs(%0 : tensor<4096x4096xbf16>) permutation = [1, 0] 
    %collapsed_30 = tensor.collapse_shape %transposed_28 [[0, 1], [2, 3]] : tensor<8x1x32x128xbf16> into tensor<8x4096xbf16>
    %48 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapsed_30 : tensor<8x4096xbf16>) outs(%1 : tensor<8x1x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x1x4096xbf16>
    %49 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_29 : tensor<4096x4096xbf16>) outs(%3 : tensor<8x4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x4096x4096xbf16>
    %50 = linalg.batch_matmul ins(%48, %49 : tensor<8x1x4096xbf16>, tensor<8x4096x4096xbf16>) outs(%5 : tensor<8x1x4096xbf16>) -> tensor<8x1x4096xbf16>
    return %50, %42 : tensor<8x1x4096xbf16>, tensor<8x32x1x1024xbf16>
  }
}
