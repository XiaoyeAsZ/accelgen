#map = affine_map<(d0, d1, d2) -> (d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> (0, 0, d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (0, d1, 0, 0)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>
module {
  func.func @main(%arg0: tensor<8x1024x4096xbf16>, %arg1: tensor<1x1024x128xbf16>, %arg2: tensor<1x1024x128xbf16>, %arg3: tensor<1x32x1x1xbf16>, %arg4: tensor<4096x4096xbf16>, %arg5: tensor<4096x4096xbf16>, %arg6: tensor<4096x4096xbf16>, %arg7: tensor<4096x4096xbf16>) -> (tensor<8x1024x4096xbf16>, tensor<8x32x1024x1024xbf16>) {
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
    %3 = tensor.empty() : tensor<8x1024x4096xbf16>
    %4 = linalg.fill ins(%cst : bf16) outs(%3 : tensor<8x1024x4096xbf16>) -> tensor<8x1024x4096xbf16>
    %5 = linalg.batch_matmul ins(%arg0, %2 : tensor<8x1024x4096xbf16>, tensor<8x4096x4096xbf16>) outs(%4 : tensor<8x1024x4096xbf16>) -> tensor<8x1024x4096xbf16>
    %expanded = tensor.expand_shape %5 [[0], [1], [2, 3]] output_shape [8, 1024, 32, 128] : tensor<8x1024x4096xbf16> into tensor<8x1024x32x128xbf16>
    %6 = tensor.empty() : tensor<8x32x1024x128xbf16>
    %transposed_3 = linalg.transpose ins(%expanded : tensor<8x1024x32x128xbf16>) outs(%6 : tensor<8x32x1024x128xbf16>) permutation = [0, 2, 1, 3] 
    %transposed_4 = linalg.transpose ins(%arg5 : tensor<4096x4096xbf16>) outs(%0 : tensor<4096x4096xbf16>) permutation = [1, 0] 
    %7 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_4 : tensor<4096x4096xbf16>) outs(%1 : tensor<8x4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x4096x4096xbf16>
    %8 = linalg.batch_matmul ins(%arg0, %7 : tensor<8x1024x4096xbf16>, tensor<8x4096x4096xbf16>) outs(%4 : tensor<8x1024x4096xbf16>) -> tensor<8x1024x4096xbf16>
    %expanded_5 = tensor.expand_shape %8 [[0], [1], [2, 3]] output_shape [8, 1024, 32, 128] : tensor<8x1024x4096xbf16> into tensor<8x1024x32x128xbf16>
    %transposed_6 = linalg.transpose ins(%expanded_5 : tensor<8x1024x32x128xbf16>) outs(%6 : tensor<8x32x1024x128xbf16>) permutation = [0, 2, 1, 3] 
    %transposed_7 = linalg.transpose ins(%arg6 : tensor<4096x4096xbf16>) outs(%0 : tensor<4096x4096xbf16>) permutation = [1, 0] 
    %9 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_7 : tensor<4096x4096xbf16>) outs(%1 : tensor<8x4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x4096x4096xbf16>
    %10 = linalg.batch_matmul ins(%arg0, %9 : tensor<8x1024x4096xbf16>, tensor<8x4096x4096xbf16>) outs(%4 : tensor<8x1024x4096xbf16>) -> tensor<8x1024x4096xbf16>
    %expanded_8 = tensor.expand_shape %10 [[0], [1], [2, 3]] output_shape [8, 1024, 32, 128] : tensor<8x1024x4096xbf16> into tensor<8x1024x32x128xbf16>
    %transposed_9 = linalg.transpose ins(%expanded_8 : tensor<8x1024x32x128xbf16>) outs(%6 : tensor<8x32x1024x128xbf16>) permutation = [0, 2, 1, 3] 
    %expanded_10 = tensor.expand_shape %arg1 [[0], [1, 2], [3]] output_shape [1, 1, 1024, 128] : tensor<1x1024x128xbf16> into tensor<1x1x1024x128xbf16>
    %expanded_11 = tensor.expand_shape %arg2 [[0], [1, 2], [3]] output_shape [1, 1, 1024, 128] : tensor<1x1024x128xbf16> into tensor<1x1x1024x128xbf16>
    %11 = linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_3, %expanded_10 : tensor<8x32x1024x128xbf16>, tensor<1x1x1024x128xbf16>) outs(%6 : tensor<8x32x1024x128xbf16>) {
    ^bb0(%in: bf16, %in_26: bf16, %out: bf16):
      %47 = arith.mulf %in, %in_26 : bf16
      linalg.yield %47 : bf16
    } -> tensor<8x32x1024x128xbf16>
    %extracted_slice = tensor.extract_slice %transposed_3[0, 0, 0, 0] [8, 32, 1024, 64] [1, 1, 1, 1] : tensor<8x32x1024x128xbf16> to tensor<8x32x1024x64xbf16>
    %extracted_slice_12 = tensor.extract_slice %transposed_3[0, 0, 0, 64] [8, 32, 1024, 64] [1, 1, 1, 1] : tensor<8x32x1024x128xbf16> to tensor<8x32x1024x64xbf16>
    %12 = tensor.empty() : tensor<8x32x1024x64xbf16>
    %13 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_12 : tensor<8x32x1024x64xbf16>) outs(%12 : tensor<8x32x1024x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %47 = arith.negf %in : bf16
      linalg.yield %47 : bf16
    } -> tensor<8x32x1024x64xbf16>
    %concat = tensor.concat dim(3) %13, %extracted_slice : (tensor<8x32x1024x64xbf16>, tensor<8x32x1024x64xbf16>) -> tensor<8x32x1024x128xbf16>
    %14 = linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat, %expanded_11 : tensor<8x32x1024x128xbf16>, tensor<1x1x1024x128xbf16>) outs(%6 : tensor<8x32x1024x128xbf16>) {
    ^bb0(%in: bf16, %in_26: bf16, %out: bf16):
      %47 = arith.mulf %in, %in_26 : bf16
      linalg.yield %47 : bf16
    } -> tensor<8x32x1024x128xbf16>
    %15 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%11, %14 : tensor<8x32x1024x128xbf16>, tensor<8x32x1024x128xbf16>) outs(%6 : tensor<8x32x1024x128xbf16>) {
    ^bb0(%in: bf16, %in_26: bf16, %out: bf16):
      %47 = arith.addf %in, %in_26 : bf16
      linalg.yield %47 : bf16
    } -> tensor<8x32x1024x128xbf16>
    %16 = linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_6, %expanded_10 : tensor<8x32x1024x128xbf16>, tensor<1x1x1024x128xbf16>) outs(%6 : tensor<8x32x1024x128xbf16>) {
    ^bb0(%in: bf16, %in_26: bf16, %out: bf16):
      %47 = arith.mulf %in, %in_26 : bf16
      linalg.yield %47 : bf16
    } -> tensor<8x32x1024x128xbf16>
    %extracted_slice_13 = tensor.extract_slice %transposed_6[0, 0, 0, 0] [8, 32, 1024, 64] [1, 1, 1, 1] : tensor<8x32x1024x128xbf16> to tensor<8x32x1024x64xbf16>
    %extracted_slice_14 = tensor.extract_slice %transposed_6[0, 0, 0, 64] [8, 32, 1024, 64] [1, 1, 1, 1] : tensor<8x32x1024x128xbf16> to tensor<8x32x1024x64xbf16>
    %17 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_14 : tensor<8x32x1024x64xbf16>) outs(%12 : tensor<8x32x1024x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %47 = arith.negf %in : bf16
      linalg.yield %47 : bf16
    } -> tensor<8x32x1024x64xbf16>
    %concat_15 = tensor.concat dim(3) %17, %extracted_slice_13 : (tensor<8x32x1024x64xbf16>, tensor<8x32x1024x64xbf16>) -> tensor<8x32x1024x128xbf16>
    %18 = linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_15, %expanded_11 : tensor<8x32x1024x128xbf16>, tensor<1x1x1024x128xbf16>) outs(%6 : tensor<8x32x1024x128xbf16>) {
    ^bb0(%in: bf16, %in_26: bf16, %out: bf16):
      %47 = arith.mulf %in, %in_26 : bf16
      linalg.yield %47 : bf16
    } -> tensor<8x32x1024x128xbf16>
    %19 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%16, %18 : tensor<8x32x1024x128xbf16>, tensor<8x32x1024x128xbf16>) outs(%6 : tensor<8x32x1024x128xbf16>) {
    ^bb0(%in: bf16, %in_26: bf16, %out: bf16):
      %47 = arith.addf %in, %in_26 : bf16
      linalg.yield %47 : bf16
    } -> tensor<8x32x1024x128xbf16>
    %20 = tensor.empty() : tensor<8x32x128x1024xbf16>
    %transposed_16 = linalg.transpose ins(%19 : tensor<8x32x1024x128xbf16>) outs(%20 : tensor<8x32x128x1024xbf16>) permutation = [0, 1, 3, 2] 
    %collapsed = tensor.collapse_shape %15 [[0, 1], [2], [3]] : tensor<8x32x1024x128xbf16> into tensor<256x1024x128xbf16>
    %collapsed_17 = tensor.collapse_shape %transposed_16 [[0, 1], [2], [3]] : tensor<8x32x128x1024xbf16> into tensor<256x128x1024xbf16>
    %21 = tensor.empty() : tensor<256x1024x1024xbf16>
    %22 = linalg.fill ins(%cst : bf16) outs(%21 : tensor<256x1024x1024xbf16>) -> tensor<256x1024x1024xbf16>
    %23 = linalg.batch_matmul ins(%collapsed, %collapsed_17 : tensor<256x1024x128xbf16>, tensor<256x128x1024xbf16>) outs(%22 : tensor<256x1024x1024xbf16>) -> tensor<256x1024x1024xbf16>
    %expanded_18 = tensor.expand_shape %23 [[0, 1], [2], [3]] output_shape [8, 32, 1024, 1024] : tensor<256x1024x1024xbf16> into tensor<8x32x1024x1024xbf16>
    %24 = tensor.empty() : tensor<8x32x1024x1024xbf16>
    %25 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_18 : tensor<8x32x1024x1024xbf16>) outs(%24 : tensor<8x32x1024x1024xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %47 = arith.truncf %cst_2 : f64 to bf16
      %48 = arith.mulf %in, %47 : bf16
      linalg.yield %48 : bf16
    } -> tensor<8x32x1024x1024xbf16>
    %26 = linalg.generic {indexing_maps = [#map2, #map4, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%25, %arg3 : tensor<8x32x1024x1024xbf16>, tensor<1x32x1x1xbf16>) outs(%24 : tensor<8x32x1024x1024xbf16>) {
    ^bb0(%in: bf16, %in_26: bf16, %out: bf16):
      %47 = arith.addf %in, %in_26 : bf16
      linalg.yield %47 : bf16
    } -> tensor<8x32x1024x1024xbf16>
    %27 = tensor.empty() : tensor<8x32x1024x1024xf32>
    %28 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%26 : tensor<8x32x1024x1024xbf16>) outs(%27 : tensor<8x32x1024x1024xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %47 = arith.extf %in : bf16 to f32
      linalg.yield %47 : f32
    } -> tensor<8x32x1024x1024xf32>
    %29 = tensor.empty() : tensor<8x32x1024xi64>
    %30 = linalg.fill ins(%c0_i64 : i64) outs(%29 : tensor<8x32x1024xi64>) -> tensor<8x32x1024xi64>
    %31 = tensor.empty() : tensor<8x32x1024xf32>
    %32 = linalg.fill ins(%cst_0 : f32) outs(%31 : tensor<8x32x1024xf32>) -> tensor<8x32x1024xf32>
    %33:2 = linalg.generic {indexing_maps = [#map2, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%28 : tensor<8x32x1024x1024xf32>) outs(%32, %30 : tensor<8x32x1024xf32>, tensor<8x32x1024xi64>) {
    ^bb0(%in: f32, %out: f32, %out_26: i64):
      %47 = linalg.index 3 : index
      %48 = arith.index_cast %47 : index to i64
      %49 = arith.maximumf %in, %out : f32
      %50 = arith.cmpf ogt, %in, %out : f32
      %51 = arith.select %50, %48, %out_26 : i64
      linalg.yield %49, %51 : f32, i64
    } -> (tensor<8x32x1024xf32>, tensor<8x32x1024xi64>)
    %expanded_19 = tensor.expand_shape %33#0 [[0], [1], [2, 3]] output_shape [8, 32, 1024, 1] : tensor<8x32x1024xf32> into tensor<8x32x1024x1xf32>
    %34 = linalg.generic {indexing_maps = [#map2, #map6, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%28, %expanded_19 : tensor<8x32x1024x1024xf32>, tensor<8x32x1024x1xf32>) outs(%27 : tensor<8x32x1024x1024xf32>) {
    ^bb0(%in: f32, %in_26: f32, %out: f32):
      %47 = arith.subf %in, %in_26 : f32
      linalg.yield %47 : f32
    } -> tensor<8x32x1024x1024xf32>
    %35 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%34 : tensor<8x32x1024x1024xf32>) outs(%27 : tensor<8x32x1024x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %47 = math.exp %in : f32
      linalg.yield %47 : f32
    } -> tensor<8x32x1024x1024xf32>
    %36 = tensor.empty() : tensor<8x32x1024x1xf32>
    %37 = linalg.fill ins(%cst_1 : f32) outs(%36 : tensor<8x32x1024x1xf32>) -> tensor<8x32x1024x1xf32>
    %38 = linalg.generic {indexing_maps = [#map2, #map6], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%35 : tensor<8x32x1024x1024xf32>) outs(%37 : tensor<8x32x1024x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %47 = arith.addf %in, %out : f32
      linalg.yield %47 : f32
    } -> tensor<8x32x1024x1xf32>
    %39 = linalg.generic {indexing_maps = [#map2, #map6, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%35, %38 : tensor<8x32x1024x1024xf32>, tensor<8x32x1024x1xf32>) outs(%27 : tensor<8x32x1024x1024xf32>) {
    ^bb0(%in: f32, %in_26: f32, %out: f32):
      %47 = arith.divf %in, %in_26 : f32
      linalg.yield %47 : f32
    } -> tensor<8x32x1024x1024xf32>
    %40 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%39 : tensor<8x32x1024x1024xf32>) outs(%24 : tensor<8x32x1024x1024xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %47 = arith.truncf %in : f32 to bf16
      linalg.yield %47 : bf16
    } -> tensor<8x32x1024x1024xbf16>
    %collapsed_20 = tensor.collapse_shape %40 [[0, 1], [2], [3]] : tensor<8x32x1024x1024xbf16> into tensor<256x1024x1024xbf16>
    %collapsed_21 = tensor.collapse_shape %transposed_9 [[0, 1], [2], [3]] : tensor<8x32x1024x128xbf16> into tensor<256x1024x128xbf16>
    %41 = tensor.empty() : tensor<256x1024x128xbf16>
    %42 = linalg.fill ins(%cst : bf16) outs(%41 : tensor<256x1024x128xbf16>) -> tensor<256x1024x128xbf16>
    %43 = linalg.batch_matmul ins(%collapsed_20, %collapsed_21 : tensor<256x1024x1024xbf16>, tensor<256x1024x128xbf16>) outs(%42 : tensor<256x1024x128xbf16>) -> tensor<256x1024x128xbf16>
    %expanded_22 = tensor.expand_shape %43 [[0, 1], [2], [3]] output_shape [8, 32, 1024, 128] : tensor<256x1024x128xbf16> into tensor<8x32x1024x128xbf16>
    %44 = tensor.empty() : tensor<8x1024x32x128xbf16>
    %transposed_23 = linalg.transpose ins(%expanded_22 : tensor<8x32x1024x128xbf16>) outs(%44 : tensor<8x1024x32x128xbf16>) permutation = [0, 2, 1, 3] 
    %collapsed_24 = tensor.collapse_shape %transposed_23 [[0], [1], [2, 3]] : tensor<8x1024x32x128xbf16> into tensor<8x1024x4096xbf16>
    %transposed_25 = linalg.transpose ins(%arg7 : tensor<4096x4096xbf16>) outs(%0 : tensor<4096x4096xbf16>) permutation = [1, 0] 
    %45 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_25 : tensor<4096x4096xbf16>) outs(%1 : tensor<8x4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x4096x4096xbf16>
    %46 = linalg.batch_matmul ins(%collapsed_24, %45 : tensor<8x1024x4096xbf16>, tensor<8x4096x4096xbf16>) outs(%4 : tensor<8x1024x4096xbf16>) -> tensor<8x1024x4096xbf16>
    return %46, %40 : tensor<8x1024x4096xbf16>, tensor<8x32x1024x1024xbf16>
  }
}
