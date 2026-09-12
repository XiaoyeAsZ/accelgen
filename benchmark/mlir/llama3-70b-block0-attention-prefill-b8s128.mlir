#map = affine_map<(d0, d1, d2) -> (d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> (0, 0, d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>
#map5 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map6 = affine_map<(d0, d1, d2, d3) -> (0, d1, 0, 0)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map8 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>
module {
  func.func @main(%arg0: tensor<8x128x8192xbf16>, %arg1: tensor<1x128x128xbf16>, %arg2: tensor<1x128x128xbf16>, %arg3: tensor<1x64x1x1xbf16>, %arg4: tensor<8192x8192xbf16>, %arg5: tensor<1024x8192xbf16>, %arg6: tensor<1024x8192xbf16>, %arg7: tensor<8192x8192xbf16>) -> (tensor<8x128x8192xbf16>, tensor<8x64x128x128xbf16>) {
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant 0.000000e+00 : bf16
    %cst_0 = arith.constant 0xFF800000 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %cst_2 = arith.constant 0.088388347648318447 : f64
    %0 = tensor.empty() : tensor<8192x8192xbf16>
    %transposed = linalg.transpose ins(%arg4 : tensor<8192x8192xbf16>) outs(%0 : tensor<8192x8192xbf16>) permutation = [1, 0] 
    %1 = tensor.empty() : tensor<8x8192x8192xbf16>
    %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed : tensor<8192x8192xbf16>) outs(%1 : tensor<8x8192x8192xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x8192x8192xbf16>
    %3 = tensor.empty() : tensor<8x128x8192xbf16>
    %4 = linalg.fill ins(%cst : bf16) outs(%3 : tensor<8x128x8192xbf16>) -> tensor<8x128x8192xbf16>
    %5 = linalg.batch_matmul ins(%arg0, %2 : tensor<8x128x8192xbf16>, tensor<8x8192x8192xbf16>) outs(%4 : tensor<8x128x8192xbf16>) -> tensor<8x128x8192xbf16>
    %expanded = tensor.expand_shape %5 [[0], [1], [2, 3]] output_shape [8, 128, 64, 128] : tensor<8x128x8192xbf16> into tensor<8x128x64x128xbf16>
    %6 = tensor.empty() : tensor<8x64x128x128xbf16>
    %transposed_3 = linalg.transpose ins(%expanded : tensor<8x128x64x128xbf16>) outs(%6 : tensor<8x64x128x128xbf16>) permutation = [0, 2, 1, 3] 
    %7 = tensor.empty() : tensor<8192x1024xbf16>
    %transposed_4 = linalg.transpose ins(%arg5 : tensor<1024x8192xbf16>) outs(%7 : tensor<8192x1024xbf16>) permutation = [1, 0] 
    %8 = tensor.empty() : tensor<8x8192x1024xbf16>
    %9 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_4 : tensor<8192x1024xbf16>) outs(%8 : tensor<8x8192x1024xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x8192x1024xbf16>
    %10 = tensor.empty() : tensor<8x128x1024xbf16>
    %11 = linalg.fill ins(%cst : bf16) outs(%10 : tensor<8x128x1024xbf16>) -> tensor<8x128x1024xbf16>
    %12 = linalg.batch_matmul ins(%arg0, %9 : tensor<8x128x8192xbf16>, tensor<8x8192x1024xbf16>) outs(%11 : tensor<8x128x1024xbf16>) -> tensor<8x128x1024xbf16>
    %expanded_5 = tensor.expand_shape %12 [[0], [1], [2, 3]] output_shape [8, 128, 8, 128] : tensor<8x128x1024xbf16> into tensor<8x128x8x128xbf16>
    %13 = tensor.empty() : tensor<8x8x128x128xbf16>
    %transposed_6 = linalg.transpose ins(%expanded_5 : tensor<8x128x8x128xbf16>) outs(%13 : tensor<8x8x128x128xbf16>) permutation = [0, 2, 1, 3] 
    %transposed_7 = linalg.transpose ins(%arg6 : tensor<1024x8192xbf16>) outs(%7 : tensor<8192x1024xbf16>) permutation = [1, 0] 
    %14 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_7 : tensor<8192x1024xbf16>) outs(%8 : tensor<8x8192x1024xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x8192x1024xbf16>
    %15 = linalg.batch_matmul ins(%arg0, %14 : tensor<8x128x8192xbf16>, tensor<8x8192x1024xbf16>) outs(%11 : tensor<8x128x1024xbf16>) -> tensor<8x128x1024xbf16>
    %expanded_8 = tensor.expand_shape %15 [[0], [1], [2, 3]] output_shape [8, 128, 8, 128] : tensor<8x128x1024xbf16> into tensor<8x128x8x128xbf16>
    %transposed_9 = linalg.transpose ins(%expanded_8 : tensor<8x128x8x128xbf16>) outs(%13 : tensor<8x8x128x128xbf16>) permutation = [0, 2, 1, 3] 
    %expanded_10 = tensor.expand_shape %arg1 [[0], [1, 2], [3]] output_shape [1, 1, 128, 128] : tensor<1x128x128xbf16> into tensor<1x1x128x128xbf16>
    %expanded_11 = tensor.expand_shape %arg2 [[0], [1, 2], [3]] output_shape [1, 1, 128, 128] : tensor<1x128x128xbf16> into tensor<1x1x128x128xbf16>
    %16 = linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_3, %expanded_10 : tensor<8x64x128x128xbf16>, tensor<1x1x128x128xbf16>) outs(%6 : tensor<8x64x128x128xbf16>) {
    ^bb0(%in: bf16, %in_27: bf16, %out: bf16):
      %52 = arith.mulf %in, %in_27 : bf16
      linalg.yield %52 : bf16
    } -> tensor<8x64x128x128xbf16>
    %extracted_slice = tensor.extract_slice %transposed_3[0, 0, 0, 0] [8, 64, 128, 64] [1, 1, 1, 1] : tensor<8x64x128x128xbf16> to tensor<8x64x128x64xbf16>
    %extracted_slice_12 = tensor.extract_slice %transposed_3[0, 0, 0, 64] [8, 64, 128, 64] [1, 1, 1, 1] : tensor<8x64x128x128xbf16> to tensor<8x64x128x64xbf16>
    %17 = tensor.empty() : tensor<8x64x128x64xbf16>
    %18 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_12 : tensor<8x64x128x64xbf16>) outs(%17 : tensor<8x64x128x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %52 = arith.negf %in : bf16
      linalg.yield %52 : bf16
    } -> tensor<8x64x128x64xbf16>
    %concat = tensor.concat dim(3) %18, %extracted_slice : (tensor<8x64x128x64xbf16>, tensor<8x64x128x64xbf16>) -> tensor<8x64x128x128xbf16>
    %19 = linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat, %expanded_11 : tensor<8x64x128x128xbf16>, tensor<1x1x128x128xbf16>) outs(%6 : tensor<8x64x128x128xbf16>) {
    ^bb0(%in: bf16, %in_27: bf16, %out: bf16):
      %52 = arith.mulf %in, %in_27 : bf16
      linalg.yield %52 : bf16
    } -> tensor<8x64x128x128xbf16>
    %20 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%16, %19 : tensor<8x64x128x128xbf16>, tensor<8x64x128x128xbf16>) outs(%6 : tensor<8x64x128x128xbf16>) {
    ^bb0(%in: bf16, %in_27: bf16, %out: bf16):
      %52 = arith.addf %in, %in_27 : bf16
      linalg.yield %52 : bf16
    } -> tensor<8x64x128x128xbf16>
    %21 = linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_6, %expanded_10 : tensor<8x8x128x128xbf16>, tensor<1x1x128x128xbf16>) outs(%13 : tensor<8x8x128x128xbf16>) {
    ^bb0(%in: bf16, %in_27: bf16, %out: bf16):
      %52 = arith.mulf %in, %in_27 : bf16
      linalg.yield %52 : bf16
    } -> tensor<8x8x128x128xbf16>
    %extracted_slice_13 = tensor.extract_slice %transposed_6[0, 0, 0, 0] [8, 8, 128, 64] [1, 1, 1, 1] : tensor<8x8x128x128xbf16> to tensor<8x8x128x64xbf16>
    %extracted_slice_14 = tensor.extract_slice %transposed_6[0, 0, 0, 64] [8, 8, 128, 64] [1, 1, 1, 1] : tensor<8x8x128x128xbf16> to tensor<8x8x128x64xbf16>
    %22 = tensor.empty() : tensor<8x8x128x64xbf16>
    %23 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_14 : tensor<8x8x128x64xbf16>) outs(%22 : tensor<8x8x128x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %52 = arith.negf %in : bf16
      linalg.yield %52 : bf16
    } -> tensor<8x8x128x64xbf16>
    %concat_15 = tensor.concat dim(3) %23, %extracted_slice_13 : (tensor<8x8x128x64xbf16>, tensor<8x8x128x64xbf16>) -> tensor<8x8x128x128xbf16>
    %24 = linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_15, %expanded_11 : tensor<8x8x128x128xbf16>, tensor<1x1x128x128xbf16>) outs(%13 : tensor<8x8x128x128xbf16>) {
    ^bb0(%in: bf16, %in_27: bf16, %out: bf16):
      %52 = arith.mulf %in, %in_27 : bf16
      linalg.yield %52 : bf16
    } -> tensor<8x8x128x128xbf16>
    %25 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%21, %24 : tensor<8x8x128x128xbf16>, tensor<8x8x128x128xbf16>) outs(%13 : tensor<8x8x128x128xbf16>) {
    ^bb0(%in: bf16, %in_27: bf16, %out: bf16):
      %52 = arith.addf %in, %in_27 : bf16
      linalg.yield %52 : bf16
    } -> tensor<8x8x128x128xbf16>
    %26 = tensor.empty() : tensor<8x8x8x128x128xbf16>
    %27 = linalg.generic {indexing_maps = [#map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%25 : tensor<8x8x128x128xbf16>) outs(%26 : tensor<8x8x8x128x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x8x8x128x128xbf16>
    %collapsed = tensor.collapse_shape %27 [[0], [1, 2], [3], [4]] : tensor<8x8x8x128x128xbf16> into tensor<8x64x128x128xbf16>
    %28 = linalg.generic {indexing_maps = [#map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%transposed_9 : tensor<8x8x128x128xbf16>) outs(%26 : tensor<8x8x8x128x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x8x8x128x128xbf16>
    %transposed_16 = linalg.transpose ins(%collapsed : tensor<8x64x128x128xbf16>) outs(%6 : tensor<8x64x128x128xbf16>) permutation = [0, 1, 3, 2] 
    %collapsed_17 = tensor.collapse_shape %20 [[0, 1], [2], [3]] : tensor<8x64x128x128xbf16> into tensor<512x128x128xbf16>
    %collapsed_18 = tensor.collapse_shape %transposed_16 [[0, 1], [2], [3]] : tensor<8x64x128x128xbf16> into tensor<512x128x128xbf16>
    %29 = tensor.empty() : tensor<512x128x128xbf16>
    %30 = linalg.fill ins(%cst : bf16) outs(%29 : tensor<512x128x128xbf16>) -> tensor<512x128x128xbf16>
    %31 = linalg.batch_matmul ins(%collapsed_17, %collapsed_18 : tensor<512x128x128xbf16>, tensor<512x128x128xbf16>) outs(%30 : tensor<512x128x128xbf16>) -> tensor<512x128x128xbf16>
    %expanded_19 = tensor.expand_shape %31 [[0, 1], [2], [3]] output_shape [8, 64, 128, 128] : tensor<512x128x128xbf16> into tensor<8x64x128x128xbf16>
    %32 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_19 : tensor<8x64x128x128xbf16>) outs(%6 : tensor<8x64x128x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %52 = arith.truncf %cst_2 : f64 to bf16
      %53 = arith.mulf %in, %52 : bf16
      linalg.yield %53 : bf16
    } -> tensor<8x64x128x128xbf16>
    %33 = linalg.generic {indexing_maps = [#map2, #map6, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%32, %arg3 : tensor<8x64x128x128xbf16>, tensor<1x64x1x1xbf16>) outs(%6 : tensor<8x64x128x128xbf16>) {
    ^bb0(%in: bf16, %in_27: bf16, %out: bf16):
      %52 = arith.addf %in, %in_27 : bf16
      linalg.yield %52 : bf16
    } -> tensor<8x64x128x128xbf16>
    %34 = tensor.empty() : tensor<8x64x128x128xf32>
    %35 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%33 : tensor<8x64x128x128xbf16>) outs(%34 : tensor<8x64x128x128xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %52 = arith.extf %in : bf16 to f32
      linalg.yield %52 : f32
    } -> tensor<8x64x128x128xf32>
    %36 = tensor.empty() : tensor<8x64x128xi64>
    %37 = linalg.fill ins(%c0_i64 : i64) outs(%36 : tensor<8x64x128xi64>) -> tensor<8x64x128xi64>
    %38 = tensor.empty() : tensor<8x64x128xf32>
    %39 = linalg.fill ins(%cst_0 : f32) outs(%38 : tensor<8x64x128xf32>) -> tensor<8x64x128xf32>
    %40:2 = linalg.generic {indexing_maps = [#map2, #map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%35 : tensor<8x64x128x128xf32>) outs(%39, %37 : tensor<8x64x128xf32>, tensor<8x64x128xi64>) {
    ^bb0(%in: f32, %out: f32, %out_27: i64):
      %52 = linalg.index 3 : index
      %53 = arith.index_cast %52 : index to i64
      %54 = arith.maximumf %in, %out : f32
      %55 = arith.cmpf ogt, %in, %out : f32
      %56 = arith.select %55, %53, %out_27 : i64
      linalg.yield %54, %56 : f32, i64
    } -> (tensor<8x64x128xf32>, tensor<8x64x128xi64>)
    %expanded_20 = tensor.expand_shape %40#0 [[0], [1], [2, 3]] output_shape [8, 64, 128, 1] : tensor<8x64x128xf32> into tensor<8x64x128x1xf32>
    %41 = linalg.generic {indexing_maps = [#map2, #map8, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%35, %expanded_20 : tensor<8x64x128x128xf32>, tensor<8x64x128x1xf32>) outs(%34 : tensor<8x64x128x128xf32>) {
    ^bb0(%in: f32, %in_27: f32, %out: f32):
      %52 = arith.subf %in, %in_27 : f32
      linalg.yield %52 : f32
    } -> tensor<8x64x128x128xf32>
    %42 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%41 : tensor<8x64x128x128xf32>) outs(%34 : tensor<8x64x128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %52 = math.exp %in : f32
      linalg.yield %52 : f32
    } -> tensor<8x64x128x128xf32>
    %43 = tensor.empty() : tensor<8x64x128x1xf32>
    %44 = linalg.fill ins(%cst_1 : f32) outs(%43 : tensor<8x64x128x1xf32>) -> tensor<8x64x128x1xf32>
    %45 = linalg.generic {indexing_maps = [#map2, #map8], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%42 : tensor<8x64x128x128xf32>) outs(%44 : tensor<8x64x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %52 = arith.addf %in, %out : f32
      linalg.yield %52 : f32
    } -> tensor<8x64x128x1xf32>
    %46 = linalg.generic {indexing_maps = [#map2, #map8, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%42, %45 : tensor<8x64x128x128xf32>, tensor<8x64x128x1xf32>) outs(%34 : tensor<8x64x128x128xf32>) {
    ^bb0(%in: f32, %in_27: f32, %out: f32):
      %52 = arith.divf %in, %in_27 : f32
      linalg.yield %52 : f32
    } -> tensor<8x64x128x128xf32>
    %47 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%46 : tensor<8x64x128x128xf32>) outs(%6 : tensor<8x64x128x128xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %52 = arith.truncf %in : f32 to bf16
      linalg.yield %52 : bf16
    } -> tensor<8x64x128x128xbf16>
    %collapsed_21 = tensor.collapse_shape %47 [[0, 1], [2], [3]] : tensor<8x64x128x128xbf16> into tensor<512x128x128xbf16>
    %collapsed_22 = tensor.collapse_shape %28 [[0, 1, 2], [3], [4]] : tensor<8x8x8x128x128xbf16> into tensor<512x128x128xbf16>
    %48 = linalg.batch_matmul ins(%collapsed_21, %collapsed_22 : tensor<512x128x128xbf16>, tensor<512x128x128xbf16>) outs(%30 : tensor<512x128x128xbf16>) -> tensor<512x128x128xbf16>
    %expanded_23 = tensor.expand_shape %48 [[0, 1], [2], [3]] output_shape [8, 64, 128, 128] : tensor<512x128x128xbf16> into tensor<8x64x128x128xbf16>
    %49 = tensor.empty() : tensor<8x128x64x128xbf16>
    %transposed_24 = linalg.transpose ins(%expanded_23 : tensor<8x64x128x128xbf16>) outs(%49 : tensor<8x128x64x128xbf16>) permutation = [0, 2, 1, 3] 
    %collapsed_25 = tensor.collapse_shape %transposed_24 [[0], [1], [2, 3]] : tensor<8x128x64x128xbf16> into tensor<8x128x8192xbf16>
    %transposed_26 = linalg.transpose ins(%arg7 : tensor<8192x8192xbf16>) outs(%0 : tensor<8192x8192xbf16>) permutation = [1, 0] 
    %50 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_26 : tensor<8192x8192xbf16>) outs(%1 : tensor<8x8192x8192xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x8192x8192xbf16>
    %51 = linalg.batch_matmul ins(%collapsed_25, %50 : tensor<8x128x8192xbf16>, tensor<8x8192x8192xbf16>) outs(%4 : tensor<8x128x8192xbf16>) -> tensor<8x128x8192xbf16>
    return %51, %47 : tensor<8x128x8192xbf16>, tensor<8x64x128x128xbf16>
  }
}
