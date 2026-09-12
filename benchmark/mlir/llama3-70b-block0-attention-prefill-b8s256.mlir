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
  func.func @main(%arg0: tensor<8x256x8192xbf16>, %arg1: tensor<1x256x128xbf16>, %arg2: tensor<1x256x128xbf16>, %arg3: tensor<1x64x1x1xbf16>, %arg4: tensor<8192x8192xbf16>, %arg5: tensor<1024x8192xbf16>, %arg6: tensor<1024x8192xbf16>, %arg7: tensor<8192x8192xbf16>) -> (tensor<8x256x8192xbf16>, tensor<8x64x256x256xbf16>) {
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
    %3 = tensor.empty() : tensor<8x256x8192xbf16>
    %4 = linalg.fill ins(%cst : bf16) outs(%3 : tensor<8x256x8192xbf16>) -> tensor<8x256x8192xbf16>
    %5 = linalg.batch_matmul ins(%arg0, %2 : tensor<8x256x8192xbf16>, tensor<8x8192x8192xbf16>) outs(%4 : tensor<8x256x8192xbf16>) -> tensor<8x256x8192xbf16>
    %expanded = tensor.expand_shape %5 [[0], [1], [2, 3]] output_shape [8, 256, 64, 128] : tensor<8x256x8192xbf16> into tensor<8x256x64x128xbf16>
    %6 = tensor.empty() : tensor<8x64x256x128xbf16>
    %transposed_3 = linalg.transpose ins(%expanded : tensor<8x256x64x128xbf16>) outs(%6 : tensor<8x64x256x128xbf16>) permutation = [0, 2, 1, 3] 
    %7 = tensor.empty() : tensor<8192x1024xbf16>
    %transposed_4 = linalg.transpose ins(%arg5 : tensor<1024x8192xbf16>) outs(%7 : tensor<8192x1024xbf16>) permutation = [1, 0] 
    %8 = tensor.empty() : tensor<8x8192x1024xbf16>
    %9 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_4 : tensor<8192x1024xbf16>) outs(%8 : tensor<8x8192x1024xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x8192x1024xbf16>
    %10 = tensor.empty() : tensor<8x256x1024xbf16>
    %11 = linalg.fill ins(%cst : bf16) outs(%10 : tensor<8x256x1024xbf16>) -> tensor<8x256x1024xbf16>
    %12 = linalg.batch_matmul ins(%arg0, %9 : tensor<8x256x8192xbf16>, tensor<8x8192x1024xbf16>) outs(%11 : tensor<8x256x1024xbf16>) -> tensor<8x256x1024xbf16>
    %expanded_5 = tensor.expand_shape %12 [[0], [1], [2, 3]] output_shape [8, 256, 8, 128] : tensor<8x256x1024xbf16> into tensor<8x256x8x128xbf16>
    %13 = tensor.empty() : tensor<8x8x256x128xbf16>
    %transposed_6 = linalg.transpose ins(%expanded_5 : tensor<8x256x8x128xbf16>) outs(%13 : tensor<8x8x256x128xbf16>) permutation = [0, 2, 1, 3] 
    %transposed_7 = linalg.transpose ins(%arg6 : tensor<1024x8192xbf16>) outs(%7 : tensor<8192x1024xbf16>) permutation = [1, 0] 
    %14 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_7 : tensor<8192x1024xbf16>) outs(%8 : tensor<8x8192x1024xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x8192x1024xbf16>
    %15 = linalg.batch_matmul ins(%arg0, %14 : tensor<8x256x8192xbf16>, tensor<8x8192x1024xbf16>) outs(%11 : tensor<8x256x1024xbf16>) -> tensor<8x256x1024xbf16>
    %expanded_8 = tensor.expand_shape %15 [[0], [1], [2, 3]] output_shape [8, 256, 8, 128] : tensor<8x256x1024xbf16> into tensor<8x256x8x128xbf16>
    %transposed_9 = linalg.transpose ins(%expanded_8 : tensor<8x256x8x128xbf16>) outs(%13 : tensor<8x8x256x128xbf16>) permutation = [0, 2, 1, 3] 
    %expanded_10 = tensor.expand_shape %arg1 [[0], [1, 2], [3]] output_shape [1, 1, 256, 128] : tensor<1x256x128xbf16> into tensor<1x1x256x128xbf16>
    %expanded_11 = tensor.expand_shape %arg2 [[0], [1, 2], [3]] output_shape [1, 1, 256, 128] : tensor<1x256x128xbf16> into tensor<1x1x256x128xbf16>
    %16 = linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_3, %expanded_10 : tensor<8x64x256x128xbf16>, tensor<1x1x256x128xbf16>) outs(%6 : tensor<8x64x256x128xbf16>) {
    ^bb0(%in: bf16, %in_27: bf16, %out: bf16):
      %56 = arith.mulf %in, %in_27 : bf16
      linalg.yield %56 : bf16
    } -> tensor<8x64x256x128xbf16>
    %extracted_slice = tensor.extract_slice %transposed_3[0, 0, 0, 0] [8, 64, 256, 64] [1, 1, 1, 1] : tensor<8x64x256x128xbf16> to tensor<8x64x256x64xbf16>
    %extracted_slice_12 = tensor.extract_slice %transposed_3[0, 0, 0, 64] [8, 64, 256, 64] [1, 1, 1, 1] : tensor<8x64x256x128xbf16> to tensor<8x64x256x64xbf16>
    %17 = tensor.empty() : tensor<8x64x256x64xbf16>
    %18 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_12 : tensor<8x64x256x64xbf16>) outs(%17 : tensor<8x64x256x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %56 = arith.negf %in : bf16
      linalg.yield %56 : bf16
    } -> tensor<8x64x256x64xbf16>
    %concat = tensor.concat dim(3) %18, %extracted_slice : (tensor<8x64x256x64xbf16>, tensor<8x64x256x64xbf16>) -> tensor<8x64x256x128xbf16>
    %19 = linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat, %expanded_11 : tensor<8x64x256x128xbf16>, tensor<1x1x256x128xbf16>) outs(%6 : tensor<8x64x256x128xbf16>) {
    ^bb0(%in: bf16, %in_27: bf16, %out: bf16):
      %56 = arith.mulf %in, %in_27 : bf16
      linalg.yield %56 : bf16
    } -> tensor<8x64x256x128xbf16>
    %20 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%16, %19 : tensor<8x64x256x128xbf16>, tensor<8x64x256x128xbf16>) outs(%6 : tensor<8x64x256x128xbf16>) {
    ^bb0(%in: bf16, %in_27: bf16, %out: bf16):
      %56 = arith.addf %in, %in_27 : bf16
      linalg.yield %56 : bf16
    } -> tensor<8x64x256x128xbf16>
    %21 = linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_6, %expanded_10 : tensor<8x8x256x128xbf16>, tensor<1x1x256x128xbf16>) outs(%13 : tensor<8x8x256x128xbf16>) {
    ^bb0(%in: bf16, %in_27: bf16, %out: bf16):
      %56 = arith.mulf %in, %in_27 : bf16
      linalg.yield %56 : bf16
    } -> tensor<8x8x256x128xbf16>
    %extracted_slice_13 = tensor.extract_slice %transposed_6[0, 0, 0, 0] [8, 8, 256, 64] [1, 1, 1, 1] : tensor<8x8x256x128xbf16> to tensor<8x8x256x64xbf16>
    %extracted_slice_14 = tensor.extract_slice %transposed_6[0, 0, 0, 64] [8, 8, 256, 64] [1, 1, 1, 1] : tensor<8x8x256x128xbf16> to tensor<8x8x256x64xbf16>
    %22 = tensor.empty() : tensor<8x8x256x64xbf16>
    %23 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_14 : tensor<8x8x256x64xbf16>) outs(%22 : tensor<8x8x256x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %56 = arith.negf %in : bf16
      linalg.yield %56 : bf16
    } -> tensor<8x8x256x64xbf16>
    %concat_15 = tensor.concat dim(3) %23, %extracted_slice_13 : (tensor<8x8x256x64xbf16>, tensor<8x8x256x64xbf16>) -> tensor<8x8x256x128xbf16>
    %24 = linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_15, %expanded_11 : tensor<8x8x256x128xbf16>, tensor<1x1x256x128xbf16>) outs(%13 : tensor<8x8x256x128xbf16>) {
    ^bb0(%in: bf16, %in_27: bf16, %out: bf16):
      %56 = arith.mulf %in, %in_27 : bf16
      linalg.yield %56 : bf16
    } -> tensor<8x8x256x128xbf16>
    %25 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%21, %24 : tensor<8x8x256x128xbf16>, tensor<8x8x256x128xbf16>) outs(%13 : tensor<8x8x256x128xbf16>) {
    ^bb0(%in: bf16, %in_27: bf16, %out: bf16):
      %56 = arith.addf %in, %in_27 : bf16
      linalg.yield %56 : bf16
    } -> tensor<8x8x256x128xbf16>
    %26 = tensor.empty() : tensor<8x8x8x256x128xbf16>
    %27 = linalg.generic {indexing_maps = [#map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%25 : tensor<8x8x256x128xbf16>) outs(%26 : tensor<8x8x8x256x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x8x8x256x128xbf16>
    %collapsed = tensor.collapse_shape %27 [[0], [1, 2], [3], [4]] : tensor<8x8x8x256x128xbf16> into tensor<8x64x256x128xbf16>
    %28 = linalg.generic {indexing_maps = [#map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%transposed_9 : tensor<8x8x256x128xbf16>) outs(%26 : tensor<8x8x8x256x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x8x8x256x128xbf16>
    %29 = tensor.empty() : tensor<8x64x128x256xbf16>
    %transposed_16 = linalg.transpose ins(%collapsed : tensor<8x64x256x128xbf16>) outs(%29 : tensor<8x64x128x256xbf16>) permutation = [0, 1, 3, 2] 
    %collapsed_17 = tensor.collapse_shape %20 [[0, 1], [2], [3]] : tensor<8x64x256x128xbf16> into tensor<512x256x128xbf16>
    %collapsed_18 = tensor.collapse_shape %transposed_16 [[0, 1], [2], [3]] : tensor<8x64x128x256xbf16> into tensor<512x128x256xbf16>
    %30 = tensor.empty() : tensor<512x256x256xbf16>
    %31 = linalg.fill ins(%cst : bf16) outs(%30 : tensor<512x256x256xbf16>) -> tensor<512x256x256xbf16>
    %32 = linalg.batch_matmul ins(%collapsed_17, %collapsed_18 : tensor<512x256x128xbf16>, tensor<512x128x256xbf16>) outs(%31 : tensor<512x256x256xbf16>) -> tensor<512x256x256xbf16>
    %expanded_19 = tensor.expand_shape %32 [[0, 1], [2], [3]] output_shape [8, 64, 256, 256] : tensor<512x256x256xbf16> into tensor<8x64x256x256xbf16>
    %33 = tensor.empty() : tensor<8x64x256x256xbf16>
    %34 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_19 : tensor<8x64x256x256xbf16>) outs(%33 : tensor<8x64x256x256xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %56 = arith.truncf %cst_2 : f64 to bf16
      %57 = arith.mulf %in, %56 : bf16
      linalg.yield %57 : bf16
    } -> tensor<8x64x256x256xbf16>
    %35 = linalg.generic {indexing_maps = [#map2, #map6, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%34, %arg3 : tensor<8x64x256x256xbf16>, tensor<1x64x1x1xbf16>) outs(%33 : tensor<8x64x256x256xbf16>) {
    ^bb0(%in: bf16, %in_27: bf16, %out: bf16):
      %56 = arith.addf %in, %in_27 : bf16
      linalg.yield %56 : bf16
    } -> tensor<8x64x256x256xbf16>
    %36 = tensor.empty() : tensor<8x64x256x256xf32>
    %37 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%35 : tensor<8x64x256x256xbf16>) outs(%36 : tensor<8x64x256x256xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %56 = arith.extf %in : bf16 to f32
      linalg.yield %56 : f32
    } -> tensor<8x64x256x256xf32>
    %38 = tensor.empty() : tensor<8x64x256xi64>
    %39 = linalg.fill ins(%c0_i64 : i64) outs(%38 : tensor<8x64x256xi64>) -> tensor<8x64x256xi64>
    %40 = tensor.empty() : tensor<8x64x256xf32>
    %41 = linalg.fill ins(%cst_0 : f32) outs(%40 : tensor<8x64x256xf32>) -> tensor<8x64x256xf32>
    %42:2 = linalg.generic {indexing_maps = [#map2, #map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%37 : tensor<8x64x256x256xf32>) outs(%41, %39 : tensor<8x64x256xf32>, tensor<8x64x256xi64>) {
    ^bb0(%in: f32, %out: f32, %out_27: i64):
      %56 = linalg.index 3 : index
      %57 = arith.index_cast %56 : index to i64
      %58 = arith.maximumf %in, %out : f32
      %59 = arith.cmpf ogt, %in, %out : f32
      %60 = arith.select %59, %57, %out_27 : i64
      linalg.yield %58, %60 : f32, i64
    } -> (tensor<8x64x256xf32>, tensor<8x64x256xi64>)
    %expanded_20 = tensor.expand_shape %42#0 [[0], [1], [2, 3]] output_shape [8, 64, 256, 1] : tensor<8x64x256xf32> into tensor<8x64x256x1xf32>
    %43 = linalg.generic {indexing_maps = [#map2, #map8, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%37, %expanded_20 : tensor<8x64x256x256xf32>, tensor<8x64x256x1xf32>) outs(%36 : tensor<8x64x256x256xf32>) {
    ^bb0(%in: f32, %in_27: f32, %out: f32):
      %56 = arith.subf %in, %in_27 : f32
      linalg.yield %56 : f32
    } -> tensor<8x64x256x256xf32>
    %44 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%43 : tensor<8x64x256x256xf32>) outs(%36 : tensor<8x64x256x256xf32>) {
    ^bb0(%in: f32, %out: f32):
      %56 = math.exp %in : f32
      linalg.yield %56 : f32
    } -> tensor<8x64x256x256xf32>
    %45 = tensor.empty() : tensor<8x64x256x1xf32>
    %46 = linalg.fill ins(%cst_1 : f32) outs(%45 : tensor<8x64x256x1xf32>) -> tensor<8x64x256x1xf32>
    %47 = linalg.generic {indexing_maps = [#map2, #map8], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%44 : tensor<8x64x256x256xf32>) outs(%46 : tensor<8x64x256x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %56 = arith.addf %in, %out : f32
      linalg.yield %56 : f32
    } -> tensor<8x64x256x1xf32>
    %48 = linalg.generic {indexing_maps = [#map2, #map8, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%44, %47 : tensor<8x64x256x256xf32>, tensor<8x64x256x1xf32>) outs(%36 : tensor<8x64x256x256xf32>) {
    ^bb0(%in: f32, %in_27: f32, %out: f32):
      %56 = arith.divf %in, %in_27 : f32
      linalg.yield %56 : f32
    } -> tensor<8x64x256x256xf32>
    %49 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%48 : tensor<8x64x256x256xf32>) outs(%33 : tensor<8x64x256x256xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %56 = arith.truncf %in : f32 to bf16
      linalg.yield %56 : bf16
    } -> tensor<8x64x256x256xbf16>
    %collapsed_21 = tensor.collapse_shape %49 [[0, 1], [2], [3]] : tensor<8x64x256x256xbf16> into tensor<512x256x256xbf16>
    %collapsed_22 = tensor.collapse_shape %28 [[0, 1, 2], [3], [4]] : tensor<8x8x8x256x128xbf16> into tensor<512x256x128xbf16>
    %50 = tensor.empty() : tensor<512x256x128xbf16>
    %51 = linalg.fill ins(%cst : bf16) outs(%50 : tensor<512x256x128xbf16>) -> tensor<512x256x128xbf16>
    %52 = linalg.batch_matmul ins(%collapsed_21, %collapsed_22 : tensor<512x256x256xbf16>, tensor<512x256x128xbf16>) outs(%51 : tensor<512x256x128xbf16>) -> tensor<512x256x128xbf16>
    %expanded_23 = tensor.expand_shape %52 [[0, 1], [2], [3]] output_shape [8, 64, 256, 128] : tensor<512x256x128xbf16> into tensor<8x64x256x128xbf16>
    %53 = tensor.empty() : tensor<8x256x64x128xbf16>
    %transposed_24 = linalg.transpose ins(%expanded_23 : tensor<8x64x256x128xbf16>) outs(%53 : tensor<8x256x64x128xbf16>) permutation = [0, 2, 1, 3] 
    %collapsed_25 = tensor.collapse_shape %transposed_24 [[0], [1], [2, 3]] : tensor<8x256x64x128xbf16> into tensor<8x256x8192xbf16>
    %transposed_26 = linalg.transpose ins(%arg7 : tensor<8192x8192xbf16>) outs(%0 : tensor<8192x8192xbf16>) permutation = [1, 0] 
    %54 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_26 : tensor<8192x8192xbf16>) outs(%1 : tensor<8x8192x8192xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x8192x8192xbf16>
    %55 = linalg.batch_matmul ins(%collapsed_25, %54 : tensor<8x256x8192xbf16>, tensor<8x8192x8192xbf16>) outs(%4 : tensor<8x256x8192xbf16>) -> tensor<8x256x8192xbf16>
    return %55, %49 : tensor<8x256x8192xbf16>, tensor<8x64x256x256xbf16>
  }
}
