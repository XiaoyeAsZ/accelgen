#map = affine_map<(d0, d1, d2) -> (d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>
#map5 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, 0)>
#map8 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map9 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>
module {
  func.func @main(%arg0: tensor<1x128x4096xbf16>, %arg1: tensor<1x128x128xbf16>, %arg2: tensor<1x128x128xbf16>, %arg3: tensor<1x32x1x1xbf16>, %arg4: tensor<4096x4096xbf16>, %arg5: tensor<1024x4096xbf16>, %arg6: tensor<1024x4096xbf16>, %arg7: tensor<4096x4096xbf16>) -> (tensor<1x128x4096xbf16>, tensor<1x32x128x128xbf16>) {
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant 0.000000e+00 : bf16
    %cst_0 = arith.constant 0xFF800000 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %cst_2 = arith.constant 0.088388347648318447 : f64
    %0 = tensor.empty() : tensor<4096x4096xbf16>
    %transposed = linalg.transpose ins(%arg4 : tensor<4096x4096xbf16>) outs(%0 : tensor<4096x4096xbf16>) permutation = [1, 0] 
    %1 = tensor.empty() : tensor<1x128x4096xbf16>
    %collapsed = tensor.collapse_shape %arg0 [[0, 1], [2]] : tensor<1x128x4096xbf16> into tensor<128x4096xbf16>
    %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapsed : tensor<128x4096xbf16>) outs(%1 : tensor<1x128x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x128x4096xbf16>
    %3 = tensor.empty() : tensor<1x4096x4096xbf16>
    %4 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed : tensor<4096x4096xbf16>) outs(%3 : tensor<1x4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x4096x4096xbf16>
    %5 = linalg.fill ins(%cst : bf16) outs(%1 : tensor<1x128x4096xbf16>) -> tensor<1x128x4096xbf16>
    %6 = linalg.batch_matmul ins(%2, %4 : tensor<1x128x4096xbf16>, tensor<1x4096x4096xbf16>) outs(%5 : tensor<1x128x4096xbf16>) -> tensor<1x128x4096xbf16>
    %expanded = tensor.expand_shape %6 [[0], [1], [2, 3]] output_shape [1, 128, 32, 128] : tensor<1x128x4096xbf16> into tensor<1x128x32x128xbf16>
    %7 = tensor.empty() : tensor<1x32x128x128xbf16>
    %transposed_3 = linalg.transpose ins(%expanded : tensor<1x128x32x128xbf16>) outs(%7 : tensor<1x32x128x128xbf16>) permutation = [0, 2, 1, 3] 
    %8 = tensor.empty() : tensor<4096x1024xbf16>
    %transposed_4 = linalg.transpose ins(%arg5 : tensor<1024x4096xbf16>) outs(%8 : tensor<4096x1024xbf16>) permutation = [1, 0] 
    %9 = tensor.empty() : tensor<1x4096x1024xbf16>
    %10 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_4 : tensor<4096x1024xbf16>) outs(%9 : tensor<1x4096x1024xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x4096x1024xbf16>
    %11 = tensor.empty() : tensor<1x128x1024xbf16>
    %12 = linalg.fill ins(%cst : bf16) outs(%11 : tensor<1x128x1024xbf16>) -> tensor<1x128x1024xbf16>
    %13 = linalg.batch_matmul ins(%2, %10 : tensor<1x128x4096xbf16>, tensor<1x4096x1024xbf16>) outs(%12 : tensor<1x128x1024xbf16>) -> tensor<1x128x1024xbf16>
    %expanded_5 = tensor.expand_shape %13 [[0], [1], [2, 3]] output_shape [1, 128, 8, 128] : tensor<1x128x1024xbf16> into tensor<1x128x8x128xbf16>
    %14 = tensor.empty() : tensor<1x8x128x128xbf16>
    %transposed_6 = linalg.transpose ins(%expanded_5 : tensor<1x128x8x128xbf16>) outs(%14 : tensor<1x8x128x128xbf16>) permutation = [0, 2, 1, 3] 
    %transposed_7 = linalg.transpose ins(%arg6 : tensor<1024x4096xbf16>) outs(%8 : tensor<4096x1024xbf16>) permutation = [1, 0] 
    %15 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_7 : tensor<4096x1024xbf16>) outs(%9 : tensor<1x4096x1024xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x4096x1024xbf16>
    %16 = linalg.batch_matmul ins(%2, %15 : tensor<1x128x4096xbf16>, tensor<1x4096x1024xbf16>) outs(%12 : tensor<1x128x1024xbf16>) -> tensor<1x128x1024xbf16>
    %expanded_8 = tensor.expand_shape %16 [[0], [1], [2, 3]] output_shape [1, 128, 8, 128] : tensor<1x128x1024xbf16> into tensor<1x128x8x128xbf16>
    %transposed_9 = linalg.transpose ins(%expanded_8 : tensor<1x128x8x128xbf16>) outs(%14 : tensor<1x8x128x128xbf16>) permutation = [0, 2, 1, 3] 
    %expanded_10 = tensor.expand_shape %arg1 [[0], [1, 2], [3]] output_shape [1, 1, 128, 128] : tensor<1x128x128xbf16> into tensor<1x1x128x128xbf16>
    %expanded_11 = tensor.expand_shape %arg2 [[0], [1, 2], [3]] output_shape [1, 1, 128, 128] : tensor<1x128x128xbf16> into tensor<1x1x128x128xbf16>
    %17 = linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_3, %expanded_10 : tensor<1x32x128x128xbf16>, tensor<1x1x128x128xbf16>) outs(%7 : tensor<1x32x128x128xbf16>) {
    ^bb0(%in: bf16, %in_32: bf16, %out: bf16):
      %58 = arith.mulf %in, %in_32 : bf16
      linalg.yield %58 : bf16
    } -> tensor<1x32x128x128xbf16>
    %extracted_slice = tensor.extract_slice %transposed_3[0, 0, 0, 0] [1, 32, 128, 64] [1, 1, 1, 1] : tensor<1x32x128x128xbf16> to tensor<1x32x128x64xbf16>
    %extracted_slice_12 = tensor.extract_slice %transposed_3[0, 0, 0, 64] [1, 32, 128, 64] [1, 1, 1, 1] : tensor<1x32x128x128xbf16> to tensor<1x32x128x64xbf16>
    %18 = tensor.empty() : tensor<1x32x128x64xbf16>
    %19 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_12 : tensor<1x32x128x64xbf16>) outs(%18 : tensor<1x32x128x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %58 = arith.negf %in : bf16
      linalg.yield %58 : bf16
    } -> tensor<1x32x128x64xbf16>
    %concat = tensor.concat dim(3) %19, %extracted_slice : (tensor<1x32x128x64xbf16>, tensor<1x32x128x64xbf16>) -> tensor<1x32x128x128xbf16>
    %20 = linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat, %expanded_11 : tensor<1x32x128x128xbf16>, tensor<1x1x128x128xbf16>) outs(%7 : tensor<1x32x128x128xbf16>) {
    ^bb0(%in: bf16, %in_32: bf16, %out: bf16):
      %58 = arith.mulf %in, %in_32 : bf16
      linalg.yield %58 : bf16
    } -> tensor<1x32x128x128xbf16>
    %21 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%17, %20 : tensor<1x32x128x128xbf16>, tensor<1x32x128x128xbf16>) outs(%7 : tensor<1x32x128x128xbf16>) {
    ^bb0(%in: bf16, %in_32: bf16, %out: bf16):
      %58 = arith.addf %in, %in_32 : bf16
      linalg.yield %58 : bf16
    } -> tensor<1x32x128x128xbf16>
    %22 = linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_6, %expanded_10 : tensor<1x8x128x128xbf16>, tensor<1x1x128x128xbf16>) outs(%14 : tensor<1x8x128x128xbf16>) {
    ^bb0(%in: bf16, %in_32: bf16, %out: bf16):
      %58 = arith.mulf %in, %in_32 : bf16
      linalg.yield %58 : bf16
    } -> tensor<1x8x128x128xbf16>
    %extracted_slice_13 = tensor.extract_slice %transposed_6[0, 0, 0, 0] [1, 8, 128, 64] [1, 1, 1, 1] : tensor<1x8x128x128xbf16> to tensor<1x8x128x64xbf16>
    %extracted_slice_14 = tensor.extract_slice %transposed_6[0, 0, 0, 64] [1, 8, 128, 64] [1, 1, 1, 1] : tensor<1x8x128x128xbf16> to tensor<1x8x128x64xbf16>
    %23 = tensor.empty() : tensor<1x8x128x64xbf16>
    %24 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_14 : tensor<1x8x128x64xbf16>) outs(%23 : tensor<1x8x128x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %58 = arith.negf %in : bf16
      linalg.yield %58 : bf16
    } -> tensor<1x8x128x64xbf16>
    %concat_15 = tensor.concat dim(3) %24, %extracted_slice_13 : (tensor<1x8x128x64xbf16>, tensor<1x8x128x64xbf16>) -> tensor<1x8x128x128xbf16>
    %25 = linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_15, %expanded_11 : tensor<1x8x128x128xbf16>, tensor<1x1x128x128xbf16>) outs(%14 : tensor<1x8x128x128xbf16>) {
    ^bb0(%in: bf16, %in_32: bf16, %out: bf16):
      %58 = arith.mulf %in, %in_32 : bf16
      linalg.yield %58 : bf16
    } -> tensor<1x8x128x128xbf16>
    %26 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%22, %25 : tensor<1x8x128x128xbf16>, tensor<1x8x128x128xbf16>) outs(%14 : tensor<1x8x128x128xbf16>) {
    ^bb0(%in: bf16, %in_32: bf16, %out: bf16):
      %58 = arith.addf %in, %in_32 : bf16
      linalg.yield %58 : bf16
    } -> tensor<1x8x128x128xbf16>
    %27 = tensor.empty() : tensor<1x8x4x128x128xbf16>
    %28 = linalg.generic {indexing_maps = [#map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%26 : tensor<1x8x128x128xbf16>) outs(%27 : tensor<1x8x4x128x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x8x4x128x128xbf16>
    %collapsed_16 = tensor.collapse_shape %28 [[0], [1, 2], [3], [4]] : tensor<1x8x4x128x128xbf16> into tensor<1x32x128x128xbf16>
    %29 = linalg.generic {indexing_maps = [#map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%transposed_9 : tensor<1x8x128x128xbf16>) outs(%27 : tensor<1x8x4x128x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x8x4x128x128xbf16>
    %transposed_17 = linalg.transpose ins(%collapsed_16 : tensor<1x32x128x128xbf16>) outs(%7 : tensor<1x32x128x128xbf16>) permutation = [0, 1, 3, 2] 
    %collapsed_18 = tensor.collapse_shape %21 [[0, 1], [2], [3]] : tensor<1x32x128x128xbf16> into tensor<32x128x128xbf16>
    %30 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%collapsed_18 : tensor<32x128x128xbf16>) outs(%7 : tensor<1x32x128x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x128x128xbf16>
    %collapsed_19 = tensor.collapse_shape %transposed_17 [[0, 1], [2], [3]] : tensor<1x32x128x128xbf16> into tensor<32x128x128xbf16>
    %31 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%collapsed_19 : tensor<32x128x128xbf16>) outs(%7 : tensor<1x32x128x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x128x128xbf16>
    %collapsed_20 = tensor.collapse_shape %30 [[0, 1], [2], [3]] : tensor<1x32x128x128xbf16> into tensor<32x128x128xbf16>
    %collapsed_21 = tensor.collapse_shape %31 [[0, 1], [2], [3]] : tensor<1x32x128x128xbf16> into tensor<32x128x128xbf16>
    %32 = tensor.empty() : tensor<32x128x128xbf16>
    %33 = linalg.fill ins(%cst : bf16) outs(%32 : tensor<32x128x128xbf16>) -> tensor<32x128x128xbf16>
    %34 = linalg.batch_matmul ins(%collapsed_20, %collapsed_21 : tensor<32x128x128xbf16>, tensor<32x128x128xbf16>) outs(%33 : tensor<32x128x128xbf16>) -> tensor<32x128x128xbf16>
    %expanded_22 = tensor.expand_shape %34 [[0, 1], [2], [3]] output_shape [1, 32, 128, 128] : tensor<32x128x128xbf16> into tensor<1x32x128x128xbf16>
    %35 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_22 : tensor<1x32x128x128xbf16>) outs(%7 : tensor<1x32x128x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %58 = arith.truncf %cst_2 : f64 to bf16
      %59 = arith.mulf %in, %58 : bf16
      linalg.yield %59 : bf16
    } -> tensor<1x32x128x128xbf16>
    %36 = linalg.generic {indexing_maps = [#map2, #map7, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%35, %arg3 : tensor<1x32x128x128xbf16>, tensor<1x32x1x1xbf16>) outs(%7 : tensor<1x32x128x128xbf16>) {
    ^bb0(%in: bf16, %in_32: bf16, %out: bf16):
      %58 = arith.addf %in, %in_32 : bf16
      linalg.yield %58 : bf16
    } -> tensor<1x32x128x128xbf16>
    %37 = tensor.empty() : tensor<1x32x128x128xf32>
    %38 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%36 : tensor<1x32x128x128xbf16>) outs(%37 : tensor<1x32x128x128xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %58 = arith.extf %in : bf16 to f32
      linalg.yield %58 : f32
    } -> tensor<1x32x128x128xf32>
    %39 = tensor.empty() : tensor<1x32x128xi64>
    %40 = linalg.fill ins(%c0_i64 : i64) outs(%39 : tensor<1x32x128xi64>) -> tensor<1x32x128xi64>
    %41 = tensor.empty() : tensor<1x32x128xf32>
    %42 = linalg.fill ins(%cst_0 : f32) outs(%41 : tensor<1x32x128xf32>) -> tensor<1x32x128xf32>
    %43:2 = linalg.generic {indexing_maps = [#map2, #map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%38 : tensor<1x32x128x128xf32>) outs(%42, %40 : tensor<1x32x128xf32>, tensor<1x32x128xi64>) {
    ^bb0(%in: f32, %out: f32, %out_32: i64):
      %58 = linalg.index 3 : index
      %59 = arith.index_cast %58 : index to i64
      %60 = arith.maximumf %in, %out : f32
      %61 = arith.cmpf ogt, %in, %out : f32
      %62 = arith.select %61, %59, %out_32 : i64
      linalg.yield %60, %62 : f32, i64
    } -> (tensor<1x32x128xf32>, tensor<1x32x128xi64>)
    %expanded_23 = tensor.expand_shape %43#0 [[0], [1], [2, 3]] output_shape [1, 32, 128, 1] : tensor<1x32x128xf32> into tensor<1x32x128x1xf32>
    %44 = linalg.generic {indexing_maps = [#map2, #map9, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%38, %expanded_23 : tensor<1x32x128x128xf32>, tensor<1x32x128x1xf32>) outs(%37 : tensor<1x32x128x128xf32>) {
    ^bb0(%in: f32, %in_32: f32, %out: f32):
      %58 = arith.subf %in, %in_32 : f32
      linalg.yield %58 : f32
    } -> tensor<1x32x128x128xf32>
    %45 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%44 : tensor<1x32x128x128xf32>) outs(%37 : tensor<1x32x128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %58 = math.exp %in : f32
      linalg.yield %58 : f32
    } -> tensor<1x32x128x128xf32>
    %46 = tensor.empty() : tensor<1x32x128x1xf32>
    %47 = linalg.fill ins(%cst_1 : f32) outs(%46 : tensor<1x32x128x1xf32>) -> tensor<1x32x128x1xf32>
    %48 = linalg.generic {indexing_maps = [#map2, #map9], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%45 : tensor<1x32x128x128xf32>) outs(%47 : tensor<1x32x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %58 = arith.addf %in, %out : f32
      linalg.yield %58 : f32
    } -> tensor<1x32x128x1xf32>
    %49 = linalg.generic {indexing_maps = [#map2, #map9, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%45, %48 : tensor<1x32x128x128xf32>, tensor<1x32x128x1xf32>) outs(%37 : tensor<1x32x128x128xf32>) {
    ^bb0(%in: f32, %in_32: f32, %out: f32):
      %58 = arith.divf %in, %in_32 : f32
      linalg.yield %58 : f32
    } -> tensor<1x32x128x128xf32>
    %50 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%49 : tensor<1x32x128x128xf32>) outs(%7 : tensor<1x32x128x128xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %58 = arith.truncf %in : f32 to bf16
      linalg.yield %58 : bf16
    } -> tensor<1x32x128x128xbf16>
    %collapsed_24 = tensor.collapse_shape %50 [[0, 1], [2], [3]] : tensor<1x32x128x128xbf16> into tensor<32x128x128xbf16>
    %51 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%collapsed_24 : tensor<32x128x128xbf16>) outs(%7 : tensor<1x32x128x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x128x128xbf16>
    %collapsed_25 = tensor.collapse_shape %29 [[0, 1, 2], [3], [4]] : tensor<1x8x4x128x128xbf16> into tensor<32x128x128xbf16>
    %52 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%collapsed_25 : tensor<32x128x128xbf16>) outs(%7 : tensor<1x32x128x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x128x128xbf16>
    %collapsed_26 = tensor.collapse_shape %51 [[0, 1], [2], [3]] : tensor<1x32x128x128xbf16> into tensor<32x128x128xbf16>
    %collapsed_27 = tensor.collapse_shape %52 [[0, 1], [2], [3]] : tensor<1x32x128x128xbf16> into tensor<32x128x128xbf16>
    %53 = linalg.batch_matmul ins(%collapsed_26, %collapsed_27 : tensor<32x128x128xbf16>, tensor<32x128x128xbf16>) outs(%33 : tensor<32x128x128xbf16>) -> tensor<32x128x128xbf16>
    %expanded_28 = tensor.expand_shape %53 [[0, 1], [2], [3]] output_shape [1, 32, 128, 128] : tensor<32x128x128xbf16> into tensor<1x32x128x128xbf16>
    %54 = tensor.empty() : tensor<1x128x32x128xbf16>
    %transposed_29 = linalg.transpose ins(%expanded_28 : tensor<1x32x128x128xbf16>) outs(%54 : tensor<1x128x32x128xbf16>) permutation = [0, 2, 1, 3] 
    %transposed_30 = linalg.transpose ins(%arg7 : tensor<4096x4096xbf16>) outs(%0 : tensor<4096x4096xbf16>) permutation = [1, 0] 
    %collapsed_31 = tensor.collapse_shape %transposed_29 [[0, 1], [2, 3]] : tensor<1x128x32x128xbf16> into tensor<128x4096xbf16>
    %55 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapsed_31 : tensor<128x4096xbf16>) outs(%1 : tensor<1x128x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x128x4096xbf16>
    %56 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_30 : tensor<4096x4096xbf16>) outs(%3 : tensor<1x4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x4096x4096xbf16>
    %57 = linalg.batch_matmul ins(%55, %56 : tensor<1x128x4096xbf16>, tensor<1x4096x4096xbf16>) outs(%5 : tensor<1x128x4096xbf16>) -> tensor<1x128x4096xbf16>
    return %57, %50 : tensor<1x128x4096xbf16>, tensor<1x32x128x128xbf16>
  }
}
