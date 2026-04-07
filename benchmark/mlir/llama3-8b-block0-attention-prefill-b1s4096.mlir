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
  func.func @main(%arg0: tensor<1x4096x4096xbf16>, %arg1: tensor<1x4096x128xbf16>, %arg2: tensor<1x4096x128xbf16>, %arg3: tensor<1x32x1x1xbf16>, %arg4: tensor<4096x4096xbf16>, %arg5: tensor<1024x4096xbf16>, %arg6: tensor<1024x4096xbf16>, %arg7: tensor<4096x4096xbf16>) -> (tensor<1x4096x4096xbf16>, tensor<1x32x4096x4096xbf16>) {
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant 0.000000e+00 : bf16
    %cst_0 = arith.constant 0xFF800000 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %cst_2 = arith.constant 0.088388347648318447 : f64
    %0 = tensor.empty() : tensor<4096x4096xbf16>
    %transposed = linalg.transpose ins(%arg4 : tensor<4096x4096xbf16>) outs(%0 : tensor<4096x4096xbf16>) permutation = [1, 0] 
    %1 = tensor.empty() : tensor<1x4096x4096xbf16>
    %collapsed = tensor.collapse_shape %arg0 [[0, 1], [2]] : tensor<1x4096x4096xbf16> into tensor<4096x4096xbf16>
    %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapsed : tensor<4096x4096xbf16>) outs(%1 : tensor<1x4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x4096x4096xbf16>
    %3 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed : tensor<4096x4096xbf16>) outs(%1 : tensor<1x4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x4096x4096xbf16>
    %4 = linalg.fill ins(%cst : bf16) outs(%1 : tensor<1x4096x4096xbf16>) -> tensor<1x4096x4096xbf16>
    %5 = linalg.batch_matmul ins(%2, %3 : tensor<1x4096x4096xbf16>, tensor<1x4096x4096xbf16>) outs(%4 : tensor<1x4096x4096xbf16>) -> tensor<1x4096x4096xbf16>
    %expanded = tensor.expand_shape %5 [[0], [1], [2, 3]] output_shape [1, 4096, 32, 128] : tensor<1x4096x4096xbf16> into tensor<1x4096x32x128xbf16>
    %6 = tensor.empty() : tensor<1x32x4096x128xbf16>
    %transposed_3 = linalg.transpose ins(%expanded : tensor<1x4096x32x128xbf16>) outs(%6 : tensor<1x32x4096x128xbf16>) permutation = [0, 2, 1, 3] 
    %7 = tensor.empty() : tensor<4096x1024xbf16>
    %transposed_4 = linalg.transpose ins(%arg5 : tensor<1024x4096xbf16>) outs(%7 : tensor<4096x1024xbf16>) permutation = [1, 0] 
    %8 = tensor.empty() : tensor<1x4096x1024xbf16>
    %9 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_4 : tensor<4096x1024xbf16>) outs(%8 : tensor<1x4096x1024xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x4096x1024xbf16>
    %10 = linalg.fill ins(%cst : bf16) outs(%8 : tensor<1x4096x1024xbf16>) -> tensor<1x4096x1024xbf16>
    %11 = linalg.batch_matmul ins(%2, %9 : tensor<1x4096x4096xbf16>, tensor<1x4096x1024xbf16>) outs(%10 : tensor<1x4096x1024xbf16>) -> tensor<1x4096x1024xbf16>
    %expanded_5 = tensor.expand_shape %11 [[0], [1], [2, 3]] output_shape [1, 4096, 8, 128] : tensor<1x4096x1024xbf16> into tensor<1x4096x8x128xbf16>
    %12 = tensor.empty() : tensor<1x8x4096x128xbf16>
    %transposed_6 = linalg.transpose ins(%expanded_5 : tensor<1x4096x8x128xbf16>) outs(%12 : tensor<1x8x4096x128xbf16>) permutation = [0, 2, 1, 3] 
    %transposed_7 = linalg.transpose ins(%arg6 : tensor<1024x4096xbf16>) outs(%7 : tensor<4096x1024xbf16>) permutation = [1, 0] 
    %13 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_7 : tensor<4096x1024xbf16>) outs(%8 : tensor<1x4096x1024xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x4096x1024xbf16>
    %14 = linalg.batch_matmul ins(%2, %13 : tensor<1x4096x4096xbf16>, tensor<1x4096x1024xbf16>) outs(%10 : tensor<1x4096x1024xbf16>) -> tensor<1x4096x1024xbf16>
    %expanded_8 = tensor.expand_shape %14 [[0], [1], [2, 3]] output_shape [1, 4096, 8, 128] : tensor<1x4096x1024xbf16> into tensor<1x4096x8x128xbf16>
    %transposed_9 = linalg.transpose ins(%expanded_8 : tensor<1x4096x8x128xbf16>) outs(%12 : tensor<1x8x4096x128xbf16>) permutation = [0, 2, 1, 3] 
    %expanded_10 = tensor.expand_shape %arg1 [[0], [1, 2], [3]] output_shape [1, 1, 4096, 128] : tensor<1x4096x128xbf16> into tensor<1x1x4096x128xbf16>
    %expanded_11 = tensor.expand_shape %arg2 [[0], [1, 2], [3]] output_shape [1, 1, 4096, 128] : tensor<1x4096x128xbf16> into tensor<1x1x4096x128xbf16>
    %15 = linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_3, %expanded_10 : tensor<1x32x4096x128xbf16>, tensor<1x1x4096x128xbf16>) outs(%6 : tensor<1x32x4096x128xbf16>) {
    ^bb0(%in: bf16, %in_32: bf16, %out: bf16):
      %60 = arith.mulf %in, %in_32 : bf16
      linalg.yield %60 : bf16
    } -> tensor<1x32x4096x128xbf16>
    %extracted_slice = tensor.extract_slice %transposed_3[0, 0, 0, 0] [1, 32, 4096, 64] [1, 1, 1, 1] : tensor<1x32x4096x128xbf16> to tensor<1x32x4096x64xbf16>
    %extracted_slice_12 = tensor.extract_slice %transposed_3[0, 0, 0, 64] [1, 32, 4096, 64] [1, 1, 1, 1] : tensor<1x32x4096x128xbf16> to tensor<1x32x4096x64xbf16>
    %16 = tensor.empty() : tensor<1x32x4096x64xbf16>
    %17 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_12 : tensor<1x32x4096x64xbf16>) outs(%16 : tensor<1x32x4096x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %60 = arith.negf %in : bf16
      linalg.yield %60 : bf16
    } -> tensor<1x32x4096x64xbf16>
    %concat = tensor.concat dim(3) %17, %extracted_slice : (tensor<1x32x4096x64xbf16>, tensor<1x32x4096x64xbf16>) -> tensor<1x32x4096x128xbf16>
    %18 = linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat, %expanded_11 : tensor<1x32x4096x128xbf16>, tensor<1x1x4096x128xbf16>) outs(%6 : tensor<1x32x4096x128xbf16>) {
    ^bb0(%in: bf16, %in_32: bf16, %out: bf16):
      %60 = arith.mulf %in, %in_32 : bf16
      linalg.yield %60 : bf16
    } -> tensor<1x32x4096x128xbf16>
    %19 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%15, %18 : tensor<1x32x4096x128xbf16>, tensor<1x32x4096x128xbf16>) outs(%6 : tensor<1x32x4096x128xbf16>) {
    ^bb0(%in: bf16, %in_32: bf16, %out: bf16):
      %60 = arith.addf %in, %in_32 : bf16
      linalg.yield %60 : bf16
    } -> tensor<1x32x4096x128xbf16>
    %20 = linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_6, %expanded_10 : tensor<1x8x4096x128xbf16>, tensor<1x1x4096x128xbf16>) outs(%12 : tensor<1x8x4096x128xbf16>) {
    ^bb0(%in: bf16, %in_32: bf16, %out: bf16):
      %60 = arith.mulf %in, %in_32 : bf16
      linalg.yield %60 : bf16
    } -> tensor<1x8x4096x128xbf16>
    %extracted_slice_13 = tensor.extract_slice %transposed_6[0, 0, 0, 0] [1, 8, 4096, 64] [1, 1, 1, 1] : tensor<1x8x4096x128xbf16> to tensor<1x8x4096x64xbf16>
    %extracted_slice_14 = tensor.extract_slice %transposed_6[0, 0, 0, 64] [1, 8, 4096, 64] [1, 1, 1, 1] : tensor<1x8x4096x128xbf16> to tensor<1x8x4096x64xbf16>
    %21 = tensor.empty() : tensor<1x8x4096x64xbf16>
    %22 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_14 : tensor<1x8x4096x64xbf16>) outs(%21 : tensor<1x8x4096x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %60 = arith.negf %in : bf16
      linalg.yield %60 : bf16
    } -> tensor<1x8x4096x64xbf16>
    %concat_15 = tensor.concat dim(3) %22, %extracted_slice_13 : (tensor<1x8x4096x64xbf16>, tensor<1x8x4096x64xbf16>) -> tensor<1x8x4096x128xbf16>
    %23 = linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_15, %expanded_11 : tensor<1x8x4096x128xbf16>, tensor<1x1x4096x128xbf16>) outs(%12 : tensor<1x8x4096x128xbf16>) {
    ^bb0(%in: bf16, %in_32: bf16, %out: bf16):
      %60 = arith.mulf %in, %in_32 : bf16
      linalg.yield %60 : bf16
    } -> tensor<1x8x4096x128xbf16>
    %24 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%20, %23 : tensor<1x8x4096x128xbf16>, tensor<1x8x4096x128xbf16>) outs(%12 : tensor<1x8x4096x128xbf16>) {
    ^bb0(%in: bf16, %in_32: bf16, %out: bf16):
      %60 = arith.addf %in, %in_32 : bf16
      linalg.yield %60 : bf16
    } -> tensor<1x8x4096x128xbf16>
    %25 = tensor.empty() : tensor<1x8x4x4096x128xbf16>
    %26 = linalg.generic {indexing_maps = [#map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%24 : tensor<1x8x4096x128xbf16>) outs(%25 : tensor<1x8x4x4096x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x8x4x4096x128xbf16>
    %collapsed_16 = tensor.collapse_shape %26 [[0], [1, 2], [3], [4]] : tensor<1x8x4x4096x128xbf16> into tensor<1x32x4096x128xbf16>
    %27 = linalg.generic {indexing_maps = [#map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%transposed_9 : tensor<1x8x4096x128xbf16>) outs(%25 : tensor<1x8x4x4096x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x8x4x4096x128xbf16>
    %28 = tensor.empty() : tensor<1x32x128x4096xbf16>
    %transposed_17 = linalg.transpose ins(%collapsed_16 : tensor<1x32x4096x128xbf16>) outs(%28 : tensor<1x32x128x4096xbf16>) permutation = [0, 1, 3, 2] 
    %collapsed_18 = tensor.collapse_shape %19 [[0, 1], [2], [3]] : tensor<1x32x4096x128xbf16> into tensor<32x4096x128xbf16>
    %29 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%collapsed_18 : tensor<32x4096x128xbf16>) outs(%6 : tensor<1x32x4096x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x4096x128xbf16>
    %collapsed_19 = tensor.collapse_shape %transposed_17 [[0, 1], [2], [3]] : tensor<1x32x128x4096xbf16> into tensor<32x128x4096xbf16>
    %30 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%collapsed_19 : tensor<32x128x4096xbf16>) outs(%28 : tensor<1x32x128x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x128x4096xbf16>
    %collapsed_20 = tensor.collapse_shape %29 [[0, 1], [2], [3]] : tensor<1x32x4096x128xbf16> into tensor<32x4096x128xbf16>
    %collapsed_21 = tensor.collapse_shape %30 [[0, 1], [2], [3]] : tensor<1x32x128x4096xbf16> into tensor<32x128x4096xbf16>
    %31 = tensor.empty() : tensor<32x4096x4096xbf16>
    %32 = linalg.fill ins(%cst : bf16) outs(%31 : tensor<32x4096x4096xbf16>) -> tensor<32x4096x4096xbf16>
    %33 = linalg.batch_matmul ins(%collapsed_20, %collapsed_21 : tensor<32x4096x128xbf16>, tensor<32x128x4096xbf16>) outs(%32 : tensor<32x4096x4096xbf16>) -> tensor<32x4096x4096xbf16>
    %expanded_22 = tensor.expand_shape %33 [[0, 1], [2], [3]] output_shape [1, 32, 4096, 4096] : tensor<32x4096x4096xbf16> into tensor<1x32x4096x4096xbf16>
    %34 = tensor.empty() : tensor<1x32x4096x4096xbf16>
    %35 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_22 : tensor<1x32x4096x4096xbf16>) outs(%34 : tensor<1x32x4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %60 = arith.truncf %cst_2 : f64 to bf16
      %61 = arith.mulf %in, %60 : bf16
      linalg.yield %61 : bf16
    } -> tensor<1x32x4096x4096xbf16>
    %36 = linalg.generic {indexing_maps = [#map2, #map7, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%35, %arg3 : tensor<1x32x4096x4096xbf16>, tensor<1x32x1x1xbf16>) outs(%34 : tensor<1x32x4096x4096xbf16>) {
    ^bb0(%in: bf16, %in_32: bf16, %out: bf16):
      %60 = arith.addf %in, %in_32 : bf16
      linalg.yield %60 : bf16
    } -> tensor<1x32x4096x4096xbf16>
    %37 = tensor.empty() : tensor<1x32x4096x4096xf32>
    %38 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%36 : tensor<1x32x4096x4096xbf16>) outs(%37 : tensor<1x32x4096x4096xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %60 = arith.extf %in : bf16 to f32
      linalg.yield %60 : f32
    } -> tensor<1x32x4096x4096xf32>
    %39 = tensor.empty() : tensor<1x32x4096xi64>
    %40 = linalg.fill ins(%c0_i64 : i64) outs(%39 : tensor<1x32x4096xi64>) -> tensor<1x32x4096xi64>
    %41 = tensor.empty() : tensor<1x32x4096xf32>
    %42 = linalg.fill ins(%cst_0 : f32) outs(%41 : tensor<1x32x4096xf32>) -> tensor<1x32x4096xf32>
    %43:2 = linalg.generic {indexing_maps = [#map2, #map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%38 : tensor<1x32x4096x4096xf32>) outs(%42, %40 : tensor<1x32x4096xf32>, tensor<1x32x4096xi64>) {
    ^bb0(%in: f32, %out: f32, %out_32: i64):
      %60 = linalg.index 3 : index
      %61 = arith.index_cast %60 : index to i64
      %62 = arith.maximumf %in, %out : f32
      %63 = arith.cmpf ogt, %in, %out : f32
      %64 = arith.select %63, %61, %out_32 : i64
      linalg.yield %62, %64 : f32, i64
    } -> (tensor<1x32x4096xf32>, tensor<1x32x4096xi64>)
    %expanded_23 = tensor.expand_shape %43#0 [[0], [1], [2, 3]] output_shape [1, 32, 4096, 1] : tensor<1x32x4096xf32> into tensor<1x32x4096x1xf32>
    %44 = linalg.generic {indexing_maps = [#map2, #map9, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%38, %expanded_23 : tensor<1x32x4096x4096xf32>, tensor<1x32x4096x1xf32>) outs(%37 : tensor<1x32x4096x4096xf32>) {
    ^bb0(%in: f32, %in_32: f32, %out: f32):
      %60 = arith.subf %in, %in_32 : f32
      linalg.yield %60 : f32
    } -> tensor<1x32x4096x4096xf32>
    %45 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%44 : tensor<1x32x4096x4096xf32>) outs(%37 : tensor<1x32x4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %60 = math.exp %in : f32
      linalg.yield %60 : f32
    } -> tensor<1x32x4096x4096xf32>
    %46 = tensor.empty() : tensor<1x32x4096x1xf32>
    %47 = linalg.fill ins(%cst_1 : f32) outs(%46 : tensor<1x32x4096x1xf32>) -> tensor<1x32x4096x1xf32>
    %48 = linalg.generic {indexing_maps = [#map2, #map9], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%45 : tensor<1x32x4096x4096xf32>) outs(%47 : tensor<1x32x4096x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %60 = arith.addf %in, %out : f32
      linalg.yield %60 : f32
    } -> tensor<1x32x4096x1xf32>
    %49 = linalg.generic {indexing_maps = [#map2, #map9, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%45, %48 : tensor<1x32x4096x4096xf32>, tensor<1x32x4096x1xf32>) outs(%37 : tensor<1x32x4096x4096xf32>) {
    ^bb0(%in: f32, %in_32: f32, %out: f32):
      %60 = arith.divf %in, %in_32 : f32
      linalg.yield %60 : f32
    } -> tensor<1x32x4096x4096xf32>
    %50 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%49 : tensor<1x32x4096x4096xf32>) outs(%34 : tensor<1x32x4096x4096xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %60 = arith.truncf %in : f32 to bf16
      linalg.yield %60 : bf16
    } -> tensor<1x32x4096x4096xbf16>
    %collapsed_24 = tensor.collapse_shape %50 [[0, 1], [2], [3]] : tensor<1x32x4096x4096xbf16> into tensor<32x4096x4096xbf16>
    %51 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%collapsed_24 : tensor<32x4096x4096xbf16>) outs(%34 : tensor<1x32x4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x4096x4096xbf16>
    %collapsed_25 = tensor.collapse_shape %27 [[0, 1, 2], [3], [4]] : tensor<1x8x4x4096x128xbf16> into tensor<32x4096x128xbf16>
    %52 = linalg.generic {indexing_maps = [#map6, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%collapsed_25 : tensor<32x4096x128xbf16>) outs(%6 : tensor<1x32x4096x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x4096x128xbf16>
    %collapsed_26 = tensor.collapse_shape %51 [[0, 1], [2], [3]] : tensor<1x32x4096x4096xbf16> into tensor<32x4096x4096xbf16>
    %collapsed_27 = tensor.collapse_shape %52 [[0, 1], [2], [3]] : tensor<1x32x4096x128xbf16> into tensor<32x4096x128xbf16>
    %53 = tensor.empty() : tensor<32x4096x128xbf16>
    %54 = linalg.fill ins(%cst : bf16) outs(%53 : tensor<32x4096x128xbf16>) -> tensor<32x4096x128xbf16>
    %55 = linalg.batch_matmul ins(%collapsed_26, %collapsed_27 : tensor<32x4096x4096xbf16>, tensor<32x4096x128xbf16>) outs(%54 : tensor<32x4096x128xbf16>) -> tensor<32x4096x128xbf16>
    %expanded_28 = tensor.expand_shape %55 [[0, 1], [2], [3]] output_shape [1, 32, 4096, 128] : tensor<32x4096x128xbf16> into tensor<1x32x4096x128xbf16>
    %56 = tensor.empty() : tensor<1x4096x32x128xbf16>
    %transposed_29 = linalg.transpose ins(%expanded_28 : tensor<1x32x4096x128xbf16>) outs(%56 : tensor<1x4096x32x128xbf16>) permutation = [0, 2, 1, 3] 
    %transposed_30 = linalg.transpose ins(%arg7 : tensor<4096x4096xbf16>) outs(%0 : tensor<4096x4096xbf16>) permutation = [1, 0] 
    %collapsed_31 = tensor.collapse_shape %transposed_29 [[0, 1], [2, 3]] : tensor<1x4096x32x128xbf16> into tensor<4096x4096xbf16>
    %57 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapsed_31 : tensor<4096x4096xbf16>) outs(%1 : tensor<1x4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x4096x4096xbf16>
    %58 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_30 : tensor<4096x4096xbf16>) outs(%1 : tensor<1x4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x4096x4096xbf16>
    %59 = linalg.batch_matmul ins(%57, %58 : tensor<1x4096x4096xbf16>, tensor<1x4096x4096xbf16>) outs(%4 : tensor<1x4096x4096xbf16>) -> tensor<1x4096x4096xbf16>
    return %59, %50 : tensor<1x4096x4096xbf16>, tensor<1x32x4096x4096xbf16>
  }
}
