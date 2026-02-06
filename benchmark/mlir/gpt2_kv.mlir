#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, 0)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d2)>
#map4 = affine_map<(d0) -> (d0)>
#map5 = affine_map<(d0, d1) -> (d0, d1)>
#map6 = affine_map<(d0, d1) -> (d1)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
#map8 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map9 = affine_map<() -> ()>
#map10 = affine_map<(d0, d1, d2, d3) -> ()>
#map11 = affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>
#map12 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map13 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>
module {
  func.func @main(%arg0: tensor<1x1x1024x1024xi1>, %arg1: tensor<bf16>, %arg2: tensor<i64>, %arg3: tensor<i64>, %arg4: tensor<i64>, %arg5: tensor<i64>, %arg6: tensor<1x16x768xbf16>, %arg7: tensor<768xbf16>, %arg8: tensor<768xbf16>, %arg9: tensor<768x2304xbf16>, %arg10: tensor<2304xbf16>, %arg11: tensor<768x768xbf16>, %arg12: tensor<768xbf16>, %arg13: tensor<768xbf16>, %arg14: tensor<768xbf16>, %arg15: tensor<768x3072xbf16>, %arg16: tensor<3072xbf16>, %arg17: tensor<3072x768xbf16>, %arg18: tensor<768xbf16>) -> (tensor<1x16x768xbf16>, tensor<1x12x16x16xbf16>) {
    %cst = arith.constant dense<6.400000e+01> : tensor<bf16>
    %c0_i64 = arith.constant 0 : i64
    %cst_0 = arith.constant 0.000000e+00 : bf16
    %cst_1 = arith.constant 0.000000e+00 : f32
    %cst_2 = arith.constant 0xFF800000 : f32
    %c3_i64 = arith.constant 3 : i64
    %cst_3 = arith.constant 4.467770e-02 : bf16
    %cst_4 = arith.constant 7.968750e-01 : bf16
    %cst_5 = arith.constant 1.000000e+00 : bf16
    %cst_6 = arith.constant 5.000000e-01 : bf16
    %cst_7 = arith.constant 1.000000e-05 : f64
    %cst_8 = arith.constant 7.680000e+02 : bf16
    %cst_9 = arith.constant dense<-3.389530e+38> : tensor<bf16>
    %0 = tensor.empty() : tensor<1x16x1xbf16>
    %1 = linalg.fill ins(%cst_0 : bf16) outs(%0 : tensor<1x16x1xbf16>) -> tensor<1x16x1xbf16>
    %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg6 : tensor<1x16x768xbf16>) outs(%1 : tensor<1x16x1xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %105 = arith.addf %in, %out : bf16
      linalg.yield %105 : bf16
    } -> tensor<1x16x1xbf16>
    %3 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x16x1xbf16>) outs(%0 : tensor<1x16x1xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %105 = arith.divf %in, %cst_8 : bf16
      linalg.yield %105 : bf16
    } -> tensor<1x16x1xbf16>
    %4 = tensor.empty() : tensor<1x16x768xbf16>
    %collapsed = tensor.collapse_shape %3 [[0], [1, 2]] : tensor<1x16x1xbf16> into tensor<1x16xbf16>
    %5 = linalg.generic {indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapsed : tensor<1x16xbf16>) outs(%4 : tensor<1x16x768xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x16x768xbf16>
    %6 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg6, %5 : tensor<1x16x768xbf16>, tensor<1x16x768xbf16>) outs(%4 : tensor<1x16x768xbf16>) {
    ^bb0(%in: bf16, %in_41: bf16, %out: bf16):
      %105 = arith.subf %in, %in_41 : bf16
      linalg.yield %105 : bf16
    } -> tensor<1x16x768xbf16>
    %7 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%6, %6 : tensor<1x16x768xbf16>, tensor<1x16x768xbf16>) outs(%4 : tensor<1x16x768xbf16>) {
    ^bb0(%in: bf16, %in_41: bf16, %out: bf16):
      %105 = arith.mulf %in, %in_41 : bf16
      linalg.yield %105 : bf16
    } -> tensor<1x16x768xbf16>
    %8 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%7 : tensor<1x16x768xbf16>) outs(%1 : tensor<1x16x1xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %105 = arith.addf %in, %out : bf16
      linalg.yield %105 : bf16
    } -> tensor<1x16x1xbf16>
    %9 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%8 : tensor<1x16x1xbf16>) outs(%0 : tensor<1x16x1xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %105 = arith.divf %in, %cst_8 : bf16
      linalg.yield %105 : bf16
    } -> tensor<1x16x1xbf16>
    %10 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%9 : tensor<1x16x1xbf16>) outs(%0 : tensor<1x16x1xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %105 = arith.truncf %cst_7 : f64 to bf16
      %106 = arith.addf %in, %105 : bf16
      linalg.yield %106 : bf16
    } -> tensor<1x16x1xbf16>
    %11 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%10 : tensor<1x16x1xbf16>) outs(%0 : tensor<1x16x1xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %105 = math.rsqrt %in : bf16
      linalg.yield %105 : bf16
    } -> tensor<1x16x1xbf16>
    %collapsed_10 = tensor.collapse_shape %11 [[0], [1, 2]] : tensor<1x16x1xbf16> into tensor<1x16xbf16>
    %12 = linalg.generic {indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapsed_10 : tensor<1x16xbf16>) outs(%4 : tensor<1x16x768xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x16x768xbf16>
    %13 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%6, %12 : tensor<1x16x768xbf16>, tensor<1x16x768xbf16>) outs(%4 : tensor<1x16x768xbf16>) {
    ^bb0(%in: bf16, %in_41: bf16, %out: bf16):
      %105 = arith.mulf %in, %in_41 : bf16
      linalg.yield %105 : bf16
    } -> tensor<1x16x768xbf16>
    %14 = linalg.generic {indexing_maps = [#map, #map3, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%13, %arg7 : tensor<1x16x768xbf16>, tensor<768xbf16>) outs(%4 : tensor<1x16x768xbf16>) {
    ^bb0(%in: bf16, %in_41: bf16, %out: bf16):
      %105 = arith.mulf %in, %in_41 : bf16
      linalg.yield %105 : bf16
    } -> tensor<1x16x768xbf16>
    %15 = linalg.generic {indexing_maps = [#map, #map3, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%14, %arg8 : tensor<1x16x768xbf16>, tensor<768xbf16>) outs(%4 : tensor<1x16x768xbf16>) {
    ^bb0(%in: bf16, %in_41: bf16, %out: bf16):
      %105 = arith.addf %in, %in_41 : bf16
      linalg.yield %105 : bf16
    } -> tensor<1x16x768xbf16>
    %collapsed_11 = tensor.collapse_shape %15 [[0, 1], [2]] : tensor<1x16x768xbf16> into tensor<16x768xbf16>
    %16 = tensor.empty() : tensor<2304xf32>
    %17 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel"]} ins(%arg10 : tensor<2304xbf16>) outs(%16 : tensor<2304xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %105 = arith.extf %in : bf16 to f32
      linalg.yield %105 : f32
    } -> tensor<2304xf32>
    %18 = tensor.empty() : tensor<16x768xf32>
    %19 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel"]} ins(%collapsed_11 : tensor<16x768xbf16>) outs(%18 : tensor<16x768xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %105 = arith.extf %in : bf16 to f32
      linalg.yield %105 : f32
    } -> tensor<16x768xf32>
    %20 = tensor.empty() : tensor<768x2304xf32>
    %21 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel"]} ins(%arg9 : tensor<768x2304xbf16>) outs(%20 : tensor<768x2304xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %105 = arith.extf %in : bf16 to f32
      linalg.yield %105 : f32
    } -> tensor<768x2304xf32>
    %22 = tensor.empty() : tensor<16x2304xf32>
    %23 = linalg.fill ins(%cst_1 : f32) outs(%22 : tensor<16x2304xf32>) -> tensor<16x2304xf32>
    %24 = linalg.matmul ins(%19, %21 : tensor<16x768xf32>, tensor<768x2304xf32>) outs(%23 : tensor<16x2304xf32>) -> tensor<16x2304xf32>
    %25 = linalg.generic {indexing_maps = [#map5, #map6, #map5], iterator_types = ["parallel", "parallel"]} ins(%24, %17 : tensor<16x2304xf32>, tensor<2304xf32>) outs(%22 : tensor<16x2304xf32>) {
    ^bb0(%in: f32, %in_41: f32, %out: f32):
      %105 = arith.addf %in, %in_41 : f32
      linalg.yield %105 : f32
    } -> tensor<16x2304xf32>
    %26 = tensor.empty() : tensor<16x2304xbf16>
    %27 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel"]} ins(%25 : tensor<16x2304xf32>) outs(%26 : tensor<16x2304xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %105 = arith.truncf %in : f32 to bf16
      linalg.yield %105 : bf16
    } -> tensor<16x2304xbf16>
    %expanded = tensor.expand_shape %27 [[0, 1], [2]] output_shape [1, 16, 2304] : tensor<16x2304xbf16> into tensor<1x16x2304xbf16>
    %extracted_slice = tensor.extract_slice %expanded[0, 0, 0] [1, 16, 768] [1, 1, 1] : tensor<1x16x2304xbf16> to tensor<1x16x768xbf16>
    %extracted_slice_12 = tensor.extract_slice %expanded[0, 0, 768] [1, 16, 768] [1, 1, 1] : tensor<1x16x2304xbf16> to tensor<1x16x768xbf16>
    %extracted_slice_13 = tensor.extract_slice %expanded[0, 0, 1536] [1, 16, 768] [1, 1, 1] : tensor<1x16x2304xbf16> to tensor<1x16x768xbf16>
    %expanded_14 = tensor.expand_shape %extracted_slice [[0], [1], [2, 3]] output_shape [1, 16, 12, 64] : tensor<1x16x768xbf16> into tensor<1x16x12x64xbf16>
    %28 = tensor.empty() : tensor<1x12x16x64xbf16>
    %transposed = linalg.transpose ins(%expanded_14 : tensor<1x16x12x64xbf16>) outs(%28 : tensor<1x12x16x64xbf16>) permutation = [0, 2, 1, 3] 
    %expanded_15 = tensor.expand_shape %extracted_slice_12 [[0], [1], [2, 3]] output_shape [1, 16, 12, 64] : tensor<1x16x768xbf16> into tensor<1x16x12x64xbf16>
    %expanded_16 = tensor.expand_shape %extracted_slice_13 [[0], [1], [2, 3]] output_shape [1, 16, 12, 64] : tensor<1x16x768xbf16> into tensor<1x16x12x64xbf16>
    %transposed_17 = linalg.transpose ins(%expanded_16 : tensor<1x16x12x64xbf16>) outs(%28 : tensor<1x12x16x64xbf16>) permutation = [0, 2, 1, 3] 
    %29 = tensor.empty() : tensor<1x12x64x16xbf16>
    %transposed_18 = linalg.transpose ins(%expanded_15 : tensor<1x16x12x64xbf16>) outs(%29 : tensor<1x12x64x16xbf16>) permutation = [0, 2, 3, 1] 
    %collapsed_19 = tensor.collapse_shape %transposed [[0, 1], [2], [3]] : tensor<1x12x16x64xbf16> into tensor<12x16x64xbf16>
    %30 = linalg.generic {indexing_maps = [#map7, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%collapsed_19 : tensor<12x16x64xbf16>) outs(%28 : tensor<1x12x16x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x12x16x64xbf16>
    %collapsed_20 = tensor.collapse_shape %transposed_18 [[0, 1], [2], [3]] : tensor<1x12x64x16xbf16> into tensor<12x64x16xbf16>
    %31 = linalg.generic {indexing_maps = [#map7, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%collapsed_20 : tensor<12x64x16xbf16>) outs(%29 : tensor<1x12x64x16xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x12x64x16xbf16>
    %collapsed_21 = tensor.collapse_shape %30 [[0, 1], [2], [3]] : tensor<1x12x16x64xbf16> into tensor<12x16x64xbf16>
    %collapsed_22 = tensor.collapse_shape %31 [[0, 1], [2], [3]] : tensor<1x12x64x16xbf16> into tensor<12x64x16xbf16>
    %32 = tensor.empty() : tensor<12x16x16xbf16>
    %33 = linalg.fill ins(%cst_0 : bf16) outs(%32 : tensor<12x16x16xbf16>) -> tensor<12x16x16xbf16>
    %34 = linalg.batch_matmul ins(%collapsed_21, %collapsed_22 : tensor<12x16x64xbf16>, tensor<12x64x16xbf16>) outs(%33 : tensor<12x16x16xbf16>) -> tensor<12x16x16xbf16>
    %expanded_23 = tensor.expand_shape %34 [[0, 1], [2], [3]] output_shape [1, 12, 16, 16] : tensor<12x16x16xbf16> into tensor<1x12x16x16xbf16>
    %35 = tensor.empty() : tensor<bf16>
    %36 = linalg.generic {indexing_maps = [#map9, #map9], iterator_types = []} ins(%cst : tensor<bf16>) outs(%35 : tensor<bf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %105 = math.sqrt %in : bf16
      linalg.yield %105 : bf16
    } -> tensor<bf16>
    %37 = tensor.empty() : tensor<1x12x16x16xbf16>
    %38 = linalg.generic {indexing_maps = [#map8, #map10, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_23, %36 : tensor<1x12x16x16xbf16>, tensor<bf16>) outs(%37 : tensor<1x12x16x16xbf16>) {
    ^bb0(%in: bf16, %in_41: bf16, %out: bf16):
      %105 = arith.divf %in, %in_41 : bf16
      linalg.yield %105 : bf16
    } -> tensor<1x12x16x16xbf16>
    %extracted_slice_24 = tensor.extract_slice %arg0[0, 0, 0, 0] [1, 1, 16, 1024] [1, 1, 1, 1] : tensor<1x1x1024x1024xi1> to tensor<1x1x16x1024xi1>
    %extracted_slice_25 = tensor.extract_slice %extracted_slice_24[0, 0, 0, 0] [1, 1, 16, 16] [1, 1, 1, 1] : tensor<1x1x16x1024xi1> to tensor<1x1x16x16xi1>
    %39 = linalg.generic {indexing_maps = [#map11, #map8, #map10, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_25, %38, %cst_9 : tensor<1x1x16x16xi1>, tensor<1x12x16x16xbf16>, tensor<bf16>) outs(%37 : tensor<1x12x16x16xbf16>) {
    ^bb0(%in: i1, %in_41: bf16, %in_42: bf16, %out: bf16):
      %105 = arith.select %in, %in_41, %in_42 : bf16
      linalg.yield %105 : bf16
    } -> tensor<1x12x16x16xbf16>
    %40 = tensor.empty() : tensor<1x12x16x16xf32>
    %41 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%39 : tensor<1x12x16x16xbf16>) outs(%40 : tensor<1x12x16x16xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %105 = arith.extf %in : bf16 to f32
      linalg.yield %105 : f32
    } -> tensor<1x12x16x16xf32>
    %42 = tensor.empty() : tensor<1x12x16xi64>
    %43 = linalg.fill ins(%c0_i64 : i64) outs(%42 : tensor<1x12x16xi64>) -> tensor<1x12x16xi64>
    %44 = tensor.empty() : tensor<1x12x16xf32>
    %45 = linalg.fill ins(%cst_2 : f32) outs(%44 : tensor<1x12x16xf32>) -> tensor<1x12x16xf32>
    %46:2 = linalg.generic {indexing_maps = [#map8, #map12, #map12], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%41 : tensor<1x12x16x16xf32>) outs(%45, %43 : tensor<1x12x16xf32>, tensor<1x12x16xi64>) {
    ^bb0(%in: f32, %out: f32, %out_41: i64):
      %105 = linalg.index 3 : index
      %106 = arith.index_cast %105 : index to i64
      %107 = arith.maximumf %in, %out : f32
      %108 = arith.cmpf ogt, %in, %out : f32
      %109 = arith.select %108, %106, %out_41 : i64
      linalg.yield %107, %109 : f32, i64
    } -> (tensor<1x12x16xf32>, tensor<1x12x16xi64>)
    %expanded_26 = tensor.expand_shape %46#0 [[0], [1], [2, 3]] output_shape [1, 12, 16, 1] : tensor<1x12x16xf32> into tensor<1x12x16x1xf32>
    %47 = linalg.generic {indexing_maps = [#map8, #map13, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%41, %expanded_26 : tensor<1x12x16x16xf32>, tensor<1x12x16x1xf32>) outs(%40 : tensor<1x12x16x16xf32>) {
    ^bb0(%in: f32, %in_41: f32, %out: f32):
      %105 = arith.subf %in, %in_41 : f32
      linalg.yield %105 : f32
    } -> tensor<1x12x16x16xf32>
    %48 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%47 : tensor<1x12x16x16xf32>) outs(%40 : tensor<1x12x16x16xf32>) {
    ^bb0(%in: f32, %out: f32):
      %105 = math.exp %in : f32
      linalg.yield %105 : f32
    } -> tensor<1x12x16x16xf32>
    %49 = tensor.empty() : tensor<1x12x16x1xf32>
    %50 = linalg.fill ins(%cst_1 : f32) outs(%49 : tensor<1x12x16x1xf32>) -> tensor<1x12x16x1xf32>
    %51 = linalg.generic {indexing_maps = [#map8, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%48 : tensor<1x12x16x16xf32>) outs(%50 : tensor<1x12x16x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %105 = arith.addf %in, %out : f32
      linalg.yield %105 : f32
    } -> tensor<1x12x16x1xf32>
    %52 = linalg.generic {indexing_maps = [#map8, #map13, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%48, %51 : tensor<1x12x16x16xf32>, tensor<1x12x16x1xf32>) outs(%40 : tensor<1x12x16x16xf32>) {
    ^bb0(%in: f32, %in_41: f32, %out: f32):
      %105 = arith.divf %in, %in_41 : f32
      linalg.yield %105 : f32
    } -> tensor<1x12x16x16xf32>
    %53 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%52 : tensor<1x12x16x16xf32>) outs(%37 : tensor<1x12x16x16xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %105 = arith.truncf %in : f32 to bf16
      linalg.yield %105 : bf16
    } -> tensor<1x12x16x16xbf16>
    %collapsed_27 = tensor.collapse_shape %53 [[0, 1], [2], [3]] : tensor<1x12x16x16xbf16> into tensor<12x16x16xbf16>
    %54 = linalg.generic {indexing_maps = [#map7, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%collapsed_27 : tensor<12x16x16xbf16>) outs(%37 : tensor<1x12x16x16xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x12x16x16xbf16>
    %collapsed_28 = tensor.collapse_shape %transposed_17 [[0, 1], [2], [3]] : tensor<1x12x16x64xbf16> into tensor<12x16x64xbf16>
    %55 = linalg.generic {indexing_maps = [#map7, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%collapsed_28 : tensor<12x16x64xbf16>) outs(%28 : tensor<1x12x16x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x12x16x64xbf16>
    %collapsed_29 = tensor.collapse_shape %54 [[0, 1], [2], [3]] : tensor<1x12x16x16xbf16> into tensor<12x16x16xbf16>
    %collapsed_30 = tensor.collapse_shape %55 [[0, 1], [2], [3]] : tensor<1x12x16x64xbf16> into tensor<12x16x64xbf16>
    %56 = tensor.empty() : tensor<12x16x64xbf16>
    %57 = linalg.fill ins(%cst_0 : bf16) outs(%56 : tensor<12x16x64xbf16>) -> tensor<12x16x64xbf16>
    %58 = linalg.batch_matmul ins(%collapsed_29, %collapsed_30 : tensor<12x16x16xbf16>, tensor<12x16x64xbf16>) outs(%57 : tensor<12x16x64xbf16>) -> tensor<12x16x64xbf16>
    %expanded_31 = tensor.expand_shape %58 [[0, 1], [2], [3]] output_shape [1, 12, 16, 64] : tensor<12x16x64xbf16> into tensor<1x12x16x64xbf16>
    %59 = tensor.empty() : tensor<1x16x12x64xbf16>
    %transposed_32 = linalg.transpose ins(%expanded_31 : tensor<1x12x16x64xbf16>) outs(%59 : tensor<1x16x12x64xbf16>) permutation = [0, 2, 1, 3] 
    %collapsed_33 = tensor.collapse_shape %transposed_32 [[0, 1], [2, 3]] : tensor<1x16x12x64xbf16> into tensor<16x768xbf16>
    %60 = tensor.empty() : tensor<768xf32>
    %61 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel"]} ins(%arg12 : tensor<768xbf16>) outs(%60 : tensor<768xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %105 = arith.extf %in : bf16 to f32
      linalg.yield %105 : f32
    } -> tensor<768xf32>
    %62 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel"]} ins(%collapsed_33 : tensor<16x768xbf16>) outs(%18 : tensor<16x768xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %105 = arith.extf %in : bf16 to f32
      linalg.yield %105 : f32
    } -> tensor<16x768xf32>
    %63 = tensor.empty() : tensor<768x768xf32>
    %64 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel"]} ins(%arg11 : tensor<768x768xbf16>) outs(%63 : tensor<768x768xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %105 = arith.extf %in : bf16 to f32
      linalg.yield %105 : f32
    } -> tensor<768x768xf32>
    %65 = linalg.fill ins(%cst_1 : f32) outs(%18 : tensor<16x768xf32>) -> tensor<16x768xf32>
    %66 = linalg.matmul ins(%62, %64 : tensor<16x768xf32>, tensor<768x768xf32>) outs(%65 : tensor<16x768xf32>) -> tensor<16x768xf32>
    %67 = linalg.generic {indexing_maps = [#map5, #map6, #map5], iterator_types = ["parallel", "parallel"]} ins(%66, %61 : tensor<16x768xf32>, tensor<768xf32>) outs(%18 : tensor<16x768xf32>) {
    ^bb0(%in: f32, %in_41: f32, %out: f32):
      %105 = arith.addf %in, %in_41 : f32
      linalg.yield %105 : f32
    } -> tensor<16x768xf32>
    %68 = tensor.empty() : tensor<16x768xbf16>
    %69 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel"]} ins(%67 : tensor<16x768xf32>) outs(%68 : tensor<16x768xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %105 = arith.truncf %in : f32 to bf16
      linalg.yield %105 : bf16
    } -> tensor<16x768xbf16>
    %expanded_34 = tensor.expand_shape %69 [[0, 1], [2]] output_shape [1, 16, 768] : tensor<16x768xbf16> into tensor<1x16x768xbf16>
    %70 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_34, %arg6 : tensor<1x16x768xbf16>, tensor<1x16x768xbf16>) outs(%4 : tensor<1x16x768xbf16>) {
    ^bb0(%in: bf16, %in_41: bf16, %out: bf16):
      %105 = arith.addf %in, %in_41 : bf16
      linalg.yield %105 : bf16
    } -> tensor<1x16x768xbf16>
    %71 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%70 : tensor<1x16x768xbf16>) outs(%1 : tensor<1x16x1xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %105 = arith.addf %in, %out : bf16
      linalg.yield %105 : bf16
    } -> tensor<1x16x1xbf16>
    %72 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%71 : tensor<1x16x1xbf16>) outs(%0 : tensor<1x16x1xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %105 = arith.divf %in, %cst_8 : bf16
      linalg.yield %105 : bf16
    } -> tensor<1x16x1xbf16>
    %collapsed_35 = tensor.collapse_shape %72 [[0], [1, 2]] : tensor<1x16x1xbf16> into tensor<1x16xbf16>
    %73 = linalg.generic {indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapsed_35 : tensor<1x16xbf16>) outs(%4 : tensor<1x16x768xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x16x768xbf16>
    %74 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%70, %73 : tensor<1x16x768xbf16>, tensor<1x16x768xbf16>) outs(%4 : tensor<1x16x768xbf16>) {
    ^bb0(%in: bf16, %in_41: bf16, %out: bf16):
      %105 = arith.subf %in, %in_41 : bf16
      linalg.yield %105 : bf16
    } -> tensor<1x16x768xbf16>
    %75 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%74, %74 : tensor<1x16x768xbf16>, tensor<1x16x768xbf16>) outs(%4 : tensor<1x16x768xbf16>) {
    ^bb0(%in: bf16, %in_41: bf16, %out: bf16):
      %105 = arith.mulf %in, %in_41 : bf16
      linalg.yield %105 : bf16
    } -> tensor<1x16x768xbf16>
    %76 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%75 : tensor<1x16x768xbf16>) outs(%1 : tensor<1x16x1xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %105 = arith.addf %in, %out : bf16
      linalg.yield %105 : bf16
    } -> tensor<1x16x1xbf16>
    %77 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%76 : tensor<1x16x1xbf16>) outs(%0 : tensor<1x16x1xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %105 = arith.divf %in, %cst_8 : bf16
      linalg.yield %105 : bf16
    } -> tensor<1x16x1xbf16>
    %78 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%77 : tensor<1x16x1xbf16>) outs(%0 : tensor<1x16x1xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %105 = arith.truncf %cst_7 : f64 to bf16
      %106 = arith.addf %in, %105 : bf16
      linalg.yield %106 : bf16
    } -> tensor<1x16x1xbf16>
    %79 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%78 : tensor<1x16x1xbf16>) outs(%0 : tensor<1x16x1xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %105 = math.rsqrt %in : bf16
      linalg.yield %105 : bf16
    } -> tensor<1x16x1xbf16>
    %collapsed_36 = tensor.collapse_shape %79 [[0], [1, 2]] : tensor<1x16x1xbf16> into tensor<1x16xbf16>
    %80 = linalg.generic {indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapsed_36 : tensor<1x16xbf16>) outs(%4 : tensor<1x16x768xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x16x768xbf16>
    %81 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%74, %80 : tensor<1x16x768xbf16>, tensor<1x16x768xbf16>) outs(%4 : tensor<1x16x768xbf16>) {
    ^bb0(%in: bf16, %in_41: bf16, %out: bf16):
      %105 = arith.mulf %in, %in_41 : bf16
      linalg.yield %105 : bf16
    } -> tensor<1x16x768xbf16>
    %82 = linalg.generic {indexing_maps = [#map, #map3, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%81, %arg13 : tensor<1x16x768xbf16>, tensor<768xbf16>) outs(%4 : tensor<1x16x768xbf16>) {
    ^bb0(%in: bf16, %in_41: bf16, %out: bf16):
      %105 = arith.mulf %in, %in_41 : bf16
      linalg.yield %105 : bf16
    } -> tensor<1x16x768xbf16>
    %83 = linalg.generic {indexing_maps = [#map, #map3, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%82, %arg14 : tensor<1x16x768xbf16>, tensor<768xbf16>) outs(%4 : tensor<1x16x768xbf16>) {
    ^bb0(%in: bf16, %in_41: bf16, %out: bf16):
      %105 = arith.addf %in, %in_41 : bf16
      linalg.yield %105 : bf16
    } -> tensor<1x16x768xbf16>
    %collapsed_37 = tensor.collapse_shape %83 [[0, 1], [2]] : tensor<1x16x768xbf16> into tensor<16x768xbf16>
    %84 = tensor.empty() : tensor<3072xf32>
    %85 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel"]} ins(%arg16 : tensor<3072xbf16>) outs(%84 : tensor<3072xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %105 = arith.extf %in : bf16 to f32
      linalg.yield %105 : f32
    } -> tensor<3072xf32>
    %86 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel"]} ins(%collapsed_37 : tensor<16x768xbf16>) outs(%18 : tensor<16x768xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %105 = arith.extf %in : bf16 to f32
      linalg.yield %105 : f32
    } -> tensor<16x768xf32>
    %87 = tensor.empty() : tensor<768x3072xf32>
    %88 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel"]} ins(%arg15 : tensor<768x3072xbf16>) outs(%87 : tensor<768x3072xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %105 = arith.extf %in : bf16 to f32
      linalg.yield %105 : f32
    } -> tensor<768x3072xf32>
    %89 = tensor.empty() : tensor<16x3072xf32>
    %90 = linalg.fill ins(%cst_1 : f32) outs(%89 : tensor<16x3072xf32>) -> tensor<16x3072xf32>
    %91 = linalg.matmul ins(%86, %88 : tensor<16x768xf32>, tensor<768x3072xf32>) outs(%90 : tensor<16x3072xf32>) -> tensor<16x3072xf32>
    %92 = linalg.generic {indexing_maps = [#map5, #map6, #map5], iterator_types = ["parallel", "parallel"]} ins(%91, %85 : tensor<16x3072xf32>, tensor<3072xf32>) outs(%89 : tensor<16x3072xf32>) {
    ^bb0(%in: f32, %in_41: f32, %out: f32):
      %105 = arith.addf %in, %in_41 : f32
      linalg.yield %105 : f32
    } -> tensor<16x3072xf32>
    %93 = tensor.empty() : tensor<16x3072xbf16>
    %94 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel"]} ins(%92 : tensor<16x3072xf32>) outs(%93 : tensor<16x3072xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %105 = arith.truncf %in : f32 to bf16
      linalg.yield %105 : bf16
    } -> tensor<16x3072xbf16>
    %expanded_38 = tensor.expand_shape %94 [[0, 1], [2]] output_shape [1, 16, 3072] : tensor<16x3072xbf16> into tensor<1x16x3072xbf16>
    %95 = tensor.empty() : tensor<1x16x3072xbf16>
    %96 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_38 : tensor<1x16x3072xbf16>) outs(%95 : tensor<1x16x3072xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %105 = math.fpowi %in, %c3_i64 : bf16, i64
      %106 = arith.mulf %105, %cst_3 : bf16
      %107 = arith.addf %in, %106 : bf16
      %108 = arith.mulf %107, %cst_4 : bf16
      %109 = math.tanh %108 : bf16
      %110 = arith.addf %109, %cst_5 : bf16
      %111 = arith.mulf %110, %cst_6 : bf16
      %112 = arith.mulf %in, %111 : bf16
      linalg.yield %112 : bf16
    } -> tensor<1x16x3072xbf16>
    %collapsed_39 = tensor.collapse_shape %96 [[0, 1], [2]] : tensor<1x16x3072xbf16> into tensor<16x3072xbf16>
    %97 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel"]} ins(%arg18 : tensor<768xbf16>) outs(%60 : tensor<768xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %105 = arith.extf %in : bf16 to f32
      linalg.yield %105 : f32
    } -> tensor<768xf32>
    %98 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel"]} ins(%collapsed_39 : tensor<16x3072xbf16>) outs(%89 : tensor<16x3072xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %105 = arith.extf %in : bf16 to f32
      linalg.yield %105 : f32
    } -> tensor<16x3072xf32>
    %99 = tensor.empty() : tensor<3072x768xf32>
    %100 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel"]} ins(%arg17 : tensor<3072x768xbf16>) outs(%99 : tensor<3072x768xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %105 = arith.extf %in : bf16 to f32
      linalg.yield %105 : f32
    } -> tensor<3072x768xf32>
    %101 = linalg.matmul ins(%98, %100 : tensor<16x3072xf32>, tensor<3072x768xf32>) outs(%65 : tensor<16x768xf32>) -> tensor<16x768xf32>
    %102 = linalg.generic {indexing_maps = [#map5, #map6, #map5], iterator_types = ["parallel", "parallel"]} ins(%101, %97 : tensor<16x768xf32>, tensor<768xf32>) outs(%18 : tensor<16x768xf32>) {
    ^bb0(%in: f32, %in_41: f32, %out: f32):
      %105 = arith.addf %in, %in_41 : f32
      linalg.yield %105 : f32
    } -> tensor<16x768xf32>
    %103 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel"]} ins(%102 : tensor<16x768xf32>) outs(%68 : tensor<16x768xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %105 = arith.truncf %in : f32 to bf16
      linalg.yield %105 : bf16
    } -> tensor<16x768xbf16>
    %expanded_40 = tensor.expand_shape %103 [[0, 1], [2]] output_shape [1, 16, 768] : tensor<16x768xbf16> into tensor<1x16x768xbf16>
    %104 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_40, %70 : tensor<1x16x768xbf16>, tensor<1x16x768xbf16>) outs(%4 : tensor<1x16x768xbf16>) {
    ^bb0(%in: bf16, %in_41: bf16, %out: bf16):
      %105 = arith.addf %in, %in_41 : bf16
      linalg.yield %105 : bf16
    } -> tensor<1x16x768xbf16>
    return %104, %53 : tensor<1x16x768xbf16>, tensor<1x12x16x16xbf16>
  }
}
