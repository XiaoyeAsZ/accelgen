#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, 0)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d2)>
#map4 = affine_map<(d0, d1) -> (d0, d1)>
#map5 = affine_map<(d0, d1) -> (d1)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map8 = affine_map<() -> ()>
#map9 = affine_map<(d0, d1, d2, d3) -> ()>
#map10 = affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>
#map11 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map12 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>
module {
  func.func @main(%arg0: tensor<1x1x1024x1024xi1>, %arg1: tensor<f32>, %arg2: tensor<i64>, %arg3: tensor<i64>, %arg4: tensor<i64>, %arg5: tensor<i64>, %arg6: tensor<1x16x768xbf16>, %arg7: tensor<768xf32>, %arg8: tensor<768xf32>, %arg9: tensor<768x2304xf32>, %arg10: tensor<2304xf32>, %arg11: tensor<768x768xf32>, %arg12: tensor<768xf32>, %arg13: tensor<768xf32>, %arg14: tensor<768xf32>, %arg15: tensor<768x3072xf32>, %arg16: tensor<3072xf32>, %arg17: tensor<3072x768xf32>, %arg18: tensor<768xf32>) -> (tensor<1x16x768xf32>, tensor<1x12x16x16xf32>) {
    %cst = arith.constant dense<6.400000e+01> : tensor<f32>
    %c0_i64 = arith.constant 0 : i64
    %cst_0 = arith.constant 0.000000e+00 : bf16
    %cst_1 = arith.constant 0.000000e+00 : f32
    %cst_2 = arith.constant 0xFF800000 : f32
    %c3_i64 = arith.constant 3 : i64
    %cst_3 = arith.constant 4.471500e-02 : f32
    %cst_4 = arith.constant 7.977240e-01 : f32
    %cst_5 = arith.constant 1.000000e+00 : f32
    %cst_6 = arith.constant 5.000000e-01 : f32
    %cst_7 = arith.constant 1.000000e-05 : f64
    %cst_8 = arith.constant 7.680000e+02 : bf16
    %cst_9 = arith.constant 7.680000e+02 : f32
    %cst_10 = arith.constant dense<-3.40282347E+38> : tensor<f32>
    %0 = tensor.empty() : tensor<1x16x1xbf16>
    %1 = linalg.fill ins(%cst_0 : bf16) outs(%0 : tensor<1x16x1xbf16>) -> tensor<1x16x1xbf16>
    %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg6 : tensor<1x16x768xbf16>) outs(%1 : tensor<1x16x1xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %80 = arith.addf %in, %out : bf16
      linalg.yield %80 : bf16
    } -> tensor<1x16x1xbf16>
    %3 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x16x1xbf16>) outs(%0 : tensor<1x16x1xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %80 = arith.divf %in, %cst_8 : bf16
      linalg.yield %80 : bf16
    } -> tensor<1x16x1xbf16>
    %4 = tensor.empty() : tensor<1x16x768xbf16>
    %collapsed = tensor.collapse_shape %3 [[0], [1, 2]] : tensor<1x16x1xbf16> into tensor<1x16xbf16>
    %5 = linalg.generic {indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapsed : tensor<1x16xbf16>) outs(%4 : tensor<1x16x768xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x16x768xbf16>
    %6 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg6, %5 : tensor<1x16x768xbf16>, tensor<1x16x768xbf16>) outs(%4 : tensor<1x16x768xbf16>) {
    ^bb0(%in: bf16, %in_42: bf16, %out: bf16):
      %80 = arith.subf %in, %in_42 : bf16
      linalg.yield %80 : bf16
    } -> tensor<1x16x768xbf16>
    %7 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%6, %6 : tensor<1x16x768xbf16>, tensor<1x16x768xbf16>) outs(%4 : tensor<1x16x768xbf16>) {
    ^bb0(%in: bf16, %in_42: bf16, %out: bf16):
      %80 = arith.mulf %in, %in_42 : bf16
      linalg.yield %80 : bf16
    } -> tensor<1x16x768xbf16>
    %8 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%7 : tensor<1x16x768xbf16>) outs(%1 : tensor<1x16x1xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %80 = arith.addf %in, %out : bf16
      linalg.yield %80 : bf16
    } -> tensor<1x16x1xbf16>
    %9 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%8 : tensor<1x16x1xbf16>) outs(%0 : tensor<1x16x1xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %80 = arith.divf %in, %cst_8 : bf16
      linalg.yield %80 : bf16
    } -> tensor<1x16x1xbf16>
    %10 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%9 : tensor<1x16x1xbf16>) outs(%0 : tensor<1x16x1xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %80 = arith.truncf %cst_7 : f64 to bf16
      %81 = arith.addf %in, %80 : bf16
      linalg.yield %81 : bf16
    } -> tensor<1x16x1xbf16>
    %11 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%10 : tensor<1x16x1xbf16>) outs(%0 : tensor<1x16x1xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %80 = math.rsqrt %in : bf16
      linalg.yield %80 : bf16
    } -> tensor<1x16x1xbf16>
    %collapsed_11 = tensor.collapse_shape %11 [[0], [1, 2]] : tensor<1x16x1xbf16> into tensor<1x16xbf16>
    %12 = linalg.generic {indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapsed_11 : tensor<1x16xbf16>) outs(%4 : tensor<1x16x768xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x16x768xbf16>
    %13 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%6, %12 : tensor<1x16x768xbf16>, tensor<1x16x768xbf16>) outs(%4 : tensor<1x16x768xbf16>) {
    ^bb0(%in: bf16, %in_42: bf16, %out: bf16):
      %80 = arith.mulf %in, %in_42 : bf16
      linalg.yield %80 : bf16
    } -> tensor<1x16x768xbf16>
    %14 = linalg.generic {indexing_maps = [#map, #map3, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%13, %arg7 : tensor<1x16x768xbf16>, tensor<768xf32>) outs(%4 : tensor<1x16x768xbf16>) {
    ^bb0(%in: bf16, %in_42: f32, %out: bf16):
      %80 = arith.truncf %in_42 : f32 to bf16
      %81 = arith.mulf %in, %80 : bf16
      linalg.yield %81 : bf16
    } -> tensor<1x16x768xbf16>
    %15 = linalg.generic {indexing_maps = [#map, #map3, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%14, %arg8 : tensor<1x16x768xbf16>, tensor<768xf32>) outs(%4 : tensor<1x16x768xbf16>) {
    ^bb0(%in: bf16, %in_42: f32, %out: bf16):
      %80 = arith.truncf %in_42 : f32 to bf16
      %81 = arith.addf %in, %80 : bf16
      linalg.yield %81 : bf16
    } -> tensor<1x16x768xbf16>
    %collapsed_12 = tensor.collapse_shape %15 [[0, 1], [2]] : tensor<1x16x768xbf16> into tensor<16x768xbf16>
    %16 = tensor.empty() : tensor<16x768xf32>
    %17 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel"]} ins(%collapsed_12 : tensor<16x768xbf16>) outs(%16 : tensor<16x768xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %80 = arith.extf %in : bf16 to f32
      linalg.yield %80 : f32
    } -> tensor<16x768xf32>
    %18 = tensor.empty() : tensor<16x2304xf32>
    %19 = linalg.fill ins(%cst_1 : f32) outs(%18 : tensor<16x2304xf32>) -> tensor<16x2304xf32>
    %20 = linalg.matmul ins(%17, %arg9 : tensor<16x768xf32>, tensor<768x2304xf32>) outs(%19 : tensor<16x2304xf32>) -> tensor<16x2304xf32>
    %21 = linalg.generic {indexing_maps = [#map4, #map5, #map4], iterator_types = ["parallel", "parallel"]} ins(%20, %arg10 : tensor<16x2304xf32>, tensor<2304xf32>) outs(%18 : tensor<16x2304xf32>) {
    ^bb0(%in: f32, %in_42: f32, %out: f32):
      %80 = arith.addf %in, %in_42 : f32
      linalg.yield %80 : f32
    } -> tensor<16x2304xf32>
    %expanded = tensor.expand_shape %21 [[0, 1], [2]] output_shape [1, 16, 2304] : tensor<16x2304xf32> into tensor<1x16x2304xf32>
    %extracted_slice = tensor.extract_slice %expanded[0, 0, 0] [1, 16, 768] [1, 1, 1] : tensor<1x16x2304xf32> to tensor<1x16x768xf32>
    %extracted_slice_13 = tensor.extract_slice %expanded[0, 0, 768] [1, 16, 768] [1, 1, 1] : tensor<1x16x2304xf32> to tensor<1x16x768xf32>
    %extracted_slice_14 = tensor.extract_slice %expanded[0, 0, 1536] [1, 16, 768] [1, 1, 1] : tensor<1x16x2304xf32> to tensor<1x16x768xf32>
    %expanded_15 = tensor.expand_shape %extracted_slice [[0], [1], [2, 3]] output_shape [1, 16, 12, 64] : tensor<1x16x768xf32> into tensor<1x16x12x64xf32>
    %22 = tensor.empty() : tensor<1x12x16x64xf32>
    %transposed = linalg.transpose ins(%expanded_15 : tensor<1x16x12x64xf32>) outs(%22 : tensor<1x12x16x64xf32>) permutation = [0, 2, 1, 3] 
    %expanded_16 = tensor.expand_shape %extracted_slice_13 [[0], [1], [2, 3]] output_shape [1, 16, 12, 64] : tensor<1x16x768xf32> into tensor<1x16x12x64xf32>
    %expanded_17 = tensor.expand_shape %extracted_slice_14 [[0], [1], [2, 3]] output_shape [1, 16, 12, 64] : tensor<1x16x768xf32> into tensor<1x16x12x64xf32>
    %transposed_18 = linalg.transpose ins(%expanded_17 : tensor<1x16x12x64xf32>) outs(%22 : tensor<1x12x16x64xf32>) permutation = [0, 2, 1, 3] 
    %23 = tensor.empty() : tensor<1x12x64x16xf32>
    %transposed_19 = linalg.transpose ins(%expanded_16 : tensor<1x16x12x64xf32>) outs(%23 : tensor<1x12x64x16xf32>) permutation = [0, 2, 3, 1] 
    %collapsed_20 = tensor.collapse_shape %transposed [[0, 1], [2], [3]] : tensor<1x12x16x64xf32> into tensor<12x16x64xf32>
    %24 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%collapsed_20 : tensor<12x16x64xf32>) outs(%22 : tensor<1x12x16x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x16x64xf32>
    %collapsed_21 = tensor.collapse_shape %transposed_19 [[0, 1], [2], [3]] : tensor<1x12x64x16xf32> into tensor<12x64x16xf32>
    %25 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%collapsed_21 : tensor<12x64x16xf32>) outs(%23 : tensor<1x12x64x16xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x64x16xf32>
    %collapsed_22 = tensor.collapse_shape %24 [[0, 1], [2], [3]] : tensor<1x12x16x64xf32> into tensor<12x16x64xf32>
    %collapsed_23 = tensor.collapse_shape %25 [[0, 1], [2], [3]] : tensor<1x12x64x16xf32> into tensor<12x64x16xf32>
    %26 = tensor.empty() : tensor<12x16x16xf32>
    %27 = linalg.fill ins(%cst_1 : f32) outs(%26 : tensor<12x16x16xf32>) -> tensor<12x16x16xf32>
    %28 = linalg.batch_matmul ins(%collapsed_22, %collapsed_23 : tensor<12x16x64xf32>, tensor<12x64x16xf32>) outs(%27 : tensor<12x16x16xf32>) -> tensor<12x16x16xf32>
    %expanded_24 = tensor.expand_shape %28 [[0, 1], [2], [3]] output_shape [1, 12, 16, 16] : tensor<12x16x16xf32> into tensor<1x12x16x16xf32>
    %29 = tensor.empty() : tensor<f32>
    %30 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = []} ins(%cst : tensor<f32>) outs(%29 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %80 = math.sqrt %in : f32
      linalg.yield %80 : f32
    } -> tensor<f32>
    %31 = tensor.empty() : tensor<1x12x16x16xf32>
    %32 = linalg.generic {indexing_maps = [#map7, #map9, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_24, %30 : tensor<1x12x16x16xf32>, tensor<f32>) outs(%31 : tensor<1x12x16x16xf32>) {
    ^bb0(%in: f32, %in_42: f32, %out: f32):
      %80 = arith.divf %in, %in_42 : f32
      linalg.yield %80 : f32
    } -> tensor<1x12x16x16xf32>
    %extracted_slice_25 = tensor.extract_slice %arg0[0, 0, 0, 0] [1, 1, 16, 1024] [1, 1, 1, 1] : tensor<1x1x1024x1024xi1> to tensor<1x1x16x1024xi1>
    %extracted_slice_26 = tensor.extract_slice %extracted_slice_25[0, 0, 0, 0] [1, 1, 16, 16] [1, 1, 1, 1] : tensor<1x1x16x1024xi1> to tensor<1x1x16x16xi1>
    %33 = linalg.generic {indexing_maps = [#map10, #map7, #map9, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_26, %32, %cst_10 : tensor<1x1x16x16xi1>, tensor<1x12x16x16xf32>, tensor<f32>) outs(%31 : tensor<1x12x16x16xf32>) {
    ^bb0(%in: i1, %in_42: f32, %in_43: f32, %out: f32):
      %80 = arith.select %in, %in_42, %in_43 : f32
      linalg.yield %80 : f32
    } -> tensor<1x12x16x16xf32>
    %34 = tensor.empty() : tensor<1x12x16xi64>
    %35 = linalg.fill ins(%c0_i64 : i64) outs(%34 : tensor<1x12x16xi64>) -> tensor<1x12x16xi64>
    %36 = tensor.empty() : tensor<1x12x16xf32>
    %37 = linalg.fill ins(%cst_2 : f32) outs(%36 : tensor<1x12x16xf32>) -> tensor<1x12x16xf32>
    %38:2 = linalg.generic {indexing_maps = [#map7, #map11, #map11], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%33 : tensor<1x12x16x16xf32>) outs(%37, %35 : tensor<1x12x16xf32>, tensor<1x12x16xi64>) {
    ^bb0(%in: f32, %out: f32, %out_42: i64):
      %80 = linalg.index 3 : index
      %81 = arith.index_cast %80 : index to i64
      %82 = arith.maximumf %in, %out : f32
      %83 = arith.cmpf ogt, %in, %out : f32
      %84 = arith.select %83, %81, %out_42 : i64
      linalg.yield %82, %84 : f32, i64
    } -> (tensor<1x12x16xf32>, tensor<1x12x16xi64>)
    %expanded_27 = tensor.expand_shape %38#0 [[0], [1], [2, 3]] output_shape [1, 12, 16, 1] : tensor<1x12x16xf32> into tensor<1x12x16x1xf32>
    %39 = linalg.generic {indexing_maps = [#map7, #map12, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%33, %expanded_27 : tensor<1x12x16x16xf32>, tensor<1x12x16x1xf32>) outs(%31 : tensor<1x12x16x16xf32>) {
    ^bb0(%in: f32, %in_42: f32, %out: f32):
      %80 = arith.subf %in, %in_42 : f32
      linalg.yield %80 : f32
    } -> tensor<1x12x16x16xf32>
    %40 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%39 : tensor<1x12x16x16xf32>) outs(%31 : tensor<1x12x16x16xf32>) {
    ^bb0(%in: f32, %out: f32):
      %80 = math.exp %in : f32
      linalg.yield %80 : f32
    } -> tensor<1x12x16x16xf32>
    %41 = tensor.empty() : tensor<1x12x16x1xf32>
    %42 = linalg.fill ins(%cst_1 : f32) outs(%41 : tensor<1x12x16x1xf32>) -> tensor<1x12x16x1xf32>
    %43 = linalg.generic {indexing_maps = [#map7, #map12], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%40 : tensor<1x12x16x16xf32>) outs(%42 : tensor<1x12x16x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %80 = arith.addf %in, %out : f32
      linalg.yield %80 : f32
    } -> tensor<1x12x16x1xf32>
    %44 = linalg.generic {indexing_maps = [#map7, #map12, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%40, %43 : tensor<1x12x16x16xf32>, tensor<1x12x16x1xf32>) outs(%31 : tensor<1x12x16x16xf32>) {
    ^bb0(%in: f32, %in_42: f32, %out: f32):
      %80 = arith.divf %in, %in_42 : f32
      linalg.yield %80 : f32
    } -> tensor<1x12x16x16xf32>
    %collapsed_28 = tensor.collapse_shape %44 [[0, 1], [2], [3]] : tensor<1x12x16x16xf32> into tensor<12x16x16xf32>
    %45 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%collapsed_28 : tensor<12x16x16xf32>) outs(%31 : tensor<1x12x16x16xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x16x16xf32>
    %collapsed_29 = tensor.collapse_shape %transposed_18 [[0, 1], [2], [3]] : tensor<1x12x16x64xf32> into tensor<12x16x64xf32>
    %46 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%collapsed_29 : tensor<12x16x64xf32>) outs(%22 : tensor<1x12x16x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x16x64xf32>
    %collapsed_30 = tensor.collapse_shape %45 [[0, 1], [2], [3]] : tensor<1x12x16x16xf32> into tensor<12x16x16xf32>
    %collapsed_31 = tensor.collapse_shape %46 [[0, 1], [2], [3]] : tensor<1x12x16x64xf32> into tensor<12x16x64xf32>
    %47 = tensor.empty() : tensor<12x16x64xf32>
    %48 = linalg.fill ins(%cst_1 : f32) outs(%47 : tensor<12x16x64xf32>) -> tensor<12x16x64xf32>
    %49 = linalg.batch_matmul ins(%collapsed_30, %collapsed_31 : tensor<12x16x16xf32>, tensor<12x16x64xf32>) outs(%48 : tensor<12x16x64xf32>) -> tensor<12x16x64xf32>
    %expanded_32 = tensor.expand_shape %49 [[0, 1], [2], [3]] output_shape [1, 12, 16, 64] : tensor<12x16x64xf32> into tensor<1x12x16x64xf32>
    %50 = tensor.empty() : tensor<1x16x12x64xf32>
    %transposed_33 = linalg.transpose ins(%expanded_32 : tensor<1x12x16x64xf32>) outs(%50 : tensor<1x16x12x64xf32>) permutation = [0, 2, 1, 3] 
    %collapsed_34 = tensor.collapse_shape %transposed_33 [[0, 1], [2, 3]] : tensor<1x16x12x64xf32> into tensor<16x768xf32>
    %51 = linalg.fill ins(%cst_1 : f32) outs(%16 : tensor<16x768xf32>) -> tensor<16x768xf32>
    %52 = linalg.matmul ins(%collapsed_34, %arg11 : tensor<16x768xf32>, tensor<768x768xf32>) outs(%51 : tensor<16x768xf32>) -> tensor<16x768xf32>
    %53 = linalg.generic {indexing_maps = [#map4, #map5, #map4], iterator_types = ["parallel", "parallel"]} ins(%52, %arg12 : tensor<16x768xf32>, tensor<768xf32>) outs(%16 : tensor<16x768xf32>) {
    ^bb0(%in: f32, %in_42: f32, %out: f32):
      %80 = arith.addf %in, %in_42 : f32
      linalg.yield %80 : f32
    } -> tensor<16x768xf32>
    %expanded_35 = tensor.expand_shape %53 [[0, 1], [2]] output_shape [1, 16, 768] : tensor<16x768xf32> into tensor<1x16x768xf32>
    %54 = tensor.empty() : tensor<1x16x768xf32>
    %55 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_35, %arg6 : tensor<1x16x768xf32>, tensor<1x16x768xbf16>) outs(%54 : tensor<1x16x768xf32>) {
    ^bb0(%in: f32, %in_42: bf16, %out: f32):
      %80 = arith.extf %in_42 : bf16 to f32
      %81 = arith.addf %in, %80 : f32
      linalg.yield %81 : f32
    } -> tensor<1x16x768xf32>
    %56 = tensor.empty() : tensor<1x16x1xf32>
    %57 = linalg.fill ins(%cst_1 : f32) outs(%56 : tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
    %58 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%55 : tensor<1x16x768xf32>) outs(%57 : tensor<1x16x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %80 = arith.addf %in, %out : f32
      linalg.yield %80 : f32
    } -> tensor<1x16x1xf32>
    %59 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%58 : tensor<1x16x1xf32>) outs(%56 : tensor<1x16x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %80 = arith.divf %in, %cst_9 : f32
      linalg.yield %80 : f32
    } -> tensor<1x16x1xf32>
    %collapsed_36 = tensor.collapse_shape %59 [[0], [1, 2]] : tensor<1x16x1xf32> into tensor<1x16xf32>
    %60 = linalg.generic {indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapsed_36 : tensor<1x16xf32>) outs(%54 : tensor<1x16x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x16x768xf32>
    %61 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%55, %60 : tensor<1x16x768xf32>, tensor<1x16x768xf32>) outs(%54 : tensor<1x16x768xf32>) {
    ^bb0(%in: f32, %in_42: f32, %out: f32):
      %80 = arith.subf %in, %in_42 : f32
      linalg.yield %80 : f32
    } -> tensor<1x16x768xf32>
    %62 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%61, %61 : tensor<1x16x768xf32>, tensor<1x16x768xf32>) outs(%54 : tensor<1x16x768xf32>) {
    ^bb0(%in: f32, %in_42: f32, %out: f32):
      %80 = arith.mulf %in, %in_42 : f32
      linalg.yield %80 : f32
    } -> tensor<1x16x768xf32>
    %63 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%62 : tensor<1x16x768xf32>) outs(%57 : tensor<1x16x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %80 = arith.addf %in, %out : f32
      linalg.yield %80 : f32
    } -> tensor<1x16x1xf32>
    %64 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%63 : tensor<1x16x1xf32>) outs(%56 : tensor<1x16x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %80 = arith.divf %in, %cst_9 : f32
      linalg.yield %80 : f32
    } -> tensor<1x16x1xf32>
    %65 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%64 : tensor<1x16x1xf32>) outs(%56 : tensor<1x16x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %80 = arith.truncf %cst_7 : f64 to f32
      %81 = arith.addf %in, %80 : f32
      linalg.yield %81 : f32
    } -> tensor<1x16x1xf32>
    %66 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%65 : tensor<1x16x1xf32>) outs(%56 : tensor<1x16x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %80 = math.rsqrt %in : f32
      linalg.yield %80 : f32
    } -> tensor<1x16x1xf32>
    %collapsed_37 = tensor.collapse_shape %66 [[0], [1, 2]] : tensor<1x16x1xf32> into tensor<1x16xf32>
    %67 = linalg.generic {indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapsed_37 : tensor<1x16xf32>) outs(%54 : tensor<1x16x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x16x768xf32>
    %68 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%61, %67 : tensor<1x16x768xf32>, tensor<1x16x768xf32>) outs(%54 : tensor<1x16x768xf32>) {
    ^bb0(%in: f32, %in_42: f32, %out: f32):
      %80 = arith.mulf %in, %in_42 : f32
      linalg.yield %80 : f32
    } -> tensor<1x16x768xf32>
    %69 = linalg.generic {indexing_maps = [#map, #map3, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%68, %arg13 : tensor<1x16x768xf32>, tensor<768xf32>) outs(%54 : tensor<1x16x768xf32>) {
    ^bb0(%in: f32, %in_42: f32, %out: f32):
      %80 = arith.mulf %in, %in_42 : f32
      linalg.yield %80 : f32
    } -> tensor<1x16x768xf32>
    %70 = linalg.generic {indexing_maps = [#map, #map3, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%69, %arg14 : tensor<1x16x768xf32>, tensor<768xf32>) outs(%54 : tensor<1x16x768xf32>) {
    ^bb0(%in: f32, %in_42: f32, %out: f32):
      %80 = arith.addf %in, %in_42 : f32
      linalg.yield %80 : f32
    } -> tensor<1x16x768xf32>
    %collapsed_38 = tensor.collapse_shape %70 [[0, 1], [2]] : tensor<1x16x768xf32> into tensor<16x768xf32>
    %71 = tensor.empty() : tensor<16x3072xf32>
    %72 = linalg.fill ins(%cst_1 : f32) outs(%71 : tensor<16x3072xf32>) -> tensor<16x3072xf32>
    %73 = linalg.matmul ins(%collapsed_38, %arg15 : tensor<16x768xf32>, tensor<768x3072xf32>) outs(%72 : tensor<16x3072xf32>) -> tensor<16x3072xf32>
    %74 = linalg.generic {indexing_maps = [#map4, #map5, #map4], iterator_types = ["parallel", "parallel"]} ins(%73, %arg16 : tensor<16x3072xf32>, tensor<3072xf32>) outs(%71 : tensor<16x3072xf32>) {
    ^bb0(%in: f32, %in_42: f32, %out: f32):
      %80 = arith.addf %in, %in_42 : f32
      linalg.yield %80 : f32
    } -> tensor<16x3072xf32>
    %expanded_39 = tensor.expand_shape %74 [[0, 1], [2]] output_shape [1, 16, 3072] : tensor<16x3072xf32> into tensor<1x16x3072xf32>
    %75 = tensor.empty() : tensor<1x16x3072xf32>
    %76 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_39 : tensor<1x16x3072xf32>) outs(%75 : tensor<1x16x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %80 = math.fpowi %in, %c3_i64 : f32, i64
      %81 = arith.mulf %80, %cst_3 : f32
      %82 = arith.addf %in, %81 : f32
      %83 = arith.mulf %82, %cst_4 : f32
      %84 = math.tanh %83 : f32
      %85 = arith.addf %84, %cst_5 : f32
      %86 = arith.mulf %85, %cst_6 : f32
      %87 = arith.mulf %in, %86 : f32
      linalg.yield %87 : f32
    } -> tensor<1x16x3072xf32>
    %collapsed_40 = tensor.collapse_shape %76 [[0, 1], [2]] : tensor<1x16x3072xf32> into tensor<16x3072xf32>
    %77 = linalg.matmul ins(%collapsed_40, %arg17 : tensor<16x3072xf32>, tensor<3072x768xf32>) outs(%51 : tensor<16x768xf32>) -> tensor<16x768xf32>
    %78 = linalg.generic {indexing_maps = [#map4, #map5, #map4], iterator_types = ["parallel", "parallel"]} ins(%77, %arg18 : tensor<16x768xf32>, tensor<768xf32>) outs(%16 : tensor<16x768xf32>) {
    ^bb0(%in: f32, %in_42: f32, %out: f32):
      %80 = arith.addf %in, %in_42 : f32
      linalg.yield %80 : f32
    } -> tensor<16x768xf32>
    %expanded_41 = tensor.expand_shape %78 [[0, 1], [2]] output_shape [1, 16, 768] : tensor<16x768xf32> into tensor<1x16x768xf32>
    %79 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_41, %55 : tensor<1x16x768xf32>, tensor<1x16x768xf32>) outs(%54 : tensor<1x16x768xf32>) {
    ^bb0(%in: f32, %in_42: f32, %out: f32):
      %80 = arith.addf %in, %in_42 : f32
      linalg.yield %80 : f32
    } -> tensor<1x16x768xf32>
    return %79, %44 : tensor<1x16x768xf32>, tensor<1x12x16x16xf32>
  }
}
