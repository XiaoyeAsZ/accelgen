#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map4 = affine_map<() -> ()>
#map5 = affine_map<(d0, d1, d2, d3) -> ()>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map8 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>
module {
  func.func @main(%arg0: tensor<1x1x1024x1024xi1>, %arg1: tensor<f32>, %arg2: tensor<i64>, %arg3: tensor<i64>, %arg4: tensor<i64>, %arg5: tensor<i64>, %arg6: tensor<1x16x768xbf16>, %arg7: tensor<768x2304xf32>, %arg8: tensor<2304xf32>, %arg9: tensor<768x768xf32>, %arg10: tensor<768xf32>) -> (tensor<1x16x768xf32>, tensor<1x12x16x16xf32>) {
    %cst = arith.constant dense<6.400000e+01> : tensor<f32>
    %c0_i64 = arith.constant 0 : i64
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 0xFF800000 : f32
    %cst_2 = arith.constant dense<-3.40282347E+38> : tensor<f32>
    %collapsed = tensor.collapse_shape %arg6 [[0, 1], [2]] : tensor<1x16x768xbf16> into tensor<16x768xbf16>
    %0 = tensor.empty() : tensor<16x768xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%collapsed : tensor<16x768xbf16>) outs(%0 : tensor<16x768xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %38 = arith.extf %in : bf16 to f32
      linalg.yield %38 : f32
    } -> tensor<16x768xf32>
    %2 = tensor.empty() : tensor<16x2304xf32>
    %3 = linalg.fill ins(%cst_0 : f32) outs(%2 : tensor<16x2304xf32>) -> tensor<16x2304xf32>
    %4 = linalg.matmul ins(%1, %arg7 : tensor<16x768xf32>, tensor<768x2304xf32>) outs(%3 : tensor<16x2304xf32>) -> tensor<16x2304xf32>
    %5 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%4, %arg8 : tensor<16x2304xf32>, tensor<2304xf32>) outs(%2 : tensor<16x2304xf32>) {
    ^bb0(%in: f32, %in_26: f32, %out: f32):
      %38 = arith.addf %in, %in_26 : f32
      linalg.yield %38 : f32
    } -> tensor<16x2304xf32>
    %expanded = tensor.expand_shape %5 [[0, 1], [2]] output_shape [1, 16, 2304] : tensor<16x2304xf32> into tensor<1x16x2304xf32>
    %extracted_slice = tensor.extract_slice %expanded[0, 0, 0] [1, 16, 768] [1, 1, 1] : tensor<1x16x2304xf32> to tensor<1x16x768xf32>
    %extracted_slice_3 = tensor.extract_slice %expanded[0, 0, 768] [1, 16, 768] [1, 1, 1] : tensor<1x16x2304xf32> to tensor<1x16x768xf32>
    %extracted_slice_4 = tensor.extract_slice %expanded[0, 0, 1536] [1, 16, 768] [1, 1, 1] : tensor<1x16x2304xf32> to tensor<1x16x768xf32>
    %expanded_5 = tensor.expand_shape %extracted_slice [[0], [1], [2, 3]] output_shape [1, 16, 12, 64] : tensor<1x16x768xf32> into tensor<1x16x12x64xf32>
    %6 = tensor.empty() : tensor<1x12x16x64xf32>
    %transposed = linalg.transpose ins(%expanded_5 : tensor<1x16x12x64xf32>) outs(%6 : tensor<1x12x16x64xf32>) permutation = [0, 2, 1, 3] 
    %expanded_6 = tensor.expand_shape %extracted_slice_3 [[0], [1], [2, 3]] output_shape [1, 16, 12, 64] : tensor<1x16x768xf32> into tensor<1x16x12x64xf32>
    %expanded_7 = tensor.expand_shape %extracted_slice_4 [[0], [1], [2, 3]] output_shape [1, 16, 12, 64] : tensor<1x16x768xf32> into tensor<1x16x12x64xf32>
    %transposed_8 = linalg.transpose ins(%expanded_7 : tensor<1x16x12x64xf32>) outs(%6 : tensor<1x12x16x64xf32>) permutation = [0, 2, 1, 3] 
    %7 = tensor.empty() : tensor<1x12x64x16xf32>
    %transposed_9 = linalg.transpose ins(%expanded_6 : tensor<1x16x12x64xf32>) outs(%7 : tensor<1x12x64x16xf32>) permutation = [0, 2, 3, 1] 
    %collapsed_10 = tensor.collapse_shape %transposed [[0, 1], [2], [3]] : tensor<1x12x16x64xf32> into tensor<12x16x64xf32>
    %8 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%collapsed_10 : tensor<12x16x64xf32>) outs(%6 : tensor<1x12x16x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x16x64xf32>
    %collapsed_11 = tensor.collapse_shape %transposed_9 [[0, 1], [2], [3]] : tensor<1x12x64x16xf32> into tensor<12x64x16xf32>
    %9 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%collapsed_11 : tensor<12x64x16xf32>) outs(%7 : tensor<1x12x64x16xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x64x16xf32>
    %collapsed_12 = tensor.collapse_shape %8 [[0, 1], [2], [3]] : tensor<1x12x16x64xf32> into tensor<12x16x64xf32>
    %collapsed_13 = tensor.collapse_shape %9 [[0, 1], [2], [3]] : tensor<1x12x64x16xf32> into tensor<12x64x16xf32>
    %10 = tensor.empty() : tensor<12x16x16xf32>
    %11 = linalg.fill ins(%cst_0 : f32) outs(%10 : tensor<12x16x16xf32>) -> tensor<12x16x16xf32>
    %12 = linalg.batch_matmul ins(%collapsed_12, %collapsed_13 : tensor<12x16x64xf32>, tensor<12x64x16xf32>) outs(%11 : tensor<12x16x16xf32>) -> tensor<12x16x16xf32>
    %expanded_14 = tensor.expand_shape %12 [[0, 1], [2], [3]] output_shape [1, 12, 16, 16] : tensor<12x16x16xf32> into tensor<1x12x16x16xf32>
    %13 = tensor.empty() : tensor<f32>
    %14 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = []} ins(%cst : tensor<f32>) outs(%13 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %38 = math.sqrt %in : f32
      linalg.yield %38 : f32
    } -> tensor<f32>
    %15 = tensor.empty() : tensor<1x12x16x16xf32>
    %16 = linalg.generic {indexing_maps = [#map3, #map5, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_14, %14 : tensor<1x12x16x16xf32>, tensor<f32>) outs(%15 : tensor<1x12x16x16xf32>) {
    ^bb0(%in: f32, %in_26: f32, %out: f32):
      %38 = arith.divf %in, %in_26 : f32
      linalg.yield %38 : f32
    } -> tensor<1x12x16x16xf32>
    %extracted_slice_15 = tensor.extract_slice %arg0[0, 0, 0, 0] [1, 1, 16, 1024] [1, 1, 1, 1] : tensor<1x1x1024x1024xi1> to tensor<1x1x16x1024xi1>
    %extracted_slice_16 = tensor.extract_slice %extracted_slice_15[0, 0, 0, 0] [1, 1, 16, 16] [1, 1, 1, 1] : tensor<1x1x16x1024xi1> to tensor<1x1x16x16xi1>
    %17 = linalg.generic {indexing_maps = [#map6, #map3, #map5, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_16, %16, %cst_2 : tensor<1x1x16x16xi1>, tensor<1x12x16x16xf32>, tensor<f32>) outs(%15 : tensor<1x12x16x16xf32>) {
    ^bb0(%in: i1, %in_26: f32, %in_27: f32, %out: f32):
      %38 = arith.select %in, %in_26, %in_27 : f32
      linalg.yield %38 : f32
    } -> tensor<1x12x16x16xf32>
    %18 = tensor.empty() : tensor<1x12x16xi64>
    %19 = linalg.fill ins(%c0_i64 : i64) outs(%18 : tensor<1x12x16xi64>) -> tensor<1x12x16xi64>
    %20 = tensor.empty() : tensor<1x12x16xf32>
    %21 = linalg.fill ins(%cst_1 : f32) outs(%20 : tensor<1x12x16xf32>) -> tensor<1x12x16xf32>
    %22:2 = linalg.generic {indexing_maps = [#map3, #map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%17 : tensor<1x12x16x16xf32>) outs(%21, %19 : tensor<1x12x16xf32>, tensor<1x12x16xi64>) {
    ^bb0(%in: f32, %out: f32, %out_26: i64):
      %38 = linalg.index 3 : index
      %39 = arith.index_cast %38 : index to i64
      %40 = arith.maximumf %in, %out : f32
      %41 = arith.cmpf ogt, %in, %out : f32
      %42 = arith.select %41, %39, %out_26 : i64
      linalg.yield %40, %42 : f32, i64
    } -> (tensor<1x12x16xf32>, tensor<1x12x16xi64>)
    %expanded_17 = tensor.expand_shape %22#0 [[0], [1], [2, 3]] output_shape [1, 12, 16, 1] : tensor<1x12x16xf32> into tensor<1x12x16x1xf32>
    %23 = linalg.generic {indexing_maps = [#map3, #map8, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%17, %expanded_17 : tensor<1x12x16x16xf32>, tensor<1x12x16x1xf32>) outs(%15 : tensor<1x12x16x16xf32>) {
    ^bb0(%in: f32, %in_26: f32, %out: f32):
      %38 = arith.subf %in, %in_26 : f32
      linalg.yield %38 : f32
    } -> tensor<1x12x16x16xf32>
    %24 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%23 : tensor<1x12x16x16xf32>) outs(%15 : tensor<1x12x16x16xf32>) {
    ^bb0(%in: f32, %out: f32):
      %38 = math.exp %in : f32
      linalg.yield %38 : f32
    } -> tensor<1x12x16x16xf32>
    %25 = tensor.empty() : tensor<1x12x16x1xf32>
    %26 = linalg.fill ins(%cst_0 : f32) outs(%25 : tensor<1x12x16x1xf32>) -> tensor<1x12x16x1xf32>
    %27 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%24 : tensor<1x12x16x16xf32>) outs(%26 : tensor<1x12x16x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %38 = arith.addf %in, %out : f32
      linalg.yield %38 : f32
    } -> tensor<1x12x16x1xf32>
    %28 = linalg.generic {indexing_maps = [#map3, #map8, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%24, %27 : tensor<1x12x16x16xf32>, tensor<1x12x16x1xf32>) outs(%15 : tensor<1x12x16x16xf32>) {
    ^bb0(%in: f32, %in_26: f32, %out: f32):
      %38 = arith.divf %in, %in_26 : f32
      linalg.yield %38 : f32
    } -> tensor<1x12x16x16xf32>
    %collapsed_18 = tensor.collapse_shape %28 [[0, 1], [2], [3]] : tensor<1x12x16x16xf32> into tensor<12x16x16xf32>
    %29 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%collapsed_18 : tensor<12x16x16xf32>) outs(%15 : tensor<1x12x16x16xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x16x16xf32>
    %collapsed_19 = tensor.collapse_shape %transposed_8 [[0, 1], [2], [3]] : tensor<1x12x16x64xf32> into tensor<12x16x64xf32>
    %30 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%collapsed_19 : tensor<12x16x64xf32>) outs(%6 : tensor<1x12x16x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x16x64xf32>
    %collapsed_20 = tensor.collapse_shape %29 [[0, 1], [2], [3]] : tensor<1x12x16x16xf32> into tensor<12x16x16xf32>
    %collapsed_21 = tensor.collapse_shape %30 [[0, 1], [2], [3]] : tensor<1x12x16x64xf32> into tensor<12x16x64xf32>
    %31 = tensor.empty() : tensor<12x16x64xf32>
    %32 = linalg.fill ins(%cst_0 : f32) outs(%31 : tensor<12x16x64xf32>) -> tensor<12x16x64xf32>
    %33 = linalg.batch_matmul ins(%collapsed_20, %collapsed_21 : tensor<12x16x16xf32>, tensor<12x16x64xf32>) outs(%32 : tensor<12x16x64xf32>) -> tensor<12x16x64xf32>
    %expanded_22 = tensor.expand_shape %33 [[0, 1], [2], [3]] output_shape [1, 12, 16, 64] : tensor<12x16x64xf32> into tensor<1x12x16x64xf32>
    %34 = tensor.empty() : tensor<1x16x12x64xf32>
    %transposed_23 = linalg.transpose ins(%expanded_22 : tensor<1x12x16x64xf32>) outs(%34 : tensor<1x16x12x64xf32>) permutation = [0, 2, 1, 3] 
    %collapsed_24 = tensor.collapse_shape %transposed_23 [[0, 1], [2, 3]] : tensor<1x16x12x64xf32> into tensor<16x768xf32>
    %35 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<16x768xf32>) -> tensor<16x768xf32>
    %36 = linalg.matmul ins(%collapsed_24, %arg9 : tensor<16x768xf32>, tensor<768x768xf32>) outs(%35 : tensor<16x768xf32>) -> tensor<16x768xf32>
    %37 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%36, %arg10 : tensor<16x768xf32>, tensor<768xf32>) outs(%0 : tensor<16x768xf32>) {
    ^bb0(%in: f32, %in_26: f32, %out: f32):
      %38 = arith.addf %in, %in_26 : f32
      linalg.yield %38 : f32
    } -> tensor<16x768xf32>
    %expanded_25 = tensor.expand_shape %37 [[0, 1], [2]] output_shape [1, 16, 768] : tensor<16x768xf32> into tensor<1x16x768xf32>
    return %expanded_25, %28 : tensor<1x16x768xf32>, tensor<1x12x16x16xf32>
  }
}
