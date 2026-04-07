#map = affine_map<(d0, d1, d2) -> (d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d3)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>
#map7 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>
#map8 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map9 = affine_map<(d0, d1, d2, d3) -> (d1, d3)>
#map10 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
#map11 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
module {
  func.func @main(%arg0: tensor<1x1x4096xbf16>, %arg1: tensor<1x1x128xbf16>, %arg2: tensor<1x1x128xbf16>, %arg3: tensor<1x32x1x1xbf16>, %arg4: tensor<1x8x1023x128xbf16>, %arg5: tensor<1x8x1023x128xbf16>, %arg6: tensor<4096x4096xbf16>, %arg7: tensor<1024x4096xbf16>, %arg8: tensor<1024x4096xbf16>, %arg9: tensor<4096x4096xbf16>, %arg10: tensor<128xbf16>, %arg11: tensor<128xbf16>) -> tensor<1x1x4096xbf16> {
    %c2_i64 = arith.constant 2 : i64
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant 0.000000e+00 : bf16
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 0xFF800000 : f32
    %cst_2 = arith.constant 0.088388347648318447 : f64
    %cst_3 = arith.constant 9.9999999999999995E-7 : f64
    %cst_4 = arith.constant 1.280000e+02 : f32
    %0 = tensor.empty() : tensor<4096x4096xbf16>
    %transposed = linalg.transpose ins(%arg6 : tensor<4096x4096xbf16>) outs(%0 : tensor<4096x4096xbf16>) permutation = [1, 0] 
    %1 = tensor.empty() : tensor<1x1x4096xbf16>
    %collapsed = tensor.collapse_shape %arg0 [[0, 1, 2]] : tensor<1x1x4096xbf16> into tensor<4096xbf16>
    %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapsed : tensor<4096xbf16>) outs(%1 : tensor<1x1x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x1x4096xbf16>
    %3 = tensor.empty() : tensor<1x4096x4096xbf16>
    %4 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed : tensor<4096x4096xbf16>) outs(%3 : tensor<1x4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x4096x4096xbf16>
    %5 = linalg.fill ins(%cst : bf16) outs(%1 : tensor<1x1x4096xbf16>) -> tensor<1x1x4096xbf16>
    %6 = linalg.batch_matmul ins(%2, %4 : tensor<1x1x4096xbf16>, tensor<1x4096x4096xbf16>) outs(%5 : tensor<1x1x4096xbf16>) -> tensor<1x1x4096xbf16>
    %expanded = tensor.expand_shape %6 [[0], [1], [2, 3]] output_shape [1, 1, 32, 128] : tensor<1x1x4096xbf16> into tensor<1x1x32x128xbf16>
    %7 = tensor.empty() : tensor<1x1x32x128xf32>
    %8 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded : tensor<1x1x32x128xbf16>) outs(%7 : tensor<1x1x32x128xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %88 = arith.extf %in : bf16 to f32
      linalg.yield %88 : f32
    } -> tensor<1x1x32x128xf32>
    %9 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%8 : tensor<1x1x32x128xf32>) outs(%7 : tensor<1x1x32x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %88 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %88 : f32
    } -> tensor<1x1x32x128xf32>
    %10 = tensor.empty() : tensor<1x1x32x1xf32>
    %11 = linalg.fill ins(%cst_0 : f32) outs(%10 : tensor<1x1x32x1xf32>) -> tensor<1x1x32x1xf32>
    %12 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%9 : tensor<1x1x32x128xf32>) outs(%11 : tensor<1x1x32x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %88 = arith.addf %in, %out : f32
      linalg.yield %88 : f32
    } -> tensor<1x1x32x1xf32>
    %13 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%12 : tensor<1x1x32x1xf32>) outs(%10 : tensor<1x1x32x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %88 = arith.divf %in, %cst_4 : f32
      linalg.yield %88 : f32
    } -> tensor<1x1x32x1xf32>
    %14 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%13 : tensor<1x1x32x1xf32>) outs(%10 : tensor<1x1x32x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %88 = arith.truncf %cst_3 : f64 to f32
      %89 = arith.addf %in, %88 : f32
      linalg.yield %89 : f32
    } -> tensor<1x1x32x1xf32>
    %15 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%14 : tensor<1x1x32x1xf32>) outs(%10 : tensor<1x1x32x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %88 = math.rsqrt %in : f32
      linalg.yield %88 : f32
    } -> tensor<1x1x32x1xf32>
    %16 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%8, %15 : tensor<1x1x32x128xf32>, tensor<1x1x32x1xf32>) outs(%7 : tensor<1x1x32x128xf32>) {
    ^bb0(%in: f32, %in_36: f32, %out: f32):
      %88 = arith.mulf %in, %in_36 : f32
      linalg.yield %88 : f32
    } -> tensor<1x1x32x128xf32>
    %17 = tensor.empty() : tensor<1x1x32x128xbf16>
    %18 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%16 : tensor<1x1x32x128xf32>) outs(%17 : tensor<1x1x32x128xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %88 = arith.truncf %in : f32 to bf16
      linalg.yield %88 : bf16
    } -> tensor<1x1x32x128xbf16>
    %19 = linalg.generic {indexing_maps = [#map5, #map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg10, %18 : tensor<128xbf16>, tensor<1x1x32x128xbf16>) outs(%17 : tensor<1x1x32x128xbf16>) {
    ^bb0(%in: bf16, %in_36: bf16, %out: bf16):
      %88 = arith.mulf %in, %in_36 : bf16
      linalg.yield %88 : bf16
    } -> tensor<1x1x32x128xbf16>
    %20 = tensor.empty() : tensor<1x32x1x128xbf16>
    %transposed_5 = linalg.transpose ins(%19 : tensor<1x1x32x128xbf16>) outs(%20 : tensor<1x32x1x128xbf16>) permutation = [0, 2, 1, 3] 
    %21 = tensor.empty() : tensor<4096x1024xbf16>
    %transposed_6 = linalg.transpose ins(%arg7 : tensor<1024x4096xbf16>) outs(%21 : tensor<4096x1024xbf16>) permutation = [1, 0] 
    %22 = tensor.empty() : tensor<1x4096x1024xbf16>
    %23 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_6 : tensor<4096x1024xbf16>) outs(%22 : tensor<1x4096x1024xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x4096x1024xbf16>
    %24 = tensor.empty() : tensor<1x1x1024xbf16>
    %25 = linalg.fill ins(%cst : bf16) outs(%24 : tensor<1x1x1024xbf16>) -> tensor<1x1x1024xbf16>
    %26 = linalg.batch_matmul ins(%2, %23 : tensor<1x1x4096xbf16>, tensor<1x4096x1024xbf16>) outs(%25 : tensor<1x1x1024xbf16>) -> tensor<1x1x1024xbf16>
    %expanded_7 = tensor.expand_shape %26 [[0], [1], [2, 3]] output_shape [1, 1, 8, 128] : tensor<1x1x1024xbf16> into tensor<1x1x8x128xbf16>
    %27 = tensor.empty() : tensor<1x1x8x128xf32>
    %28 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_7 : tensor<1x1x8x128xbf16>) outs(%27 : tensor<1x1x8x128xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %88 = arith.extf %in : bf16 to f32
      linalg.yield %88 : f32
    } -> tensor<1x1x8x128xf32>
    %29 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%28 : tensor<1x1x8x128xf32>) outs(%27 : tensor<1x1x8x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %88 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %88 : f32
    } -> tensor<1x1x8x128xf32>
    %30 = tensor.empty() : tensor<1x1x8x1xf32>
    %31 = linalg.fill ins(%cst_0 : f32) outs(%30 : tensor<1x1x8x1xf32>) -> tensor<1x1x8x1xf32>
    %32 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%29 : tensor<1x1x8x128xf32>) outs(%31 : tensor<1x1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %88 = arith.addf %in, %out : f32
      linalg.yield %88 : f32
    } -> tensor<1x1x8x1xf32>
    %33 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%32 : tensor<1x1x8x1xf32>) outs(%30 : tensor<1x1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %88 = arith.divf %in, %cst_4 : f32
      linalg.yield %88 : f32
    } -> tensor<1x1x8x1xf32>
    %34 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%33 : tensor<1x1x8x1xf32>) outs(%30 : tensor<1x1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %88 = arith.truncf %cst_3 : f64 to f32
      %89 = arith.addf %in, %88 : f32
      linalg.yield %89 : f32
    } -> tensor<1x1x8x1xf32>
    %35 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%34 : tensor<1x1x8x1xf32>) outs(%30 : tensor<1x1x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %88 = math.rsqrt %in : f32
      linalg.yield %88 : f32
    } -> tensor<1x1x8x1xf32>
    %36 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%28, %35 : tensor<1x1x8x128xf32>, tensor<1x1x8x1xf32>) outs(%27 : tensor<1x1x8x128xf32>) {
    ^bb0(%in: f32, %in_36: f32, %out: f32):
      %88 = arith.mulf %in, %in_36 : f32
      linalg.yield %88 : f32
    } -> tensor<1x1x8x128xf32>
    %37 = tensor.empty() : tensor<1x1x8x128xbf16>
    %38 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%36 : tensor<1x1x8x128xf32>) outs(%37 : tensor<1x1x8x128xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %88 = arith.truncf %in : f32 to bf16
      linalg.yield %88 : bf16
    } -> tensor<1x1x8x128xbf16>
    %39 = linalg.generic {indexing_maps = [#map5, #map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg11, %38 : tensor<128xbf16>, tensor<1x1x8x128xbf16>) outs(%37 : tensor<1x1x8x128xbf16>) {
    ^bb0(%in: bf16, %in_36: bf16, %out: bf16):
      %88 = arith.mulf %in, %in_36 : bf16
      linalg.yield %88 : bf16
    } -> tensor<1x1x8x128xbf16>
    %40 = tensor.empty() : tensor<1x8x1x128xbf16>
    %transposed_8 = linalg.transpose ins(%39 : tensor<1x1x8x128xbf16>) outs(%40 : tensor<1x8x1x128xbf16>) permutation = [0, 2, 1, 3] 
    %transposed_9 = linalg.transpose ins(%arg8 : tensor<1024x4096xbf16>) outs(%21 : tensor<4096x1024xbf16>) permutation = [1, 0] 
    %41 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_9 : tensor<4096x1024xbf16>) outs(%22 : tensor<1x4096x1024xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x4096x1024xbf16>
    %42 = linalg.batch_matmul ins(%2, %41 : tensor<1x1x4096xbf16>, tensor<1x4096x1024xbf16>) outs(%25 : tensor<1x1x1024xbf16>) -> tensor<1x1x1024xbf16>
    %expanded_10 = tensor.expand_shape %42 [[0], [1], [2, 3]] output_shape [1, 1, 8, 128] : tensor<1x1x1024xbf16> into tensor<1x1x8x128xbf16>
    %transposed_11 = linalg.transpose ins(%expanded_10 : tensor<1x1x8x128xbf16>) outs(%40 : tensor<1x8x1x128xbf16>) permutation = [0, 2, 1, 3] 
    %expanded_12 = tensor.expand_shape %arg1 [[0], [1, 2], [3]] output_shape [1, 1, 1, 128] : tensor<1x1x128xbf16> into tensor<1x1x1x128xbf16>
    %expanded_13 = tensor.expand_shape %arg2 [[0], [1, 2], [3]] output_shape [1, 1, 1, 128] : tensor<1x1x128xbf16> into tensor<1x1x1x128xbf16>
    %43 = linalg.generic {indexing_maps = [#map3, #map6, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_5, %expanded_12 : tensor<1x32x1x128xbf16>, tensor<1x1x1x128xbf16>) outs(%20 : tensor<1x32x1x128xbf16>) {
    ^bb0(%in: bf16, %in_36: bf16, %out: bf16):
      %88 = arith.mulf %in, %in_36 : bf16
      linalg.yield %88 : bf16
    } -> tensor<1x32x1x128xbf16>
    %extracted_slice = tensor.extract_slice %transposed_5[0, 0, 0, 0] [1, 32, 1, 64] [1, 1, 1, 1] : tensor<1x32x1x128xbf16> to tensor<1x32x1x64xbf16>
    %extracted_slice_14 = tensor.extract_slice %transposed_5[0, 0, 0, 64] [1, 32, 1, 64] [1, 1, 1, 1] : tensor<1x32x1x128xbf16> to tensor<1x32x1x64xbf16>
    %44 = tensor.empty() : tensor<1x32x1x64xbf16>
    %45 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_14 : tensor<1x32x1x64xbf16>) outs(%44 : tensor<1x32x1x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %88 = arith.negf %in : bf16
      linalg.yield %88 : bf16
    } -> tensor<1x32x1x64xbf16>
    %concat = tensor.concat dim(3) %45, %extracted_slice : (tensor<1x32x1x64xbf16>, tensor<1x32x1x64xbf16>) -> tensor<1x32x1x128xbf16>
    %46 = linalg.generic {indexing_maps = [#map3, #map6, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat, %expanded_13 : tensor<1x32x1x128xbf16>, tensor<1x1x1x128xbf16>) outs(%20 : tensor<1x32x1x128xbf16>) {
    ^bb0(%in: bf16, %in_36: bf16, %out: bf16):
      %88 = arith.mulf %in, %in_36 : bf16
      linalg.yield %88 : bf16
    } -> tensor<1x32x1x128xbf16>
    %47 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%43, %46 : tensor<1x32x1x128xbf16>, tensor<1x32x1x128xbf16>) outs(%20 : tensor<1x32x1x128xbf16>) {
    ^bb0(%in: bf16, %in_36: bf16, %out: bf16):
      %88 = arith.addf %in, %in_36 : bf16
      linalg.yield %88 : bf16
    } -> tensor<1x32x1x128xbf16>
    %48 = linalg.generic {indexing_maps = [#map3, #map6, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_8, %expanded_12 : tensor<1x8x1x128xbf16>, tensor<1x1x1x128xbf16>) outs(%40 : tensor<1x8x1x128xbf16>) {
    ^bb0(%in: bf16, %in_36: bf16, %out: bf16):
      %88 = arith.mulf %in, %in_36 : bf16
      linalg.yield %88 : bf16
    } -> tensor<1x8x1x128xbf16>
    %extracted_slice_15 = tensor.extract_slice %transposed_8[0, 0, 0, 0] [1, 8, 1, 64] [1, 1, 1, 1] : tensor<1x8x1x128xbf16> to tensor<1x8x1x64xbf16>
    %extracted_slice_16 = tensor.extract_slice %transposed_8[0, 0, 0, 64] [1, 8, 1, 64] [1, 1, 1, 1] : tensor<1x8x1x128xbf16> to tensor<1x8x1x64xbf16>
    %49 = tensor.empty() : tensor<1x8x1x64xbf16>
    %50 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_16 : tensor<1x8x1x64xbf16>) outs(%49 : tensor<1x8x1x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %88 = arith.negf %in : bf16
      linalg.yield %88 : bf16
    } -> tensor<1x8x1x64xbf16>
    %concat_17 = tensor.concat dim(3) %50, %extracted_slice_15 : (tensor<1x8x1x64xbf16>, tensor<1x8x1x64xbf16>) -> tensor<1x8x1x128xbf16>
    %51 = linalg.generic {indexing_maps = [#map3, #map6, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_17, %expanded_13 : tensor<1x8x1x128xbf16>, tensor<1x1x1x128xbf16>) outs(%40 : tensor<1x8x1x128xbf16>) {
    ^bb0(%in: bf16, %in_36: bf16, %out: bf16):
      %88 = arith.mulf %in, %in_36 : bf16
      linalg.yield %88 : bf16
    } -> tensor<1x8x1x128xbf16>
    %52 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%48, %51 : tensor<1x8x1x128xbf16>, tensor<1x8x1x128xbf16>) outs(%40 : tensor<1x8x1x128xbf16>) {
    ^bb0(%in: bf16, %in_36: bf16, %out: bf16):
      %88 = arith.addf %in, %in_36 : bf16
      linalg.yield %88 : bf16
    } -> tensor<1x8x1x128xbf16>
    %concat_18 = tensor.concat dim(2) %arg4, %52 : (tensor<1x8x1023x128xbf16>, tensor<1x8x1x128xbf16>) -> tensor<1x8x1024x128xbf16>
    %concat_19 = tensor.concat dim(2) %arg5, %transposed_11 : (tensor<1x8x1023x128xbf16>, tensor<1x8x1x128xbf16>) -> tensor<1x8x1024x128xbf16>
    %53 = tensor.empty() : tensor<1x8x4x1024x128xbf16>
    %54 = linalg.generic {indexing_maps = [#map7, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%concat_18 : tensor<1x8x1024x128xbf16>) outs(%53 : tensor<1x8x4x1024x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x8x4x1024x128xbf16>
    %collapsed_20 = tensor.collapse_shape %54 [[0], [1, 2], [3], [4]] : tensor<1x8x4x1024x128xbf16> into tensor<1x32x1024x128xbf16>
    %55 = linalg.generic {indexing_maps = [#map7, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%concat_19 : tensor<1x8x1024x128xbf16>) outs(%53 : tensor<1x8x4x1024x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x8x4x1024x128xbf16>
    %56 = tensor.empty() : tensor<1x32x128x1024xbf16>
    %transposed_21 = linalg.transpose ins(%collapsed_20 : tensor<1x32x1024x128xbf16>) outs(%56 : tensor<1x32x128x1024xbf16>) permutation = [0, 1, 3, 2] 
    %collapsed_22 = tensor.collapse_shape %47 [[0, 1, 2], [3]] : tensor<1x32x1x128xbf16> into tensor<32x128xbf16>
    %57 = linalg.generic {indexing_maps = [#map9, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%collapsed_22 : tensor<32x128xbf16>) outs(%20 : tensor<1x32x1x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x1x128xbf16>
    %collapsed_23 = tensor.collapse_shape %transposed_21 [[0, 1], [2], [3]] : tensor<1x32x128x1024xbf16> into tensor<32x128x1024xbf16>
    %58 = linalg.generic {indexing_maps = [#map10, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%collapsed_23 : tensor<32x128x1024xbf16>) outs(%56 : tensor<1x32x128x1024xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x128x1024xbf16>
    %collapsed_24 = tensor.collapse_shape %57 [[0, 1], [2], [3]] : tensor<1x32x1x128xbf16> into tensor<32x1x128xbf16>
    %collapsed_25 = tensor.collapse_shape %58 [[0, 1], [2], [3]] : tensor<1x32x128x1024xbf16> into tensor<32x128x1024xbf16>
    %59 = tensor.empty() : tensor<32x1x1024xbf16>
    %60 = linalg.fill ins(%cst : bf16) outs(%59 : tensor<32x1x1024xbf16>) -> tensor<32x1x1024xbf16>
    %61 = linalg.batch_matmul ins(%collapsed_24, %collapsed_25 : tensor<32x1x128xbf16>, tensor<32x128x1024xbf16>) outs(%60 : tensor<32x1x1024xbf16>) -> tensor<32x1x1024xbf16>
    %expanded_26 = tensor.expand_shape %61 [[0, 1], [2], [3]] output_shape [1, 32, 1, 1024] : tensor<32x1x1024xbf16> into tensor<1x32x1x1024xbf16>
    %62 = tensor.empty() : tensor<1x32x1x1024xbf16>
    %63 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_26 : tensor<1x32x1x1024xbf16>) outs(%62 : tensor<1x32x1x1024xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %88 = arith.truncf %cst_2 : f64 to bf16
      %89 = arith.mulf %in, %88 : bf16
      linalg.yield %89 : bf16
    } -> tensor<1x32x1x1024xbf16>
    %64 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%63, %arg3 : tensor<1x32x1x1024xbf16>, tensor<1x32x1x1xbf16>) outs(%62 : tensor<1x32x1x1024xbf16>) {
    ^bb0(%in: bf16, %in_36: bf16, %out: bf16):
      %88 = arith.addf %in, %in_36 : bf16
      linalg.yield %88 : bf16
    } -> tensor<1x32x1x1024xbf16>
    %65 = tensor.empty() : tensor<1x32x1x1024xf32>
    %66 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%64 : tensor<1x32x1x1024xbf16>) outs(%65 : tensor<1x32x1x1024xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %88 = arith.extf %in : bf16 to f32
      linalg.yield %88 : f32
    } -> tensor<1x32x1x1024xf32>
    %67 = tensor.empty() : tensor<1x32x1xi64>
    %68 = linalg.fill ins(%c0_i64 : i64) outs(%67 : tensor<1x32x1xi64>) -> tensor<1x32x1xi64>
    %69 = tensor.empty() : tensor<1x32x1xf32>
    %70 = linalg.fill ins(%cst_1 : f32) outs(%69 : tensor<1x32x1xf32>) -> tensor<1x32x1xf32>
    %71:2 = linalg.generic {indexing_maps = [#map3, #map11, #map11], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%66 : tensor<1x32x1x1024xf32>) outs(%70, %68 : tensor<1x32x1xf32>, tensor<1x32x1xi64>) {
    ^bb0(%in: f32, %out: f32, %out_36: i64):
      %88 = linalg.index 3 : index
      %89 = arith.index_cast %88 : index to i64
      %90 = arith.maximumf %in, %out : f32
      %91 = arith.cmpf ogt, %in, %out : f32
      %92 = arith.select %91, %89, %out_36 : i64
      linalg.yield %90, %92 : f32, i64
    } -> (tensor<1x32x1xf32>, tensor<1x32x1xi64>)
    %expanded_27 = tensor.expand_shape %71#0 [[0], [1], [2, 3]] output_shape [1, 32, 1, 1] : tensor<1x32x1xf32> into tensor<1x32x1x1xf32>
    %72 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%66, %expanded_27 : tensor<1x32x1x1024xf32>, tensor<1x32x1x1xf32>) outs(%65 : tensor<1x32x1x1024xf32>) {
    ^bb0(%in: f32, %in_36: f32, %out: f32):
      %88 = arith.subf %in, %in_36 : f32
      linalg.yield %88 : f32
    } -> tensor<1x32x1x1024xf32>
    %73 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%72 : tensor<1x32x1x1024xf32>) outs(%65 : tensor<1x32x1x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %88 = math.exp %in : f32
      linalg.yield %88 : f32
    } -> tensor<1x32x1x1024xf32>
    %74 = tensor.empty() : tensor<1x32x1x1xf32>
    %75 = linalg.fill ins(%cst_0 : f32) outs(%74 : tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %76 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%73 : tensor<1x32x1x1024xf32>) outs(%75 : tensor<1x32x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %88 = arith.addf %in, %out : f32
      linalg.yield %88 : f32
    } -> tensor<1x32x1x1xf32>
    %77 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%73, %76 : tensor<1x32x1x1024xf32>, tensor<1x32x1x1xf32>) outs(%65 : tensor<1x32x1x1024xf32>) {
    ^bb0(%in: f32, %in_36: f32, %out: f32):
      %88 = arith.divf %in, %in_36 : f32
      linalg.yield %88 : f32
    } -> tensor<1x32x1x1024xf32>
    %78 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%77 : tensor<1x32x1x1024xf32>) outs(%62 : tensor<1x32x1x1024xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %88 = arith.truncf %in : f32 to bf16
      linalg.yield %88 : bf16
    } -> tensor<1x32x1x1024xbf16>
    %collapsed_28 = tensor.collapse_shape %78 [[0, 1, 2], [3]] : tensor<1x32x1x1024xbf16> into tensor<32x1024xbf16>
    %79 = linalg.generic {indexing_maps = [#map9, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%collapsed_28 : tensor<32x1024xbf16>) outs(%62 : tensor<1x32x1x1024xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x1x1024xbf16>
    %80 = tensor.empty() : tensor<1x32x1024x128xbf16>
    %collapsed_29 = tensor.collapse_shape %55 [[0, 1, 2], [3], [4]] : tensor<1x8x4x1024x128xbf16> into tensor<32x1024x128xbf16>
    %81 = linalg.generic {indexing_maps = [#map10, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%collapsed_29 : tensor<32x1024x128xbf16>) outs(%80 : tensor<1x32x1024x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x1024x128xbf16>
    %collapsed_30 = tensor.collapse_shape %79 [[0, 1], [2], [3]] : tensor<1x32x1x1024xbf16> into tensor<32x1x1024xbf16>
    %collapsed_31 = tensor.collapse_shape %81 [[0, 1], [2], [3]] : tensor<1x32x1024x128xbf16> into tensor<32x1024x128xbf16>
    %82 = tensor.empty() : tensor<32x1x128xbf16>
    %83 = linalg.fill ins(%cst : bf16) outs(%82 : tensor<32x1x128xbf16>) -> tensor<32x1x128xbf16>
    %84 = linalg.batch_matmul ins(%collapsed_30, %collapsed_31 : tensor<32x1x1024xbf16>, tensor<32x1024x128xbf16>) outs(%83 : tensor<32x1x128xbf16>) -> tensor<32x1x128xbf16>
    %expanded_32 = tensor.expand_shape %84 [[0, 1], [2], [3]] output_shape [1, 32, 1, 128] : tensor<32x1x128xbf16> into tensor<1x32x1x128xbf16>
    %transposed_33 = linalg.transpose ins(%expanded_32 : tensor<1x32x1x128xbf16>) outs(%17 : tensor<1x1x32x128xbf16>) permutation = [0, 2, 1, 3] 
    %transposed_34 = linalg.transpose ins(%arg9 : tensor<4096x4096xbf16>) outs(%0 : tensor<4096x4096xbf16>) permutation = [1, 0] 
    %collapsed_35 = tensor.collapse_shape %transposed_33 [[0, 1, 2, 3]] : tensor<1x1x32x128xbf16> into tensor<4096xbf16>
    %85 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapsed_35 : tensor<4096xbf16>) outs(%1 : tensor<1x1x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x1x4096xbf16>
    %86 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_34 : tensor<4096x4096xbf16>) outs(%3 : tensor<1x4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x4096x4096xbf16>
    %87 = linalg.batch_matmul ins(%85, %86 : tensor<1x1x4096xbf16>, tensor<1x4096x4096xbf16>) outs(%5 : tensor<1x1x4096xbf16>) -> tensor<1x1x4096xbf16>
    return %87 : tensor<1x1x4096xbf16>
  }
}
