#map = affine_map<(d0, d1, d2) -> (d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d3)>
#map5 = affine_map<(d0, d1, d2, d3) -> (0, 0, d2, d3)>
#map6 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>
#map7 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map8 = affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
#map9 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
module {
  func.func @main(%arg0: tensor<8x512x4096xbf16>, %arg1: tensor<1x512x128xbf16>, %arg2: tensor<1x512x128xbf16>, %arg3: tensor<1x64x512x512xbf16>, %arg4: tensor<8192x4096xbf16>, %arg5: tensor<512x4096xbf16>, %arg6: tensor<512x4096xbf16>, %arg7: tensor<4096x8192xbf16>, %arg8: tensor<128xbf16>, %arg9: tensor<128xbf16>) -> tensor<8x512x4096xbf16> {
    %c2_i64 = arith.constant 2 : i64
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant 0.000000e+00 : bf16
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 0xFF800000 : f32
    %cst_2 = arith.constant 0.088388347648318447 : f64
    %cst_3 = arith.constant 9.9999999999999995E-7 : f64
    %cst_4 = arith.constant 1.280000e+02 : f32
    %0 = tensor.empty() : tensor<4096x8192xbf16>
    %transposed = linalg.transpose ins(%arg4 : tensor<8192x4096xbf16>) outs(%0 : tensor<4096x8192xbf16>) permutation = [1, 0] 
    %1 = tensor.empty() : tensor<8x4096x8192xbf16>
    %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed : tensor<4096x8192xbf16>) outs(%1 : tensor<8x4096x8192xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x4096x8192xbf16>
    %3 = tensor.empty() : tensor<8x512x8192xbf16>
    %4 = linalg.fill ins(%cst : bf16) outs(%3 : tensor<8x512x8192xbf16>) -> tensor<8x512x8192xbf16>
    %5 = linalg.batch_matmul ins(%arg0, %2 : tensor<8x512x4096xbf16>, tensor<8x4096x8192xbf16>) outs(%4 : tensor<8x512x8192xbf16>) -> tensor<8x512x8192xbf16>
    %expanded = tensor.expand_shape %5 [[0], [1], [2, 3]] output_shape [8, 512, 64, 128] : tensor<8x512x8192xbf16> into tensor<8x512x64x128xbf16>
    %6 = tensor.empty() : tensor<8x512x64x128xf32>
    %7 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded : tensor<8x512x64x128xbf16>) outs(%6 : tensor<8x512x64x128xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %85 = arith.extf %in : bf16 to f32
      linalg.yield %85 : f32
    } -> tensor<8x512x64x128xf32>
    %8 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%7 : tensor<8x512x64x128xf32>) outs(%6 : tensor<8x512x64x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %85 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %85 : f32
    } -> tensor<8x512x64x128xf32>
    %9 = tensor.empty() : tensor<8x512x64x1xf32>
    %10 = linalg.fill ins(%cst_0 : f32) outs(%9 : tensor<8x512x64x1xf32>) -> tensor<8x512x64x1xf32>
    %11 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%8 : tensor<8x512x64x128xf32>) outs(%10 : tensor<8x512x64x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %85 = arith.addf %in, %out : f32
      linalg.yield %85 : f32
    } -> tensor<8x512x64x1xf32>
    %12 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%11 : tensor<8x512x64x1xf32>) outs(%9 : tensor<8x512x64x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %85 = arith.divf %in, %cst_4 : f32
      linalg.yield %85 : f32
    } -> tensor<8x512x64x1xf32>
    %13 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%12 : tensor<8x512x64x1xf32>) outs(%9 : tensor<8x512x64x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %85 = arith.truncf %cst_3 : f64 to f32
      %86 = arith.addf %in, %85 : f32
      linalg.yield %86 : f32
    } -> tensor<8x512x64x1xf32>
    %14 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%13 : tensor<8x512x64x1xf32>) outs(%9 : tensor<8x512x64x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %85 = math.rsqrt %in : f32
      linalg.yield %85 : f32
    } -> tensor<8x512x64x1xf32>
    %15 = linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%7, %14 : tensor<8x512x64x128xf32>, tensor<8x512x64x1xf32>) outs(%6 : tensor<8x512x64x128xf32>) {
    ^bb0(%in: f32, %in_29: f32, %out: f32):
      %85 = arith.mulf %in, %in_29 : f32
      linalg.yield %85 : f32
    } -> tensor<8x512x64x128xf32>
    %16 = tensor.empty() : tensor<8x512x64x128xbf16>
    %17 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%15 : tensor<8x512x64x128xf32>) outs(%16 : tensor<8x512x64x128xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %85 = arith.truncf %in : f32 to bf16
      linalg.yield %85 : bf16
    } -> tensor<8x512x64x128xbf16>
    %18 = linalg.generic {indexing_maps = [#map4, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg8, %17 : tensor<128xbf16>, tensor<8x512x64x128xbf16>) outs(%16 : tensor<8x512x64x128xbf16>) {
    ^bb0(%in: bf16, %in_29: bf16, %out: bf16):
      %85 = arith.mulf %in, %in_29 : bf16
      linalg.yield %85 : bf16
    } -> tensor<8x512x64x128xbf16>
    %19 = tensor.empty() : tensor<8x64x512x128xbf16>
    %transposed_5 = linalg.transpose ins(%18 : tensor<8x512x64x128xbf16>) outs(%19 : tensor<8x64x512x128xbf16>) permutation = [0, 2, 1, 3] 
    %20 = tensor.empty() : tensor<4096x512xbf16>
    %transposed_6 = linalg.transpose ins(%arg5 : tensor<512x4096xbf16>) outs(%20 : tensor<4096x512xbf16>) permutation = [1, 0] 
    %21 = tensor.empty() : tensor<8x4096x512xbf16>
    %22 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_6 : tensor<4096x512xbf16>) outs(%21 : tensor<8x4096x512xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x4096x512xbf16>
    %23 = tensor.empty() : tensor<8x512x512xbf16>
    %24 = linalg.fill ins(%cst : bf16) outs(%23 : tensor<8x512x512xbf16>) -> tensor<8x512x512xbf16>
    %25 = linalg.batch_matmul ins(%arg0, %22 : tensor<8x512x4096xbf16>, tensor<8x4096x512xbf16>) outs(%24 : tensor<8x512x512xbf16>) -> tensor<8x512x512xbf16>
    %expanded_7 = tensor.expand_shape %25 [[0], [1], [2, 3]] output_shape [8, 512, 4, 128] : tensor<8x512x512xbf16> into tensor<8x512x4x128xbf16>
    %26 = tensor.empty() : tensor<8x512x4x128xf32>
    %27 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_7 : tensor<8x512x4x128xbf16>) outs(%26 : tensor<8x512x4x128xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %85 = arith.extf %in : bf16 to f32
      linalg.yield %85 : f32
    } -> tensor<8x512x4x128xf32>
    %28 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%27 : tensor<8x512x4x128xf32>) outs(%26 : tensor<8x512x4x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %85 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %85 : f32
    } -> tensor<8x512x4x128xf32>
    %29 = tensor.empty() : tensor<8x512x4x1xf32>
    %30 = linalg.fill ins(%cst_0 : f32) outs(%29 : tensor<8x512x4x1xf32>) -> tensor<8x512x4x1xf32>
    %31 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%28 : tensor<8x512x4x128xf32>) outs(%30 : tensor<8x512x4x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %85 = arith.addf %in, %out : f32
      linalg.yield %85 : f32
    } -> tensor<8x512x4x1xf32>
    %32 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%31 : tensor<8x512x4x1xf32>) outs(%29 : tensor<8x512x4x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %85 = arith.divf %in, %cst_4 : f32
      linalg.yield %85 : f32
    } -> tensor<8x512x4x1xf32>
    %33 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%32 : tensor<8x512x4x1xf32>) outs(%29 : tensor<8x512x4x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %85 = arith.truncf %cst_3 : f64 to f32
      %86 = arith.addf %in, %85 : f32
      linalg.yield %86 : f32
    } -> tensor<8x512x4x1xf32>
    %34 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%33 : tensor<8x512x4x1xf32>) outs(%29 : tensor<8x512x4x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %85 = math.rsqrt %in : f32
      linalg.yield %85 : f32
    } -> tensor<8x512x4x1xf32>
    %35 = linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%27, %34 : tensor<8x512x4x128xf32>, tensor<8x512x4x1xf32>) outs(%26 : tensor<8x512x4x128xf32>) {
    ^bb0(%in: f32, %in_29: f32, %out: f32):
      %85 = arith.mulf %in, %in_29 : f32
      linalg.yield %85 : f32
    } -> tensor<8x512x4x128xf32>
    %36 = tensor.empty() : tensor<8x512x4x128xbf16>
    %37 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%35 : tensor<8x512x4x128xf32>) outs(%36 : tensor<8x512x4x128xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %85 = arith.truncf %in : f32 to bf16
      linalg.yield %85 : bf16
    } -> tensor<8x512x4x128xbf16>
    %38 = linalg.generic {indexing_maps = [#map4, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg9, %37 : tensor<128xbf16>, tensor<8x512x4x128xbf16>) outs(%36 : tensor<8x512x4x128xbf16>) {
    ^bb0(%in: bf16, %in_29: bf16, %out: bf16):
      %85 = arith.mulf %in, %in_29 : bf16
      linalg.yield %85 : bf16
    } -> tensor<8x512x4x128xbf16>
    %39 = tensor.empty() : tensor<8x4x512x128xbf16>
    %transposed_8 = linalg.transpose ins(%38 : tensor<8x512x4x128xbf16>) outs(%39 : tensor<8x4x512x128xbf16>) permutation = [0, 2, 1, 3] 
    %transposed_9 = linalg.transpose ins(%arg6 : tensor<512x4096xbf16>) outs(%20 : tensor<4096x512xbf16>) permutation = [1, 0] 
    %40 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_9 : tensor<4096x512xbf16>) outs(%21 : tensor<8x4096x512xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x4096x512xbf16>
    %41 = linalg.batch_matmul ins(%arg0, %40 : tensor<8x512x4096xbf16>, tensor<8x4096x512xbf16>) outs(%24 : tensor<8x512x512xbf16>) -> tensor<8x512x512xbf16>
    %expanded_10 = tensor.expand_shape %41 [[0], [1], [2, 3]] output_shape [8, 512, 4, 128] : tensor<8x512x512xbf16> into tensor<8x512x4x128xbf16>
    %transposed_11 = linalg.transpose ins(%expanded_10 : tensor<8x512x4x128xbf16>) outs(%39 : tensor<8x4x512x128xbf16>) permutation = [0, 2, 1, 3] 
    %expanded_12 = tensor.expand_shape %arg1 [[0], [1, 2], [3]] output_shape [1, 1, 512, 128] : tensor<1x512x128xbf16> into tensor<1x1x512x128xbf16>
    %expanded_13 = tensor.expand_shape %arg2 [[0], [1, 2], [3]] output_shape [1, 1, 512, 128] : tensor<1x512x128xbf16> into tensor<1x1x512x128xbf16>
    %42 = linalg.generic {indexing_maps = [#map2, #map5, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_5, %expanded_12 : tensor<8x64x512x128xbf16>, tensor<1x1x512x128xbf16>) outs(%19 : tensor<8x64x512x128xbf16>) {
    ^bb0(%in: bf16, %in_29: bf16, %out: bf16):
      %85 = arith.mulf %in, %in_29 : bf16
      linalg.yield %85 : bf16
    } -> tensor<8x64x512x128xbf16>
    %extracted_slice = tensor.extract_slice %transposed_5[0, 0, 0, 0] [8, 64, 512, 64] [1, 1, 1, 1] : tensor<8x64x512x128xbf16> to tensor<8x64x512x64xbf16>
    %extracted_slice_14 = tensor.extract_slice %transposed_5[0, 0, 0, 64] [8, 64, 512, 64] [1, 1, 1, 1] : tensor<8x64x512x128xbf16> to tensor<8x64x512x64xbf16>
    %43 = tensor.empty() : tensor<8x64x512x64xbf16>
    %44 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_14 : tensor<8x64x512x64xbf16>) outs(%43 : tensor<8x64x512x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %85 = arith.negf %in : bf16
      linalg.yield %85 : bf16
    } -> tensor<8x64x512x64xbf16>
    %concat = tensor.concat dim(3) %44, %extracted_slice : (tensor<8x64x512x64xbf16>, tensor<8x64x512x64xbf16>) -> tensor<8x64x512x128xbf16>
    %45 = linalg.generic {indexing_maps = [#map2, #map5, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat, %expanded_13 : tensor<8x64x512x128xbf16>, tensor<1x1x512x128xbf16>) outs(%19 : tensor<8x64x512x128xbf16>) {
    ^bb0(%in: bf16, %in_29: bf16, %out: bf16):
      %85 = arith.mulf %in, %in_29 : bf16
      linalg.yield %85 : bf16
    } -> tensor<8x64x512x128xbf16>
    %46 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%42, %45 : tensor<8x64x512x128xbf16>, tensor<8x64x512x128xbf16>) outs(%19 : tensor<8x64x512x128xbf16>) {
    ^bb0(%in: bf16, %in_29: bf16, %out: bf16):
      %85 = arith.addf %in, %in_29 : bf16
      linalg.yield %85 : bf16
    } -> tensor<8x64x512x128xbf16>
    %47 = linalg.generic {indexing_maps = [#map2, #map5, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_8, %expanded_12 : tensor<8x4x512x128xbf16>, tensor<1x1x512x128xbf16>) outs(%39 : tensor<8x4x512x128xbf16>) {
    ^bb0(%in: bf16, %in_29: bf16, %out: bf16):
      %85 = arith.mulf %in, %in_29 : bf16
      linalg.yield %85 : bf16
    } -> tensor<8x4x512x128xbf16>
    %extracted_slice_15 = tensor.extract_slice %transposed_8[0, 0, 0, 0] [8, 4, 512, 64] [1, 1, 1, 1] : tensor<8x4x512x128xbf16> to tensor<8x4x512x64xbf16>
    %extracted_slice_16 = tensor.extract_slice %transposed_8[0, 0, 0, 64] [8, 4, 512, 64] [1, 1, 1, 1] : tensor<8x4x512x128xbf16> to tensor<8x4x512x64xbf16>
    %48 = tensor.empty() : tensor<8x4x512x64xbf16>
    %49 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_16 : tensor<8x4x512x64xbf16>) outs(%48 : tensor<8x4x512x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %85 = arith.negf %in : bf16
      linalg.yield %85 : bf16
    } -> tensor<8x4x512x64xbf16>
    %concat_17 = tensor.concat dim(3) %49, %extracted_slice_15 : (tensor<8x4x512x64xbf16>, tensor<8x4x512x64xbf16>) -> tensor<8x4x512x128xbf16>
    %50 = linalg.generic {indexing_maps = [#map2, #map5, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_17, %expanded_13 : tensor<8x4x512x128xbf16>, tensor<1x1x512x128xbf16>) outs(%39 : tensor<8x4x512x128xbf16>) {
    ^bb0(%in: bf16, %in_29: bf16, %out: bf16):
      %85 = arith.mulf %in, %in_29 : bf16
      linalg.yield %85 : bf16
    } -> tensor<8x4x512x128xbf16>
    %51 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%47, %50 : tensor<8x4x512x128xbf16>, tensor<8x4x512x128xbf16>) outs(%39 : tensor<8x4x512x128xbf16>) {
    ^bb0(%in: bf16, %in_29: bf16, %out: bf16):
      %85 = arith.addf %in, %in_29 : bf16
      linalg.yield %85 : bf16
    } -> tensor<8x4x512x128xbf16>
    %52 = tensor.empty() : tensor<8x4x16x512x128xbf16>
    %53 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%51 : tensor<8x4x512x128xbf16>) outs(%52 : tensor<8x4x16x512x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x4x16x512x128xbf16>
    %collapsed = tensor.collapse_shape %53 [[0], [1, 2], [3], [4]] : tensor<8x4x16x512x128xbf16> into tensor<8x64x512x128xbf16>
    %54 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%transposed_11 : tensor<8x4x512x128xbf16>) outs(%52 : tensor<8x4x16x512x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x4x16x512x128xbf16>
    %55 = tensor.empty() : tensor<8x64x128x512xbf16>
    %transposed_18 = linalg.transpose ins(%collapsed : tensor<8x64x512x128xbf16>) outs(%55 : tensor<8x64x128x512xbf16>) permutation = [0, 1, 3, 2] 
    %collapsed_19 = tensor.collapse_shape %46 [[0, 1], [2], [3]] : tensor<8x64x512x128xbf16> into tensor<512x512x128xbf16>
    %collapsed_20 = tensor.collapse_shape %transposed_18 [[0, 1], [2], [3]] : tensor<8x64x128x512xbf16> into tensor<512x128x512xbf16>
    %56 = tensor.empty() : tensor<512x512x512xbf16>
    %57 = linalg.fill ins(%cst : bf16) outs(%56 : tensor<512x512x512xbf16>) -> tensor<512x512x512xbf16>
    %58 = linalg.batch_matmul ins(%collapsed_19, %collapsed_20 : tensor<512x512x128xbf16>, tensor<512x128x512xbf16>) outs(%57 : tensor<512x512x512xbf16>) -> tensor<512x512x512xbf16>
    %expanded_21 = tensor.expand_shape %58 [[0, 1], [2], [3]] output_shape [8, 64, 512, 512] : tensor<512x512x512xbf16> into tensor<8x64x512x512xbf16>
    %59 = tensor.empty() : tensor<8x64x512x512xbf16>
    %60 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_21 : tensor<8x64x512x512xbf16>) outs(%59 : tensor<8x64x512x512xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %85 = arith.truncf %cst_2 : f64 to bf16
      %86 = arith.mulf %in, %85 : bf16
      linalg.yield %86 : bf16
    } -> tensor<8x64x512x512xbf16>
    %61 = linalg.generic {indexing_maps = [#map2, #map8, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%60, %arg3 : tensor<8x64x512x512xbf16>, tensor<1x64x512x512xbf16>) outs(%59 : tensor<8x64x512x512xbf16>) {
    ^bb0(%in: bf16, %in_29: bf16, %out: bf16):
      %85 = arith.addf %in, %in_29 : bf16
      linalg.yield %85 : bf16
    } -> tensor<8x64x512x512xbf16>
    %62 = tensor.empty() : tensor<8x64x512x512xf32>
    %63 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%61 : tensor<8x64x512x512xbf16>) outs(%62 : tensor<8x64x512x512xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %85 = arith.extf %in : bf16 to f32
      linalg.yield %85 : f32
    } -> tensor<8x64x512x512xf32>
    %64 = tensor.empty() : tensor<8x64x512xi64>
    %65 = linalg.fill ins(%c0_i64 : i64) outs(%64 : tensor<8x64x512xi64>) -> tensor<8x64x512xi64>
    %66 = tensor.empty() : tensor<8x64x512xf32>
    %67 = linalg.fill ins(%cst_1 : f32) outs(%66 : tensor<8x64x512xf32>) -> tensor<8x64x512xf32>
    %68:2 = linalg.generic {indexing_maps = [#map2, #map9, #map9], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%63 : tensor<8x64x512x512xf32>) outs(%67, %65 : tensor<8x64x512xf32>, tensor<8x64x512xi64>) {
    ^bb0(%in: f32, %out: f32, %out_29: i64):
      %85 = linalg.index 3 : index
      %86 = arith.index_cast %85 : index to i64
      %87 = arith.maximumf %in, %out : f32
      %88 = arith.cmpf ogt, %in, %out : f32
      %89 = arith.select %88, %86, %out_29 : i64
      linalg.yield %87, %89 : f32, i64
    } -> (tensor<8x64x512xf32>, tensor<8x64x512xi64>)
    %expanded_22 = tensor.expand_shape %68#0 [[0], [1], [2, 3]] output_shape [8, 64, 512, 1] : tensor<8x64x512xf32> into tensor<8x64x512x1xf32>
    %69 = linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%63, %expanded_22 : tensor<8x64x512x512xf32>, tensor<8x64x512x1xf32>) outs(%62 : tensor<8x64x512x512xf32>) {
    ^bb0(%in: f32, %in_29: f32, %out: f32):
      %85 = arith.subf %in, %in_29 : f32
      linalg.yield %85 : f32
    } -> tensor<8x64x512x512xf32>
    %70 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%69 : tensor<8x64x512x512xf32>) outs(%62 : tensor<8x64x512x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %85 = math.exp %in : f32
      linalg.yield %85 : f32
    } -> tensor<8x64x512x512xf32>
    %71 = tensor.empty() : tensor<8x64x512x1xf32>
    %72 = linalg.fill ins(%cst_0 : f32) outs(%71 : tensor<8x64x512x1xf32>) -> tensor<8x64x512x1xf32>
    %73 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%70 : tensor<8x64x512x512xf32>) outs(%72 : tensor<8x64x512x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %85 = arith.addf %in, %out : f32
      linalg.yield %85 : f32
    } -> tensor<8x64x512x1xf32>
    %74 = linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%70, %73 : tensor<8x64x512x512xf32>, tensor<8x64x512x1xf32>) outs(%62 : tensor<8x64x512x512xf32>) {
    ^bb0(%in: f32, %in_29: f32, %out: f32):
      %85 = arith.divf %in, %in_29 : f32
      linalg.yield %85 : f32
    } -> tensor<8x64x512x512xf32>
    %75 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%74 : tensor<8x64x512x512xf32>) outs(%59 : tensor<8x64x512x512xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %85 = arith.truncf %in : f32 to bf16
      linalg.yield %85 : bf16
    } -> tensor<8x64x512x512xbf16>
    %collapsed_23 = tensor.collapse_shape %75 [[0, 1], [2], [3]] : tensor<8x64x512x512xbf16> into tensor<512x512x512xbf16>
    %collapsed_24 = tensor.collapse_shape %54 [[0, 1, 2], [3], [4]] : tensor<8x4x16x512x128xbf16> into tensor<512x512x128xbf16>
    %76 = tensor.empty() : tensor<512x512x128xbf16>
    %77 = linalg.fill ins(%cst : bf16) outs(%76 : tensor<512x512x128xbf16>) -> tensor<512x512x128xbf16>
    %78 = linalg.batch_matmul ins(%collapsed_23, %collapsed_24 : tensor<512x512x512xbf16>, tensor<512x512x128xbf16>) outs(%77 : tensor<512x512x128xbf16>) -> tensor<512x512x128xbf16>
    %expanded_25 = tensor.expand_shape %78 [[0, 1], [2], [3]] output_shape [8, 64, 512, 128] : tensor<512x512x128xbf16> into tensor<8x64x512x128xbf16>
    %transposed_26 = linalg.transpose ins(%expanded_25 : tensor<8x64x512x128xbf16>) outs(%16 : tensor<8x512x64x128xbf16>) permutation = [0, 2, 1, 3] 
    %collapsed_27 = tensor.collapse_shape %transposed_26 [[0], [1], [2, 3]] : tensor<8x512x64x128xbf16> into tensor<8x512x8192xbf16>
    %79 = tensor.empty() : tensor<8192x4096xbf16>
    %transposed_28 = linalg.transpose ins(%arg7 : tensor<4096x8192xbf16>) outs(%79 : tensor<8192x4096xbf16>) permutation = [1, 0] 
    %80 = tensor.empty() : tensor<8x8192x4096xbf16>
    %81 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_28 : tensor<8192x4096xbf16>) outs(%80 : tensor<8x8192x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x8192x4096xbf16>
    %82 = tensor.empty() : tensor<8x512x4096xbf16>
    %83 = linalg.fill ins(%cst : bf16) outs(%82 : tensor<8x512x4096xbf16>) -> tensor<8x512x4096xbf16>
    %84 = linalg.batch_matmul ins(%collapsed_27, %81 : tensor<8x512x8192xbf16>, tensor<8x8192x4096xbf16>) outs(%83 : tensor<8x512x4096xbf16>) -> tensor<8x512x4096xbf16>
    return %84 : tensor<8x512x4096xbf16>
  }
}
