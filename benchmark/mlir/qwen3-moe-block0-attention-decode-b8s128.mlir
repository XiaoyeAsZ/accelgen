#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d3)>
#map6 = affine_map<(d0, d1, d2, d3) -> (0, 0, d2, d3)>
#map7 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>
#map8 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map9 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map10 = affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
#map11 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
module {
  func.func @main(%arg0: tensor<8x1x4096xbf16>, %arg1: tensor<1x1x128xbf16>, %arg2: tensor<1x1x128xbf16>, %arg3: tensor<1x64x1x128xbf16>, %arg4: tensor<8x4x127x128xbf16>, %arg5: tensor<8x4x127x128xbf16>, %arg6: tensor<8192x4096xbf16>, %arg7: tensor<512x4096xbf16>, %arg8: tensor<512x4096xbf16>, %arg9: tensor<4096x8192xbf16>, %arg10: tensor<128xbf16>, %arg11: tensor<128xbf16>) -> tensor<8x1x4096xbf16> {
    %c2_i64 = arith.constant 2 : i64
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant 0.000000e+00 : bf16
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 0xFF800000 : f32
    %cst_2 = arith.constant 0.088388347648318447 : f64
    %cst_3 = arith.constant 9.9999999999999995E-7 : f64
    %cst_4 = arith.constant 1.280000e+02 : f32
    %0 = tensor.empty() : tensor<4096x8192xbf16>
    %transposed = linalg.transpose ins(%arg6 : tensor<8192x4096xbf16>) outs(%0 : tensor<4096x8192xbf16>) permutation = [1, 0] 
    %1 = tensor.empty() : tensor<8x1x4096xbf16>
    %collapsed = tensor.collapse_shape %arg0 [[0, 1], [2]] : tensor<8x1x4096xbf16> into tensor<8x4096xbf16>
    %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapsed : tensor<8x4096xbf16>) outs(%1 : tensor<8x1x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x1x4096xbf16>
    %3 = tensor.empty() : tensor<8x4096x8192xbf16>
    %4 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed : tensor<4096x8192xbf16>) outs(%3 : tensor<8x4096x8192xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x4096x8192xbf16>
    %5 = tensor.empty() : tensor<8x1x8192xbf16>
    %6 = linalg.fill ins(%cst : bf16) outs(%5 : tensor<8x1x8192xbf16>) -> tensor<8x1x8192xbf16>
    %7 = linalg.batch_matmul ins(%2, %4 : tensor<8x1x4096xbf16>, tensor<8x4096x8192xbf16>) outs(%6 : tensor<8x1x8192xbf16>) -> tensor<8x1x8192xbf16>
    %expanded = tensor.expand_shape %7 [[0], [1], [2, 3]] output_shape [8, 1, 64, 128] : tensor<8x1x8192xbf16> into tensor<8x1x64x128xbf16>
    %8 = tensor.empty() : tensor<8x1x64x128xf32>
    %9 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded : tensor<8x1x64x128xbf16>) outs(%8 : tensor<8x1x64x128xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %86 = arith.extf %in : bf16 to f32
      linalg.yield %86 : f32
    } -> tensor<8x1x64x128xf32>
    %10 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%9 : tensor<8x1x64x128xf32>) outs(%8 : tensor<8x1x64x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %86 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %86 : f32
    } -> tensor<8x1x64x128xf32>
    %11 = tensor.empty() : tensor<8x1x64x1xf32>
    %12 = linalg.fill ins(%cst_0 : f32) outs(%11 : tensor<8x1x64x1xf32>) -> tensor<8x1x64x1xf32>
    %13 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%10 : tensor<8x1x64x128xf32>) outs(%12 : tensor<8x1x64x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %86 = arith.addf %in, %out : f32
      linalg.yield %86 : f32
    } -> tensor<8x1x64x1xf32>
    %14 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%13 : tensor<8x1x64x1xf32>) outs(%11 : tensor<8x1x64x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %86 = arith.divf %in, %cst_4 : f32
      linalg.yield %86 : f32
    } -> tensor<8x1x64x1xf32>
    %15 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%14 : tensor<8x1x64x1xf32>) outs(%11 : tensor<8x1x64x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %86 = arith.truncf %cst_3 : f64 to f32
      %87 = arith.addf %in, %86 : f32
      linalg.yield %87 : f32
    } -> tensor<8x1x64x1xf32>
    %16 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%15 : tensor<8x1x64x1xf32>) outs(%11 : tensor<8x1x64x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %86 = math.rsqrt %in : f32
      linalg.yield %86 : f32
    } -> tensor<8x1x64x1xf32>
    %17 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%9, %16 : tensor<8x1x64x128xf32>, tensor<8x1x64x1xf32>) outs(%8 : tensor<8x1x64x128xf32>) {
    ^bb0(%in: f32, %in_34: f32, %out: f32):
      %86 = arith.mulf %in, %in_34 : f32
      linalg.yield %86 : f32
    } -> tensor<8x1x64x128xf32>
    %18 = tensor.empty() : tensor<8x1x64x128xbf16>
    %19 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%17 : tensor<8x1x64x128xf32>) outs(%18 : tensor<8x1x64x128xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %86 = arith.truncf %in : f32 to bf16
      linalg.yield %86 : bf16
    } -> tensor<8x1x64x128xbf16>
    %20 = linalg.generic {indexing_maps = [#map5, #map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg10, %19 : tensor<128xbf16>, tensor<8x1x64x128xbf16>) outs(%18 : tensor<8x1x64x128xbf16>) {
    ^bb0(%in: bf16, %in_34: bf16, %out: bf16):
      %86 = arith.mulf %in, %in_34 : bf16
      linalg.yield %86 : bf16
    } -> tensor<8x1x64x128xbf16>
    %21 = tensor.empty() : tensor<8x64x1x128xbf16>
    %transposed_5 = linalg.transpose ins(%20 : tensor<8x1x64x128xbf16>) outs(%21 : tensor<8x64x1x128xbf16>) permutation = [0, 2, 1, 3] 
    %22 = tensor.empty() : tensor<4096x512xbf16>
    %transposed_6 = linalg.transpose ins(%arg7 : tensor<512x4096xbf16>) outs(%22 : tensor<4096x512xbf16>) permutation = [1, 0] 
    %23 = tensor.empty() : tensor<8x4096x512xbf16>
    %24 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_6 : tensor<4096x512xbf16>) outs(%23 : tensor<8x4096x512xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x4096x512xbf16>
    %25 = tensor.empty() : tensor<8x1x512xbf16>
    %26 = linalg.fill ins(%cst : bf16) outs(%25 : tensor<8x1x512xbf16>) -> tensor<8x1x512xbf16>
    %27 = linalg.batch_matmul ins(%2, %24 : tensor<8x1x4096xbf16>, tensor<8x4096x512xbf16>) outs(%26 : tensor<8x1x512xbf16>) -> tensor<8x1x512xbf16>
    %expanded_7 = tensor.expand_shape %27 [[0], [1], [2, 3]] output_shape [8, 1, 4, 128] : tensor<8x1x512xbf16> into tensor<8x1x4x128xbf16>
    %28 = tensor.empty() : tensor<8x1x4x128xf32>
    %29 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_7 : tensor<8x1x4x128xbf16>) outs(%28 : tensor<8x1x4x128xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %86 = arith.extf %in : bf16 to f32
      linalg.yield %86 : f32
    } -> tensor<8x1x4x128xf32>
    %30 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%29 : tensor<8x1x4x128xf32>) outs(%28 : tensor<8x1x4x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %86 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %86 : f32
    } -> tensor<8x1x4x128xf32>
    %31 = tensor.empty() : tensor<8x1x4x1xf32>
    %32 = linalg.fill ins(%cst_0 : f32) outs(%31 : tensor<8x1x4x1xf32>) -> tensor<8x1x4x1xf32>
    %33 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%30 : tensor<8x1x4x128xf32>) outs(%32 : tensor<8x1x4x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %86 = arith.addf %in, %out : f32
      linalg.yield %86 : f32
    } -> tensor<8x1x4x1xf32>
    %34 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%33 : tensor<8x1x4x1xf32>) outs(%31 : tensor<8x1x4x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %86 = arith.divf %in, %cst_4 : f32
      linalg.yield %86 : f32
    } -> tensor<8x1x4x1xf32>
    %35 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%34 : tensor<8x1x4x1xf32>) outs(%31 : tensor<8x1x4x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %86 = arith.truncf %cst_3 : f64 to f32
      %87 = arith.addf %in, %86 : f32
      linalg.yield %87 : f32
    } -> tensor<8x1x4x1xf32>
    %36 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%35 : tensor<8x1x4x1xf32>) outs(%31 : tensor<8x1x4x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %86 = math.rsqrt %in : f32
      linalg.yield %86 : f32
    } -> tensor<8x1x4x1xf32>
    %37 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%29, %36 : tensor<8x1x4x128xf32>, tensor<8x1x4x1xf32>) outs(%28 : tensor<8x1x4x128xf32>) {
    ^bb0(%in: f32, %in_34: f32, %out: f32):
      %86 = arith.mulf %in, %in_34 : f32
      linalg.yield %86 : f32
    } -> tensor<8x1x4x128xf32>
    %38 = tensor.empty() : tensor<8x1x4x128xbf16>
    %39 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%37 : tensor<8x1x4x128xf32>) outs(%38 : tensor<8x1x4x128xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %86 = arith.truncf %in : f32 to bf16
      linalg.yield %86 : bf16
    } -> tensor<8x1x4x128xbf16>
    %40 = linalg.generic {indexing_maps = [#map5, #map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg11, %39 : tensor<128xbf16>, tensor<8x1x4x128xbf16>) outs(%38 : tensor<8x1x4x128xbf16>) {
    ^bb0(%in: bf16, %in_34: bf16, %out: bf16):
      %86 = arith.mulf %in, %in_34 : bf16
      linalg.yield %86 : bf16
    } -> tensor<8x1x4x128xbf16>
    %41 = tensor.empty() : tensor<8x4x1x128xbf16>
    %transposed_8 = linalg.transpose ins(%40 : tensor<8x1x4x128xbf16>) outs(%41 : tensor<8x4x1x128xbf16>) permutation = [0, 2, 1, 3] 
    %transposed_9 = linalg.transpose ins(%arg8 : tensor<512x4096xbf16>) outs(%22 : tensor<4096x512xbf16>) permutation = [1, 0] 
    %42 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_9 : tensor<4096x512xbf16>) outs(%23 : tensor<8x4096x512xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x4096x512xbf16>
    %43 = linalg.batch_matmul ins(%2, %42 : tensor<8x1x4096xbf16>, tensor<8x4096x512xbf16>) outs(%26 : tensor<8x1x512xbf16>) -> tensor<8x1x512xbf16>
    %expanded_10 = tensor.expand_shape %43 [[0], [1], [2, 3]] output_shape [8, 1, 4, 128] : tensor<8x1x512xbf16> into tensor<8x1x4x128xbf16>
    %transposed_11 = linalg.transpose ins(%expanded_10 : tensor<8x1x4x128xbf16>) outs(%41 : tensor<8x4x1x128xbf16>) permutation = [0, 2, 1, 3] 
    %expanded_12 = tensor.expand_shape %arg1 [[0], [1, 2], [3]] output_shape [1, 1, 1, 128] : tensor<1x1x128xbf16> into tensor<1x1x1x128xbf16>
    %expanded_13 = tensor.expand_shape %arg2 [[0], [1, 2], [3]] output_shape [1, 1, 1, 128] : tensor<1x1x128xbf16> into tensor<1x1x1x128xbf16>
    %44 = linalg.generic {indexing_maps = [#map3, #map6, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_5, %expanded_12 : tensor<8x64x1x128xbf16>, tensor<1x1x1x128xbf16>) outs(%21 : tensor<8x64x1x128xbf16>) {
    ^bb0(%in: bf16, %in_34: bf16, %out: bf16):
      %86 = arith.mulf %in, %in_34 : bf16
      linalg.yield %86 : bf16
    } -> tensor<8x64x1x128xbf16>
    %extracted_slice = tensor.extract_slice %transposed_5[0, 0, 0, 0] [8, 64, 1, 64] [1, 1, 1, 1] : tensor<8x64x1x128xbf16> to tensor<8x64x1x64xbf16>
    %extracted_slice_14 = tensor.extract_slice %transposed_5[0, 0, 0, 64] [8, 64, 1, 64] [1, 1, 1, 1] : tensor<8x64x1x128xbf16> to tensor<8x64x1x64xbf16>
    %45 = tensor.empty() : tensor<8x64x1x64xbf16>
    %46 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_14 : tensor<8x64x1x64xbf16>) outs(%45 : tensor<8x64x1x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %86 = arith.negf %in : bf16
      linalg.yield %86 : bf16
    } -> tensor<8x64x1x64xbf16>
    %concat = tensor.concat dim(3) %46, %extracted_slice : (tensor<8x64x1x64xbf16>, tensor<8x64x1x64xbf16>) -> tensor<8x64x1x128xbf16>
    %47 = linalg.generic {indexing_maps = [#map3, #map6, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat, %expanded_13 : tensor<8x64x1x128xbf16>, tensor<1x1x1x128xbf16>) outs(%21 : tensor<8x64x1x128xbf16>) {
    ^bb0(%in: bf16, %in_34: bf16, %out: bf16):
      %86 = arith.mulf %in, %in_34 : bf16
      linalg.yield %86 : bf16
    } -> tensor<8x64x1x128xbf16>
    %48 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%44, %47 : tensor<8x64x1x128xbf16>, tensor<8x64x1x128xbf16>) outs(%21 : tensor<8x64x1x128xbf16>) {
    ^bb0(%in: bf16, %in_34: bf16, %out: bf16):
      %86 = arith.addf %in, %in_34 : bf16
      linalg.yield %86 : bf16
    } -> tensor<8x64x1x128xbf16>
    %49 = linalg.generic {indexing_maps = [#map3, #map6, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_8, %expanded_12 : tensor<8x4x1x128xbf16>, tensor<1x1x1x128xbf16>) outs(%41 : tensor<8x4x1x128xbf16>) {
    ^bb0(%in: bf16, %in_34: bf16, %out: bf16):
      %86 = arith.mulf %in, %in_34 : bf16
      linalg.yield %86 : bf16
    } -> tensor<8x4x1x128xbf16>
    %extracted_slice_15 = tensor.extract_slice %transposed_8[0, 0, 0, 0] [8, 4, 1, 64] [1, 1, 1, 1] : tensor<8x4x1x128xbf16> to tensor<8x4x1x64xbf16>
    %extracted_slice_16 = tensor.extract_slice %transposed_8[0, 0, 0, 64] [8, 4, 1, 64] [1, 1, 1, 1] : tensor<8x4x1x128xbf16> to tensor<8x4x1x64xbf16>
    %50 = tensor.empty() : tensor<8x4x1x64xbf16>
    %51 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_16 : tensor<8x4x1x64xbf16>) outs(%50 : tensor<8x4x1x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %86 = arith.negf %in : bf16
      linalg.yield %86 : bf16
    } -> tensor<8x4x1x64xbf16>
    %concat_17 = tensor.concat dim(3) %51, %extracted_slice_15 : (tensor<8x4x1x64xbf16>, tensor<8x4x1x64xbf16>) -> tensor<8x4x1x128xbf16>
    %52 = linalg.generic {indexing_maps = [#map3, #map6, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_17, %expanded_13 : tensor<8x4x1x128xbf16>, tensor<1x1x1x128xbf16>) outs(%41 : tensor<8x4x1x128xbf16>) {
    ^bb0(%in: bf16, %in_34: bf16, %out: bf16):
      %86 = arith.mulf %in, %in_34 : bf16
      linalg.yield %86 : bf16
    } -> tensor<8x4x1x128xbf16>
    %53 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%49, %52 : tensor<8x4x1x128xbf16>, tensor<8x4x1x128xbf16>) outs(%41 : tensor<8x4x1x128xbf16>) {
    ^bb0(%in: bf16, %in_34: bf16, %out: bf16):
      %86 = arith.addf %in, %in_34 : bf16
      linalg.yield %86 : bf16
    } -> tensor<8x4x1x128xbf16>
    %concat_18 = tensor.concat dim(2) %arg4, %53 : (tensor<8x4x127x128xbf16>, tensor<8x4x1x128xbf16>) -> tensor<8x4x128x128xbf16>
    %concat_19 = tensor.concat dim(2) %arg5, %transposed_11 : (tensor<8x4x127x128xbf16>, tensor<8x4x1x128xbf16>) -> tensor<8x4x128x128xbf16>
    %54 = tensor.empty() : tensor<8x4x16x128x128xbf16>
    %55 = linalg.generic {indexing_maps = [#map7, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%concat_18 : tensor<8x4x128x128xbf16>) outs(%54 : tensor<8x4x16x128x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x4x16x128x128xbf16>
    %collapsed_20 = tensor.collapse_shape %55 [[0], [1, 2], [3], [4]] : tensor<8x4x16x128x128xbf16> into tensor<8x64x128x128xbf16>
    %56 = linalg.generic {indexing_maps = [#map7, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%concat_19 : tensor<8x4x128x128xbf16>) outs(%54 : tensor<8x4x16x128x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x4x16x128x128xbf16>
    %57 = tensor.empty() : tensor<8x64x128x128xbf16>
    %transposed_21 = linalg.transpose ins(%collapsed_20 : tensor<8x64x128x128xbf16>) outs(%57 : tensor<8x64x128x128xbf16>) permutation = [0, 1, 3, 2] 
    %collapsed_22 = tensor.collapse_shape %48 [[0], [1, 2], [3]] : tensor<8x64x1x128xbf16> into tensor<8x64x128xbf16>
    %58 = linalg.generic {indexing_maps = [#map9, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%collapsed_22 : tensor<8x64x128xbf16>) outs(%21 : tensor<8x64x1x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x64x1x128xbf16>
    %collapsed_23 = tensor.collapse_shape %58 [[0, 1], [2], [3]] : tensor<8x64x1x128xbf16> into tensor<512x1x128xbf16>
    %collapsed_24 = tensor.collapse_shape %transposed_21 [[0, 1], [2], [3]] : tensor<8x64x128x128xbf16> into tensor<512x128x128xbf16>
    %59 = tensor.empty() : tensor<512x1x128xbf16>
    %60 = linalg.fill ins(%cst : bf16) outs(%59 : tensor<512x1x128xbf16>) -> tensor<512x1x128xbf16>
    %61 = linalg.batch_matmul ins(%collapsed_23, %collapsed_24 : tensor<512x1x128xbf16>, tensor<512x128x128xbf16>) outs(%60 : tensor<512x1x128xbf16>) -> tensor<512x1x128xbf16>
    %expanded_25 = tensor.expand_shape %61 [[0, 1], [2], [3]] output_shape [8, 64, 1, 128] : tensor<512x1x128xbf16> into tensor<8x64x1x128xbf16>
    %62 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_25 : tensor<8x64x1x128xbf16>) outs(%21 : tensor<8x64x1x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %86 = arith.truncf %cst_2 : f64 to bf16
      %87 = arith.mulf %in, %86 : bf16
      linalg.yield %87 : bf16
    } -> tensor<8x64x1x128xbf16>
    %63 = linalg.generic {indexing_maps = [#map3, #map10, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%62, %arg3 : tensor<8x64x1x128xbf16>, tensor<1x64x1x128xbf16>) outs(%21 : tensor<8x64x1x128xbf16>) {
    ^bb0(%in: bf16, %in_34: bf16, %out: bf16):
      %86 = arith.addf %in, %in_34 : bf16
      linalg.yield %86 : bf16
    } -> tensor<8x64x1x128xbf16>
    %64 = tensor.empty() : tensor<8x64x1x128xf32>
    %65 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%63 : tensor<8x64x1x128xbf16>) outs(%64 : tensor<8x64x1x128xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %86 = arith.extf %in : bf16 to f32
      linalg.yield %86 : f32
    } -> tensor<8x64x1x128xf32>
    %66 = tensor.empty() : tensor<8x64x1xi64>
    %67 = linalg.fill ins(%c0_i64 : i64) outs(%66 : tensor<8x64x1xi64>) -> tensor<8x64x1xi64>
    %68 = tensor.empty() : tensor<8x64x1xf32>
    %69 = linalg.fill ins(%cst_1 : f32) outs(%68 : tensor<8x64x1xf32>) -> tensor<8x64x1xf32>
    %70:2 = linalg.generic {indexing_maps = [#map3, #map11, #map11], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%65 : tensor<8x64x1x128xf32>) outs(%69, %67 : tensor<8x64x1xf32>, tensor<8x64x1xi64>) {
    ^bb0(%in: f32, %out: f32, %out_34: i64):
      %86 = linalg.index 3 : index
      %87 = arith.index_cast %86 : index to i64
      %88 = arith.maximumf %in, %out : f32
      %89 = arith.cmpf ogt, %in, %out : f32
      %90 = arith.select %89, %87, %out_34 : i64
      linalg.yield %88, %90 : f32, i64
    } -> (tensor<8x64x1xf32>, tensor<8x64x1xi64>)
    %expanded_26 = tensor.expand_shape %70#0 [[0], [1], [2, 3]] output_shape [8, 64, 1, 1] : tensor<8x64x1xf32> into tensor<8x64x1x1xf32>
    %71 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%65, %expanded_26 : tensor<8x64x1x128xf32>, tensor<8x64x1x1xf32>) outs(%64 : tensor<8x64x1x128xf32>) {
    ^bb0(%in: f32, %in_34: f32, %out: f32):
      %86 = arith.subf %in, %in_34 : f32
      linalg.yield %86 : f32
    } -> tensor<8x64x1x128xf32>
    %72 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%71 : tensor<8x64x1x128xf32>) outs(%64 : tensor<8x64x1x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %86 = math.exp %in : f32
      linalg.yield %86 : f32
    } -> tensor<8x64x1x128xf32>
    %73 = tensor.empty() : tensor<8x64x1x1xf32>
    %74 = linalg.fill ins(%cst_0 : f32) outs(%73 : tensor<8x64x1x1xf32>) -> tensor<8x64x1x1xf32>
    %75 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%72 : tensor<8x64x1x128xf32>) outs(%74 : tensor<8x64x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %86 = arith.addf %in, %out : f32
      linalg.yield %86 : f32
    } -> tensor<8x64x1x1xf32>
    %76 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%72, %75 : tensor<8x64x1x128xf32>, tensor<8x64x1x1xf32>) outs(%64 : tensor<8x64x1x128xf32>) {
    ^bb0(%in: f32, %in_34: f32, %out: f32):
      %86 = arith.divf %in, %in_34 : f32
      linalg.yield %86 : f32
    } -> tensor<8x64x1x128xf32>
    %77 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%76 : tensor<8x64x1x128xf32>) outs(%21 : tensor<8x64x1x128xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %86 = arith.truncf %in : f32 to bf16
      linalg.yield %86 : bf16
    } -> tensor<8x64x1x128xbf16>
    %collapsed_27 = tensor.collapse_shape %77 [[0], [1, 2], [3]] : tensor<8x64x1x128xbf16> into tensor<8x64x128xbf16>
    %78 = linalg.generic {indexing_maps = [#map9, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%collapsed_27 : tensor<8x64x128xbf16>) outs(%21 : tensor<8x64x1x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x64x1x128xbf16>
    %collapsed_28 = tensor.collapse_shape %78 [[0, 1], [2], [3]] : tensor<8x64x1x128xbf16> into tensor<512x1x128xbf16>
    %collapsed_29 = tensor.collapse_shape %56 [[0, 1, 2], [3], [4]] : tensor<8x4x16x128x128xbf16> into tensor<512x128x128xbf16>
    %79 = linalg.batch_matmul ins(%collapsed_28, %collapsed_29 : tensor<512x1x128xbf16>, tensor<512x128x128xbf16>) outs(%60 : tensor<512x1x128xbf16>) -> tensor<512x1x128xbf16>
    %expanded_30 = tensor.expand_shape %79 [[0, 1], [2], [3]] output_shape [8, 64, 1, 128] : tensor<512x1x128xbf16> into tensor<8x64x1x128xbf16>
    %transposed_31 = linalg.transpose ins(%expanded_30 : tensor<8x64x1x128xbf16>) outs(%18 : tensor<8x1x64x128xbf16>) permutation = [0, 2, 1, 3] 
    %80 = tensor.empty() : tensor<8192x4096xbf16>
    %transposed_32 = linalg.transpose ins(%arg9 : tensor<4096x8192xbf16>) outs(%80 : tensor<8192x4096xbf16>) permutation = [1, 0] 
    %collapsed_33 = tensor.collapse_shape %transposed_31 [[0, 1], [2, 3]] : tensor<8x1x64x128xbf16> into tensor<8x8192xbf16>
    %81 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapsed_33 : tensor<8x8192xbf16>) outs(%5 : tensor<8x1x8192xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x1x8192xbf16>
    %82 = tensor.empty() : tensor<8x8192x4096xbf16>
    %83 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_32 : tensor<8192x4096xbf16>) outs(%82 : tensor<8x8192x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x8192x4096xbf16>
    %84 = linalg.fill ins(%cst : bf16) outs(%1 : tensor<8x1x4096xbf16>) -> tensor<8x1x4096xbf16>
    %85 = linalg.batch_matmul ins(%81, %83 : tensor<8x1x8192xbf16>, tensor<8x8192x4096xbf16>) outs(%84 : tensor<8x1x4096xbf16>) -> tensor<8x1x4096xbf16>
    return %85 : tensor<8x1x4096xbf16>
  }
}
