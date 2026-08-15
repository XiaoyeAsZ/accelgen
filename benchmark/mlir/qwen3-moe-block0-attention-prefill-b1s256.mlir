#map = affine_map<(d0, d1, d2) -> (d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d3)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>
#map6 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>
#map7 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map8 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
#map9 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
module {
  func.func @main(%arg0: tensor<1x256x4096xbf16>, %arg1: tensor<1x256x128xbf16>, %arg2: tensor<1x256x128xbf16>, %arg3: tensor<1x64x256x256xbf16>, %arg4: tensor<8192x4096xbf16>, %arg5: tensor<512x4096xbf16>, %arg6: tensor<512x4096xbf16>, %arg7: tensor<4096x8192xbf16>, %arg8: tensor<128xbf16>, %arg9: tensor<128xbf16>) -> tensor<1x256x4096xbf16> {
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
    %1 = tensor.empty() : tensor<1x256x4096xbf16>
    %collapsed = tensor.collapse_shape %arg0 [[0, 1], [2]] : tensor<1x256x4096xbf16> into tensor<256x4096xbf16>
    %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapsed : tensor<256x4096xbf16>) outs(%1 : tensor<1x256x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x256x4096xbf16>
    %3 = tensor.empty() : tensor<1x4096x8192xbf16>
    %4 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed : tensor<4096x8192xbf16>) outs(%3 : tensor<1x4096x8192xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x4096x8192xbf16>
    %5 = tensor.empty() : tensor<1x256x8192xbf16>
    %6 = linalg.fill ins(%cst : bf16) outs(%5 : tensor<1x256x8192xbf16>) -> tensor<1x256x8192xbf16>
    %7 = linalg.batch_matmul ins(%2, %4 : tensor<1x256x4096xbf16>, tensor<1x4096x8192xbf16>) outs(%6 : tensor<1x256x8192xbf16>) -> tensor<1x256x8192xbf16>
    %expanded = tensor.expand_shape %7 [[0], [1], [2, 3]] output_shape [1, 256, 64, 128] : tensor<1x256x8192xbf16> into tensor<1x256x64x128xbf16>
    %8 = tensor.empty() : tensor<1x256x64x128xf32>
    %9 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded : tensor<1x256x64x128xbf16>) outs(%8 : tensor<1x256x64x128xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %91 = arith.extf %in : bf16 to f32
      linalg.yield %91 : f32
    } -> tensor<1x256x64x128xf32>
    %10 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%9 : tensor<1x256x64x128xf32>) outs(%8 : tensor<1x256x64x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %91 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %91 : f32
    } -> tensor<1x256x64x128xf32>
    %11 = tensor.empty() : tensor<1x256x64x1xf32>
    %12 = linalg.fill ins(%cst_0 : f32) outs(%11 : tensor<1x256x64x1xf32>) -> tensor<1x256x64x1xf32>
    %13 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%10 : tensor<1x256x64x128xf32>) outs(%12 : tensor<1x256x64x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %91 = arith.addf %in, %out : f32
      linalg.yield %91 : f32
    } -> tensor<1x256x64x1xf32>
    %14 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%13 : tensor<1x256x64x1xf32>) outs(%11 : tensor<1x256x64x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %91 = arith.divf %in, %cst_4 : f32
      linalg.yield %91 : f32
    } -> tensor<1x256x64x1xf32>
    %15 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%14 : tensor<1x256x64x1xf32>) outs(%11 : tensor<1x256x64x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %91 = arith.truncf %cst_3 : f64 to f32
      %92 = arith.addf %in, %91 : f32
      linalg.yield %92 : f32
    } -> tensor<1x256x64x1xf32>
    %16 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%15 : tensor<1x256x64x1xf32>) outs(%11 : tensor<1x256x64x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %91 = math.rsqrt %in : f32
      linalg.yield %91 : f32
    } -> tensor<1x256x64x1xf32>
    %17 = linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%9, %16 : tensor<1x256x64x128xf32>, tensor<1x256x64x1xf32>) outs(%8 : tensor<1x256x64x128xf32>) {
    ^bb0(%in: f32, %in_34: f32, %out: f32):
      %91 = arith.mulf %in, %in_34 : f32
      linalg.yield %91 : f32
    } -> tensor<1x256x64x128xf32>
    %18 = tensor.empty() : tensor<1x256x64x128xbf16>
    %19 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%17 : tensor<1x256x64x128xf32>) outs(%18 : tensor<1x256x64x128xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %91 = arith.truncf %in : f32 to bf16
      linalg.yield %91 : bf16
    } -> tensor<1x256x64x128xbf16>
    %20 = linalg.generic {indexing_maps = [#map4, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg8, %19 : tensor<128xbf16>, tensor<1x256x64x128xbf16>) outs(%18 : tensor<1x256x64x128xbf16>) {
    ^bb0(%in: bf16, %in_34: bf16, %out: bf16):
      %91 = arith.mulf %in, %in_34 : bf16
      linalg.yield %91 : bf16
    } -> tensor<1x256x64x128xbf16>
    %21 = tensor.empty() : tensor<1x64x256x128xbf16>
    %transposed_5 = linalg.transpose ins(%20 : tensor<1x256x64x128xbf16>) outs(%21 : tensor<1x64x256x128xbf16>) permutation = [0, 2, 1, 3] 
    %22 = tensor.empty() : tensor<4096x512xbf16>
    %transposed_6 = linalg.transpose ins(%arg5 : tensor<512x4096xbf16>) outs(%22 : tensor<4096x512xbf16>) permutation = [1, 0] 
    %23 = tensor.empty() : tensor<1x4096x512xbf16>
    %24 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_6 : tensor<4096x512xbf16>) outs(%23 : tensor<1x4096x512xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x4096x512xbf16>
    %25 = tensor.empty() : tensor<1x256x512xbf16>
    %26 = linalg.fill ins(%cst : bf16) outs(%25 : tensor<1x256x512xbf16>) -> tensor<1x256x512xbf16>
    %27 = linalg.batch_matmul ins(%2, %24 : tensor<1x256x4096xbf16>, tensor<1x4096x512xbf16>) outs(%26 : tensor<1x256x512xbf16>) -> tensor<1x256x512xbf16>
    %expanded_7 = tensor.expand_shape %27 [[0], [1], [2, 3]] output_shape [1, 256, 4, 128] : tensor<1x256x512xbf16> into tensor<1x256x4x128xbf16>
    %28 = tensor.empty() : tensor<1x256x4x128xf32>
    %29 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_7 : tensor<1x256x4x128xbf16>) outs(%28 : tensor<1x256x4x128xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %91 = arith.extf %in : bf16 to f32
      linalg.yield %91 : f32
    } -> tensor<1x256x4x128xf32>
    %30 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%29 : tensor<1x256x4x128xf32>) outs(%28 : tensor<1x256x4x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %91 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %91 : f32
    } -> tensor<1x256x4x128xf32>
    %31 = tensor.empty() : tensor<1x256x4x1xf32>
    %32 = linalg.fill ins(%cst_0 : f32) outs(%31 : tensor<1x256x4x1xf32>) -> tensor<1x256x4x1xf32>
    %33 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%30 : tensor<1x256x4x128xf32>) outs(%32 : tensor<1x256x4x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %91 = arith.addf %in, %out : f32
      linalg.yield %91 : f32
    } -> tensor<1x256x4x1xf32>
    %34 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%33 : tensor<1x256x4x1xf32>) outs(%31 : tensor<1x256x4x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %91 = arith.divf %in, %cst_4 : f32
      linalg.yield %91 : f32
    } -> tensor<1x256x4x1xf32>
    %35 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%34 : tensor<1x256x4x1xf32>) outs(%31 : tensor<1x256x4x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %91 = arith.truncf %cst_3 : f64 to f32
      %92 = arith.addf %in, %91 : f32
      linalg.yield %92 : f32
    } -> tensor<1x256x4x1xf32>
    %36 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%35 : tensor<1x256x4x1xf32>) outs(%31 : tensor<1x256x4x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %91 = math.rsqrt %in : f32
      linalg.yield %91 : f32
    } -> tensor<1x256x4x1xf32>
    %37 = linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%29, %36 : tensor<1x256x4x128xf32>, tensor<1x256x4x1xf32>) outs(%28 : tensor<1x256x4x128xf32>) {
    ^bb0(%in: f32, %in_34: f32, %out: f32):
      %91 = arith.mulf %in, %in_34 : f32
      linalg.yield %91 : f32
    } -> tensor<1x256x4x128xf32>
    %38 = tensor.empty() : tensor<1x256x4x128xbf16>
    %39 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%37 : tensor<1x256x4x128xf32>) outs(%38 : tensor<1x256x4x128xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %91 = arith.truncf %in : f32 to bf16
      linalg.yield %91 : bf16
    } -> tensor<1x256x4x128xbf16>
    %40 = linalg.generic {indexing_maps = [#map4, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg9, %39 : tensor<128xbf16>, tensor<1x256x4x128xbf16>) outs(%38 : tensor<1x256x4x128xbf16>) {
    ^bb0(%in: bf16, %in_34: bf16, %out: bf16):
      %91 = arith.mulf %in, %in_34 : bf16
      linalg.yield %91 : bf16
    } -> tensor<1x256x4x128xbf16>
    %41 = tensor.empty() : tensor<1x4x256x128xbf16>
    %transposed_8 = linalg.transpose ins(%40 : tensor<1x256x4x128xbf16>) outs(%41 : tensor<1x4x256x128xbf16>) permutation = [0, 2, 1, 3] 
    %transposed_9 = linalg.transpose ins(%arg6 : tensor<512x4096xbf16>) outs(%22 : tensor<4096x512xbf16>) permutation = [1, 0] 
    %42 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_9 : tensor<4096x512xbf16>) outs(%23 : tensor<1x4096x512xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x4096x512xbf16>
    %43 = linalg.batch_matmul ins(%2, %42 : tensor<1x256x4096xbf16>, tensor<1x4096x512xbf16>) outs(%26 : tensor<1x256x512xbf16>) -> tensor<1x256x512xbf16>
    %expanded_10 = tensor.expand_shape %43 [[0], [1], [2, 3]] output_shape [1, 256, 4, 128] : tensor<1x256x512xbf16> into tensor<1x256x4x128xbf16>
    %transposed_11 = linalg.transpose ins(%expanded_10 : tensor<1x256x4x128xbf16>) outs(%41 : tensor<1x4x256x128xbf16>) permutation = [0, 2, 1, 3] 
    %expanded_12 = tensor.expand_shape %arg1 [[0], [1, 2], [3]] output_shape [1, 1, 256, 128] : tensor<1x256x128xbf16> into tensor<1x1x256x128xbf16>
    %expanded_13 = tensor.expand_shape %arg2 [[0], [1, 2], [3]] output_shape [1, 1, 256, 128] : tensor<1x256x128xbf16> into tensor<1x1x256x128xbf16>
    %44 = linalg.generic {indexing_maps = [#map2, #map5, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_5, %expanded_12 : tensor<1x64x256x128xbf16>, tensor<1x1x256x128xbf16>) outs(%21 : tensor<1x64x256x128xbf16>) {
    ^bb0(%in: bf16, %in_34: bf16, %out: bf16):
      %91 = arith.mulf %in, %in_34 : bf16
      linalg.yield %91 : bf16
    } -> tensor<1x64x256x128xbf16>
    %extracted_slice = tensor.extract_slice %transposed_5[0, 0, 0, 0] [1, 64, 256, 64] [1, 1, 1, 1] : tensor<1x64x256x128xbf16> to tensor<1x64x256x64xbf16>
    %extracted_slice_14 = tensor.extract_slice %transposed_5[0, 0, 0, 64] [1, 64, 256, 64] [1, 1, 1, 1] : tensor<1x64x256x128xbf16> to tensor<1x64x256x64xbf16>
    %45 = tensor.empty() : tensor<1x64x256x64xbf16>
    %46 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_14 : tensor<1x64x256x64xbf16>) outs(%45 : tensor<1x64x256x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %91 = arith.negf %in : bf16
      linalg.yield %91 : bf16
    } -> tensor<1x64x256x64xbf16>
    %concat = tensor.concat dim(3) %46, %extracted_slice : (tensor<1x64x256x64xbf16>, tensor<1x64x256x64xbf16>) -> tensor<1x64x256x128xbf16>
    %47 = linalg.generic {indexing_maps = [#map2, #map5, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat, %expanded_13 : tensor<1x64x256x128xbf16>, tensor<1x1x256x128xbf16>) outs(%21 : tensor<1x64x256x128xbf16>) {
    ^bb0(%in: bf16, %in_34: bf16, %out: bf16):
      %91 = arith.mulf %in, %in_34 : bf16
      linalg.yield %91 : bf16
    } -> tensor<1x64x256x128xbf16>
    %48 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%44, %47 : tensor<1x64x256x128xbf16>, tensor<1x64x256x128xbf16>) outs(%21 : tensor<1x64x256x128xbf16>) {
    ^bb0(%in: bf16, %in_34: bf16, %out: bf16):
      %91 = arith.addf %in, %in_34 : bf16
      linalg.yield %91 : bf16
    } -> tensor<1x64x256x128xbf16>
    %49 = linalg.generic {indexing_maps = [#map2, #map5, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_8, %expanded_12 : tensor<1x4x256x128xbf16>, tensor<1x1x256x128xbf16>) outs(%41 : tensor<1x4x256x128xbf16>) {
    ^bb0(%in: bf16, %in_34: bf16, %out: bf16):
      %91 = arith.mulf %in, %in_34 : bf16
      linalg.yield %91 : bf16
    } -> tensor<1x4x256x128xbf16>
    %extracted_slice_15 = tensor.extract_slice %transposed_8[0, 0, 0, 0] [1, 4, 256, 64] [1, 1, 1, 1] : tensor<1x4x256x128xbf16> to tensor<1x4x256x64xbf16>
    %extracted_slice_16 = tensor.extract_slice %transposed_8[0, 0, 0, 64] [1, 4, 256, 64] [1, 1, 1, 1] : tensor<1x4x256x128xbf16> to tensor<1x4x256x64xbf16>
    %50 = tensor.empty() : tensor<1x4x256x64xbf16>
    %51 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_16 : tensor<1x4x256x64xbf16>) outs(%50 : tensor<1x4x256x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %91 = arith.negf %in : bf16
      linalg.yield %91 : bf16
    } -> tensor<1x4x256x64xbf16>
    %concat_17 = tensor.concat dim(3) %51, %extracted_slice_15 : (tensor<1x4x256x64xbf16>, tensor<1x4x256x64xbf16>) -> tensor<1x4x256x128xbf16>
    %52 = linalg.generic {indexing_maps = [#map2, #map5, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_17, %expanded_13 : tensor<1x4x256x128xbf16>, tensor<1x1x256x128xbf16>) outs(%41 : tensor<1x4x256x128xbf16>) {
    ^bb0(%in: bf16, %in_34: bf16, %out: bf16):
      %91 = arith.mulf %in, %in_34 : bf16
      linalg.yield %91 : bf16
    } -> tensor<1x4x256x128xbf16>
    %53 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%49, %52 : tensor<1x4x256x128xbf16>, tensor<1x4x256x128xbf16>) outs(%41 : tensor<1x4x256x128xbf16>) {
    ^bb0(%in: bf16, %in_34: bf16, %out: bf16):
      %91 = arith.addf %in, %in_34 : bf16
      linalg.yield %91 : bf16
    } -> tensor<1x4x256x128xbf16>
    %54 = tensor.empty() : tensor<1x4x16x256x128xbf16>
    %55 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%53 : tensor<1x4x256x128xbf16>) outs(%54 : tensor<1x4x16x256x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x4x16x256x128xbf16>
    %collapsed_18 = tensor.collapse_shape %55 [[0], [1, 2], [3], [4]] : tensor<1x4x16x256x128xbf16> into tensor<1x64x256x128xbf16>
    %56 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%transposed_11 : tensor<1x4x256x128xbf16>) outs(%54 : tensor<1x4x16x256x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x4x16x256x128xbf16>
    %57 = tensor.empty() : tensor<1x64x128x256xbf16>
    %transposed_19 = linalg.transpose ins(%collapsed_18 : tensor<1x64x256x128xbf16>) outs(%57 : tensor<1x64x128x256xbf16>) permutation = [0, 1, 3, 2] 
    %collapsed_20 = tensor.collapse_shape %48 [[0, 1], [2], [3]] : tensor<1x64x256x128xbf16> into tensor<64x256x128xbf16>
    %58 = linalg.generic {indexing_maps = [#map8, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%collapsed_20 : tensor<64x256x128xbf16>) outs(%21 : tensor<1x64x256x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x64x256x128xbf16>
    %collapsed_21 = tensor.collapse_shape %transposed_19 [[0, 1], [2], [3]] : tensor<1x64x128x256xbf16> into tensor<64x128x256xbf16>
    %59 = linalg.generic {indexing_maps = [#map8, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%collapsed_21 : tensor<64x128x256xbf16>) outs(%57 : tensor<1x64x128x256xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x64x128x256xbf16>
    %collapsed_22 = tensor.collapse_shape %58 [[0, 1], [2], [3]] : tensor<1x64x256x128xbf16> into tensor<64x256x128xbf16>
    %collapsed_23 = tensor.collapse_shape %59 [[0, 1], [2], [3]] : tensor<1x64x128x256xbf16> into tensor<64x128x256xbf16>
    %60 = tensor.empty() : tensor<64x256x256xbf16>
    %61 = linalg.fill ins(%cst : bf16) outs(%60 : tensor<64x256x256xbf16>) -> tensor<64x256x256xbf16>
    %62 = linalg.batch_matmul ins(%collapsed_22, %collapsed_23 : tensor<64x256x128xbf16>, tensor<64x128x256xbf16>) outs(%61 : tensor<64x256x256xbf16>) -> tensor<64x256x256xbf16>
    %expanded_24 = tensor.expand_shape %62 [[0, 1], [2], [3]] output_shape [1, 64, 256, 256] : tensor<64x256x256xbf16> into tensor<1x64x256x256xbf16>
    %63 = tensor.empty() : tensor<1x64x256x256xbf16>
    %64 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_24 : tensor<1x64x256x256xbf16>) outs(%63 : tensor<1x64x256x256xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %91 = arith.truncf %cst_2 : f64 to bf16
      %92 = arith.mulf %in, %91 : bf16
      linalg.yield %92 : bf16
    } -> tensor<1x64x256x256xbf16>
    %65 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%64, %arg3 : tensor<1x64x256x256xbf16>, tensor<1x64x256x256xbf16>) outs(%63 : tensor<1x64x256x256xbf16>) {
    ^bb0(%in: bf16, %in_34: bf16, %out: bf16):
      %91 = arith.addf %in, %in_34 : bf16
      linalg.yield %91 : bf16
    } -> tensor<1x64x256x256xbf16>
    %66 = tensor.empty() : tensor<1x64x256x256xf32>
    %67 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%65 : tensor<1x64x256x256xbf16>) outs(%66 : tensor<1x64x256x256xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %91 = arith.extf %in : bf16 to f32
      linalg.yield %91 : f32
    } -> tensor<1x64x256x256xf32>
    %68 = tensor.empty() : tensor<1x64x256xi64>
    %69 = linalg.fill ins(%c0_i64 : i64) outs(%68 : tensor<1x64x256xi64>) -> tensor<1x64x256xi64>
    %70 = tensor.empty() : tensor<1x64x256xf32>
    %71 = linalg.fill ins(%cst_1 : f32) outs(%70 : tensor<1x64x256xf32>) -> tensor<1x64x256xf32>
    %72:2 = linalg.generic {indexing_maps = [#map2, #map9, #map9], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%67 : tensor<1x64x256x256xf32>) outs(%71, %69 : tensor<1x64x256xf32>, tensor<1x64x256xi64>) {
    ^bb0(%in: f32, %out: f32, %out_34: i64):
      %91 = linalg.index 3 : index
      %92 = arith.index_cast %91 : index to i64
      %93 = arith.maximumf %in, %out : f32
      %94 = arith.cmpf ogt, %in, %out : f32
      %95 = arith.select %94, %92, %out_34 : i64
      linalg.yield %93, %95 : f32, i64
    } -> (tensor<1x64x256xf32>, tensor<1x64x256xi64>)
    %expanded_25 = tensor.expand_shape %72#0 [[0], [1], [2, 3]] output_shape [1, 64, 256, 1] : tensor<1x64x256xf32> into tensor<1x64x256x1xf32>
    %73 = linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%67, %expanded_25 : tensor<1x64x256x256xf32>, tensor<1x64x256x1xf32>) outs(%66 : tensor<1x64x256x256xf32>) {
    ^bb0(%in: f32, %in_34: f32, %out: f32):
      %91 = arith.subf %in, %in_34 : f32
      linalg.yield %91 : f32
    } -> tensor<1x64x256x256xf32>
    %74 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%73 : tensor<1x64x256x256xf32>) outs(%66 : tensor<1x64x256x256xf32>) {
    ^bb0(%in: f32, %out: f32):
      %91 = math.exp %in : f32
      linalg.yield %91 : f32
    } -> tensor<1x64x256x256xf32>
    %75 = tensor.empty() : tensor<1x64x256x1xf32>
    %76 = linalg.fill ins(%cst_0 : f32) outs(%75 : tensor<1x64x256x1xf32>) -> tensor<1x64x256x1xf32>
    %77 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%74 : tensor<1x64x256x256xf32>) outs(%76 : tensor<1x64x256x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %91 = arith.addf %in, %out : f32
      linalg.yield %91 : f32
    } -> tensor<1x64x256x1xf32>
    %78 = linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%74, %77 : tensor<1x64x256x256xf32>, tensor<1x64x256x1xf32>) outs(%66 : tensor<1x64x256x256xf32>) {
    ^bb0(%in: f32, %in_34: f32, %out: f32):
      %91 = arith.divf %in, %in_34 : f32
      linalg.yield %91 : f32
    } -> tensor<1x64x256x256xf32>
    %79 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%78 : tensor<1x64x256x256xf32>) outs(%63 : tensor<1x64x256x256xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %91 = arith.truncf %in : f32 to bf16
      linalg.yield %91 : bf16
    } -> tensor<1x64x256x256xbf16>
    %collapsed_26 = tensor.collapse_shape %79 [[0, 1], [2], [3]] : tensor<1x64x256x256xbf16> into tensor<64x256x256xbf16>
    %80 = linalg.generic {indexing_maps = [#map8, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%collapsed_26 : tensor<64x256x256xbf16>) outs(%63 : tensor<1x64x256x256xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x64x256x256xbf16>
    %collapsed_27 = tensor.collapse_shape %56 [[0, 1, 2], [3], [4]] : tensor<1x4x16x256x128xbf16> into tensor<64x256x128xbf16>
    %81 = linalg.generic {indexing_maps = [#map8, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%collapsed_27 : tensor<64x256x128xbf16>) outs(%21 : tensor<1x64x256x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x64x256x128xbf16>
    %collapsed_28 = tensor.collapse_shape %80 [[0, 1], [2], [3]] : tensor<1x64x256x256xbf16> into tensor<64x256x256xbf16>
    %collapsed_29 = tensor.collapse_shape %81 [[0, 1], [2], [3]] : tensor<1x64x256x128xbf16> into tensor<64x256x128xbf16>
    %82 = tensor.empty() : tensor<64x256x128xbf16>
    %83 = linalg.fill ins(%cst : bf16) outs(%82 : tensor<64x256x128xbf16>) -> tensor<64x256x128xbf16>
    %84 = linalg.batch_matmul ins(%collapsed_28, %collapsed_29 : tensor<64x256x256xbf16>, tensor<64x256x128xbf16>) outs(%83 : tensor<64x256x128xbf16>) -> tensor<64x256x128xbf16>
    %expanded_30 = tensor.expand_shape %84 [[0, 1], [2], [3]] output_shape [1, 64, 256, 128] : tensor<64x256x128xbf16> into tensor<1x64x256x128xbf16>
    %transposed_31 = linalg.transpose ins(%expanded_30 : tensor<1x64x256x128xbf16>) outs(%18 : tensor<1x256x64x128xbf16>) permutation = [0, 2, 1, 3] 
    %85 = tensor.empty() : tensor<8192x4096xbf16>
    %transposed_32 = linalg.transpose ins(%arg7 : tensor<4096x8192xbf16>) outs(%85 : tensor<8192x4096xbf16>) permutation = [1, 0] 
    %collapsed_33 = tensor.collapse_shape %transposed_31 [[0, 1], [2, 3]] : tensor<1x256x64x128xbf16> into tensor<256x8192xbf16>
    %86 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapsed_33 : tensor<256x8192xbf16>) outs(%5 : tensor<1x256x8192xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x256x8192xbf16>
    %87 = tensor.empty() : tensor<1x8192x4096xbf16>
    %88 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_32 : tensor<8192x4096xbf16>) outs(%87 : tensor<1x8192x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x8192x4096xbf16>
    %89 = linalg.fill ins(%cst : bf16) outs(%1 : tensor<1x256x4096xbf16>) -> tensor<1x256x4096xbf16>
    %90 = linalg.batch_matmul ins(%86, %88 : tensor<1x256x8192xbf16>, tensor<1x8192x4096xbf16>) outs(%89 : tensor<1x256x4096xbf16>) -> tensor<1x256x4096xbf16>
    return %90 : tensor<1x256x4096xbf16>
  }
}
