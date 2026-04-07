#map = affine_map<(d0, d1, d2) -> (d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d3)>
#map5 = affine_map<(d0, d1, d2, d3) -> (0, 0, d2, d3)>
#map6 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>
#map7 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map8 = affine_map<(d0, d1, d2, d3) -> (0, d1, 0, 0)>
#map9 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
module {
  func.func @main(%arg0: tensor<8x4096x4096xbf16>, %arg1: tensor<1x4096x128xbf16>, %arg2: tensor<1x4096x128xbf16>, %arg3: tensor<1x32x1x1xbf16>, %arg4: tensor<4096x4096xbf16>, %arg5: tensor<1024x4096xbf16>, %arg6: tensor<1024x4096xbf16>, %arg7: tensor<4096x4096xbf16>, %arg8: tensor<128xbf16>, %arg9: tensor<128xbf16>) -> tensor<8x4096x4096xbf16> {
    %c2_i64 = arith.constant 2 : i64
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant 0.000000e+00 : bf16
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 0xFF800000 : f32
    %cst_2 = arith.constant 0.088388347648318447 : f64
    %cst_3 = arith.constant 9.9999999999999995E-7 : f64
    %cst_4 = arith.constant 1.280000e+02 : f32
    %0 = tensor.empty() : tensor<4096x4096xbf16>
    %transposed = linalg.transpose ins(%arg4 : tensor<4096x4096xbf16>) outs(%0 : tensor<4096x4096xbf16>) permutation = [1, 0] 
    %1 = tensor.empty() : tensor<8x4096x4096xbf16>
    %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed : tensor<4096x4096xbf16>) outs(%1 : tensor<8x4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x4096x4096xbf16>
    %3 = linalg.fill ins(%cst : bf16) outs(%1 : tensor<8x4096x4096xbf16>) -> tensor<8x4096x4096xbf16>
    %4 = linalg.batch_matmul ins(%arg0, %2 : tensor<8x4096x4096xbf16>, tensor<8x4096x4096xbf16>) outs(%3 : tensor<8x4096x4096xbf16>) -> tensor<8x4096x4096xbf16>
    %expanded = tensor.expand_shape %4 [[0], [1], [2, 3]] output_shape [8, 4096, 32, 128] : tensor<8x4096x4096xbf16> into tensor<8x4096x32x128xbf16>
    %5 = tensor.empty() : tensor<8x4096x32x128xf32>
    %6 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded : tensor<8x4096x32x128xbf16>) outs(%5 : tensor<8x4096x32x128xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %79 = arith.extf %in : bf16 to f32
      linalg.yield %79 : f32
    } -> tensor<8x4096x32x128xf32>
    %7 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%6 : tensor<8x4096x32x128xf32>) outs(%5 : tensor<8x4096x32x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %79 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %79 : f32
    } -> tensor<8x4096x32x128xf32>
    %8 = tensor.empty() : tensor<8x4096x32x1xf32>
    %9 = linalg.fill ins(%cst_0 : f32) outs(%8 : tensor<8x4096x32x1xf32>) -> tensor<8x4096x32x1xf32>
    %10 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%7 : tensor<8x4096x32x128xf32>) outs(%9 : tensor<8x4096x32x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %79 = arith.addf %in, %out : f32
      linalg.yield %79 : f32
    } -> tensor<8x4096x32x1xf32>
    %11 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%10 : tensor<8x4096x32x1xf32>) outs(%8 : tensor<8x4096x32x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %79 = arith.divf %in, %cst_4 : f32
      linalg.yield %79 : f32
    } -> tensor<8x4096x32x1xf32>
    %12 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%11 : tensor<8x4096x32x1xf32>) outs(%8 : tensor<8x4096x32x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %79 = arith.truncf %cst_3 : f64 to f32
      %80 = arith.addf %in, %79 : f32
      linalg.yield %80 : f32
    } -> tensor<8x4096x32x1xf32>
    %13 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%12 : tensor<8x4096x32x1xf32>) outs(%8 : tensor<8x4096x32x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %79 = math.rsqrt %in : f32
      linalg.yield %79 : f32
    } -> tensor<8x4096x32x1xf32>
    %14 = linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%6, %13 : tensor<8x4096x32x128xf32>, tensor<8x4096x32x1xf32>) outs(%5 : tensor<8x4096x32x128xf32>) {
    ^bb0(%in: f32, %in_29: f32, %out: f32):
      %79 = arith.mulf %in, %in_29 : f32
      linalg.yield %79 : f32
    } -> tensor<8x4096x32x128xf32>
    %15 = tensor.empty() : tensor<8x4096x32x128xbf16>
    %16 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%14 : tensor<8x4096x32x128xf32>) outs(%15 : tensor<8x4096x32x128xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %79 = arith.truncf %in : f32 to bf16
      linalg.yield %79 : bf16
    } -> tensor<8x4096x32x128xbf16>
    %17 = linalg.generic {indexing_maps = [#map4, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg8, %16 : tensor<128xbf16>, tensor<8x4096x32x128xbf16>) outs(%15 : tensor<8x4096x32x128xbf16>) {
    ^bb0(%in: bf16, %in_29: bf16, %out: bf16):
      %79 = arith.mulf %in, %in_29 : bf16
      linalg.yield %79 : bf16
    } -> tensor<8x4096x32x128xbf16>
    %18 = tensor.empty() : tensor<8x32x4096x128xbf16>
    %transposed_5 = linalg.transpose ins(%17 : tensor<8x4096x32x128xbf16>) outs(%18 : tensor<8x32x4096x128xbf16>) permutation = [0, 2, 1, 3] 
    %19 = tensor.empty() : tensor<4096x1024xbf16>
    %transposed_6 = linalg.transpose ins(%arg5 : tensor<1024x4096xbf16>) outs(%19 : tensor<4096x1024xbf16>) permutation = [1, 0] 
    %20 = tensor.empty() : tensor<8x4096x1024xbf16>
    %21 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_6 : tensor<4096x1024xbf16>) outs(%20 : tensor<8x4096x1024xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x4096x1024xbf16>
    %22 = linalg.fill ins(%cst : bf16) outs(%20 : tensor<8x4096x1024xbf16>) -> tensor<8x4096x1024xbf16>
    %23 = linalg.batch_matmul ins(%arg0, %21 : tensor<8x4096x4096xbf16>, tensor<8x4096x1024xbf16>) outs(%22 : tensor<8x4096x1024xbf16>) -> tensor<8x4096x1024xbf16>
    %expanded_7 = tensor.expand_shape %23 [[0], [1], [2, 3]] output_shape [8, 4096, 8, 128] : tensor<8x4096x1024xbf16> into tensor<8x4096x8x128xbf16>
    %24 = tensor.empty() : tensor<8x4096x8x128xf32>
    %25 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_7 : tensor<8x4096x8x128xbf16>) outs(%24 : tensor<8x4096x8x128xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %79 = arith.extf %in : bf16 to f32
      linalg.yield %79 : f32
    } -> tensor<8x4096x8x128xf32>
    %26 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%25 : tensor<8x4096x8x128xf32>) outs(%24 : tensor<8x4096x8x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %79 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %79 : f32
    } -> tensor<8x4096x8x128xf32>
    %27 = tensor.empty() : tensor<8x4096x8x1xf32>
    %28 = linalg.fill ins(%cst_0 : f32) outs(%27 : tensor<8x4096x8x1xf32>) -> tensor<8x4096x8x1xf32>
    %29 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%26 : tensor<8x4096x8x128xf32>) outs(%28 : tensor<8x4096x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %79 = arith.addf %in, %out : f32
      linalg.yield %79 : f32
    } -> tensor<8x4096x8x1xf32>
    %30 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%29 : tensor<8x4096x8x1xf32>) outs(%27 : tensor<8x4096x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %79 = arith.divf %in, %cst_4 : f32
      linalg.yield %79 : f32
    } -> tensor<8x4096x8x1xf32>
    %31 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%30 : tensor<8x4096x8x1xf32>) outs(%27 : tensor<8x4096x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %79 = arith.truncf %cst_3 : f64 to f32
      %80 = arith.addf %in, %79 : f32
      linalg.yield %80 : f32
    } -> tensor<8x4096x8x1xf32>
    %32 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%31 : tensor<8x4096x8x1xf32>) outs(%27 : tensor<8x4096x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %79 = math.rsqrt %in : f32
      linalg.yield %79 : f32
    } -> tensor<8x4096x8x1xf32>
    %33 = linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%25, %32 : tensor<8x4096x8x128xf32>, tensor<8x4096x8x1xf32>) outs(%24 : tensor<8x4096x8x128xf32>) {
    ^bb0(%in: f32, %in_29: f32, %out: f32):
      %79 = arith.mulf %in, %in_29 : f32
      linalg.yield %79 : f32
    } -> tensor<8x4096x8x128xf32>
    %34 = tensor.empty() : tensor<8x4096x8x128xbf16>
    %35 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%33 : tensor<8x4096x8x128xf32>) outs(%34 : tensor<8x4096x8x128xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %79 = arith.truncf %in : f32 to bf16
      linalg.yield %79 : bf16
    } -> tensor<8x4096x8x128xbf16>
    %36 = linalg.generic {indexing_maps = [#map4, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg9, %35 : tensor<128xbf16>, tensor<8x4096x8x128xbf16>) outs(%34 : tensor<8x4096x8x128xbf16>) {
    ^bb0(%in: bf16, %in_29: bf16, %out: bf16):
      %79 = arith.mulf %in, %in_29 : bf16
      linalg.yield %79 : bf16
    } -> tensor<8x4096x8x128xbf16>
    %37 = tensor.empty() : tensor<8x8x4096x128xbf16>
    %transposed_8 = linalg.transpose ins(%36 : tensor<8x4096x8x128xbf16>) outs(%37 : tensor<8x8x4096x128xbf16>) permutation = [0, 2, 1, 3] 
    %transposed_9 = linalg.transpose ins(%arg6 : tensor<1024x4096xbf16>) outs(%19 : tensor<4096x1024xbf16>) permutation = [1, 0] 
    %38 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_9 : tensor<4096x1024xbf16>) outs(%20 : tensor<8x4096x1024xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x4096x1024xbf16>
    %39 = linalg.batch_matmul ins(%arg0, %38 : tensor<8x4096x4096xbf16>, tensor<8x4096x1024xbf16>) outs(%22 : tensor<8x4096x1024xbf16>) -> tensor<8x4096x1024xbf16>
    %expanded_10 = tensor.expand_shape %39 [[0], [1], [2, 3]] output_shape [8, 4096, 8, 128] : tensor<8x4096x1024xbf16> into tensor<8x4096x8x128xbf16>
    %transposed_11 = linalg.transpose ins(%expanded_10 : tensor<8x4096x8x128xbf16>) outs(%37 : tensor<8x8x4096x128xbf16>) permutation = [0, 2, 1, 3] 
    %expanded_12 = tensor.expand_shape %arg1 [[0], [1, 2], [3]] output_shape [1, 1, 4096, 128] : tensor<1x4096x128xbf16> into tensor<1x1x4096x128xbf16>
    %expanded_13 = tensor.expand_shape %arg2 [[0], [1, 2], [3]] output_shape [1, 1, 4096, 128] : tensor<1x4096x128xbf16> into tensor<1x1x4096x128xbf16>
    %40 = linalg.generic {indexing_maps = [#map2, #map5, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_5, %expanded_12 : tensor<8x32x4096x128xbf16>, tensor<1x1x4096x128xbf16>) outs(%18 : tensor<8x32x4096x128xbf16>) {
    ^bb0(%in: bf16, %in_29: bf16, %out: bf16):
      %79 = arith.mulf %in, %in_29 : bf16
      linalg.yield %79 : bf16
    } -> tensor<8x32x4096x128xbf16>
    %extracted_slice = tensor.extract_slice %transposed_5[0, 0, 0, 0] [8, 32, 4096, 64] [1, 1, 1, 1] : tensor<8x32x4096x128xbf16> to tensor<8x32x4096x64xbf16>
    %extracted_slice_14 = tensor.extract_slice %transposed_5[0, 0, 0, 64] [8, 32, 4096, 64] [1, 1, 1, 1] : tensor<8x32x4096x128xbf16> to tensor<8x32x4096x64xbf16>
    %41 = tensor.empty() : tensor<8x32x4096x64xbf16>
    %42 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_14 : tensor<8x32x4096x64xbf16>) outs(%41 : tensor<8x32x4096x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %79 = arith.negf %in : bf16
      linalg.yield %79 : bf16
    } -> tensor<8x32x4096x64xbf16>
    %concat = tensor.concat dim(3) %42, %extracted_slice : (tensor<8x32x4096x64xbf16>, tensor<8x32x4096x64xbf16>) -> tensor<8x32x4096x128xbf16>
    %43 = linalg.generic {indexing_maps = [#map2, #map5, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat, %expanded_13 : tensor<8x32x4096x128xbf16>, tensor<1x1x4096x128xbf16>) outs(%18 : tensor<8x32x4096x128xbf16>) {
    ^bb0(%in: bf16, %in_29: bf16, %out: bf16):
      %79 = arith.mulf %in, %in_29 : bf16
      linalg.yield %79 : bf16
    } -> tensor<8x32x4096x128xbf16>
    %44 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%40, %43 : tensor<8x32x4096x128xbf16>, tensor<8x32x4096x128xbf16>) outs(%18 : tensor<8x32x4096x128xbf16>) {
    ^bb0(%in: bf16, %in_29: bf16, %out: bf16):
      %79 = arith.addf %in, %in_29 : bf16
      linalg.yield %79 : bf16
    } -> tensor<8x32x4096x128xbf16>
    %45 = linalg.generic {indexing_maps = [#map2, #map5, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_8, %expanded_12 : tensor<8x8x4096x128xbf16>, tensor<1x1x4096x128xbf16>) outs(%37 : tensor<8x8x4096x128xbf16>) {
    ^bb0(%in: bf16, %in_29: bf16, %out: bf16):
      %79 = arith.mulf %in, %in_29 : bf16
      linalg.yield %79 : bf16
    } -> tensor<8x8x4096x128xbf16>
    %extracted_slice_15 = tensor.extract_slice %transposed_8[0, 0, 0, 0] [8, 8, 4096, 64] [1, 1, 1, 1] : tensor<8x8x4096x128xbf16> to tensor<8x8x4096x64xbf16>
    %extracted_slice_16 = tensor.extract_slice %transposed_8[0, 0, 0, 64] [8, 8, 4096, 64] [1, 1, 1, 1] : tensor<8x8x4096x128xbf16> to tensor<8x8x4096x64xbf16>
    %46 = tensor.empty() : tensor<8x8x4096x64xbf16>
    %47 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_16 : tensor<8x8x4096x64xbf16>) outs(%46 : tensor<8x8x4096x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %79 = arith.negf %in : bf16
      linalg.yield %79 : bf16
    } -> tensor<8x8x4096x64xbf16>
    %concat_17 = tensor.concat dim(3) %47, %extracted_slice_15 : (tensor<8x8x4096x64xbf16>, tensor<8x8x4096x64xbf16>) -> tensor<8x8x4096x128xbf16>
    %48 = linalg.generic {indexing_maps = [#map2, #map5, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_17, %expanded_13 : tensor<8x8x4096x128xbf16>, tensor<1x1x4096x128xbf16>) outs(%37 : tensor<8x8x4096x128xbf16>) {
    ^bb0(%in: bf16, %in_29: bf16, %out: bf16):
      %79 = arith.mulf %in, %in_29 : bf16
      linalg.yield %79 : bf16
    } -> tensor<8x8x4096x128xbf16>
    %49 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%45, %48 : tensor<8x8x4096x128xbf16>, tensor<8x8x4096x128xbf16>) outs(%37 : tensor<8x8x4096x128xbf16>) {
    ^bb0(%in: bf16, %in_29: bf16, %out: bf16):
      %79 = arith.addf %in, %in_29 : bf16
      linalg.yield %79 : bf16
    } -> tensor<8x8x4096x128xbf16>
    %50 = tensor.empty() : tensor<8x8x4x4096x128xbf16>
    %51 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%49 : tensor<8x8x4096x128xbf16>) outs(%50 : tensor<8x8x4x4096x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x8x4x4096x128xbf16>
    %collapsed = tensor.collapse_shape %51 [[0], [1, 2], [3], [4]] : tensor<8x8x4x4096x128xbf16> into tensor<8x32x4096x128xbf16>
    %52 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%transposed_11 : tensor<8x8x4096x128xbf16>) outs(%50 : tensor<8x8x4x4096x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x8x4x4096x128xbf16>
    %53 = tensor.empty() : tensor<8x32x128x4096xbf16>
    %transposed_18 = linalg.transpose ins(%collapsed : tensor<8x32x4096x128xbf16>) outs(%53 : tensor<8x32x128x4096xbf16>) permutation = [0, 1, 3, 2] 
    %collapsed_19 = tensor.collapse_shape %44 [[0, 1], [2], [3]] : tensor<8x32x4096x128xbf16> into tensor<256x4096x128xbf16>
    %collapsed_20 = tensor.collapse_shape %transposed_18 [[0, 1], [2], [3]] : tensor<8x32x128x4096xbf16> into tensor<256x128x4096xbf16>
    %54 = tensor.empty() : tensor<256x4096x4096xbf16>
    %55 = linalg.fill ins(%cst : bf16) outs(%54 : tensor<256x4096x4096xbf16>) -> tensor<256x4096x4096xbf16>
    %56 = linalg.batch_matmul ins(%collapsed_19, %collapsed_20 : tensor<256x4096x128xbf16>, tensor<256x128x4096xbf16>) outs(%55 : tensor<256x4096x4096xbf16>) -> tensor<256x4096x4096xbf16>
    %expanded_21 = tensor.expand_shape %56 [[0, 1], [2], [3]] output_shape [8, 32, 4096, 4096] : tensor<256x4096x4096xbf16> into tensor<8x32x4096x4096xbf16>
    %57 = tensor.empty() : tensor<8x32x4096x4096xbf16>
    %58 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_21 : tensor<8x32x4096x4096xbf16>) outs(%57 : tensor<8x32x4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %79 = arith.truncf %cst_2 : f64 to bf16
      %80 = arith.mulf %in, %79 : bf16
      linalg.yield %80 : bf16
    } -> tensor<8x32x4096x4096xbf16>
    %59 = linalg.generic {indexing_maps = [#map2, #map8, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%58, %arg3 : tensor<8x32x4096x4096xbf16>, tensor<1x32x1x1xbf16>) outs(%57 : tensor<8x32x4096x4096xbf16>) {
    ^bb0(%in: bf16, %in_29: bf16, %out: bf16):
      %79 = arith.addf %in, %in_29 : bf16
      linalg.yield %79 : bf16
    } -> tensor<8x32x4096x4096xbf16>
    %60 = tensor.empty() : tensor<8x32x4096x4096xf32>
    %61 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%59 : tensor<8x32x4096x4096xbf16>) outs(%60 : tensor<8x32x4096x4096xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %79 = arith.extf %in : bf16 to f32
      linalg.yield %79 : f32
    } -> tensor<8x32x4096x4096xf32>
    %62 = tensor.empty() : tensor<8x32x4096xi64>
    %63 = linalg.fill ins(%c0_i64 : i64) outs(%62 : tensor<8x32x4096xi64>) -> tensor<8x32x4096xi64>
    %64 = tensor.empty() : tensor<8x32x4096xf32>
    %65 = linalg.fill ins(%cst_1 : f32) outs(%64 : tensor<8x32x4096xf32>) -> tensor<8x32x4096xf32>
    %66:2 = linalg.generic {indexing_maps = [#map2, #map9, #map9], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%61 : tensor<8x32x4096x4096xf32>) outs(%65, %63 : tensor<8x32x4096xf32>, tensor<8x32x4096xi64>) {
    ^bb0(%in: f32, %out: f32, %out_29: i64):
      %79 = linalg.index 3 : index
      %80 = arith.index_cast %79 : index to i64
      %81 = arith.maximumf %in, %out : f32
      %82 = arith.cmpf ogt, %in, %out : f32
      %83 = arith.select %82, %80, %out_29 : i64
      linalg.yield %81, %83 : f32, i64
    } -> (tensor<8x32x4096xf32>, tensor<8x32x4096xi64>)
    %expanded_22 = tensor.expand_shape %66#0 [[0], [1], [2, 3]] output_shape [8, 32, 4096, 1] : tensor<8x32x4096xf32> into tensor<8x32x4096x1xf32>
    %67 = linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%61, %expanded_22 : tensor<8x32x4096x4096xf32>, tensor<8x32x4096x1xf32>) outs(%60 : tensor<8x32x4096x4096xf32>) {
    ^bb0(%in: f32, %in_29: f32, %out: f32):
      %79 = arith.subf %in, %in_29 : f32
      linalg.yield %79 : f32
    } -> tensor<8x32x4096x4096xf32>
    %68 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%67 : tensor<8x32x4096x4096xf32>) outs(%60 : tensor<8x32x4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %79 = math.exp %in : f32
      linalg.yield %79 : f32
    } -> tensor<8x32x4096x4096xf32>
    %69 = tensor.empty() : tensor<8x32x4096x1xf32>
    %70 = linalg.fill ins(%cst_0 : f32) outs(%69 : tensor<8x32x4096x1xf32>) -> tensor<8x32x4096x1xf32>
    %71 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%68 : tensor<8x32x4096x4096xf32>) outs(%70 : tensor<8x32x4096x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %79 = arith.addf %in, %out : f32
      linalg.yield %79 : f32
    } -> tensor<8x32x4096x1xf32>
    %72 = linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%68, %71 : tensor<8x32x4096x4096xf32>, tensor<8x32x4096x1xf32>) outs(%60 : tensor<8x32x4096x4096xf32>) {
    ^bb0(%in: f32, %in_29: f32, %out: f32):
      %79 = arith.divf %in, %in_29 : f32
      linalg.yield %79 : f32
    } -> tensor<8x32x4096x4096xf32>
    %73 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%72 : tensor<8x32x4096x4096xf32>) outs(%57 : tensor<8x32x4096x4096xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %79 = arith.truncf %in : f32 to bf16
      linalg.yield %79 : bf16
    } -> tensor<8x32x4096x4096xbf16>
    %collapsed_23 = tensor.collapse_shape %73 [[0, 1], [2], [3]] : tensor<8x32x4096x4096xbf16> into tensor<256x4096x4096xbf16>
    %collapsed_24 = tensor.collapse_shape %52 [[0, 1, 2], [3], [4]] : tensor<8x8x4x4096x128xbf16> into tensor<256x4096x128xbf16>
    %74 = tensor.empty() : tensor<256x4096x128xbf16>
    %75 = linalg.fill ins(%cst : bf16) outs(%74 : tensor<256x4096x128xbf16>) -> tensor<256x4096x128xbf16>
    %76 = linalg.batch_matmul ins(%collapsed_23, %collapsed_24 : tensor<256x4096x4096xbf16>, tensor<256x4096x128xbf16>) outs(%75 : tensor<256x4096x128xbf16>) -> tensor<256x4096x128xbf16>
    %expanded_25 = tensor.expand_shape %76 [[0, 1], [2], [3]] output_shape [8, 32, 4096, 128] : tensor<256x4096x128xbf16> into tensor<8x32x4096x128xbf16>
    %transposed_26 = linalg.transpose ins(%expanded_25 : tensor<8x32x4096x128xbf16>) outs(%15 : tensor<8x4096x32x128xbf16>) permutation = [0, 2, 1, 3] 
    %collapsed_27 = tensor.collapse_shape %transposed_26 [[0], [1], [2, 3]] : tensor<8x4096x32x128xbf16> into tensor<8x4096x4096xbf16>
    %transposed_28 = linalg.transpose ins(%arg7 : tensor<4096x4096xbf16>) outs(%0 : tensor<4096x4096xbf16>) permutation = [1, 0] 
    %77 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_28 : tensor<4096x4096xbf16>) outs(%1 : tensor<8x4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x4096x4096xbf16>
    %78 = linalg.batch_matmul ins(%collapsed_27, %77 : tensor<8x4096x4096xbf16>, tensor<8x4096x4096xbf16>) outs(%3 : tensor<8x4096x4096xbf16>) -> tensor<8x4096x4096xbf16>
    return %78 : tensor<8x4096x4096xbf16>
  }
}
