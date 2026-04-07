#map = affine_map<(d0, d1, d2) -> (d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d3)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>
#map6 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>
#map7 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map8 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
#map9 = affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, 0)>
#map10 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
module {
  func.func @main(%arg0: tensor<1x4096x4096xbf16>, %arg1: tensor<1x4096x128xbf16>, %arg2: tensor<1x4096x128xbf16>, %arg3: tensor<1x32x1x1xbf16>, %arg4: tensor<4096x4096xbf16>, %arg5: tensor<1024x4096xbf16>, %arg6: tensor<1024x4096xbf16>, %arg7: tensor<4096x4096xbf16>, %arg8: tensor<128xbf16>, %arg9: tensor<128xbf16>) -> tensor<1x4096x4096xbf16> {
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
    %6 = tensor.empty() : tensor<1x4096x32x128xf32>
    %7 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded : tensor<1x4096x32x128xbf16>) outs(%6 : tensor<1x4096x32x128xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %85 = arith.extf %in : bf16 to f32
      linalg.yield %85 : f32
    } -> tensor<1x4096x32x128xf32>
    %8 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%7 : tensor<1x4096x32x128xf32>) outs(%6 : tensor<1x4096x32x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %85 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %85 : f32
    } -> tensor<1x4096x32x128xf32>
    %9 = tensor.empty() : tensor<1x4096x32x1xf32>
    %10 = linalg.fill ins(%cst_0 : f32) outs(%9 : tensor<1x4096x32x1xf32>) -> tensor<1x4096x32x1xf32>
    %11 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%8 : tensor<1x4096x32x128xf32>) outs(%10 : tensor<1x4096x32x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %85 = arith.addf %in, %out : f32
      linalg.yield %85 : f32
    } -> tensor<1x4096x32x1xf32>
    %12 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%11 : tensor<1x4096x32x1xf32>) outs(%9 : tensor<1x4096x32x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %85 = arith.divf %in, %cst_4 : f32
      linalg.yield %85 : f32
    } -> tensor<1x4096x32x1xf32>
    %13 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%12 : tensor<1x4096x32x1xf32>) outs(%9 : tensor<1x4096x32x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %85 = arith.truncf %cst_3 : f64 to f32
      %86 = arith.addf %in, %85 : f32
      linalg.yield %86 : f32
    } -> tensor<1x4096x32x1xf32>
    %14 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%13 : tensor<1x4096x32x1xf32>) outs(%9 : tensor<1x4096x32x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %85 = math.rsqrt %in : f32
      linalg.yield %85 : f32
    } -> tensor<1x4096x32x1xf32>
    %15 = linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%7, %14 : tensor<1x4096x32x128xf32>, tensor<1x4096x32x1xf32>) outs(%6 : tensor<1x4096x32x128xf32>) {
    ^bb0(%in: f32, %in_34: f32, %out: f32):
      %85 = arith.mulf %in, %in_34 : f32
      linalg.yield %85 : f32
    } -> tensor<1x4096x32x128xf32>
    %16 = tensor.empty() : tensor<1x4096x32x128xbf16>
    %17 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%15 : tensor<1x4096x32x128xf32>) outs(%16 : tensor<1x4096x32x128xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %85 = arith.truncf %in : f32 to bf16
      linalg.yield %85 : bf16
    } -> tensor<1x4096x32x128xbf16>
    %18 = linalg.generic {indexing_maps = [#map4, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg8, %17 : tensor<128xbf16>, tensor<1x4096x32x128xbf16>) outs(%16 : tensor<1x4096x32x128xbf16>) {
    ^bb0(%in: bf16, %in_34: bf16, %out: bf16):
      %85 = arith.mulf %in, %in_34 : bf16
      linalg.yield %85 : bf16
    } -> tensor<1x4096x32x128xbf16>
    %19 = tensor.empty() : tensor<1x32x4096x128xbf16>
    %transposed_5 = linalg.transpose ins(%18 : tensor<1x4096x32x128xbf16>) outs(%19 : tensor<1x32x4096x128xbf16>) permutation = [0, 2, 1, 3] 
    %20 = tensor.empty() : tensor<4096x1024xbf16>
    %transposed_6 = linalg.transpose ins(%arg5 : tensor<1024x4096xbf16>) outs(%20 : tensor<4096x1024xbf16>) permutation = [1, 0] 
    %21 = tensor.empty() : tensor<1x4096x1024xbf16>
    %22 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_6 : tensor<4096x1024xbf16>) outs(%21 : tensor<1x4096x1024xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x4096x1024xbf16>
    %23 = linalg.fill ins(%cst : bf16) outs(%21 : tensor<1x4096x1024xbf16>) -> tensor<1x4096x1024xbf16>
    %24 = linalg.batch_matmul ins(%2, %22 : tensor<1x4096x4096xbf16>, tensor<1x4096x1024xbf16>) outs(%23 : tensor<1x4096x1024xbf16>) -> tensor<1x4096x1024xbf16>
    %expanded_7 = tensor.expand_shape %24 [[0], [1], [2, 3]] output_shape [1, 4096, 8, 128] : tensor<1x4096x1024xbf16> into tensor<1x4096x8x128xbf16>
    %25 = tensor.empty() : tensor<1x4096x8x128xf32>
    %26 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_7 : tensor<1x4096x8x128xbf16>) outs(%25 : tensor<1x4096x8x128xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %85 = arith.extf %in : bf16 to f32
      linalg.yield %85 : f32
    } -> tensor<1x4096x8x128xf32>
    %27 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%26 : tensor<1x4096x8x128xf32>) outs(%25 : tensor<1x4096x8x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %85 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %85 : f32
    } -> tensor<1x4096x8x128xf32>
    %28 = tensor.empty() : tensor<1x4096x8x1xf32>
    %29 = linalg.fill ins(%cst_0 : f32) outs(%28 : tensor<1x4096x8x1xf32>) -> tensor<1x4096x8x1xf32>
    %30 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%27 : tensor<1x4096x8x128xf32>) outs(%29 : tensor<1x4096x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %85 = arith.addf %in, %out : f32
      linalg.yield %85 : f32
    } -> tensor<1x4096x8x1xf32>
    %31 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%30 : tensor<1x4096x8x1xf32>) outs(%28 : tensor<1x4096x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %85 = arith.divf %in, %cst_4 : f32
      linalg.yield %85 : f32
    } -> tensor<1x4096x8x1xf32>
    %32 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%31 : tensor<1x4096x8x1xf32>) outs(%28 : tensor<1x4096x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %85 = arith.truncf %cst_3 : f64 to f32
      %86 = arith.addf %in, %85 : f32
      linalg.yield %86 : f32
    } -> tensor<1x4096x8x1xf32>
    %33 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%32 : tensor<1x4096x8x1xf32>) outs(%28 : tensor<1x4096x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %85 = math.rsqrt %in : f32
      linalg.yield %85 : f32
    } -> tensor<1x4096x8x1xf32>
    %34 = linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%26, %33 : tensor<1x4096x8x128xf32>, tensor<1x4096x8x1xf32>) outs(%25 : tensor<1x4096x8x128xf32>) {
    ^bb0(%in: f32, %in_34: f32, %out: f32):
      %85 = arith.mulf %in, %in_34 : f32
      linalg.yield %85 : f32
    } -> tensor<1x4096x8x128xf32>
    %35 = tensor.empty() : tensor<1x4096x8x128xbf16>
    %36 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%34 : tensor<1x4096x8x128xf32>) outs(%35 : tensor<1x4096x8x128xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %85 = arith.truncf %in : f32 to bf16
      linalg.yield %85 : bf16
    } -> tensor<1x4096x8x128xbf16>
    %37 = linalg.generic {indexing_maps = [#map4, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg9, %36 : tensor<128xbf16>, tensor<1x4096x8x128xbf16>) outs(%35 : tensor<1x4096x8x128xbf16>) {
    ^bb0(%in: bf16, %in_34: bf16, %out: bf16):
      %85 = arith.mulf %in, %in_34 : bf16
      linalg.yield %85 : bf16
    } -> tensor<1x4096x8x128xbf16>
    %38 = tensor.empty() : tensor<1x8x4096x128xbf16>
    %transposed_8 = linalg.transpose ins(%37 : tensor<1x4096x8x128xbf16>) outs(%38 : tensor<1x8x4096x128xbf16>) permutation = [0, 2, 1, 3] 
    %transposed_9 = linalg.transpose ins(%arg6 : tensor<1024x4096xbf16>) outs(%20 : tensor<4096x1024xbf16>) permutation = [1, 0] 
    %39 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_9 : tensor<4096x1024xbf16>) outs(%21 : tensor<1x4096x1024xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x4096x1024xbf16>
    %40 = linalg.batch_matmul ins(%2, %39 : tensor<1x4096x4096xbf16>, tensor<1x4096x1024xbf16>) outs(%23 : tensor<1x4096x1024xbf16>) -> tensor<1x4096x1024xbf16>
    %expanded_10 = tensor.expand_shape %40 [[0], [1], [2, 3]] output_shape [1, 4096, 8, 128] : tensor<1x4096x1024xbf16> into tensor<1x4096x8x128xbf16>
    %transposed_11 = linalg.transpose ins(%expanded_10 : tensor<1x4096x8x128xbf16>) outs(%38 : tensor<1x8x4096x128xbf16>) permutation = [0, 2, 1, 3] 
    %expanded_12 = tensor.expand_shape %arg1 [[0], [1, 2], [3]] output_shape [1, 1, 4096, 128] : tensor<1x4096x128xbf16> into tensor<1x1x4096x128xbf16>
    %expanded_13 = tensor.expand_shape %arg2 [[0], [1, 2], [3]] output_shape [1, 1, 4096, 128] : tensor<1x4096x128xbf16> into tensor<1x1x4096x128xbf16>
    %41 = linalg.generic {indexing_maps = [#map2, #map5, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_5, %expanded_12 : tensor<1x32x4096x128xbf16>, tensor<1x1x4096x128xbf16>) outs(%19 : tensor<1x32x4096x128xbf16>) {
    ^bb0(%in: bf16, %in_34: bf16, %out: bf16):
      %85 = arith.mulf %in, %in_34 : bf16
      linalg.yield %85 : bf16
    } -> tensor<1x32x4096x128xbf16>
    %extracted_slice = tensor.extract_slice %transposed_5[0, 0, 0, 0] [1, 32, 4096, 64] [1, 1, 1, 1] : tensor<1x32x4096x128xbf16> to tensor<1x32x4096x64xbf16>
    %extracted_slice_14 = tensor.extract_slice %transposed_5[0, 0, 0, 64] [1, 32, 4096, 64] [1, 1, 1, 1] : tensor<1x32x4096x128xbf16> to tensor<1x32x4096x64xbf16>
    %42 = tensor.empty() : tensor<1x32x4096x64xbf16>
    %43 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_14 : tensor<1x32x4096x64xbf16>) outs(%42 : tensor<1x32x4096x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %85 = arith.negf %in : bf16
      linalg.yield %85 : bf16
    } -> tensor<1x32x4096x64xbf16>
    %concat = tensor.concat dim(3) %43, %extracted_slice : (tensor<1x32x4096x64xbf16>, tensor<1x32x4096x64xbf16>) -> tensor<1x32x4096x128xbf16>
    %44 = linalg.generic {indexing_maps = [#map2, #map5, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat, %expanded_13 : tensor<1x32x4096x128xbf16>, tensor<1x1x4096x128xbf16>) outs(%19 : tensor<1x32x4096x128xbf16>) {
    ^bb0(%in: bf16, %in_34: bf16, %out: bf16):
      %85 = arith.mulf %in, %in_34 : bf16
      linalg.yield %85 : bf16
    } -> tensor<1x32x4096x128xbf16>
    %45 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%41, %44 : tensor<1x32x4096x128xbf16>, tensor<1x32x4096x128xbf16>) outs(%19 : tensor<1x32x4096x128xbf16>) {
    ^bb0(%in: bf16, %in_34: bf16, %out: bf16):
      %85 = arith.addf %in, %in_34 : bf16
      linalg.yield %85 : bf16
    } -> tensor<1x32x4096x128xbf16>
    %46 = linalg.generic {indexing_maps = [#map2, #map5, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_8, %expanded_12 : tensor<1x8x4096x128xbf16>, tensor<1x1x4096x128xbf16>) outs(%38 : tensor<1x8x4096x128xbf16>) {
    ^bb0(%in: bf16, %in_34: bf16, %out: bf16):
      %85 = arith.mulf %in, %in_34 : bf16
      linalg.yield %85 : bf16
    } -> tensor<1x8x4096x128xbf16>
    %extracted_slice_15 = tensor.extract_slice %transposed_8[0, 0, 0, 0] [1, 8, 4096, 64] [1, 1, 1, 1] : tensor<1x8x4096x128xbf16> to tensor<1x8x4096x64xbf16>
    %extracted_slice_16 = tensor.extract_slice %transposed_8[0, 0, 0, 64] [1, 8, 4096, 64] [1, 1, 1, 1] : tensor<1x8x4096x128xbf16> to tensor<1x8x4096x64xbf16>
    %47 = tensor.empty() : tensor<1x8x4096x64xbf16>
    %48 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_16 : tensor<1x8x4096x64xbf16>) outs(%47 : tensor<1x8x4096x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %85 = arith.negf %in : bf16
      linalg.yield %85 : bf16
    } -> tensor<1x8x4096x64xbf16>
    %concat_17 = tensor.concat dim(3) %48, %extracted_slice_15 : (tensor<1x8x4096x64xbf16>, tensor<1x8x4096x64xbf16>) -> tensor<1x8x4096x128xbf16>
    %49 = linalg.generic {indexing_maps = [#map2, #map5, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_17, %expanded_13 : tensor<1x8x4096x128xbf16>, tensor<1x1x4096x128xbf16>) outs(%38 : tensor<1x8x4096x128xbf16>) {
    ^bb0(%in: bf16, %in_34: bf16, %out: bf16):
      %85 = arith.mulf %in, %in_34 : bf16
      linalg.yield %85 : bf16
    } -> tensor<1x8x4096x128xbf16>
    %50 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%46, %49 : tensor<1x8x4096x128xbf16>, tensor<1x8x4096x128xbf16>) outs(%38 : tensor<1x8x4096x128xbf16>) {
    ^bb0(%in: bf16, %in_34: bf16, %out: bf16):
      %85 = arith.addf %in, %in_34 : bf16
      linalg.yield %85 : bf16
    } -> tensor<1x8x4096x128xbf16>
    %51 = tensor.empty() : tensor<1x8x4x4096x128xbf16>
    %52 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%50 : tensor<1x8x4096x128xbf16>) outs(%51 : tensor<1x8x4x4096x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x8x4x4096x128xbf16>
    %collapsed_18 = tensor.collapse_shape %52 [[0], [1, 2], [3], [4]] : tensor<1x8x4x4096x128xbf16> into tensor<1x32x4096x128xbf16>
    %53 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%transposed_11 : tensor<1x8x4096x128xbf16>) outs(%51 : tensor<1x8x4x4096x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x8x4x4096x128xbf16>
    %54 = tensor.empty() : tensor<1x32x128x4096xbf16>
    %transposed_19 = linalg.transpose ins(%collapsed_18 : tensor<1x32x4096x128xbf16>) outs(%54 : tensor<1x32x128x4096xbf16>) permutation = [0, 1, 3, 2] 
    %collapsed_20 = tensor.collapse_shape %45 [[0, 1], [2], [3]] : tensor<1x32x4096x128xbf16> into tensor<32x4096x128xbf16>
    %55 = linalg.generic {indexing_maps = [#map8, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%collapsed_20 : tensor<32x4096x128xbf16>) outs(%19 : tensor<1x32x4096x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x4096x128xbf16>
    %collapsed_21 = tensor.collapse_shape %transposed_19 [[0, 1], [2], [3]] : tensor<1x32x128x4096xbf16> into tensor<32x128x4096xbf16>
    %56 = linalg.generic {indexing_maps = [#map8, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%collapsed_21 : tensor<32x128x4096xbf16>) outs(%54 : tensor<1x32x128x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x128x4096xbf16>
    %collapsed_22 = tensor.collapse_shape %55 [[0, 1], [2], [3]] : tensor<1x32x4096x128xbf16> into tensor<32x4096x128xbf16>
    %collapsed_23 = tensor.collapse_shape %56 [[0, 1], [2], [3]] : tensor<1x32x128x4096xbf16> into tensor<32x128x4096xbf16>
    %57 = tensor.empty() : tensor<32x4096x4096xbf16>
    %58 = linalg.fill ins(%cst : bf16) outs(%57 : tensor<32x4096x4096xbf16>) -> tensor<32x4096x4096xbf16>
    %59 = linalg.batch_matmul ins(%collapsed_22, %collapsed_23 : tensor<32x4096x128xbf16>, tensor<32x128x4096xbf16>) outs(%58 : tensor<32x4096x4096xbf16>) -> tensor<32x4096x4096xbf16>
    %expanded_24 = tensor.expand_shape %59 [[0, 1], [2], [3]] output_shape [1, 32, 4096, 4096] : tensor<32x4096x4096xbf16> into tensor<1x32x4096x4096xbf16>
    %60 = tensor.empty() : tensor<1x32x4096x4096xbf16>
    %61 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_24 : tensor<1x32x4096x4096xbf16>) outs(%60 : tensor<1x32x4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %85 = arith.truncf %cst_2 : f64 to bf16
      %86 = arith.mulf %in, %85 : bf16
      linalg.yield %86 : bf16
    } -> tensor<1x32x4096x4096xbf16>
    %62 = linalg.generic {indexing_maps = [#map2, #map9, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%61, %arg3 : tensor<1x32x4096x4096xbf16>, tensor<1x32x1x1xbf16>) outs(%60 : tensor<1x32x4096x4096xbf16>) {
    ^bb0(%in: bf16, %in_34: bf16, %out: bf16):
      %85 = arith.addf %in, %in_34 : bf16
      linalg.yield %85 : bf16
    } -> tensor<1x32x4096x4096xbf16>
    %63 = tensor.empty() : tensor<1x32x4096x4096xf32>
    %64 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%62 : tensor<1x32x4096x4096xbf16>) outs(%63 : tensor<1x32x4096x4096xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %85 = arith.extf %in : bf16 to f32
      linalg.yield %85 : f32
    } -> tensor<1x32x4096x4096xf32>
    %65 = tensor.empty() : tensor<1x32x4096xi64>
    %66 = linalg.fill ins(%c0_i64 : i64) outs(%65 : tensor<1x32x4096xi64>) -> tensor<1x32x4096xi64>
    %67 = tensor.empty() : tensor<1x32x4096xf32>
    %68 = linalg.fill ins(%cst_1 : f32) outs(%67 : tensor<1x32x4096xf32>) -> tensor<1x32x4096xf32>
    %69:2 = linalg.generic {indexing_maps = [#map2, #map10, #map10], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%64 : tensor<1x32x4096x4096xf32>) outs(%68, %66 : tensor<1x32x4096xf32>, tensor<1x32x4096xi64>) {
    ^bb0(%in: f32, %out: f32, %out_34: i64):
      %85 = linalg.index 3 : index
      %86 = arith.index_cast %85 : index to i64
      %87 = arith.maximumf %in, %out : f32
      %88 = arith.cmpf ogt, %in, %out : f32
      %89 = arith.select %88, %86, %out_34 : i64
      linalg.yield %87, %89 : f32, i64
    } -> (tensor<1x32x4096xf32>, tensor<1x32x4096xi64>)
    %expanded_25 = tensor.expand_shape %69#0 [[0], [1], [2, 3]] output_shape [1, 32, 4096, 1] : tensor<1x32x4096xf32> into tensor<1x32x4096x1xf32>
    %70 = linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%64, %expanded_25 : tensor<1x32x4096x4096xf32>, tensor<1x32x4096x1xf32>) outs(%63 : tensor<1x32x4096x4096xf32>) {
    ^bb0(%in: f32, %in_34: f32, %out: f32):
      %85 = arith.subf %in, %in_34 : f32
      linalg.yield %85 : f32
    } -> tensor<1x32x4096x4096xf32>
    %71 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%70 : tensor<1x32x4096x4096xf32>) outs(%63 : tensor<1x32x4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %85 = math.exp %in : f32
      linalg.yield %85 : f32
    } -> tensor<1x32x4096x4096xf32>
    %72 = tensor.empty() : tensor<1x32x4096x1xf32>
    %73 = linalg.fill ins(%cst_0 : f32) outs(%72 : tensor<1x32x4096x1xf32>) -> tensor<1x32x4096x1xf32>
    %74 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%71 : tensor<1x32x4096x4096xf32>) outs(%73 : tensor<1x32x4096x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %85 = arith.addf %in, %out : f32
      linalg.yield %85 : f32
    } -> tensor<1x32x4096x1xf32>
    %75 = linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%71, %74 : tensor<1x32x4096x4096xf32>, tensor<1x32x4096x1xf32>) outs(%63 : tensor<1x32x4096x4096xf32>) {
    ^bb0(%in: f32, %in_34: f32, %out: f32):
      %85 = arith.divf %in, %in_34 : f32
      linalg.yield %85 : f32
    } -> tensor<1x32x4096x4096xf32>
    %76 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%75 : tensor<1x32x4096x4096xf32>) outs(%60 : tensor<1x32x4096x4096xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %85 = arith.truncf %in : f32 to bf16
      linalg.yield %85 : bf16
    } -> tensor<1x32x4096x4096xbf16>
    %collapsed_26 = tensor.collapse_shape %76 [[0, 1], [2], [3]] : tensor<1x32x4096x4096xbf16> into tensor<32x4096x4096xbf16>
    %77 = linalg.generic {indexing_maps = [#map8, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%collapsed_26 : tensor<32x4096x4096xbf16>) outs(%60 : tensor<1x32x4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x4096x4096xbf16>
    %collapsed_27 = tensor.collapse_shape %53 [[0, 1, 2], [3], [4]] : tensor<1x8x4x4096x128xbf16> into tensor<32x4096x128xbf16>
    %78 = linalg.generic {indexing_maps = [#map8, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%collapsed_27 : tensor<32x4096x128xbf16>) outs(%19 : tensor<1x32x4096x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x32x4096x128xbf16>
    %collapsed_28 = tensor.collapse_shape %77 [[0, 1], [2], [3]] : tensor<1x32x4096x4096xbf16> into tensor<32x4096x4096xbf16>
    %collapsed_29 = tensor.collapse_shape %78 [[0, 1], [2], [3]] : tensor<1x32x4096x128xbf16> into tensor<32x4096x128xbf16>
    %79 = tensor.empty() : tensor<32x4096x128xbf16>
    %80 = linalg.fill ins(%cst : bf16) outs(%79 : tensor<32x4096x128xbf16>) -> tensor<32x4096x128xbf16>
    %81 = linalg.batch_matmul ins(%collapsed_28, %collapsed_29 : tensor<32x4096x4096xbf16>, tensor<32x4096x128xbf16>) outs(%80 : tensor<32x4096x128xbf16>) -> tensor<32x4096x128xbf16>
    %expanded_30 = tensor.expand_shape %81 [[0, 1], [2], [3]] output_shape [1, 32, 4096, 128] : tensor<32x4096x128xbf16> into tensor<1x32x4096x128xbf16>
    %transposed_31 = linalg.transpose ins(%expanded_30 : tensor<1x32x4096x128xbf16>) outs(%16 : tensor<1x4096x32x128xbf16>) permutation = [0, 2, 1, 3] 
    %transposed_32 = linalg.transpose ins(%arg7 : tensor<4096x4096xbf16>) outs(%0 : tensor<4096x4096xbf16>) permutation = [1, 0] 
    %collapsed_33 = tensor.collapse_shape %transposed_31 [[0, 1], [2, 3]] : tensor<1x4096x32x128xbf16> into tensor<4096x4096xbf16>
    %82 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapsed_33 : tensor<4096x4096xbf16>) outs(%1 : tensor<1x4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x4096x4096xbf16>
    %83 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_32 : tensor<4096x4096xbf16>) outs(%1 : tensor<1x4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1x4096x4096xbf16>
    %84 = linalg.batch_matmul ins(%82, %83 : tensor<1x4096x4096xbf16>, tensor<1x4096x4096xbf16>) outs(%4 : tensor<1x4096x4096xbf16>) -> tensor<1x4096x4096xbf16>
    return %84 : tensor<1x4096x4096xbf16>
  }
}
