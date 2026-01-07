#map = affine_map<(d0, d1) -> (d1, d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map4 = affine_map<(d0, d1, d2) -> ()>
#map5 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map8 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#map9 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map10 = affine_map<(d0, d1, d2, d3) -> (0, 0, 0, d3)>
#map11 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>
#map12 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map13 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#map14 = affine_map<(d0, d1, d2, d3) -> (0, d1, 0, 0)>
#map15 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>
#map16 = affine_map<(d0, d1, d2, d3) -> ()>
module {
  func.func @main(%arg0: tensor<8x1024x4096xbf16>, %arg1: tensor<1x1x128xbf16>, %arg2: tensor<1x1x128xbf16>, %arg3: tensor<1x32x1x1xbf16>, %arg4: tensor<4096x4096xbf16>, %arg5: tensor<1024x4096xbf16>, %arg6: tensor<1024x4096xbf16>, %arg7: tensor<4096x4096xbf16>) -> (tensor<8x1024x4096xbf16>, tensor<8x32x1024x1024xbf16>) {
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant 0.000000e+00 : bf16
    %cst_0 = arith.constant 0xFF800000 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %cst_2 = arith.constant 0.088388347648318447 : f64
    %0 = tensor.empty() : tensor<4096x4096xbf16>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg4 : tensor<4096x4096xbf16>) outs(%0 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %2 = tensor.empty() : tensor<8x4096x4096xbf16>
    %3 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<4096x4096xbf16>) outs(%2 : tensor<8x4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x4096x4096xbf16>
    %4 = tensor.empty() : tensor<8x1024x4096xbf16>
    %5 = linalg.generic {indexing_maps = [#map4, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst : bf16) outs(%4 : tensor<8x1024x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x1024x4096xbf16>
    %6 = linalg.generic {indexing_maps = [#map5, #map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg0, %3 : tensor<8x1024x4096xbf16>, tensor<8x4096x4096xbf16>) outs(%5 : tensor<8x1024x4096xbf16>) {
    ^bb0(%in: bf16, %in_19: bf16, %out: bf16):
      %65 = arith.mulf %in, %in_19 : bf16
      %66 = arith.addf %out, %65 : bf16
      linalg.yield %66 : bf16
    } -> tensor<8x1024x4096xbf16>
    %expanded = tensor.expand_shape %6 [[0], [1], [2, 3]] output_shape [8, 1024, 32, 128] : tensor<8x1024x4096xbf16> into tensor<8x1024x32x128xbf16>
    %7 = tensor.empty() : tensor<8x32x1024x128xbf16>
    %8 = linalg.generic {indexing_maps = [#map8, #map9], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded : tensor<8x1024x32x128xbf16>) outs(%7 : tensor<8x32x1024x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x32x1024x128xbf16>
    %9 = tensor.empty() : tensor<4096x1024xbf16>
    %10 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg5 : tensor<1024x4096xbf16>) outs(%9 : tensor<4096x1024xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x1024xbf16>
    %11 = tensor.empty() : tensor<8x4096x1024xbf16>
    %12 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%10 : tensor<4096x1024xbf16>) outs(%11 : tensor<8x4096x1024xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x4096x1024xbf16>
    %13 = tensor.empty() : tensor<8x1024x1024xbf16>
    %14 = linalg.generic {indexing_maps = [#map4, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst : bf16) outs(%13 : tensor<8x1024x1024xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x1024x1024xbf16>
    %15 = linalg.generic {indexing_maps = [#map5, #map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg0, %12 : tensor<8x1024x4096xbf16>, tensor<8x4096x1024xbf16>) outs(%14 : tensor<8x1024x1024xbf16>) {
    ^bb0(%in: bf16, %in_19: bf16, %out: bf16):
      %65 = arith.mulf %in, %in_19 : bf16
      %66 = arith.addf %out, %65 : bf16
      linalg.yield %66 : bf16
    } -> tensor<8x1024x1024xbf16>
    %expanded_3 = tensor.expand_shape %15 [[0], [1], [2, 3]] output_shape [8, 1024, 8, 128] : tensor<8x1024x1024xbf16> into tensor<8x1024x8x128xbf16>
    %16 = tensor.empty() : tensor<8x8x1024x128xbf16>
    %17 = linalg.generic {indexing_maps = [#map8, #map9], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_3 : tensor<8x1024x8x128xbf16>) outs(%16 : tensor<8x8x1024x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x8x1024x128xbf16>
    %18 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg6 : tensor<1024x4096xbf16>) outs(%9 : tensor<4096x1024xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x1024xbf16>
    %19 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%18 : tensor<4096x1024xbf16>) outs(%11 : tensor<8x4096x1024xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x4096x1024xbf16>
    %20 = linalg.generic {indexing_maps = [#map5, #map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg0, %19 : tensor<8x1024x4096xbf16>, tensor<8x4096x1024xbf16>) outs(%14 : tensor<8x1024x1024xbf16>) {
    ^bb0(%in: bf16, %in_19: bf16, %out: bf16):
      %65 = arith.mulf %in, %in_19 : bf16
      %66 = arith.addf %out, %65 : bf16
      linalg.yield %66 : bf16
    } -> tensor<8x1024x1024xbf16>
    %expanded_4 = tensor.expand_shape %20 [[0], [1], [2, 3]] output_shape [8, 1024, 8, 128] : tensor<8x1024x1024xbf16> into tensor<8x1024x8x128xbf16>
    %21 = linalg.generic {indexing_maps = [#map8, #map9], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_4 : tensor<8x1024x8x128xbf16>) outs(%16 : tensor<8x8x1024x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x8x1024x128xbf16>
    %expanded_5 = tensor.expand_shape %arg1 [[0], [1, 2], [3]] output_shape [1, 1, 1, 128] : tensor<1x1x128xbf16> into tensor<1x1x1x128xbf16>
    %expanded_6 = tensor.expand_shape %arg2 [[0], [1, 2], [3]] output_shape [1, 1, 1, 128] : tensor<1x1x128xbf16> into tensor<1x1x1x128xbf16>
    %22 = linalg.generic {indexing_maps = [#map9, #map10, #map9], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%8, %expanded_5 : tensor<8x32x1024x128xbf16>, tensor<1x1x1x128xbf16>) outs(%7 : tensor<8x32x1024x128xbf16>) {
    ^bb0(%in: bf16, %in_19: bf16, %out: bf16):
      %65 = arith.mulf %in, %in_19 : bf16
      linalg.yield %65 : bf16
    } -> tensor<8x32x1024x128xbf16>
    %extracted_slice = tensor.extract_slice %8[0, 0, 0, 0] [8, 32, 1024, 64] [1, 1, 1, 1] : tensor<8x32x1024x128xbf16> to tensor<8x32x1024x64xbf16>
    %extracted_slice_7 = tensor.extract_slice %8[0, 0, 0, 64] [8, 32, 1024, 64] [1, 1, 1, 1] : tensor<8x32x1024x128xbf16> to tensor<8x32x1024x64xbf16>
    %23 = tensor.empty() : tensor<8x32x1024x64xbf16>
    %24 = linalg.generic {indexing_maps = [#map9, #map9], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_7 : tensor<8x32x1024x64xbf16>) outs(%23 : tensor<8x32x1024x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %65 = arith.negf %in : bf16
      linalg.yield %65 : bf16
    } -> tensor<8x32x1024x64xbf16>
    %concat = tensor.concat dim(3) %24, %extracted_slice : (tensor<8x32x1024x64xbf16>, tensor<8x32x1024x64xbf16>) -> tensor<8x32x1024x128xbf16>
    %25 = linalg.generic {indexing_maps = [#map9, #map10, #map9], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat, %expanded_6 : tensor<8x32x1024x128xbf16>, tensor<1x1x1x128xbf16>) outs(%7 : tensor<8x32x1024x128xbf16>) {
    ^bb0(%in: bf16, %in_19: bf16, %out: bf16):
      %65 = arith.mulf %in, %in_19 : bf16
      linalg.yield %65 : bf16
    } -> tensor<8x32x1024x128xbf16>
    %26 = linalg.generic {indexing_maps = [#map9, #map9, #map9], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%22, %25 : tensor<8x32x1024x128xbf16>, tensor<8x32x1024x128xbf16>) outs(%7 : tensor<8x32x1024x128xbf16>) {
    ^bb0(%in: bf16, %in_19: bf16, %out: bf16):
      %65 = arith.addf %in, %in_19 : bf16
      linalg.yield %65 : bf16
    } -> tensor<8x32x1024x128xbf16>
    %27 = linalg.generic {indexing_maps = [#map9, #map10, #map9], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%17, %expanded_5 : tensor<8x8x1024x128xbf16>, tensor<1x1x1x128xbf16>) outs(%16 : tensor<8x8x1024x128xbf16>) {
    ^bb0(%in: bf16, %in_19: bf16, %out: bf16):
      %65 = arith.mulf %in, %in_19 : bf16
      linalg.yield %65 : bf16
    } -> tensor<8x8x1024x128xbf16>
    %extracted_slice_8 = tensor.extract_slice %17[0, 0, 0, 0] [8, 8, 1024, 64] [1, 1, 1, 1] : tensor<8x8x1024x128xbf16> to tensor<8x8x1024x64xbf16>
    %extracted_slice_9 = tensor.extract_slice %17[0, 0, 0, 64] [8, 8, 1024, 64] [1, 1, 1, 1] : tensor<8x8x1024x128xbf16> to tensor<8x8x1024x64xbf16>
    %28 = tensor.empty() : tensor<8x8x1024x64xbf16>
    %29 = linalg.generic {indexing_maps = [#map9, #map9], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_9 : tensor<8x8x1024x64xbf16>) outs(%28 : tensor<8x8x1024x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %65 = arith.negf %in : bf16
      linalg.yield %65 : bf16
    } -> tensor<8x8x1024x64xbf16>
    %concat_10 = tensor.concat dim(3) %29, %extracted_slice_8 : (tensor<8x8x1024x64xbf16>, tensor<8x8x1024x64xbf16>) -> tensor<8x8x1024x128xbf16>
    %30 = linalg.generic {indexing_maps = [#map9, #map10, #map9], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_10, %expanded_6 : tensor<8x8x1024x128xbf16>, tensor<1x1x1x128xbf16>) outs(%16 : tensor<8x8x1024x128xbf16>) {
    ^bb0(%in: bf16, %in_19: bf16, %out: bf16):
      %65 = arith.mulf %in, %in_19 : bf16
      linalg.yield %65 : bf16
    } -> tensor<8x8x1024x128xbf16>
    %31 = linalg.generic {indexing_maps = [#map9, #map9, #map9], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%27, %30 : tensor<8x8x1024x128xbf16>, tensor<8x8x1024x128xbf16>) outs(%16 : tensor<8x8x1024x128xbf16>) {
    ^bb0(%in: bf16, %in_19: bf16, %out: bf16):
      %65 = arith.addf %in, %in_19 : bf16
      linalg.yield %65 : bf16
    } -> tensor<8x8x1024x128xbf16>
    %32 = tensor.empty() : tensor<8x8x4x1024x128xbf16>
    %33 = linalg.generic {indexing_maps = [#map11, #map12], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%31 : tensor<8x8x1024x128xbf16>) outs(%32 : tensor<8x8x4x1024x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x8x4x1024x128xbf16>
    %collapsed = tensor.collapse_shape %33 [[0], [1, 2], [3], [4]] : tensor<8x8x4x1024x128xbf16> into tensor<8x32x1024x128xbf16>
    %34 = linalg.generic {indexing_maps = [#map11, #map12], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%21 : tensor<8x8x1024x128xbf16>) outs(%32 : tensor<8x8x4x1024x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x8x4x1024x128xbf16>
    %35 = tensor.empty() : tensor<8x32x128x1024xbf16>
    %36 = linalg.generic {indexing_maps = [#map13, #map9], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%collapsed : tensor<8x32x1024x128xbf16>) outs(%35 : tensor<8x32x128x1024xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x32x128x1024xbf16>
    %collapsed_11 = tensor.collapse_shape %26 [[0, 1], [2], [3]] : tensor<8x32x1024x128xbf16> into tensor<256x1024x128xbf16>
    %collapsed_12 = tensor.collapse_shape %36 [[0, 1], [2], [3]] : tensor<8x32x128x1024xbf16> into tensor<256x128x1024xbf16>
    %37 = tensor.empty() : tensor<256x1024x1024xbf16>
    %38 = linalg.generic {indexing_maps = [#map4, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst : bf16) outs(%37 : tensor<256x1024x1024xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<256x1024x1024xbf16>
    %39 = linalg.generic {indexing_maps = [#map5, #map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%collapsed_11, %collapsed_12 : tensor<256x1024x128xbf16>, tensor<256x128x1024xbf16>) outs(%38 : tensor<256x1024x1024xbf16>) {
    ^bb0(%in: bf16, %in_19: bf16, %out: bf16):
      %65 = arith.mulf %in, %in_19 : bf16
      %66 = arith.addf %out, %65 : bf16
      linalg.yield %66 : bf16
    } -> tensor<256x1024x1024xbf16>
    %expanded_13 = tensor.expand_shape %39 [[0, 1], [2], [3]] output_shape [8, 32, 1024, 1024] : tensor<256x1024x1024xbf16> into tensor<8x32x1024x1024xbf16>
    %40 = tensor.empty() : tensor<8x32x1024x1024xbf16>
    %41 = linalg.generic {indexing_maps = [#map9, #map9], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_13 : tensor<8x32x1024x1024xbf16>) outs(%40 : tensor<8x32x1024x1024xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %65 = arith.truncf %cst_2 : f64 to bf16
      %66 = arith.mulf %in, %65 : bf16
      linalg.yield %66 : bf16
    } -> tensor<8x32x1024x1024xbf16>
    %42 = linalg.generic {indexing_maps = [#map9, #map14, #map9], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%41, %arg3 : tensor<8x32x1024x1024xbf16>, tensor<1x32x1x1xbf16>) outs(%40 : tensor<8x32x1024x1024xbf16>) {
    ^bb0(%in: bf16, %in_19: bf16, %out: bf16):
      %65 = arith.addf %in, %in_19 : bf16
      linalg.yield %65 : bf16
    } -> tensor<8x32x1024x1024xbf16>
    %43 = tensor.empty() : tensor<8x32x1024x1024xf32>
    %44 = linalg.generic {indexing_maps = [#map9, #map9], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%42 : tensor<8x32x1024x1024xbf16>) outs(%43 : tensor<8x32x1024x1024xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %65 = arith.extf %in : bf16 to f32
      linalg.yield %65 : f32
    } -> tensor<8x32x1024x1024xf32>
    %45 = tensor.empty() : tensor<8x32x1024xi64>
    %46 = linalg.generic {indexing_maps = [#map4, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%c0_i64 : i64) outs(%45 : tensor<8x32x1024xi64>) {
    ^bb0(%in: i64, %out: i64):
      linalg.yield %in : i64
    } -> tensor<8x32x1024xi64>
    %47 = tensor.empty() : tensor<8x32x1024xf32>
    %48 = linalg.generic {indexing_maps = [#map4, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_0 : f32) outs(%47 : tensor<8x32x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<8x32x1024xf32>
    %49:2 = linalg.generic {indexing_maps = [#map9, #map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%44 : tensor<8x32x1024x1024xf32>) outs(%48, %46 : tensor<8x32x1024xf32>, tensor<8x32x1024xi64>) {
    ^bb0(%in: f32, %out: f32, %out_19: i64):
      %65 = linalg.index 3 : index
      %66 = arith.index_cast %65 : index to i64
      %67 = arith.maximumf %in, %out : f32
      %68 = arith.cmpf ogt, %in, %out : f32
      %69 = arith.select %68, %66, %out_19 : i64
      linalg.yield %67, %69 : f32, i64
    } -> (tensor<8x32x1024xf32>, tensor<8x32x1024xi64>)
    %expanded_14 = tensor.expand_shape %49#0 [[0], [1], [2, 3]] output_shape [8, 32, 1024, 1] : tensor<8x32x1024xf32> into tensor<8x32x1024x1xf32>
    %50 = linalg.generic {indexing_maps = [#map9, #map15, #map9], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%44, %expanded_14 : tensor<8x32x1024x1024xf32>, tensor<8x32x1024x1xf32>) outs(%43 : tensor<8x32x1024x1024xf32>) {
    ^bb0(%in: f32, %in_19: f32, %out: f32):
      %65 = arith.subf %in, %in_19 : f32
      linalg.yield %65 : f32
    } -> tensor<8x32x1024x1024xf32>
    %51 = linalg.generic {indexing_maps = [#map9, #map9], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%50 : tensor<8x32x1024x1024xf32>) outs(%43 : tensor<8x32x1024x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %65 = math.exp %in : f32
      linalg.yield %65 : f32
    } -> tensor<8x32x1024x1024xf32>
    %52 = tensor.empty() : tensor<8x32x1024x1xf32>
    %53 = linalg.generic {indexing_maps = [#map16, #map9], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_1 : f32) outs(%52 : tensor<8x32x1024x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<8x32x1024x1xf32>
    %54 = linalg.generic {indexing_maps = [#map9, #map15], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%51 : tensor<8x32x1024x1024xf32>) outs(%53 : tensor<8x32x1024x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %65 = arith.addf %in, %out : f32
      linalg.yield %65 : f32
    } -> tensor<8x32x1024x1xf32>
    %55 = linalg.generic {indexing_maps = [#map9, #map15, #map9], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%51, %54 : tensor<8x32x1024x1024xf32>, tensor<8x32x1024x1xf32>) outs(%43 : tensor<8x32x1024x1024xf32>) {
    ^bb0(%in: f32, %in_19: f32, %out: f32):
      %65 = arith.divf %in, %in_19 : f32
      linalg.yield %65 : f32
    } -> tensor<8x32x1024x1024xf32>
    %56 = linalg.generic {indexing_maps = [#map9, #map9], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%55 : tensor<8x32x1024x1024xf32>) outs(%40 : tensor<8x32x1024x1024xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %65 = arith.truncf %in : f32 to bf16
      linalg.yield %65 : bf16
    } -> tensor<8x32x1024x1024xbf16>
    %collapsed_15 = tensor.collapse_shape %56 [[0, 1], [2], [3]] : tensor<8x32x1024x1024xbf16> into tensor<256x1024x1024xbf16>
    %collapsed_16 = tensor.collapse_shape %34 [[0, 1, 2], [3], [4]] : tensor<8x8x4x1024x128xbf16> into tensor<256x1024x128xbf16>
    %57 = tensor.empty() : tensor<256x1024x128xbf16>
    %58 = linalg.generic {indexing_maps = [#map4, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst : bf16) outs(%57 : tensor<256x1024x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<256x1024x128xbf16>
    %59 = linalg.generic {indexing_maps = [#map5, #map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%collapsed_15, %collapsed_16 : tensor<256x1024x1024xbf16>, tensor<256x1024x128xbf16>) outs(%58 : tensor<256x1024x128xbf16>) {
    ^bb0(%in: bf16, %in_19: bf16, %out: bf16):
      %65 = arith.mulf %in, %in_19 : bf16
      %66 = arith.addf %out, %65 : bf16
      linalg.yield %66 : bf16
    } -> tensor<256x1024x128xbf16>
    %expanded_17 = tensor.expand_shape %59 [[0, 1], [2], [3]] output_shape [8, 32, 1024, 128] : tensor<256x1024x128xbf16> into tensor<8x32x1024x128xbf16>
    %60 = tensor.empty() : tensor<8x1024x32x128xbf16>
    %61 = linalg.generic {indexing_maps = [#map8, #map9], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_17 : tensor<8x32x1024x128xbf16>) outs(%60 : tensor<8x1024x32x128xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x1024x32x128xbf16>
    %collapsed_18 = tensor.collapse_shape %61 [[0], [1], [2, 3]] : tensor<8x1024x32x128xbf16> into tensor<8x1024x4096xbf16>
    %62 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg7 : tensor<4096x4096xbf16>) outs(%0 : tensor<4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    %63 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%62 : tensor<4096x4096xbf16>) outs(%2 : tensor<8x4096x4096xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x4096x4096xbf16>
    %64 = linalg.generic {indexing_maps = [#map5, #map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%collapsed_18, %63 : tensor<8x1024x4096xbf16>, tensor<8x4096x4096xbf16>) outs(%5 : tensor<8x1024x4096xbf16>) {
    ^bb0(%in: bf16, %in_19: bf16, %out: bf16):
      %65 = arith.mulf %in, %in_19 : bf16
      %66 = arith.addf %out, %65 : bf16
      linalg.yield %66 : bf16
    } -> tensor<8x1024x4096xbf16>
    return %64, %56 : tensor<8x1024x4096xbf16>, tensor<8x32x1024x1024xbf16>
  }
}
