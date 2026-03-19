#map = affine_map<(d0, d1, d2) -> ()>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d3, d2)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map7 = affine_map<(d0, d1, d2, d3) -> (0, 0, d2, d3)>
#map8 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>
#map9 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map10 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#map11 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map12 = affine_map<(d0, d1, d2, d3) -> (0, d1, 0, 0)>
#map13 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>
#map14 = affine_map<(d0, d1, d2, d3) -> ()>
module {
  func.func @main(%arg0: tensor<8x1024x4096xbf16>, %arg1: tensor<1x1024x128xbf16>, %arg2: tensor<1x1024x128xbf16>, %arg3: tensor<1x32x1x1xbf16>, %arg4: tensor<4096x4096xbf16>, %arg5: tensor<4096x1024xbf16>, %arg6: tensor<4096x1024xbf16>, %arg7: tensor<4096x4096xbf16>) -> (tensor<8x1024x4096xbf16>, tensor<8x32x1024x1024xbf16>) {
    %cst = arith.constant 0.000000e+00 : bf16
    %cst_0 = arith.constant 0xFF800000 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %cst_2 = arith.constant 0.088388347648318447 : f64
    %0 = tensor.empty() : tensor<8x1024x4096xbf16>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst : bf16) outs(%0 : tensor<8x1024x4096xbf16>) attrs =  {accelgen.const_broadcast = true, accelgen.memory_transformation = true} {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x1024x4096xbf16>
    %2 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg0, %arg4 : tensor<8x1024x4096xbf16>, tensor<4096x4096xbf16>) outs(%1 : tensor<8x1024x4096xbf16>) {
    ^bb0(%in: bf16, %in_19: bf16, %out: bf16):
      %51 = arith.mulf %in, %in_19 : bf16
      %52 = arith.addf %out, %51 : bf16
      linalg.yield %52 : bf16
    } -> tensor<8x1024x4096xbf16>
    %expanded = tensor.expand_shape %2 [[0], [1], [2, 3]] output_shape [8, 1024, 32, 128] : tensor<8x1024x4096xbf16> into tensor<8x1024x32x128xbf16>
    %3 = tensor.empty() : tensor<8x32x1024x128xbf16>
    %4 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded : tensor<8x1024x32x128xbf16>) outs(%3 : tensor<8x32x1024x128xbf16>) attrs =  {accelgen.memory_transformation = true, accelgen.transpose = true} {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x32x1024x128xbf16>
    %5 = tensor.empty() : tensor<8x1024x1024xbf16>
    %6 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst : bf16) outs(%5 : tensor<8x1024x1024xbf16>) attrs =  {accelgen.const_broadcast = true, accelgen.memory_transformation = true} {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x1024x1024xbf16>
    %7 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg0, %arg5 : tensor<8x1024x4096xbf16>, tensor<4096x1024xbf16>) outs(%6 : tensor<8x1024x1024xbf16>) {
    ^bb0(%in: bf16, %in_19: bf16, %out: bf16):
      %51 = arith.mulf %in, %in_19 : bf16
      %52 = arith.addf %out, %51 : bf16
      linalg.yield %52 : bf16
    } -> tensor<8x1024x1024xbf16>
    %expanded_3 = tensor.expand_shape %7 [[0], [1], [2, 3]] output_shape [8, 1024, 8, 128] : tensor<8x1024x1024xbf16> into tensor<8x1024x8x128xbf16>
    %8 = tensor.empty() : tensor<8x8x1024x128xbf16>
    %9 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_3 : tensor<8x1024x8x128xbf16>) outs(%8 : tensor<8x8x1024x128xbf16>) attrs =  {accelgen.memory_transformation = true, accelgen.transpose = true} {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x8x1024x128xbf16>
    %10 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg0, %arg6 : tensor<8x1024x4096xbf16>, tensor<4096x1024xbf16>) outs(%6 : tensor<8x1024x1024xbf16>) {
    ^bb0(%in: bf16, %in_19: bf16, %out: bf16):
      %51 = arith.mulf %in, %in_19 : bf16
      %52 = arith.addf %out, %51 : bf16
      linalg.yield %52 : bf16
    } -> tensor<8x1024x1024xbf16>
    %expanded_4 = tensor.expand_shape %10 [[0], [1], [2, 3]] output_shape [8, 1024, 8, 128] : tensor<8x1024x1024xbf16> into tensor<8x1024x8x128xbf16>
    %11 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_4 : tensor<8x1024x8x128xbf16>) outs(%8 : tensor<8x8x1024x128xbf16>) attrs =  {accelgen.memory_transformation = true, accelgen.transpose = true} {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x8x1024x128xbf16>
    %expanded_5 = tensor.expand_shape %arg1 [[0], [1, 2], [3]] output_shape [1, 1, 1024, 128] : tensor<1x1024x128xbf16> into tensor<1x1x1024x128xbf16>
    %expanded_6 = tensor.expand_shape %arg2 [[0], [1, 2], [3]] output_shape [1, 1, 1024, 128] : tensor<1x1024x128xbf16> into tensor<1x1x1024x128xbf16>
    %12 = linalg.generic {indexing_maps = [#map6, #map7, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4, %expanded_5 : tensor<8x32x1024x128xbf16>, tensor<1x1x1024x128xbf16>) outs(%3 : tensor<8x32x1024x128xbf16>) {
    ^bb0(%in: bf16, %in_19: bf16, %out: bf16):
      %51 = arith.mulf %in, %in_19 : bf16
      linalg.yield %51 : bf16
    } -> tensor<8x32x1024x128xbf16>
    %extracted_slice = tensor.extract_slice %4[0, 0, 0, 0] [8, 32, 1024, 64] [1, 1, 1, 1] : tensor<8x32x1024x128xbf16> to tensor<8x32x1024x64xbf16>
    %extracted_slice_7 = tensor.extract_slice %4[0, 0, 0, 64] [8, 32, 1024, 64] [1, 1, 1, 1] : tensor<8x32x1024x128xbf16> to tensor<8x32x1024x64xbf16>
    %13 = tensor.empty() : tensor<8x32x1024x64xbf16>
    %14 = linalg.generic {indexing_maps = [#map6, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_7 : tensor<8x32x1024x64xbf16>) outs(%13 : tensor<8x32x1024x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %51 = arith.negf %in : bf16
      linalg.yield %51 : bf16
    } -> tensor<8x32x1024x64xbf16>
    %concat = tensor.concat dim(3) %14, %extracted_slice : (tensor<8x32x1024x64xbf16>, tensor<8x32x1024x64xbf16>) -> tensor<8x32x1024x128xbf16>
    %15 = linalg.generic {indexing_maps = [#map6, #map7, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat, %expanded_6 : tensor<8x32x1024x128xbf16>, tensor<1x1x1024x128xbf16>) outs(%3 : tensor<8x32x1024x128xbf16>) {
    ^bb0(%in: bf16, %in_19: bf16, %out: bf16):
      %51 = arith.mulf %in, %in_19 : bf16
      linalg.yield %51 : bf16
    } -> tensor<8x32x1024x128xbf16>
    %16 = linalg.generic {indexing_maps = [#map6, #map6, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%12, %15 : tensor<8x32x1024x128xbf16>, tensor<8x32x1024x128xbf16>) outs(%3 : tensor<8x32x1024x128xbf16>) {
    ^bb0(%in: bf16, %in_19: bf16, %out: bf16):
      %51 = arith.addf %in, %in_19 : bf16
      linalg.yield %51 : bf16
    } -> tensor<8x32x1024x128xbf16>
    %17 = linalg.generic {indexing_maps = [#map6, #map7, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%9, %expanded_5 : tensor<8x8x1024x128xbf16>, tensor<1x1x1024x128xbf16>) outs(%8 : tensor<8x8x1024x128xbf16>) {
    ^bb0(%in: bf16, %in_19: bf16, %out: bf16):
      %51 = arith.mulf %in, %in_19 : bf16
      linalg.yield %51 : bf16
    } -> tensor<8x8x1024x128xbf16>
    %extracted_slice_8 = tensor.extract_slice %9[0, 0, 0, 0] [8, 8, 1024, 64] [1, 1, 1, 1] : tensor<8x8x1024x128xbf16> to tensor<8x8x1024x64xbf16>
    %extracted_slice_9 = tensor.extract_slice %9[0, 0, 0, 64] [8, 8, 1024, 64] [1, 1, 1, 1] : tensor<8x8x1024x128xbf16> to tensor<8x8x1024x64xbf16>
    %18 = tensor.empty() : tensor<8x8x1024x64xbf16>
    %19 = linalg.generic {indexing_maps = [#map6, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_9 : tensor<8x8x1024x64xbf16>) outs(%18 : tensor<8x8x1024x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %51 = arith.negf %in : bf16
      linalg.yield %51 : bf16
    } -> tensor<8x8x1024x64xbf16>
    %concat_10 = tensor.concat dim(3) %19, %extracted_slice_8 : (tensor<8x8x1024x64xbf16>, tensor<8x8x1024x64xbf16>) -> tensor<8x8x1024x128xbf16>
    %20 = linalg.generic {indexing_maps = [#map6, #map7, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_10, %expanded_6 : tensor<8x8x1024x128xbf16>, tensor<1x1x1024x128xbf16>) outs(%8 : tensor<8x8x1024x128xbf16>) {
    ^bb0(%in: bf16, %in_19: bf16, %out: bf16):
      %51 = arith.mulf %in, %in_19 : bf16
      linalg.yield %51 : bf16
    } -> tensor<8x8x1024x128xbf16>
    %21 = linalg.generic {indexing_maps = [#map6, #map6, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%17, %20 : tensor<8x8x1024x128xbf16>, tensor<8x8x1024x128xbf16>) outs(%8 : tensor<8x8x1024x128xbf16>) {
    ^bb0(%in: bf16, %in_19: bf16, %out: bf16):
      %51 = arith.addf %in, %in_19 : bf16
      linalg.yield %51 : bf16
    } -> tensor<8x8x1024x128xbf16>
    %22 = tensor.empty() : tensor<8x8x4x1024x128xbf16>
    %23 = linalg.generic {indexing_maps = [#map8, #map9], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%21 : tensor<8x8x1024x128xbf16>) outs(%22 : tensor<8x8x4x1024x128xbf16>) attrs =  {accelgen.expand = true, accelgen.memory_transformation = true} {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x8x4x1024x128xbf16>
    %collapsed = tensor.collapse_shape %23 [[0], [1, 2], [3], [4]] : tensor<8x8x4x1024x128xbf16> into tensor<8x32x1024x128xbf16>
    %24 = linalg.generic {indexing_maps = [#map8, #map9], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%11 : tensor<8x8x1024x128xbf16>) outs(%22 : tensor<8x8x4x1024x128xbf16>) attrs =  {accelgen.expand = true, accelgen.memory_transformation = true} {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x8x4x1024x128xbf16>
    %25 = tensor.empty() : tensor<8x32x128x1024xbf16>
    %26 = linalg.generic {indexing_maps = [#map10, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%collapsed : tensor<8x32x1024x128xbf16>) outs(%25 : tensor<8x32x128x1024xbf16>) attrs =  {accelgen.memory_transformation = true, accelgen.transpose = true} {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x32x128x1024xbf16>
    %collapsed_11 = tensor.collapse_shape %16 [[0, 1], [2], [3]] : tensor<8x32x1024x128xbf16> into tensor<256x1024x128xbf16>
    %collapsed_12 = tensor.collapse_shape %26 [[0, 1], [2], [3]] : tensor<8x32x128x1024xbf16> into tensor<256x128x1024xbf16>
    %27 = tensor.empty() : tensor<256x1024x1024xbf16>
    %28 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst : bf16) outs(%27 : tensor<256x1024x1024xbf16>) attrs =  {accelgen.const_broadcast = true, accelgen.memory_transformation = true} {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<256x1024x1024xbf16>
    %29 = linalg.generic {indexing_maps = [#map2, #map11, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%collapsed_11, %collapsed_12 : tensor<256x1024x128xbf16>, tensor<256x128x1024xbf16>) outs(%28 : tensor<256x1024x1024xbf16>) {
    ^bb0(%in: bf16, %in_19: bf16, %out: bf16):
      %51 = arith.mulf %in, %in_19 : bf16
      %52 = arith.addf %out, %51 : bf16
      linalg.yield %52 : bf16
    } -> tensor<256x1024x1024xbf16>
    %expanded_13 = tensor.expand_shape %29 [[0, 1], [2], [3]] output_shape [8, 32, 1024, 1024] : tensor<256x1024x1024xbf16> into tensor<8x32x1024x1024xbf16>
    %30 = tensor.empty() : tensor<8x32x1024x1024xbf16>
    %31 = linalg.generic {indexing_maps = [#map6, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_13 : tensor<8x32x1024x1024xbf16>) outs(%30 : tensor<8x32x1024x1024xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %51 = arith.truncf %cst_2 : f64 to bf16
      %52 = arith.mulf %in, %51 : bf16
      linalg.yield %52 : bf16
    } -> tensor<8x32x1024x1024xbf16>
    %32 = linalg.generic {indexing_maps = [#map6, #map12, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%31, %arg3 : tensor<8x32x1024x1024xbf16>, tensor<1x32x1x1xbf16>) outs(%30 : tensor<8x32x1024x1024xbf16>) {
    ^bb0(%in: bf16, %in_19: bf16, %out: bf16):
      %51 = arith.addf %in, %in_19 : bf16
      linalg.yield %51 : bf16
    } -> tensor<8x32x1024x1024xbf16>
    %33 = tensor.empty() : tensor<8x32x1024x1024xf32>
    %34 = linalg.generic {indexing_maps = [#map6, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%32 : tensor<8x32x1024x1024xbf16>) outs(%33 : tensor<8x32x1024x1024xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %51 = arith.extf %in : bf16 to f32
      linalg.yield %51 : f32
    } -> tensor<8x32x1024x1024xf32>
    %35 = tensor.empty() : tensor<8x32x1024xf32>
    %36 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_0 : f32) outs(%35 : tensor<8x32x1024xf32>) attrs =  {accelgen.const_broadcast = true, accelgen.memory_transformation = true} {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<8x32x1024xf32>
    %37 = linalg.generic {indexing_maps = [#map6, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%34 : tensor<8x32x1024x1024xf32>) outs(%36 : tensor<8x32x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %51 = arith.maximumf %in, %out : f32
      linalg.yield %51 : f32
    } -> tensor<8x32x1024xf32>
    %expanded_14 = tensor.expand_shape %37 [[0], [1], [2, 3]] output_shape [8, 32, 1024, 1] : tensor<8x32x1024xf32> into tensor<8x32x1024x1xf32>
    %38 = linalg.generic {indexing_maps = [#map6, #map13, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%34, %expanded_14 : tensor<8x32x1024x1024xf32>, tensor<8x32x1024x1xf32>) outs(%33 : tensor<8x32x1024x1024xf32>) {
    ^bb0(%in: f32, %in_19: f32, %out: f32):
      %51 = arith.subf %in, %in_19 : f32
      linalg.yield %51 : f32
    } -> tensor<8x32x1024x1024xf32>
    %39 = linalg.generic {indexing_maps = [#map6, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%38 : tensor<8x32x1024x1024xf32>) outs(%33 : tensor<8x32x1024x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %51 = math.exp %in : f32
      linalg.yield %51 : f32
    } -> tensor<8x32x1024x1024xf32>
    %40 = tensor.empty() : tensor<8x32x1024x1xf32>
    %41 = linalg.generic {indexing_maps = [#map14, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_1 : f32) outs(%40 : tensor<8x32x1024x1xf32>) attrs =  {accelgen.const_broadcast = true, accelgen.memory_transformation = true} {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<8x32x1024x1xf32>
    %42 = linalg.generic {indexing_maps = [#map6, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%39 : tensor<8x32x1024x1024xf32>) outs(%41 : tensor<8x32x1024x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %51 = arith.addf %in, %out : f32
      linalg.yield %51 : f32
    } -> tensor<8x32x1024x1xf32>
    %43 = linalg.generic {indexing_maps = [#map6, #map13, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%39, %42 : tensor<8x32x1024x1024xf32>, tensor<8x32x1024x1xf32>) outs(%33 : tensor<8x32x1024x1024xf32>) {
    ^bb0(%in: f32, %in_19: f32, %out: f32):
      %51 = arith.divf %in, %in_19 : f32
      linalg.yield %51 : f32
    } -> tensor<8x32x1024x1024xf32>
    %44 = linalg.generic {indexing_maps = [#map6, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%43 : tensor<8x32x1024x1024xf32>) outs(%30 : tensor<8x32x1024x1024xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %51 = arith.truncf %in : f32 to bf16
      linalg.yield %51 : bf16
    } -> tensor<8x32x1024x1024xbf16>
    %collapsed_15 = tensor.collapse_shape %44 [[0, 1], [2], [3]] : tensor<8x32x1024x1024xbf16> into tensor<256x1024x1024xbf16>
    %collapsed_16 = tensor.collapse_shape %24 [[0, 1, 2], [3], [4]] : tensor<8x8x4x1024x128xbf16> into tensor<256x1024x128xbf16>
    %45 = tensor.empty() : tensor<256x1024x128xbf16>
    %46 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst : bf16) outs(%45 : tensor<256x1024x128xbf16>) attrs =  {accelgen.const_broadcast = true, accelgen.memory_transformation = true} {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<256x1024x128xbf16>
    %47 = linalg.generic {indexing_maps = [#map2, #map11, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%collapsed_15, %collapsed_16 : tensor<256x1024x1024xbf16>, tensor<256x1024x128xbf16>) outs(%46 : tensor<256x1024x128xbf16>) {
    ^bb0(%in: bf16, %in_19: bf16, %out: bf16):
      %51 = arith.mulf %in, %in_19 : bf16
      %52 = arith.addf %out, %51 : bf16
      linalg.yield %52 : bf16
    } -> tensor<256x1024x128xbf16>
    %expanded_17 = tensor.expand_shape %47 [[0, 1], [2], [3]] output_shape [8, 32, 1024, 128] : tensor<256x1024x128xbf16> into tensor<8x32x1024x128xbf16>
    %48 = tensor.empty() : tensor<8x1024x32x128xbf16>
    %49 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_17 : tensor<8x32x1024x128xbf16>) outs(%48 : tensor<8x1024x32x128xbf16>) attrs =  {accelgen.memory_transformation = true, accelgen.transpose = true} {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x1024x32x128xbf16>
    %collapsed_18 = tensor.collapse_shape %49 [[0], [1], [2, 3]] : tensor<8x1024x32x128xbf16> into tensor<8x1024x4096xbf16>
    %50 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%collapsed_18, %arg7 : tensor<8x1024x4096xbf16>, tensor<4096x4096xbf16>) outs(%1 : tensor<8x1024x4096xbf16>) {
    ^bb0(%in: bf16, %in_19: bf16, %out: bf16):
      %51 = arith.mulf %in, %in_19 : bf16
      %52 = arith.addf %out, %51 : bf16
      linalg.yield %52 : bf16
    } -> tensor<8x1024x4096xbf16>
    return %50, %44 : tensor<8x1024x4096xbf16>, tensor<8x32x1024x1024xbf16>
  }
}

