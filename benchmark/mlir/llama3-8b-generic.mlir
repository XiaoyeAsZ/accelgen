#map = affine_map<(d0, d1, d2) -> (d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> ()>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map8 = affine_map<(d0, d1, d2, d3) -> (0, 0, 0, d3)>
#map9 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>
#map10 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map11 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#map12 = affine_map<(d0, d1, d2, d3) -> (0, d1, 0, 0)>
#map13 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>
#map14 = affine_map<(d0, d1, d2, d3) -> ()>
module {
  func.func @main(%arg0: tensor<8x1024x4096xbf16>, %arg1: tensor<1x1x128xbf16>, %arg2: tensor<1x1x128xbf16>, %arg3: tensor<1x32x1x1xbf16>, %arg4: tensor<4096x4096xbf16>, %arg5: tensor<4096x1024xbf16>, %arg6: tensor<4096x1024xbf16>, %arg7: tensor<4096x4096xbf16>) -> (tensor<8x1024x4096xbf16>, tensor<8x32x1024x1024xbf16>) {
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant 0.000000e+00 : bf16
    %cst_0 = arith.constant 0xFF800000 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %cst_2 = arith.constant 0.088388347648318447 : f64
    %0 = tensor.empty() : tensor<8x4096x4096xbf16>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg4 : tensor<4096x4096xbf16>) outs(%0 : tensor<8x4096x4096xbf16>) attrs =  {accelgen.expand = true, accelgen.memory_transformation = true} {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x4096x4096xbf16>
    %2 = tensor.empty() : tensor<8x1024x4096xbf16>
    %3 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst : bf16) outs(%2 : tensor<8x1024x4096xbf16>) attrs =  {accelgen.const_broadcast = true, accelgen.memory_transformation = true} {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x1024x4096xbf16>
    %4 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg0, %1 : tensor<8x1024x4096xbf16>, tensor<8x4096x4096xbf16>) outs(%3 : tensor<8x1024x4096xbf16>) {
    ^bb0(%in: bf16, %in_19: bf16, %out: bf16):
      %59 = arith.mulf %in, %in_19 : bf16
      %60 = arith.addf %out, %59 : bf16
      linalg.yield %60 : bf16
    } -> tensor<8x1024x4096xbf16>
    %expanded = tensor.expand_shape %4 [[0], [1], [2, 3]] output_shape [8, 1024, 32, 128] : tensor<8x1024x4096xbf16> into tensor<8x1024x32x128xbf16>
    %5 = tensor.empty() : tensor<8x32x1024x128xbf16>
    %6 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded : tensor<8x1024x32x128xbf16>) outs(%5 : tensor<8x32x1024x128xbf16>) attrs =  {accelgen.memory_transformation = true, accelgen.transpose = true} {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x32x1024x128xbf16>
    %7 = tensor.empty() : tensor<8x4096x1024xbf16>
    %8 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg5 : tensor<4096x1024xbf16>) outs(%7 : tensor<8x4096x1024xbf16>) attrs =  {accelgen.expand = true, accelgen.memory_transformation = true} {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x4096x1024xbf16>
    %9 = tensor.empty() : tensor<8x1024x1024xbf16>
    %10 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst : bf16) outs(%9 : tensor<8x1024x1024xbf16>) attrs =  {accelgen.const_broadcast = true, accelgen.memory_transformation = true} {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x1024x1024xbf16>
    %11 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg0, %8 : tensor<8x1024x4096xbf16>, tensor<8x4096x1024xbf16>) outs(%10 : tensor<8x1024x1024xbf16>) {
    ^bb0(%in: bf16, %in_19: bf16, %out: bf16):
      %59 = arith.mulf %in, %in_19 : bf16
      %60 = arith.addf %out, %59 : bf16
      linalg.yield %60 : bf16
    } -> tensor<8x1024x1024xbf16>
    %expanded_3 = tensor.expand_shape %11 [[0], [1], [2, 3]] output_shape [8, 1024, 8, 128] : tensor<8x1024x1024xbf16> into tensor<8x1024x8x128xbf16>
    %12 = tensor.empty() : tensor<8x8x1024x128xbf16>
    %13 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_3 : tensor<8x1024x8x128xbf16>) outs(%12 : tensor<8x8x1024x128xbf16>) attrs =  {accelgen.memory_transformation = true, accelgen.transpose = true} {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x8x1024x128xbf16>
    %14 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg6 : tensor<4096x1024xbf16>) outs(%7 : tensor<8x4096x1024xbf16>) attrs =  {accelgen.expand = true, accelgen.memory_transformation = true} {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x4096x1024xbf16>
    %15 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg0, %14 : tensor<8x1024x4096xbf16>, tensor<8x4096x1024xbf16>) outs(%10 : tensor<8x1024x1024xbf16>) {
    ^bb0(%in: bf16, %in_19: bf16, %out: bf16):
      %59 = arith.mulf %in, %in_19 : bf16
      %60 = arith.addf %out, %59 : bf16
      linalg.yield %60 : bf16
    } -> tensor<8x1024x1024xbf16>
    %expanded_4 = tensor.expand_shape %15 [[0], [1], [2, 3]] output_shape [8, 1024, 8, 128] : tensor<8x1024x1024xbf16> into tensor<8x1024x8x128xbf16>
    %16 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_4 : tensor<8x1024x8x128xbf16>) outs(%12 : tensor<8x8x1024x128xbf16>) attrs =  {accelgen.memory_transformation = true, accelgen.transpose = true} {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x8x1024x128xbf16>
    %expanded_5 = tensor.expand_shape %arg1 [[0], [1, 2], [3]] output_shape [1, 1, 1, 128] : tensor<1x1x128xbf16> into tensor<1x1x1x128xbf16>
    %expanded_6 = tensor.expand_shape %arg2 [[0], [1, 2], [3]] output_shape [1, 1, 1, 128] : tensor<1x1x128xbf16> into tensor<1x1x1x128xbf16>
    %17 = linalg.generic {indexing_maps = [#map7, #map8, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%6, %expanded_5 : tensor<8x32x1024x128xbf16>, tensor<1x1x1x128xbf16>) outs(%5 : tensor<8x32x1024x128xbf16>) {
    ^bb0(%in: bf16, %in_19: bf16, %out: bf16):
      %59 = arith.mulf %in, %in_19 : bf16
      linalg.yield %59 : bf16
    } -> tensor<8x32x1024x128xbf16>
    %extracted_slice = tensor.extract_slice %6[0, 0, 0, 0] [8, 32, 1024, 64] [1, 1, 1, 1] : tensor<8x32x1024x128xbf16> to tensor<8x32x1024x64xbf16>
    %extracted_slice_7 = tensor.extract_slice %6[0, 0, 0, 64] [8, 32, 1024, 64] [1, 1, 1, 1] : tensor<8x32x1024x128xbf16> to tensor<8x32x1024x64xbf16>
    %18 = tensor.empty() : tensor<8x32x1024x64xbf16>
    %19 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_7 : tensor<8x32x1024x64xbf16>) outs(%18 : tensor<8x32x1024x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %59 = arith.negf %in : bf16
      linalg.yield %59 : bf16
    } -> tensor<8x32x1024x64xbf16>
    %concat = tensor.concat dim(3) %19, %extracted_slice : (tensor<8x32x1024x64xbf16>, tensor<8x32x1024x64xbf16>) -> tensor<8x32x1024x128xbf16>
    %20 = linalg.generic {indexing_maps = [#map7, #map8, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat, %expanded_6 : tensor<8x32x1024x128xbf16>, tensor<1x1x1x128xbf16>) outs(%5 : tensor<8x32x1024x128xbf16>) {
    ^bb0(%in: bf16, %in_19: bf16, %out: bf16):
      %59 = arith.mulf %in, %in_19 : bf16
      linalg.yield %59 : bf16
    } -> tensor<8x32x1024x128xbf16>
    %21 = linalg.generic {indexing_maps = [#map7, #map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%17, %20 : tensor<8x32x1024x128xbf16>, tensor<8x32x1024x128xbf16>) outs(%5 : tensor<8x32x1024x128xbf16>) {
    ^bb0(%in: bf16, %in_19: bf16, %out: bf16):
      %59 = arith.addf %in, %in_19 : bf16
      linalg.yield %59 : bf16
    } -> tensor<8x32x1024x128xbf16>
    %22 = linalg.generic {indexing_maps = [#map7, #map8, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%13, %expanded_5 : tensor<8x8x1024x128xbf16>, tensor<1x1x1x128xbf16>) outs(%12 : tensor<8x8x1024x128xbf16>) {
    ^bb0(%in: bf16, %in_19: bf16, %out: bf16):
      %59 = arith.mulf %in, %in_19 : bf16
      linalg.yield %59 : bf16
    } -> tensor<8x8x1024x128xbf16>
    %extracted_slice_8 = tensor.extract_slice %13[0, 0, 0, 0] [8, 8, 1024, 64] [1, 1, 1, 1] : tensor<8x8x1024x128xbf16> to tensor<8x8x1024x64xbf16>
    %extracted_slice_9 = tensor.extract_slice %13[0, 0, 0, 64] [8, 8, 1024, 64] [1, 1, 1, 1] : tensor<8x8x1024x128xbf16> to tensor<8x8x1024x64xbf16>
    %23 = tensor.empty() : tensor<8x8x1024x64xbf16>
    %24 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_9 : tensor<8x8x1024x64xbf16>) outs(%23 : tensor<8x8x1024x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %59 = arith.negf %in : bf16
      linalg.yield %59 : bf16
    } -> tensor<8x8x1024x64xbf16>
    %concat_10 = tensor.concat dim(3) %24, %extracted_slice_8 : (tensor<8x8x1024x64xbf16>, tensor<8x8x1024x64xbf16>) -> tensor<8x8x1024x128xbf16>
    %25 = linalg.generic {indexing_maps = [#map7, #map8, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat_10, %expanded_6 : tensor<8x8x1024x128xbf16>, tensor<1x1x1x128xbf16>) outs(%12 : tensor<8x8x1024x128xbf16>) {
    ^bb0(%in: bf16, %in_19: bf16, %out: bf16):
      %59 = arith.mulf %in, %in_19 : bf16
      linalg.yield %59 : bf16
    } -> tensor<8x8x1024x128xbf16>
    %26 = linalg.generic {indexing_maps = [#map7, #map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%22, %25 : tensor<8x8x1024x128xbf16>, tensor<8x8x1024x128xbf16>) outs(%12 : tensor<8x8x1024x128xbf16>) {
    ^bb0(%in: bf16, %in_19: bf16, %out: bf16):
      %59 = arith.addf %in, %in_19 : bf16
      linalg.yield %59 : bf16
    } -> tensor<8x8x1024x128xbf16>
    %27 = tensor.empty() : tensor<8x8x4x1024x128xbf16>
    %28 = linalg.generic {indexing_maps = [#map9, #map10], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%26 : tensor<8x8x1024x128xbf16>) outs(%27 : tensor<8x8x4x1024x128xbf16>) attrs =  {accelgen.expand = true, accelgen.memory_transformation = true} {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x8x4x1024x128xbf16>
    %collapsed = tensor.collapse_shape %28 [[0], [1, 2], [3], [4]] : tensor<8x8x4x1024x128xbf16> into tensor<8x32x1024x128xbf16>
    %29 = linalg.generic {indexing_maps = [#map9, #map10], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%16 : tensor<8x8x1024x128xbf16>) outs(%27 : tensor<8x8x4x1024x128xbf16>) attrs =  {accelgen.expand = true, accelgen.memory_transformation = true} {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x8x4x1024x128xbf16>
    %30 = tensor.empty() : tensor<8x32x128x1024xbf16>
    %31 = linalg.generic {indexing_maps = [#map11, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%collapsed : tensor<8x32x1024x128xbf16>) outs(%30 : tensor<8x32x128x1024xbf16>) attrs =  {accelgen.memory_transformation = true, accelgen.transpose = true} {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x32x128x1024xbf16>
    %collapsed_11 = tensor.collapse_shape %21 [[0, 1], [2], [3]] : tensor<8x32x1024x128xbf16> into tensor<256x1024x128xbf16>
    %collapsed_12 = tensor.collapse_shape %31 [[0, 1], [2], [3]] : tensor<8x32x128x1024xbf16> into tensor<256x128x1024xbf16>
    %32 = tensor.empty() : tensor<256x1024x1024xbf16>
    %33 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst : bf16) outs(%32 : tensor<256x1024x1024xbf16>) attrs =  {accelgen.const_broadcast = true, accelgen.memory_transformation = true} {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<256x1024x1024xbf16>
    %34 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%collapsed_11, %collapsed_12 : tensor<256x1024x128xbf16>, tensor<256x128x1024xbf16>) outs(%33 : tensor<256x1024x1024xbf16>) {
    ^bb0(%in: bf16, %in_19: bf16, %out: bf16):
      %59 = arith.mulf %in, %in_19 : bf16
      %60 = arith.addf %out, %59 : bf16
      linalg.yield %60 : bf16
    } -> tensor<256x1024x1024xbf16>
    %expanded_13 = tensor.expand_shape %34 [[0, 1], [2], [3]] output_shape [8, 32, 1024, 1024] : tensor<256x1024x1024xbf16> into tensor<8x32x1024x1024xbf16>
    %35 = tensor.empty() : tensor<8x32x1024x1024xbf16>
    %36 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_13 : tensor<8x32x1024x1024xbf16>) outs(%35 : tensor<8x32x1024x1024xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %59 = arith.truncf %cst_2 : f64 to bf16
      %60 = arith.mulf %in, %59 : bf16
      linalg.yield %60 : bf16
    } -> tensor<8x32x1024x1024xbf16>
    %37 = linalg.generic {indexing_maps = [#map7, #map12, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%36, %arg3 : tensor<8x32x1024x1024xbf16>, tensor<1x32x1x1xbf16>) outs(%35 : tensor<8x32x1024x1024xbf16>) {
    ^bb0(%in: bf16, %in_19: bf16, %out: bf16):
      %59 = arith.addf %in, %in_19 : bf16
      linalg.yield %59 : bf16
    } -> tensor<8x32x1024x1024xbf16>
    %38 = tensor.empty() : tensor<8x32x1024x1024xf32>
    %39 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%37 : tensor<8x32x1024x1024xbf16>) outs(%38 : tensor<8x32x1024x1024xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %59 = arith.extf %in : bf16 to f32
      linalg.yield %59 : f32
    } -> tensor<8x32x1024x1024xf32>
    %40 = tensor.empty() : tensor<8x32x1024xi64>
    %41 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%c0_i64 : i64) outs(%40 : tensor<8x32x1024xi64>) attrs =  {accelgen.const_broadcast = true, accelgen.memory_transformation = true} {
    ^bb0(%in: i64, %out: i64):
      linalg.yield %in : i64
    } -> tensor<8x32x1024xi64>
    %42 = tensor.empty() : tensor<8x32x1024xf32>
    %43 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_0 : f32) outs(%42 : tensor<8x32x1024xf32>) attrs =  {accelgen.const_broadcast = true, accelgen.memory_transformation = true} {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<8x32x1024xf32>
    %44:2 = linalg.generic {indexing_maps = [#map7, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%39 : tensor<8x32x1024x1024xf32>) outs(%43, %41 : tensor<8x32x1024xf32>, tensor<8x32x1024xi64>) {
    ^bb0(%in: f32, %out: f32, %out_19: i64):
      %59 = linalg.index 3 : index
      %60 = arith.index_cast %59 : index to i64
      %61 = arith.maximumf %in, %out : f32
      %62 = arith.cmpf ogt, %in, %out : f32
      %63 = arith.select %62, %60, %out_19 : i64
      linalg.yield %61, %63 : f32, i64
    } -> (tensor<8x32x1024xf32>, tensor<8x32x1024xi64>)
    %expanded_14 = tensor.expand_shape %44#0 [[0], [1], [2, 3]] output_shape [8, 32, 1024, 1] : tensor<8x32x1024xf32> into tensor<8x32x1024x1xf32>
    %45 = linalg.generic {indexing_maps = [#map7, #map13, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%39, %expanded_14 : tensor<8x32x1024x1024xf32>, tensor<8x32x1024x1xf32>) outs(%38 : tensor<8x32x1024x1024xf32>) {
    ^bb0(%in: f32, %in_19: f32, %out: f32):
      %59 = arith.subf %in, %in_19 : f32
      linalg.yield %59 : f32
    } -> tensor<8x32x1024x1024xf32>
    %46 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%45 : tensor<8x32x1024x1024xf32>) outs(%38 : tensor<8x32x1024x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %59 = math.exp %in : f32
      linalg.yield %59 : f32
    } -> tensor<8x32x1024x1024xf32>
    %47 = tensor.empty() : tensor<8x32x1024x1xf32>
    %48 = linalg.generic {indexing_maps = [#map14, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_1 : f32) outs(%47 : tensor<8x32x1024x1xf32>) attrs =  {accelgen.const_broadcast = true, accelgen.memory_transformation = true} {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<8x32x1024x1xf32>
    %49 = linalg.generic {indexing_maps = [#map7, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%46 : tensor<8x32x1024x1024xf32>) outs(%48 : tensor<8x32x1024x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %59 = arith.addf %in, %out : f32
      linalg.yield %59 : f32
    } -> tensor<8x32x1024x1xf32>
    %50 = linalg.generic {indexing_maps = [#map7, #map13, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%46, %49 : tensor<8x32x1024x1024xf32>, tensor<8x32x1024x1xf32>) outs(%38 : tensor<8x32x1024x1024xf32>) {
    ^bb0(%in: f32, %in_19: f32, %out: f32):
      %59 = arith.divf %in, %in_19 : f32
      linalg.yield %59 : f32
    } -> tensor<8x32x1024x1024xf32>
    %51 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%50 : tensor<8x32x1024x1024xf32>) outs(%35 : tensor<8x32x1024x1024xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %59 = arith.truncf %in : f32 to bf16
      linalg.yield %59 : bf16
    } -> tensor<8x32x1024x1024xbf16>
    %collapsed_15 = tensor.collapse_shape %51 [[0, 1], [2], [3]] : tensor<8x32x1024x1024xbf16> into tensor<256x1024x1024xbf16>
    %collapsed_16 = tensor.collapse_shape %29 [[0, 1, 2], [3], [4]] : tensor<8x8x4x1024x128xbf16> into tensor<256x1024x128xbf16>
    %52 = tensor.empty() : tensor<256x1024x128xbf16>
    %53 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst : bf16) outs(%52 : tensor<256x1024x128xbf16>) attrs =  {accelgen.const_broadcast = true, accelgen.memory_transformation = true} {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<256x1024x128xbf16>
    %54 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%collapsed_15, %collapsed_16 : tensor<256x1024x1024xbf16>, tensor<256x1024x128xbf16>) outs(%53 : tensor<256x1024x128xbf16>) {
    ^bb0(%in: bf16, %in_19: bf16, %out: bf16):
      %59 = arith.mulf %in, %in_19 : bf16
      %60 = arith.addf %out, %59 : bf16
      linalg.yield %60 : bf16
    } -> tensor<256x1024x128xbf16>
    %expanded_17 = tensor.expand_shape %54 [[0, 1], [2], [3]] output_shape [8, 32, 1024, 128] : tensor<256x1024x128xbf16> into tensor<8x32x1024x128xbf16>
    %55 = tensor.empty() : tensor<8x1024x32x128xbf16>
    %56 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_17 : tensor<8x32x1024x128xbf16>) outs(%55 : tensor<8x1024x32x128xbf16>) attrs =  {accelgen.memory_transformation = true, accelgen.transpose = true} {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x1024x32x128xbf16>
    %collapsed_18 = tensor.collapse_shape %56 [[0], [1], [2, 3]] : tensor<8x1024x32x128xbf16> into tensor<8x1024x4096xbf16>
    %57 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg7 : tensor<4096x4096xbf16>) outs(%0 : tensor<8x4096x4096xbf16>) attrs =  {accelgen.expand = true, accelgen.memory_transformation = true} {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x4096x4096xbf16>
    %58 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%collapsed_18, %57 : tensor<8x1024x4096xbf16>, tensor<8x4096x4096xbf16>) outs(%3 : tensor<8x1024x4096xbf16>) {
    ^bb0(%in: bf16, %in_19: bf16, %out: bf16):
      %59 = arith.mulf %in, %in_19 : bf16
      %60 = arith.addf %out, %59 : bf16
      linalg.yield %60 : bf16
    } -> tensor<8x1024x4096xbf16>
    return %58, %51 : tensor<8x1024x4096xbf16>, tensor<8x32x1024x1024xbf16>
  }
}
