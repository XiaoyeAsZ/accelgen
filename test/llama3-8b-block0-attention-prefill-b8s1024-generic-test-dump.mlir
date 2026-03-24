#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (0, 0, d2, d3)>
module {
  func.func @Cluster_0(%arg0: tensor<8x32x1024x128xbf16>, %arg1: tensor<1x1x1024x128xbf16>, %arg2: bf16, %arg3: f64, %arg4: f32, %arg5: f32) -> tensor<8x32x1024x128xbf16> attributes {cycles = 7.864320e+05 : f64, externalAccess = 0x4198000000000000 : f64, sramAccess = 0x4188000000000000 : f64} {
    %0 = tensor.empty() : tensor<8x32x1024x128xbf16>
    %1 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : tensor<8x32x1024x128xbf16>, tensor<1x1x1024x128xbf16>) outs(%0 : tensor<8x32x1024x128xbf16>) attrs =  {inner_order = [3 : ui32, 2 : ui32, 0 : ui32, 1 : ui32], loop_bound = [8 : ui32, 32 : ui32, 1024 : ui32, 128 : ui32], outer_order = [3 : ui32, 2 : ui32, 1 : ui32, 0 : ui32], tiling_size = [1 : ui32, 1 : ui32, 16 : ui32, 128 : ui32], unroll_factor = [1 : ui32, 1 : ui32, 16 : ui32, 128 : ui32]} {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %2 = arith.mulf %in, %in_0 : bf16
      linalg.yield %2 : bf16
    } -> tensor<8x32x1024x128xbf16>
    return %1 : tensor<8x32x1024x128xbf16>
  }
  func.func @Cluster_1(%arg0: tensor<8x8x1024x128xbf16>, %arg1: tensor<1x1x1024x128xbf16>, %arg2: bf16, %arg3: f64, %arg4: f32, %arg5: f32, %arg6: tensor<8x8x1024x128xbf16>, %arg7: bf16, %arg8: f64, %arg9: f32, %arg10: f32, %arg11: tensor<8x32x1024x128xbf16>, %arg12: tensor<1x1x1024x128xbf16>, %arg13: bf16, %arg14: f64, %arg15: f32, %arg16: f32) -> (tensor<8x8x1024x128xbf16>, tensor<8x32x1024x128xbf16>) attributes {cycles = 0x4138000000000000 : f64, externalAccess = 0x41A8000000000000 : f64, sramAccess = 0x41A2000000000000 : f64} {
    %0 = tensor.empty() : tensor<8x8x1024x128xbf16>
    %1 = tensor.empty() : tensor<8x8x1024x128xbf16>
    %2 = tensor.empty() : tensor<8x32x1024x128xbf16>
    %3 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg12 : tensor<8x8x1024x128xbf16>, tensor<1x1x1024x128xbf16>) outs(%1 : tensor<8x8x1024x128xbf16>) attrs =  {inner_order = [3 : ui32, 2 : ui32, 0 : ui32, 1 : ui32], loop_bound = [8 : ui32, 8 : ui32, 1024 : ui32, 128 : ui32], outer_order = [3 : ui32, 2 : ui32, 1 : ui32, 0 : ui32], tiling_size = [1 : ui32, 1 : ui32, 16 : ui32, 128 : ui32], unroll_factor = [1 : ui32, 1 : ui32, 16 : ui32, 128 : ui32]} {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %6 = arith.mulf %in, %in_0 : bf16
      linalg.yield %6 : bf16
    } -> tensor<8x8x1024x128xbf16>
    %4 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg11, %arg12 : tensor<8x32x1024x128xbf16>, tensor<1x1x1024x128xbf16>) outs(%2 : tensor<8x32x1024x128xbf16>) attrs =  {inner_order = [3 : ui32, 2 : ui32, 0 : ui32, 1 : ui32], loop_bound = [8 : ui32, 32 : ui32, 1024 : ui32, 128 : ui32], outer_order = [3 : ui32, 2 : ui32, 1 : ui32, 0 : ui32], tiling_size = [1 : ui32, 1 : ui32, 16 : ui32, 128 : ui32], unroll_factor = [1 : ui32, 1 : ui32, 16 : ui32, 128 : ui32]} {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %6 = arith.mulf %in, %in_0 : bf16
      linalg.yield %6 : bf16
    } -> tensor<8x32x1024x128xbf16>
    %5 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3, %arg6 : tensor<8x8x1024x128xbf16>, tensor<8x8x1024x128xbf16>) outs(%1 : tensor<8x8x1024x128xbf16>) attrs =  {inner_order = [3 : ui32, 2 : ui32, 0 : ui32, 1 : ui32], loop_bound = [8 : ui32, 8 : ui32, 1024 : ui32, 128 : ui32], outer_order = [3 : ui32, 2 : ui32, 1 : ui32, 0 : ui32], tiling_size = [1 : ui32, 1 : ui32, 16 : ui32, 128 : ui32], unroll_factor = [1 : ui32, 1 : ui32, 16 : ui32, 128 : ui32]} {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %6 = arith.addf %in, %in_0 : bf16
      linalg.yield %6 : bf16
    } -> tensor<8x8x1024x128xbf16>
    return %5, %4 : tensor<8x8x1024x128xbf16>, tensor<8x32x1024x128xbf16>
  }
}

