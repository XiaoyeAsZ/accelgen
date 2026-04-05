#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>
#map5 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#map8 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>
#map9 = affine_map<(d0, d1, d2, d3) -> (0, d1, 0, 0)>
#map10 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#map11 = affine_map<(d0, d1, d2, d3) -> (0, 0, d2, d3)>
module {
  func.func @Cluster_0(%arg0: tensor<8x1024x4096xbf16>, %arg1: tensor<4096x4096xbf16>, %arg2: f32, %arg3: f64, %arg4: bf16, %arg5: f32) -> tensor<8x1024x4096xbf16> attributes {cycles = 0x4180000000000000 : f64, externalAccess = 0x41BA000000000000 : f64, sramAccess = 0x41E8800000000000 : f64} {
    %0 = tensor.empty() : tensor<8x1024x4096xbf16>
    %1 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<8x1024x4096xbf16>, tensor<4096x4096xbf16>) outs(%0 : tensor<8x1024x4096xbf16>) attrs =  {inner_order = [3 : ui32, 2 : ui32, 0 : ui32, 1 : ui32], loop_bound = [8 : ui32, 1024 : ui32, 4096 : ui32, 4096 : ui32], outer_order = [2 : ui32, 1 : ui32, 3 : ui32, 0 : ui32], tiling_size = [8 : ui32, 128 : ui32, 512 : ui32, 32 : ui32], unroll_factor = [1 : ui32, 1 : ui32, 128 : ui32, 32 : ui32]} {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %2 = arith.mulf %in, %in_0 : bf16
      %3 = arith.addf %out, %2 : bf16
      linalg.yield %3 : bf16
    } -> tensor<8x1024x4096xbf16>
    return %1 : tensor<8x1024x4096xbf16>
  }
  func.func @Cluster_1(%arg0: tensor<8x8x1024x128xbf16>, %arg1: f32, %arg2: f64, %arg3: bf16, %arg4: f32, %arg5: tensor<8x32x1024x1024xf32>, %arg6: f32, %arg7: f64, %arg8: bf16, %arg9: f32, %arg10: f32, %arg11: f64, %arg12: bf16, %arg13: f32, %arg14: f32, %arg15: f64, %arg16: bf16, %arg17: f32) -> tensor<8x1024x32x128xbf16> attributes {cycles = 0x4160000000000000 : f64, externalAccess = 0x41C9000000000000 : f64, sramAccess = 0x41F0400000000000 : f64} {
    %0 = tensor.empty() : tensor<8x8x4x1024x128xbf16>
    %1 = tensor.empty() : tensor<8x32x1024x1024xbf16>
    %2 = tensor.empty() : tensor<256x1024x128xbf16>
    %3 = tensor.empty() : tensor<8x1024x32x128xbf16>
    %4 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg5 : tensor<8x32x1024x1024xf32>) outs(%1 : tensor<8x32x1024x1024xbf16>) attrs =  {inner_order = [3 : ui32, 2 : ui32, 0 : ui32, 1 : ui32], loop_bound = [8 : ui32, 32 : ui32, 1024 : ui32, 1024 : ui32], outer_order = [3 : ui32, 0 : ui32, 1 : ui32, 2 : ui32], tiling_size = [1 : ui32, 2 : ui32, 32 : ui32, 1024 : ui32], unroll_factor = [1 : ui32, 1 : ui32, 4 : ui32, 8 : ui32]} {
    ^bb0(%in: f32, %out: bf16):
      %8 = arith.truncf %in : f32 to bf16
      linalg.yield %8 : bf16
    } -> tensor<8x32x1024x1024xbf16>
    %5 = linalg.generic {indexing_maps = [#map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<8x8x1024x128xbf16>) outs(%0 : tensor<8x8x4x1024x128xbf16>) attrs =  {accelgen.expand = true, accelgen.memory_transformation = true, inner_order = [4 : ui32, 3 : ui32, 0 : ui32, 1 : ui32, 2 : ui32], loop_bound = [8 : ui32, 8 : ui32, 4 : ui32, 1024 : ui32, 128 : ui32], outer_order = [4 : ui32, 3 : ui32, 0 : ui32, 1 : ui32, 2 : ui32], tiling_size = [1 : ui32, 1 : ui32, 2 : ui32, 1024 : ui32, 128 : ui32], unroll_factor = [1 : ui32, 1 : ui32, 1 : ui32, 32 : ui32, 32 : ui32]} {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x8x4x1024x128xbf16>
    %collapsed = tensor.collapse_shape %4 [[0, 1], [2], [3]] : tensor<8x32x1024x1024xbf16> into tensor<256x1024x1024xbf16>
    %collapsed_0 = tensor.collapse_shape %5 [[0, 1, 2], [3], [4]] : tensor<8x8x4x1024x128xbf16> into tensor<256x1024x128xbf16>
    %6 = linalg.generic {indexing_maps = [#map, #map6, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%collapsed, %collapsed_0 : tensor<256x1024x1024xbf16>, tensor<256x1024x128xbf16>) outs(%2 : tensor<256x1024x128xbf16>) attrs =  {inner_order = [3 : ui32, 2 : ui32, 0 : ui32, 1 : ui32], loop_bound = [256 : ui32, 1024 : ui32, 128 : ui32, 1024 : ui32], outer_order = [2 : ui32, 3 : ui32, 0 : ui32, 1 : ui32], tiling_size = [2 : ui32, 32 : ui32, 128 : ui32, 1024 : ui32], unroll_factor = [1 : ui32, 1 : ui32, 64 : ui32, 64 : ui32]} {
    ^bb0(%in: bf16, %in_1: bf16, %out: bf16):
      %8 = arith.mulf %in, %in_1 : bf16
      %9 = arith.addf %out, %8 : bf16
      linalg.yield %9 : bf16
    } -> tensor<256x1024x128xbf16>
    %expanded = tensor.expand_shape %6 [[0, 1], [2], [3]] output_shape [8, 32, 1024, 128] : tensor<256x1024x128xbf16> into tensor<8x32x1024x128xbf16>
    %7 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded : tensor<8x32x1024x128xbf16>) outs(%3 : tensor<8x1024x32x128xbf16>) attrs =  {accelgen.memory_transformation = true, accelgen.transpose = true, inner_order = [3 : ui32, 1 : ui32, 0 : ui32, 2 : ui32], loop_bound = [8 : ui32, 1024 : ui32, 32 : ui32, 128 : ui32], outer_order = [3 : ui32, 0 : ui32, 2 : ui32, 1 : ui32], tiling_size = [1 : ui32, 32 : ui32, 2 : ui32, 128 : ui32], unroll_factor = [1 : ui32, 32 : ui32, 1 : ui32, 32 : ui32]} {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x1024x32x128xbf16>
    return %7 : tensor<8x1024x32x128xbf16>
  }
  func.func @Cluster_2(%arg0: tensor<8x1024x4096xbf16>, %arg1: tensor<4096x1024xbf16>, %arg2: f32, %arg3: f64, %arg4: bf16, %arg5: f32, %arg6: f32, %arg7: f64, %arg8: bf16, %arg9: f32) -> tensor<8x8x1024x128xbf16> attributes {cycles = 0x4160000000000000 : f64, externalAccess = 0x41A1000000000000 : f64, sramAccess = 0x41C9400000000000 : f64} {
    %0 = tensor.empty() : tensor<8x1024x1024xbf16>
    %1 = tensor.empty() : tensor<8x8x1024x128xbf16>
    %2 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<8x1024x4096xbf16>, tensor<4096x1024xbf16>) outs(%0 : tensor<8x1024x1024xbf16>) attrs =  {inner_order = [3 : ui32, 2 : ui32, 0 : ui32, 1 : ui32], loop_bound = [8 : ui32, 1024 : ui32, 1024 : ui32, 4096 : ui32], outer_order = [2 : ui32, 1 : ui32, 3 : ui32, 0 : ui32], tiling_size = [8 : ui32, 64 : ui32, 512 : ui32, 32 : ui32], unroll_factor = [1 : ui32, 1 : ui32, 128 : ui32, 32 : ui32]} {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %4 = arith.mulf %in, %in_0 : bf16
      %5 = arith.addf %out, %4 : bf16
      linalg.yield %5 : bf16
    } -> tensor<8x1024x1024xbf16>
    %expanded = tensor.expand_shape %2 [[0], [1], [2, 3]] output_shape [8, 1024, 8, 128] : tensor<8x1024x1024xbf16> into tensor<8x1024x8x128xbf16>
    %3 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded : tensor<8x1024x8x128xbf16>) outs(%1 : tensor<8x8x1024x128xbf16>) attrs =  {accelgen.memory_transformation = true, accelgen.transpose = true, inner_order = [3 : ui32, 1 : ui32, 0 : ui32, 2 : ui32], loop_bound = [8 : ui32, 8 : ui32, 1024 : ui32, 128 : ui32], outer_order = [1 : ui32, 3 : ui32, 2 : ui32, 0 : ui32], tiling_size = [8 : ui32, 4 : ui32, 64 : ui32, 128 : ui32], unroll_factor = [1 : ui32, 4 : ui32, 1 : ui32, 128 : ui32]} {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x8x1024x128xbf16>
    return %3 : tensor<8x8x1024x128xbf16>
  }
  func.func @Cluster_3(%arg0: tensor<8x32x1024x1024xf32>, %arg1: tensor<8x32x1024x1xf32>, %arg2: f32, %arg3: f64, %arg4: bf16, %arg5: f32, %arg6: f32, %arg7: f64, %arg8: bf16, %arg9: f32, %arg10: f32, %arg11: f64, %arg12: bf16, %arg13: f32, %arg14: f32, %arg15: f64, %arg16: bf16, %arg17: f32) -> tensor<8x32x1024x1024xf32> attributes {cycles = 0x4158000000000000 : f64, externalAccess = 0x41C8000000000000 : f64, sramAccess = 0x41E0400000000000 : f64} {
    %0 = tensor.empty() : tensor<8x32x1024x1024xf32>
    %1 = tensor.empty() : tensor<8x32x1024x1024xf32>
    %2 = tensor.empty() : tensor<8x32x1024x1xf32>
    %3 = tensor.empty() : tensor<8x32x1024x1024xf32>
    %4 = linalg.generic {indexing_maps = [#map3, #map8, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : tensor<8x32x1024x1024xf32>, tensor<8x32x1024x1xf32>) outs(%3 : tensor<8x32x1024x1024xf32>) attrs =  {inner_order = [3 : ui32, 2 : ui32, 0 : ui32, 1 : ui32], loop_bound = [8 : ui32, 32 : ui32, 1024 : ui32, 1024 : ui32], outer_order = [3 : ui32, 2 : ui32, 1 : ui32, 0 : ui32], tiling_size = [1 : ui32, 1 : ui32, 1 : ui32, 1024 : ui32], unroll_factor = [1 : ui32, 1 : ui32, 1 : ui32, 64 : ui32]} {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %8 = arith.subf %in, %in_0 : f32
      linalg.yield %8 : f32
    } -> tensor<8x32x1024x1024xf32>
    %5 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4 : tensor<8x32x1024x1024xf32>) outs(%3 : tensor<8x32x1024x1024xf32>) attrs =  {inner_order = [3 : ui32, 2 : ui32, 0 : ui32, 1 : ui32], loop_bound = [8 : ui32, 32 : ui32, 1024 : ui32, 1024 : ui32], outer_order = [3 : ui32, 2 : ui32, 1 : ui32, 0 : ui32], tiling_size = [1 : ui32, 1 : ui32, 1 : ui32, 1024 : ui32], unroll_factor = [1 : ui32, 1 : ui32, 1 : ui32, 1024 : ui32]} {
    ^bb0(%in: f32, %out: f32):
      %8 = math.exp %in : f32
      linalg.yield %8 : f32
    } -> tensor<8x32x1024x1024xf32>
    %6 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%5 : tensor<8x32x1024x1024xf32>) outs(%2 : tensor<8x32x1024x1xf32>) attrs =  {inner_order = [3 : ui32, 2 : ui32, 0 : ui32, 1 : ui32], loop_bound = [8 : ui32, 32 : ui32, 1024 : ui32, 1024 : ui32], outer_order = [3 : ui32, 2 : ui32, 1 : ui32, 0 : ui32], tiling_size = [1 : ui32, 1 : ui32, 1 : ui32, 1024 : ui32], unroll_factor = [1 : ui32, 1 : ui32, 1 : ui32, 64 : ui32]} {
    ^bb0(%in: f32, %out: f32):
      %8 = arith.addf %in, %out : f32
      linalg.yield %8 : f32
    } -> tensor<8x32x1024x1xf32>
    %7 = linalg.generic {indexing_maps = [#map3, #map8, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%5, %6 : tensor<8x32x1024x1024xf32>, tensor<8x32x1024x1xf32>) outs(%3 : tensor<8x32x1024x1024xf32>) attrs =  {inner_order = [3 : ui32, 2 : ui32, 0 : ui32, 1 : ui32], loop_bound = [8 : ui32, 32 : ui32, 1024 : ui32, 1024 : ui32], outer_order = [3 : ui32, 2 : ui32, 1 : ui32, 0 : ui32], tiling_size = [1 : ui32, 1 : ui32, 1 : ui32, 1024 : ui32], unroll_factor = [1 : ui32, 1 : ui32, 1 : ui32, 64 : ui32]} {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %8 = arith.divf %in, %in_0 : f32
      linalg.yield %8 : f32
    } -> tensor<8x32x1024x1024xf32>
    return %7 : tensor<8x32x1024x1024xf32>
  }
  func.func @Cluster_4(%arg0: tensor<8x32x1024x1024xbf16>, %arg1: f32, %arg2: f64, %arg3: bf16, %arg4: f32, %arg5: tensor<1x32x1x1xbf16>, %arg6: f32, %arg7: f64, %arg8: bf16, %arg9: f32, %arg10: f32, %arg11: f64, %arg12: bf16, %arg13: f32, %arg14: f32, %arg15: f64, %arg16: bf16, %arg17: f32) -> tensor<8x32x1024xf32> attributes {cycles = 4.198400e+06 : f64, externalAccess = 0x41C0020000000000 : f64, sramAccess = 0x41E0400000000000 : f64} {
    %0 = tensor.empty() : tensor<8x32x1024x1024xbf16>
    %1 = tensor.empty() : tensor<8x32x1024x1024xbf16>
    %2 = tensor.empty() : tensor<8x32x1024x1024xf32>
    %3 = tensor.empty() : tensor<8x32x1024xf32>
    %4 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<8x32x1024x1024xbf16>) outs(%1 : tensor<8x32x1024x1024xbf16>) attrs =  {inner_order = [3 : ui32, 2 : ui32, 0 : ui32, 1 : ui32], loop_bound = [8 : ui32, 32 : ui32, 1024 : ui32, 1024 : ui32], outer_order = [3 : ui32, 2 : ui32, 1 : ui32, 0 : ui32], tiling_size = [1 : ui32, 1 : ui32, 64 : ui32, 1024 : ui32], unroll_factor = [1 : ui32, 1 : ui32, 8 : ui32, 8 : ui32]} {
    ^bb0(%in: bf16, %out: bf16):
      %8 = arith.truncf %arg15 : f64 to bf16
      %9 = arith.mulf %in, %8 : bf16
      linalg.yield %9 : bf16
    } -> tensor<8x32x1024x1024xbf16>
    %5 = linalg.generic {indexing_maps = [#map3, #map9, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4, %arg5 : tensor<8x32x1024x1024xbf16>, tensor<1x32x1x1xbf16>) outs(%1 : tensor<8x32x1024x1024xbf16>) attrs =  {inner_order = [3 : ui32, 2 : ui32, 0 : ui32, 1 : ui32], loop_bound = [8 : ui32, 32 : ui32, 1024 : ui32, 1024 : ui32], outer_order = [3 : ui32, 2 : ui32, 1 : ui32, 0 : ui32], tiling_size = [1 : ui32, 1 : ui32, 64 : ui32, 1024 : ui32], unroll_factor = [1 : ui32, 1 : ui32, 8 : ui32, 8 : ui32]} {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %8 = arith.addf %in, %in_0 : bf16
      linalg.yield %8 : bf16
    } -> tensor<8x32x1024x1024xbf16>
    %6 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%5 : tensor<8x32x1024x1024xbf16>) outs(%2 : tensor<8x32x1024x1024xf32>) attrs =  {inner_order = [3 : ui32, 2 : ui32, 0 : ui32, 1 : ui32], loop_bound = [8 : ui32, 32 : ui32, 1024 : ui32, 1024 : ui32], outer_order = [3 : ui32, 2 : ui32, 1 : ui32, 0 : ui32], tiling_size = [1 : ui32, 1 : ui32, 64 : ui32, 1024 : ui32], unroll_factor = [1 : ui32, 1 : ui32, 8 : ui32, 8 : ui32]} {
    ^bb0(%in: bf16, %out: f32):
      %8 = arith.extf %in : bf16 to f32
      linalg.yield %8 : f32
    } -> tensor<8x32x1024x1024xf32>
    %7 = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%6 : tensor<8x32x1024x1024xf32>) outs(%3 : tensor<8x32x1024xf32>) attrs =  {inner_order = [3 : ui32, 2 : ui32, 0 : ui32, 1 : ui32], loop_bound = [8 : ui32, 32 : ui32, 1024 : ui32, 1024 : ui32], outer_order = [3 : ui32, 2 : ui32, 1 : ui32, 0 : ui32], tiling_size = [1 : ui32, 1 : ui32, 64 : ui32, 1024 : ui32], unroll_factor = [1 : ui32, 1 : ui32, 8 : ui32, 8 : ui32]} {
    ^bb0(%in: f32, %out: f32):
      %8 = arith.maximumf %in, %out : f32
      linalg.yield %8 : f32
    } -> tensor<8x32x1024xf32>
    return %7 : tensor<8x32x1024xf32>
  }
  func.func @Cluster_5(%arg0: tensor<8x8x1024x128xbf16>, %arg1: f32, %arg2: f64, %arg3: bf16, %arg4: f32, %arg5: f32, %arg6: f64, %arg7: bf16, %arg8: f32, %arg9: tensor<256x1024x128xbf16>, %arg10: f32, %arg11: f64, %arg12: bf16, %arg13: f32) -> tensor<256x1024x1024xbf16> attributes {cycles = 0x4160000000000000 : f64, externalAccess = 0x41CC000000000000 : f64, sramAccess = 0x4203000000000000 : f64} {
    %0 = tensor.empty() : tensor<8x8x4x1024x128xbf16>
    %1 = tensor.empty() : tensor<8x32x128x1024xbf16>
    %2 = tensor.empty() : tensor<256x1024x1024xbf16>
    %3 = linalg.generic {indexing_maps = [#map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<8x8x1024x128xbf16>) outs(%0 : tensor<8x8x4x1024x128xbf16>) attrs =  {accelgen.expand = true, accelgen.memory_transformation = true, inner_order = [4 : ui32, 3 : ui32, 0 : ui32, 1 : ui32, 2 : ui32], loop_bound = [8 : ui32, 8 : ui32, 4 : ui32, 1024 : ui32, 128 : ui32], outer_order = [4 : ui32, 3 : ui32, 0 : ui32, 1 : ui32, 2 : ui32], tiling_size = [1 : ui32, 1 : ui32, 4 : ui32, 256 : ui32, 128 : ui32], unroll_factor = [1 : ui32, 1 : ui32, 1 : ui32, 32 : ui32, 32 : ui32]} {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x8x4x1024x128xbf16>
    %collapsed = tensor.collapse_shape %3 [[0], [1, 2], [3], [4]] : tensor<8x8x4x1024x128xbf16> into tensor<8x32x1024x128xbf16>
    %4 = linalg.generic {indexing_maps = [#map10, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%collapsed : tensor<8x32x1024x128xbf16>) outs(%1 : tensor<8x32x128x1024xbf16>) attrs =  {accelgen.memory_transformation = true, accelgen.transpose = true, inner_order = [3 : ui32, 2 : ui32, 0 : ui32, 1 : ui32], loop_bound = [8 : ui32, 32 : ui32, 128 : ui32, 1024 : ui32], outer_order = [2 : ui32, 3 : ui32, 0 : ui32, 1 : ui32], tiling_size = [1 : ui32, 4 : ui32, 128 : ui32, 256 : ui32], unroll_factor = [1 : ui32, 1 : ui32, 32 : ui32, 32 : ui32]} {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x32x128x1024xbf16>
    %collapsed_0 = tensor.collapse_shape %4 [[0, 1], [2], [3]] : tensor<8x32x128x1024xbf16> into tensor<256x128x1024xbf16>
    %5 = linalg.generic {indexing_maps = [#map, #map6, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg9, %collapsed_0 : tensor<256x1024x128xbf16>, tensor<256x128x1024xbf16>) outs(%2 : tensor<256x1024x1024xbf16>) attrs =  {inner_order = [3 : ui32, 2 : ui32, 0 : ui32, 1 : ui32], loop_bound = [256 : ui32, 1024 : ui32, 1024 : ui32, 128 : ui32], outer_order = [3 : ui32, 2 : ui32, 0 : ui32, 1 : ui32], tiling_size = [4 : ui32, 16 : ui32, 256 : ui32, 128 : ui32], unroll_factor = [1 : ui32, 1 : ui32, 64 : ui32, 64 : ui32]} {
    ^bb0(%in: bf16, %in_1: bf16, %out: bf16):
      %6 = arith.mulf %in, %in_1 : bf16
      %7 = arith.addf %out, %6 : bf16
      linalg.yield %7 : bf16
    } -> tensor<256x1024x1024xbf16>
    return %5 : tensor<256x1024x1024xbf16>
  }
  func.func @Cluster_6(%arg0: tensor<8x1024x32x128xbf16>, %arg1: tensor<1x1x1024x128xbf16>, %arg2: f32, %arg3: f64, %arg4: bf16, %arg5: f32, %arg6: tensor<8x32x1024x128xbf16>, %arg7: f32, %arg8: f64, %arg9: bf16, %arg10: f32) -> tensor<8x32x1024x128xbf16> attributes {cycles = 0x4130000000000000 : f64, externalAccess = 0x41A0000000000000 : f64, sramAccess = 0x4198000000000000 : f64} {
    %0 = tensor.empty() : tensor<8x32x1024x128xbf16>
    %1 = tensor.empty() : tensor<8x32x1024x128xbf16>
    %2 = linalg.generic {indexing_maps = [#map7, #map11, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : tensor<8x1024x32x128xbf16>, tensor<1x1x1024x128xbf16>) outs(%1 : tensor<8x32x1024x128xbf16>) attrs =  {inner_order = [3 : ui32, 1 : ui32, 0 : ui32, 2 : ui32], loop_bound = [8 : ui32, 32 : ui32, 1024 : ui32, 128 : ui32], outer_order = [3 : ui32, 2 : ui32, 1 : ui32, 0 : ui32], tiling_size = [1 : ui32, 1 : ui32, 16 : ui32, 128 : ui32], unroll_factor = [1 : ui32, 1 : ui32, 1 : ui32, 128 : ui32]} {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %4 = arith.mulf %in, %in_0 : bf16
      linalg.yield %4 : bf16
    } -> tensor<8x32x1024x128xbf16>
    %3 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2, %arg6 : tensor<8x32x1024x128xbf16>, tensor<8x32x1024x128xbf16>) outs(%1 : tensor<8x32x1024x128xbf16>) attrs =  {inner_order = [3 : ui32, 2 : ui32, 0 : ui32, 1 : ui32], loop_bound = [8 : ui32, 32 : ui32, 1024 : ui32, 128 : ui32], outer_order = [3 : ui32, 2 : ui32, 1 : ui32, 0 : ui32], tiling_size = [1 : ui32, 1 : ui32, 16 : ui32, 128 : ui32], unroll_factor = [1 : ui32, 1 : ui32, 16 : ui32, 128 : ui32]} {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %4 = arith.addf %in, %in_0 : bf16
      linalg.yield %4 : bf16
    } -> tensor<8x32x1024x128xbf16>
    return %3 : tensor<8x32x1024x128xbf16>
  }
  func.func @Cluster_7(%arg0: tensor<8x1024x8x128xbf16>, %arg1: tensor<1x1x1024x128xbf16>, %arg2: f32, %arg3: f64, %arg4: bf16, %arg5: f32, %arg6: tensor<8x8x1024x128xbf16>, %arg7: f32, %arg8: f64, %arg9: bf16, %arg10: f32) -> tensor<8x8x1024x128xbf16> attributes {cycles = 2.621440e+05 : f64, externalAccess = 0x4180000000000000 : f64, sramAccess = 0x4178000000000000 : f64} {
    %0 = tensor.empty() : tensor<8x8x1024x128xbf16>
    %1 = tensor.empty() : tensor<8x8x1024x128xbf16>
    %2 = linalg.generic {indexing_maps = [#map7, #map11, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : tensor<8x1024x8x128xbf16>, tensor<1x1x1024x128xbf16>) outs(%1 : tensor<8x8x1024x128xbf16>) attrs =  {inner_order = [3 : ui32, 1 : ui32, 0 : ui32, 2 : ui32], loop_bound = [8 : ui32, 8 : ui32, 1024 : ui32, 128 : ui32], outer_order = [3 : ui32, 2 : ui32, 1 : ui32, 0 : ui32], tiling_size = [1 : ui32, 1 : ui32, 1 : ui32, 128 : ui32], unroll_factor = [1 : ui32, 1 : ui32, 1 : ui32, 128 : ui32]} {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %4 = arith.mulf %in, %in_0 : bf16
      linalg.yield %4 : bf16
    } -> tensor<8x8x1024x128xbf16>
    %3 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2, %arg6 : tensor<8x8x1024x128xbf16>, tensor<8x8x1024x128xbf16>) outs(%1 : tensor<8x8x1024x128xbf16>) attrs =  {inner_order = [3 : ui32, 2 : ui32, 0 : ui32, 1 : ui32], loop_bound = [8 : ui32, 8 : ui32, 1024 : ui32, 128 : ui32], outer_order = [3 : ui32, 2 : ui32, 1 : ui32, 0 : ui32], tiling_size = [1 : ui32, 1 : ui32, 1 : ui32, 128 : ui32], unroll_factor = [1 : ui32, 1 : ui32, 1 : ui32, 128 : ui32]} {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %4 = arith.addf %in, %in_0 : bf16
      linalg.yield %4 : bf16
    } -> tensor<8x8x1024x128xbf16>
    return %3 : tensor<8x8x1024x128xbf16>
  }
  func.func @Cluster_8(%arg0: tensor<8x1024x4096xbf16>, %arg1: tensor<4096x4096xbf16>, %arg2: f32, %arg3: f64, %arg4: bf16, %arg5: f32, %arg6: f32, %arg7: f64, %arg8: bf16, %arg9: f32, %arg10: f32, %arg11: f64, %arg12: bf16, %arg13: f32, %arg14: tensor<1x1x1024x128xbf16>, %arg15: f32, %arg16: f64, %arg17: bf16, %arg18: f32) -> tensor<8x32x1024x128xbf16> attributes {cycles = 0x4170000000000000 : f64, externalAccess = 0x41D2000000000000 : f64, sramAccess = 0x41E1C00000000000 : f64} {
    %0 = tensor.empty() : tensor<8x1024x4096xbf16>
    %1 = tensor.empty() : tensor<8x32x1024x128xbf16>
    %2 = tensor.empty() : tensor<8x32x1024x64xbf16>
    %3 = tensor.empty() : tensor<8x32x1024x128xbf16>
    %4 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<8x1024x4096xbf16>, tensor<4096x4096xbf16>) outs(%0 : tensor<8x1024x4096xbf16>) attrs =  {inner_order = [3 : ui32, 2 : ui32, 0 : ui32, 1 : ui32], loop_bound = [8 : ui32, 1024 : ui32, 4096 : ui32, 4096 : ui32], outer_order = [2 : ui32, 1 : ui32, 3 : ui32, 0 : ui32], tiling_size = [8 : ui32, 128 : ui32, 64 : ui32, 64 : ui32], unroll_factor = [1 : ui32, 1 : ui32, 64 : ui32, 64 : ui32]} {
    ^bb0(%in: bf16, %in_1: bf16, %out: bf16):
      %8 = arith.mulf %in, %in_1 : bf16
      %9 = arith.addf %out, %8 : bf16
      linalg.yield %9 : bf16
    } -> tensor<8x1024x4096xbf16>
    %expanded = tensor.expand_shape %4 [[0], [1], [2, 3]] output_shape [8, 1024, 32, 128] : tensor<8x1024x4096xbf16> into tensor<8x1024x32x128xbf16>
    %5 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded : tensor<8x1024x32x128xbf16>) outs(%3 : tensor<8x32x1024x128xbf16>) attrs =  {accelgen.memory_transformation = true, accelgen.transpose = true, inner_order = [3 : ui32, 1 : ui32, 0 : ui32, 2 : ui32], loop_bound = [8 : ui32, 32 : ui32, 1024 : ui32, 128 : ui32], outer_order = [1 : ui32, 3 : ui32, 2 : ui32, 0 : ui32], tiling_size = [8 : ui32, 1 : ui32, 128 : ui32, 64 : ui32], unroll_factor = [1 : ui32, 1 : ui32, 1 : ui32, 64 : ui32]} {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x32x1024x128xbf16>
    %extracted_slice = tensor.extract_slice %5[0, 0, 0, 64] [8, 32, 1024, 64] [1, 1, 1, 1] : tensor<8x32x1024x128xbf16> to tensor<8x32x1024x64xbf16>
    %extracted_slice_0 = tensor.extract_slice %5[0, 0, 0, 0] [8, 32, 1024, 64] [1, 1, 1, 1] : tensor<8x32x1024x128xbf16> to tensor<8x32x1024x64xbf16>
    %6 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice : tensor<8x32x1024x64xbf16>) outs(%2 : tensor<8x32x1024x64xbf16>) attrs =  {inner_order = [3 : ui32, 2 : ui32, 0 : ui32, 1 : ui32], loop_bound = [8 : ui32, 32 : ui32, 1024 : ui32, 64 : ui32], outer_order = [1 : ui32, 3 : ui32, 2 : ui32, 0 : ui32], tiling_size = [8 : ui32, 1 : ui32, 128 : ui32, 64 : ui32], unroll_factor = [1 : ui32, 1 : ui32, 1 : ui32, 1 : ui32]} {
    ^bb0(%in: bf16, %out: bf16):
      %8 = arith.negf %in : bf16
      linalg.yield %8 : bf16
    } -> tensor<8x32x1024x64xbf16>
    %concat = tensor.concat dim(3) %6, %extracted_slice_0 : (tensor<8x32x1024x64xbf16>, tensor<8x32x1024x64xbf16>) -> tensor<8x32x1024x128xbf16>
    %7 = linalg.generic {indexing_maps = [#map3, #map11, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat, %arg14 : tensor<8x32x1024x128xbf16>, tensor<1x1x1024x128xbf16>) outs(%3 : tensor<8x32x1024x128xbf16>) attrs =  {inner_order = [3 : ui32, 2 : ui32, 0 : ui32, 1 : ui32], loop_bound = [8 : ui32, 32 : ui32, 1024 : ui32, 128 : ui32], outer_order = [1 : ui32, 3 : ui32, 2 : ui32, 0 : ui32], tiling_size = [8 : ui32, 1 : ui32, 128 : ui32, 128 : ui32], unroll_factor = [1 : ui32, 1 : ui32, 1 : ui32, 2 : ui32]} {
    ^bb0(%in: bf16, %in_1: bf16, %out: bf16):
      %8 = arith.mulf %in, %in_1 : bf16
      linalg.yield %8 : bf16
    } -> tensor<8x32x1024x128xbf16>
    return %7 : tensor<8x32x1024x128xbf16>
  }
  func.func @Cluster_9(%arg0: tensor<8x1024x4096xbf16>, %arg1: tensor<4096x1024xbf16>, %arg2: f32, %arg3: f64, %arg4: bf16, %arg5: f32, %arg6: f32, %arg7: f64, %arg8: bf16, %arg9: f32, %arg10: f32, %arg11: f64, %arg12: bf16, %arg13: f32, %arg14: tensor<1x1x1024x128xbf16>, %arg15: f32, %arg16: f64, %arg17: bf16, %arg18: f32) -> tensor<8x8x1024x128xbf16> attributes {cycles = 0x4150000000000000 : f64, externalAccess = 0x41B2000000000000 : f64, sramAccess = 0x41C1C00000000000 : f64} {
    %0 = tensor.empty() : tensor<8x1024x1024xbf16>
    %1 = tensor.empty() : tensor<8x8x1024x128xbf16>
    %2 = tensor.empty() : tensor<8x8x1024x64xbf16>
    %3 = tensor.empty() : tensor<8x8x1024x128xbf16>
    %4 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<8x1024x4096xbf16>, tensor<4096x1024xbf16>) outs(%0 : tensor<8x1024x1024xbf16>) attrs =  {inner_order = [3 : ui32, 2 : ui32, 0 : ui32, 1 : ui32], loop_bound = [8 : ui32, 1024 : ui32, 1024 : ui32, 4096 : ui32], outer_order = [2 : ui32, 1 : ui32, 3 : ui32, 0 : ui32], tiling_size = [8 : ui32, 128 : ui32, 64 : ui32, 128 : ui32], unroll_factor = [1 : ui32, 1 : ui32, 64 : ui32, 64 : ui32]} {
    ^bb0(%in: bf16, %in_1: bf16, %out: bf16):
      %8 = arith.mulf %in, %in_1 : bf16
      %9 = arith.addf %out, %8 : bf16
      linalg.yield %9 : bf16
    } -> tensor<8x1024x1024xbf16>
    %expanded = tensor.expand_shape %4 [[0], [1], [2, 3]] output_shape [8, 1024, 8, 128] : tensor<8x1024x1024xbf16> into tensor<8x1024x8x128xbf16>
    %5 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded : tensor<8x1024x8x128xbf16>) outs(%3 : tensor<8x8x1024x128xbf16>) attrs =  {accelgen.memory_transformation = true, accelgen.transpose = true, inner_order = [3 : ui32, 1 : ui32, 0 : ui32, 2 : ui32], loop_bound = [8 : ui32, 8 : ui32, 1024 : ui32, 128 : ui32], outer_order = [1 : ui32, 3 : ui32, 2 : ui32, 0 : ui32], tiling_size = [8 : ui32, 1 : ui32, 128 : ui32, 64 : ui32], unroll_factor = [1 : ui32, 1 : ui32, 1 : ui32, 64 : ui32]} {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x8x1024x128xbf16>
    %extracted_slice = tensor.extract_slice %5[0, 0, 0, 64] [8, 8, 1024, 64] [1, 1, 1, 1] : tensor<8x8x1024x128xbf16> to tensor<8x8x1024x64xbf16>
    %extracted_slice_0 = tensor.extract_slice %5[0, 0, 0, 0] [8, 8, 1024, 64] [1, 1, 1, 1] : tensor<8x8x1024x128xbf16> to tensor<8x8x1024x64xbf16>
    %6 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice : tensor<8x8x1024x64xbf16>) outs(%2 : tensor<8x8x1024x64xbf16>) attrs =  {inner_order = [3 : ui32, 2 : ui32, 0 : ui32, 1 : ui32], loop_bound = [8 : ui32, 8 : ui32, 1024 : ui32, 64 : ui32], outer_order = [1 : ui32, 3 : ui32, 2 : ui32, 0 : ui32], tiling_size = [8 : ui32, 1 : ui32, 128 : ui32, 64 : ui32], unroll_factor = [1 : ui32, 1 : ui32, 1 : ui32, 1 : ui32]} {
    ^bb0(%in: bf16, %out: bf16):
      %8 = arith.negf %in : bf16
      linalg.yield %8 : bf16
    } -> tensor<8x8x1024x64xbf16>
    %concat = tensor.concat dim(3) %6, %extracted_slice_0 : (tensor<8x8x1024x64xbf16>, tensor<8x8x1024x64xbf16>) -> tensor<8x8x1024x128xbf16>
    %7 = linalg.generic {indexing_maps = [#map3, #map11, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%concat, %arg14 : tensor<8x8x1024x128xbf16>, tensor<1x1x1024x128xbf16>) outs(%3 : tensor<8x8x1024x128xbf16>) attrs =  {inner_order = [3 : ui32, 2 : ui32, 0 : ui32, 1 : ui32], loop_bound = [8 : ui32, 8 : ui32, 1024 : ui32, 128 : ui32], outer_order = [1 : ui32, 3 : ui32, 2 : ui32, 0 : ui32], tiling_size = [8 : ui32, 1 : ui32, 128 : ui32, 128 : ui32], unroll_factor = [1 : ui32, 1 : ui32, 1 : ui32, 2 : ui32]} {
    ^bb0(%in: bf16, %in_1: bf16, %out: bf16):
      %8 = arith.mulf %in, %in_1 : bf16
      linalg.yield %8 : bf16
    } -> tensor<8x8x1024x128xbf16>
    return %7 : tensor<8x8x1024x128xbf16>
  }
}
