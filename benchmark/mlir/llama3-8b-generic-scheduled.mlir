#map = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d3, d2)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
module {
  func.func @Cluster_0(%arg0: f64, %arg1: i64, %arg2: bf16, %arg3: tensor<8x1024x32x128xbf16>, %arg4: f32, %arg5: tensor<8x32x1024x128xbf16>, %arg6: f32) -> tensor<8x32x1024x128xbf16> {
    %0 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg3 : tensor<8x1024x32x128xbf16>) outs(%arg5 : tensor<8x32x1024x128xbf16>) attrs =  {accelgen.memory_transformation = true, accelgen.transpose = true, inner_order = [0 : ui32, 1 : ui32, 2 : ui32, 3 : ui32], loop_bound = [8 : ui32, 32 : ui32, 1024 : ui32, 128 : ui32], outer_order = [0 : ui32, 1 : ui32, 2 : ui32, 3 : ui32], tiling_size = [1 : ui32, 1 : ui32, 1 : ui32, 1 : ui32], unroll_factor = [1 : ui32, 1 : ui32, 1 : ui32, 1 : ui32]} {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x32x1024x128xbf16>
    return %arg5 : tensor<8x32x1024x128xbf16>
  }
  func.func @Cluster_1(%arg0: f64, %arg1: tensor<4096x1024xbf16>, %arg2: tensor<8x1024x1024xbf16>, %arg3: tensor<4096x4096xbf16>, %arg4: i64, %arg5: tensor<4096x1024xbf16>, %arg6: bf16, %arg7: f32, %arg8: tensor<8x1024x4096xbf16>, %arg9: f32, %arg10: tensor<8x1024x4096xbf16>) -> tensor<8x1024x4096xbf16> {
    %0 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg8, %arg3 : tensor<8x1024x4096xbf16>, tensor<4096x4096xbf16>) outs(%arg10 : tensor<8x1024x4096xbf16>) attrs =  {inner_order = [0 : ui32, 1 : ui32, 2 : ui32, 3 : ui32], loop_bound = [8 : ui32, 1024 : ui32, 4096 : ui32, 4096 : ui32], outer_order = [0 : ui32, 1 : ui32, 2 : ui32, 3 : ui32], tiling_size = [1 : ui32, 256 : ui32, 128 : ui32, 1 : ui32], unroll_factor = [1 : ui32, 1 : ui32, 1 : ui32, 1 : ui32]} {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %3 = arith.mulf %in, %in_0 : bf16
      %4 = arith.addf %out, %3 : bf16
      linalg.yield %4 : bf16
    } -> tensor<8x1024x4096xbf16>
    %1 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg8, %arg5 : tensor<8x1024x4096xbf16>, tensor<4096x1024xbf16>) outs(%arg2 : tensor<8x1024x1024xbf16>) attrs =  {inner_order = [0 : ui32, 1 : ui32, 2 : ui32, 3 : ui32], loop_bound = [8 : ui32, 1024 : ui32, 1024 : ui32, 4096 : ui32], outer_order = [0 : ui32, 1 : ui32, 2 : ui32, 3 : ui32], tiling_size = [1 : ui32, 256 : ui32, 32 : ui32, 1 : ui32], unroll_factor = [1 : ui32, 1 : ui32, 1 : ui32, 1 : ui32]} {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %3 = arith.mulf %in, %in_0 : bf16
      %4 = arith.addf %out, %3 : bf16
      linalg.yield %4 : bf16
    } -> tensor<8x1024x1024xbf16>
    %2 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg8, %arg1 : tensor<8x1024x4096xbf16>, tensor<4096x1024xbf16>) outs(%arg2 : tensor<8x1024x1024xbf16>) attrs =  {inner_order = [0 : ui32, 1 : ui32, 2 : ui32, 3 : ui32], loop_bound = [8 : ui32, 1024 : ui32, 1024 : ui32, 4096 : ui32], outer_order = [0 : ui32, 1 : ui32, 2 : ui32, 3 : ui32], tiling_size = [1 : ui32, 256 : ui32, 32 : ui32, 1 : ui32], unroll_factor = [1 : ui32, 1 : ui32, 1 : ui32, 1 : ui32]} {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %3 = arith.mulf %in, %in_0 : bf16
      %4 = arith.addf %out, %3 : bf16
      linalg.yield %4 : bf16
    } -> tensor<8x1024x1024xbf16>
    return %arg10 : tensor<8x1024x4096xbf16>
  }
}

