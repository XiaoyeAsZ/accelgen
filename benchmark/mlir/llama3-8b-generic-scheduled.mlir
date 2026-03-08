#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module {
  func.func @Cluster_0(%arg0: tensor<4096x1024xbf16>, %arg1: tensor<8x1024x1024xbf16>, %arg2: f32, %arg3: tensor<8x1024x4096xbf16>, %arg4: tensor<4096x1024xbf16>, %arg5: f64, %arg6: i64, %arg7: bf16, %arg8: f32) {
    %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg3, %arg4 : tensor<8x1024x4096xbf16>, tensor<4096x1024xbf16>) outs(%arg1 : tensor<8x1024x1024xbf16>) attrs =  {inner_order = [0 : ui32, 1 : ui32, 2 : ui32, 3 : ui32], loop_bound = [8 : ui32, 1024 : ui32, 1024 : ui32, 4096 : ui32], outer_order = [2 : ui32, 1 : ui32, 0 : ui32, 3 : ui32], tiling_size = [1 : ui32, 256 : ui32, 64 : ui32, 1 : ui32], unroll_factor = [1 : ui32, 64 : ui32, 32 : ui32, 1 : ui32]} {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %2 = arith.mulf %in, %in_0 : bf16
      %3 = arith.addf %out, %2 : bf16
      linalg.yield %3 : bf16
    } -> tensor<8x1024x1024xbf16>
    %1 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg3, %arg0 : tensor<8x1024x4096xbf16>, tensor<4096x1024xbf16>) outs(%arg1 : tensor<8x1024x1024xbf16>) attrs =  {inner_order = [0 : ui32, 1 : ui32, 2 : ui32, 3 : ui32], loop_bound = [8 : ui32, 1024 : ui32, 1024 : ui32, 4096 : ui32], outer_order = [2 : ui32, 1 : ui32, 0 : ui32, 3 : ui32], tiling_size = [1 : ui32, 256 : ui32, 64 : ui32, 1 : ui32], unroll_factor = [1 : ui32, 64 : ui32, 32 : ui32, 1 : ui32]} {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %2 = arith.mulf %in, %in_0 : bf16
      %3 = arith.addf %out, %2 : bf16
      linalg.yield %3 : bf16
    } -> tensor<8x1024x1024xbf16>
    return
  }
  func.func @Cluster_1(%arg0: tensor<8x32x1024x128xbf16>, %arg1: f32, %arg2: tensor<8x1024x32x128xbf16>, %arg3: f64, %arg4: i64, %arg5: bf16, %arg6: f32) -> tensor<8x32x1024x128xbf16> {
    %0 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2 : tensor<8x1024x32x128xbf16>) outs(%arg0 : tensor<8x32x1024x128xbf16>) attrs =  {accelgen.memory_transformation = true, accelgen.transpose = true, inner_order = [0 : ui32, 1 : ui32, 2 : ui32, 3 : ui32], loop_bound = [8 : ui32, 32 : ui32, 1024 : ui32, 128 : ui32], outer_order = [3 : ui32, 2 : ui32, 1 : ui32, 0 : ui32], tiling_size = [1 : ui32, 1 : ui32, 1 : ui32, 1 : ui32], unroll_factor = [8 : ui32, 16 : ui32, 1 : ui32, 1 : ui32]} {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x32x1024x128xbf16>
    return %arg0 : tensor<8x32x1024x128xbf16>
  }
  func.func @Cluster_2(%arg0: tensor<8x1024x4096xbf16>, %arg1: f32, %arg2: tensor<8x1024x4096xbf16>, %arg3: tensor<4096x4096xbf16>, %arg4: f64, %arg5: i64, %arg6: bf16, %arg7: f32) -> tensor<8x1024x4096xbf16> {
    %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg2, %arg3 : tensor<8x1024x4096xbf16>, tensor<4096x4096xbf16>) outs(%arg0 : tensor<8x1024x4096xbf16>) attrs =  {inner_order = [0 : ui32, 1 : ui32, 2 : ui32, 3 : ui32], loop_bound = [8 : ui32, 1024 : ui32, 4096 : ui32, 4096 : ui32], outer_order = [2 : ui32, 1 : ui32, 0 : ui32, 3 : ui32], tiling_size = [1 : ui32, 128 : ui32, 128 : ui32, 1 : ui32], unroll_factor = [1 : ui32, 64 : ui32, 64 : ui32, 1 : ui32]} {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %1 = arith.mulf %in, %in_0 : bf16
      %2 = arith.addf %out, %1 : bf16
      linalg.yield %2 : bf16
    } -> tensor<8x1024x4096xbf16>
    return %arg0 : tensor<8x1024x4096xbf16>
  }
}

