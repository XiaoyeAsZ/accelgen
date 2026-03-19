#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
module {
  func.func @Cluster_0(%arg0: tensor<8x32x1024x1024xf32>, %arg1: tensor<8x32x1024x1xf32>, %arg2: f32, %arg3: f32, %arg4: f64, %arg5: bf16) -> tensor<8x32x1024x1024xf32> {
    %0 = tensor.empty() : tensor<8x32x1024x1024xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : tensor<8x32x1024x1024xf32>, tensor<8x32x1024x1xf32>) outs(%0 : tensor<8x32x1024x1024xf32>) attrs =  {inner_order = [0 : ui32, 1 : ui32, 2 : ui32, 3 : ui32], loop_bound = [8 : ui32, 32 : ui32, 1024 : ui32, 1024 : ui32], outer_order = [3 : ui32, 2 : ui32, 1 : ui32, 0 : ui32], tiling_size = [1 : ui32, 1 : ui32, 4 : ui32, 64 : ui32], unroll_factor = [1 : ui32, 1 : ui32, 4 : ui32, 8 : ui32]} {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %2 = arith.subf %in, %in_0 : f32
      linalg.yield %2 : f32
    } -> tensor<8x32x1024x1024xf32>
    return %1 : tensor<8x32x1024x1024xf32>
  }
  func.func @Cluster_1(%arg0: tensor<8x32x1024x1024xbf16>, %arg1: f32, %arg2: f32, %arg3: f64, %arg4: bf16, %arg5: f32, %arg6: f32, %arg7: f64, %arg8: bf16) -> tensor<8x32x1024xf32> {
    %0 = tensor.empty() : tensor<8x32x1024x1024xf32>
    %1 = tensor.empty() : tensor<8x32x1024xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<8x32x1024x1024xbf16>) outs(%0 : tensor<8x32x1024x1024xf32>) attrs =  {inner_order = [0 : ui32, 1 : ui32, 2 : ui32, 3 : ui32], loop_bound = [8 : ui32, 32 : ui32, 1024 : ui32, 1024 : ui32], outer_order = [2 : ui32, 1 : ui32, 0 : ui32, 3 : ui32], tiling_size = [1 : ui32, 1 : ui32, 128 : ui32, 32 : ui32], unroll_factor = [1 : ui32, 1 : ui32, 8 : ui32, 8 : ui32]} {
    ^bb0(%in: bf16, %out: f32):
      %4 = arith.extf %in : bf16 to f32
      linalg.yield %4 : f32
    } -> tensor<8x32x1024x1024xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%2 : tensor<8x32x1024x1024xf32>) outs(%1 : tensor<8x32x1024xf32>) attrs =  {inner_order = [0 : ui32, 1 : ui32, 2 : ui32, 3 : ui32], loop_bound = [8 : ui32, 32 : ui32, 1024 : ui32, 1024 : ui32], outer_order = [2 : ui32, 1 : ui32, 0 : ui32, 3 : ui32], tiling_size = [1 : ui32, 1 : ui32, 128 : ui32, 32 : ui32], unroll_factor = [1 : ui32, 1 : ui32, 8 : ui32, 8 : ui32]} {
    ^bb0(%in: f32, %out: f32):
      %4 = arith.maximumf %in, %out : f32
      linalg.yield %4 : f32
    } -> tensor<8x32x1024xf32>
    return %3 : tensor<8x32x1024xf32>
  }
}

