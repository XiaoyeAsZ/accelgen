#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
module {
  func.func @Cluster_0(%arg0: tensor<8x32x1024x128xbf16>, %arg1: tensor<8x32x128x1024xbf16>) -> tensor<8x32x1024x1024xbf16> {
    %0 = tensor.empty() : tensor<8x32x1024x1024xbf16>
    %1 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<8x32x1024x128xbf16>, tensor<8x32x128x1024xbf16>) outs(%0 : tensor<8x32x1024x1024xbf16>) attrs =  {inner_order = [0 : ui32, 1 : ui32, 2 : ui32, 3 : ui32, 4 : ui32], loop_bound = [8 : ui32, 32 : ui32, 1024 : ui32, 1024 : ui32, 128 : ui32], outer_order = [3 : ui32, 2 : ui32, 1 : ui32, 0 : ui32, 4 : ui32], tiling_size = [1 : ui32, 1 : ui32, 128 : ui32, 128 : ui32, 32 : ui32], unroll_factor = [1 : ui32, 1 : ui32, 1 : ui32, 128 : ui32, 32 : ui32]} {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %2 = arith.mulf %in, %in_0 : bf16
      %3 = arith.addf %out, %2 : bf16
      linalg.yield %3 : bf16
    } -> tensor<8x32x1024x1024xbf16>
    return %1 : tensor<8x32x1024x1024xbf16>
  }
}

