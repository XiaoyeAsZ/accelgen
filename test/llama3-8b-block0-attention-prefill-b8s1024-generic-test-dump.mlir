#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>
module {
  func.func @Cluster_0(%arg0: tensor<8x32x1024x1024xf32>) -> tensor<8x32x1024x1xf32> attributes {cycles = 5.242880e+05 : f64, externalAccess = 0x41B8000000000000 : f64, flops = 0x41B0000000000000 : f64, sramAccess = 0x4190000000000000 : f64} {
    %0 = tensor.empty() : tensor<8x32x1024x1xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg0 : tensor<8x32x1024x1024xf32>) outs(%0 : tensor<8x32x1024x1xf32>) attrs =  {inner_order = [3 : ui32, 2 : ui32, 0 : ui32, 1 : ui32], loop_bound = [8 : ui32, 32 : ui32, 1024 : ui32, 1024 : ui32], outer_order = [3 : ui32, 2 : ui32, 1 : ui32, 0 : ui32], tiling_size = [1 : ui32, 1 : ui32, 1 : ui32, 512 : ui32], unroll_factor = [1 : ui32, 1 : ui32, 1 : ui32, 512 : ui32]} {
    ^bb0(%in: f32, %out: f32):
      %2 = arith.addf %in, %out : f32
      linalg.yield %2 : f32
    } -> tensor<8x32x1024x1xf32>
    return %1 : tensor<8x32x1024x1xf32>
  }
}
