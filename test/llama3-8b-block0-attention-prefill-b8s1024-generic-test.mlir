#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#map6 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>
#map7 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map8 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>
#map9 = affine_map<(d0, d1, d2, d3) -> (0, d1, 0, 0)>
#map10 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#map11 = affine_map<(d0, d1, d2, d3) -> (0, 0, d2, d3)>
func.func @Cluster_3(%arg0: tensor<8x32x1024x1024xf32>, %arg1: tensor<8x32x1024x1xf32>, %arg2: f64, %arg3: bf16, %arg4: f32, %arg5: f32, %arg6: f64, %arg7: bf16, %arg8: f32, %arg9: f32, %arg10: f64, %arg11: bf16, %arg12: f32, %arg13: f32, %arg14: f64, %arg15: bf16, %arg16: f32, %arg17: f32) -> tensor<8x32x1024x1xf32> attributes {cycles = 0x4150000000000000 : f64, externalAccess = 0x41C8000000000000 : f64, flops = 0x41D0000000000000 : f64, sramAccess = 0x41E4000000000000 : f64} {
    %0 = tensor.empty() : tensor<8x32x1024x1024xf32>
    %1 = tensor.empty() : tensor<8x32x1024x1024xf32>
    %2 = tensor.empty() : tensor<8x32x1024x1xf32>
    %3 = tensor.empty() : tensor<8x32x1024x1024xf32>
    %6 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg0 : tensor<8x32x1024x1024xf32>) outs(%2 : tensor<8x32x1024x1xf32>) attrs =  {inner_order = [3 : ui32, 2 : ui32, 0 : ui32, 1 : ui32], loop_bound = [8 : ui32, 32 : ui32, 1024 : ui32, 1024 : ui32], outer_order = [3 : ui32, 2 : ui32, 1 : ui32, 0 : ui32], tiling_size = [1 : ui32, 1 : ui32, 1 : ui32, 1024 : ui32], unroll_factor = [1 : ui32, 1 : ui32, 1 : ui32, 64 : ui32]} {
    ^bb0(%in: f32, %out: f32):
      %8 = arith.addf %in, %out : f32
      linalg.yield %8 : f32
    } -> tensor<8x32x1024x1xf32>
    return %6 : tensor<8x32x1024x1xf32>
  }