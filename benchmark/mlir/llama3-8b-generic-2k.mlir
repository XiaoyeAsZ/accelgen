#map = affine_map<(d0, d1, d2) -> ()>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d3, d2)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map7 = affine_map<(d0, d1, d2, d3) -> (0, 0, 0, d3)>
#map8 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>
#map9 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map10 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#map11 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map12 = affine_map<(d0, d1, d2, d3) -> (0, d1, 0, 0)>
#map13 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>
#map14 = affine_map<(d0, d1, d2, d3) -> ()>
module {
  func.func @main(%arg0: tensor<8x1024x4096xbf16>, %arg1: tensor<1x1x128xbf16>, %arg2: tensor<1x1x128xbf16>, %arg3: tensor<1x32x1x1xbf16>, %arg4: tensor<4096x4096xbf16>, %arg5: tensor<4096x1024xbf16>, %arg6: tensor<4096x1024xbf16>, %arg7: tensor<4096x4096xbf16>) -> () {
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant 0.000000e+00 : bf16
    %cst_0 = arith.constant 0xFF800000 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %cst_2 = arith.constant 0.088388347648318447 : f64
    %5 = tensor.empty() : tensor<8x1024x1024xbf16>
    %6 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst : bf16) outs(%5 : tensor<8x1024x1024xbf16>) attrs =  {accelgen.const_broadcast = true, accelgen.memory_transformation = true} {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x1024x1024xbf16>
    %7 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg0, %arg5 : tensor<8x1024x4096xbf16>, tensor<4096x1024xbf16>) outs(%6 : tensor<8x1024x1024xbf16>) {
    ^bb0(%in: bf16, %in_19: bf16, %out: bf16):
      %53 = arith.mulf %in, %in_19 : bf16
      %54 = arith.addf %out, %53 : bf16
      linalg.yield %54 : bf16
    } -> tensor<8x1024x1024xbf16>
    %10 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg0, %arg6 : tensor<8x1024x4096xbf16>, tensor<4096x1024xbf16>) outs(%6 : tensor<8x1024x1024xbf16>) {
    ^bb0(%in: bf16, %in_19: bf16, %out: bf16):
      %53 = arith.mulf %in, %in_19 : bf16
      %54 = arith.addf %out, %53 : bf16
      linalg.yield %54 : bf16
    } -> tensor<8x1024x1024xbf16>
   
    return
  }
}

