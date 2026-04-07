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
module {
  func.func @main(%arg0: tensor<8x1024x32x128xbf16>, %arg1: tensor<1x1x1024x128xbf16>, %arg2: tensor<8x32x1024x128xbf16>) -> tensor<8x32x1024x128xbf16> {
    %0 = tensor.empty() : tensor<8x32x1024x128xbf16>
    %1 = tensor.empty() : tensor<8x32x1024x128xbf16>
    %2 = linalg.generic {indexing_maps = [#map5, #map11, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : tensor<8x1024x32x128xbf16>, tensor<1x1x1024x128xbf16>) outs(%1 : tensor<8x32x1024x128xbf16>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %4 = arith.mulf %in, %in_0 : bf16
      linalg.yield %4 : bf16
    } -> tensor<8x32x1024x128xbf16>
    %3 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2, %arg2 : tensor<8x32x1024x128xbf16>, tensor<8x32x1024x128xbf16>) outs(%1 : tensor<8x32x1024x128xbf16>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %4 = arith.addf %in, %in_0 : bf16
      linalg.yield %4 : bf16
    } -> tensor<8x32x1024x128xbf16>
    return %3 : tensor<8x32x1024x128xbf16>
  }
}

