#map = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (0, 0, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module {
  func.func @main(%arg0: tensor<8x1024x4096xbf16>, %arg1: tensor<1x1x1024x128xbf16>) -> tensor<8x32x1024x128xbf16> {
    %expanded = tensor.expand_shape %arg0 [[0], [1], [2, 3]] output_shape [8, 1024, 32, 128] : tensor<8x1024x4096xbf16> into tensor<8x1024x32x128xbf16>
    %0 = tensor.empty() : tensor<8x32x1024x128xbf16>
    %1 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded, %arg1 : tensor<8x1024x32x128xbf16>, tensor<1x1x1024x128xbf16>) outs(%0 : tensor<8x32x1024x128xbf16>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %2 = arith.mulf %in, %in_0 : bf16
      linalg.yield %2 : bf16
    } -> tensor<8x32x1024x128xbf16>
    return %1 : tensor<8x32x1024x128xbf16>
  }
}
