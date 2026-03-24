#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2) -> ()>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d3, d2)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#map7 = affine_map<(d0, d1, d2, d3) -> (0, 0, d2, d3)>
#map8 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>
#map9 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map10 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>
#map11 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
#map12 = affine_map<(d0, d1, d2, d3) -> (0, d1, d2, 0)>
#map13 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>
#map14 = affine_map<(d0, d1, d2, d3) -> ()>
#map15 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d5)>
#map16 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d5, d4)>
#map17 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4)>
#map18 = affine_map<(d0, d1, d2) -> (d0, d2)>
module {
  func.func @main(%arg0: tensor<8x32x1x1024xbf16>) -> (tensor<8x8x4x1x1024xbf16>) {
    %collapsed_20 = tensor.collapse_shape %arg0 [[0, 1], [2], [3]] : tensor<8x32x1x1024xbf16> into tensor<256x1x1024xbf16>
    %expanded_21 = tensor.expand_shape %collapsed_20 [[0, 1, 2], [3], [4]] output_shape [8, 8, 4, 1, 1024] : tensor<256x1x1024xbf16> into tensor<8x8x4x1x1024xbf16>
    return %expanded_21 : tensor<8x8x4x1x1024xbf16>
  }
}

