#map = affine_map<(d0, d1, d2) -> ()>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d3, d2)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map7 = affine_map<(d0, d1, d2, d3) -> (0, 0, d2, d3)>
#map8 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>
#map9 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map10 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>
#map11 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
#map12 = affine_map<(d0, d1, d2, d3) -> (0, d1, 0, 0)>
#map13 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>
#map14 = affine_map<(d0, d1, d2, d3) -> ()>
#map15 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d5)>
#map16 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d5, d4)>
#map17 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4)>
#map18 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>
#map19 = affine_map<(d0, d1, d2, d3, d4) -> (d3, d4, d2)>
#map20 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
module {
  func.func @main(%arg0: tensor<8x32x1024x128xbf16>, %arg1: tensor<1x1x1024x128xbf16>, %arg2: tensor<8x32x1024x128xbf16>, %arg3: tensor<8x8x1024x128xbf16>) {
    %3 = tensor.empty() : tensor<8x32x1024x128xbf16>
    %12 = linalg.generic {indexing_maps = [#map6, #map7, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : tensor<8x32x1024x128xbf16>, tensor<1x1x1024x128xbf16>) outs(%3 : tensor<8x32x1024x128xbf16>) {
    ^bb0(%in: bf16, %in_20: bf16, %out: bf16):
      %50 = arith.mulf %in, %in_20 : bf16
      linalg.yield %50 : bf16
    } -> tensor<8x32x1024x128xbf16>
    %16 = linalg.generic {indexing_maps = [#map6, #map6, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%12, %arg2 : tensor<8x32x1024x128xbf16>, tensor<8x32x1024x128xbf16>) outs(%3 : tensor<8x32x1024x128xbf16>) {
    ^bb0(%in: bf16, %in_20: bf16, %out: bf16):
      %50 = arith.addf %in, %in_20 : bf16
      linalg.yield %50 : bf16
    } -> tensor<8x32x1024x128xbf16>
    %21 = tensor.empty() : tensor<8x8x1024x128xbf16>
    %expanded_11 = tensor.empty() : tensor<8x8x4x128x1024xbf16>
    %23 = linalg.generic {indexing_maps = [#map8, #map9], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%arg3 : tensor<8x8x1024x128xbf16>) outs(%expanded_11 : tensor<8x8x4x128x1024xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x8x4x128x1024xbf16>
    %collapsed = tensor.collapse_shape %23 [[0], [1, 2], [3], [4]] : tensor<8x8x4x128x1024xbf16> into tensor<8x32x128x1024xbf16>
    %expanded_12 = tensor.empty() : tensor<8x32x1024x1024xbf16>
    %26 = linalg.generic {indexing_maps = [#map10, #map8, #map11], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%16, %collapsed : tensor<8x32x1024x128xbf16>, tensor<8x32x128x1024xbf16>) outs(%expanded_12 : tensor<8x32x1024x1024xbf16>) {
    ^bb0(%in: bf16, %in_20: bf16, %out: bf16):
      %50 = arith.mulf %in, %in_20 : bf16
      %51 = arith.addf %out, %50 : bf16
      linalg.yield %51 : bf16
    } -> tensor<8x32x1024x1024xbf16>
    return
  }
}

