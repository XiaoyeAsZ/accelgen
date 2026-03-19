#map = affine_map<(d0, d1, d2) -> ()>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d3, d2)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map7 = affine_map<(d0, d1, d2, d3) -> (0, 0, d2, d3)>
#map8 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>
#map9 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map10 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#map11 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map12 = affine_map<(d0, d1, d2, d3) -> (0, d1, 0, 0)>
#map13 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>
#map14 = affine_map<(d0, d1, d2, d3) -> ()>
module {
  func.func @main(%arg0: tensor<8x32x1024x1024xbf16>) {
    %cst = arith.constant 0.000000e+00 : bf16
    %cst_0 = arith.constant 0xFF800000 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %cst_2 = arith.constant 0.088388347648318447 : f64
    %0 = tensor.empty() : tensor<8x32x1024x1024xf32>
    %34 = linalg.generic {indexing_maps = [#map6, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<8x32x1024x1024xbf16>) outs(%0 : tensor<8x32x1024x1024xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %51 = arith.extf %in : bf16 to f32
      linalg.yield %51 : f32
    } -> tensor<8x32x1024x1024xf32>
    %1 = tensor.empty() : tensor<8x32x1024xf32>
    %37 = linalg.generic {indexing_maps = [#map6, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%34 : tensor<8x32x1024x1024xf32>) outs(%1 : tensor<8x32x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %51 = arith.maximumf %in, %out : f32
      linalg.yield %51 : f32
    } -> tensor<8x32x1024xf32>
    %expanded_14 = tensor.expand_shape %37 [[0], [1], [2, 3]] output_shape [8, 32, 1024, 1] : tensor<8x32x1024xf32> into tensor<8x32x1024x1xf32>
    %33 = tensor.empty() : tensor<8x32x1024x1024xf32>
    %38 = linalg.generic {indexing_maps = [#map6, #map13, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%34, %expanded_14 : tensor<8x32x1024x1024xf32>, tensor<8x32x1024x1xf32>) outs(%33 : tensor<8x32x1024x1024xf32>) {
    ^bb0(%in: f32, %in_19: f32, %out: f32):
      %51 = arith.subf %in, %in_19 : f32
      linalg.yield %51 : f32
    } -> tensor<8x32x1024x1024xf32>
    return
  }
}

