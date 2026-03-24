#map = affine_map<(d0, d1) -> (d1, d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map4 = affine_map<(d0, d1, d2) -> ()>
#map5 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map8 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#map9 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map10 = affine_map<(d0, d1, d2, d3) -> (0, 0, d2, d3)>
#map11 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>
#map12 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map13 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#map14 = affine_map<(d0, d1, d2, d3) -> (0, d1, 0, 0)>
#map15 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>
#map16 = affine_map<(d0, d1, d2, d3) -> ()>
module {
  func.func @main(%arg0: tensor<1x1024x128xbf16>, %arg1: tensor<1x1024x128xbf16>, %arg2: tensor<8x32x1024x128xbf16>, %arg3: tensor<8x32x1024x128xbf16>, %arg4: tensor<8x8x1024x128xbf16>, %arg5: tensor<8x8x1024x128xbf16>) {
    %cst = arith.constant 0.000000e+00 : bf16
    %cst_0 = arith.constant 0xFF800000 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %cst_2 = arith.constant 0.088388347648318447 : f64
    
    %expanded_5 = tensor.expand_shape %arg0 [[0], [1, 2], [3]] output_shape [1, 1, 1024, 128] : tensor<1x1024x128xbf16> into tensor<1x1x1024x128xbf16>
    %expanded_6 = tensor.expand_shape %arg1 [[0], [1, 2], [3]] output_shape [1, 1, 1024, 128] : tensor<1x1024x128xbf16> into tensor<1x1x1024x128xbf16>
    %7 = tensor.empty() : tensor<8x32x1024x128xbf16>
    %22 = linalg.generic {indexing_maps = [#map9, #map10, #map9], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2, %expanded_5 : tensor<8x32x1024x128xbf16>, tensor<1x1x1024x128xbf16>) outs(%7 : tensor<8x32x1024x128xbf16>) {
    ^bb0(%in: bf16, %in_19: bf16, %out: bf16):
      %63 = arith.mulf %in, %in_19 : bf16
      linalg.yield %63 : bf16
    } -> tensor<8x32x1024x128xbf16>
    %concat = tensor.concat dim(3) %24, %extracted_slice : (tensor<8x32x1024x64xbf16>, tensor<8x32x1024x64xbf16>) -> tensor<8x32x1024x128xbf16>
    %25 = linalg.generic {indexing_maps = [#map9, #map10, #map9], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg3, %expanded_6 : tensor<8x32x1024x128xbf16>, tensor<1x1x1024x128xbf16>) outs(%7 : tensor<8x32x1024x128xbf16>) {
    ^bb0(%in: bf16, %in_19: bf16, %out: bf16):
      %63 = arith.mulf %in, %in_19 : bf16
      linalg.yield %63 : bf16
    } -> tensor<8x32x1024x128xbf16>
    %16 = tensor.empty() : tensor<8x8x1024x128xbf16>
    %27 = linalg.generic {indexing_maps = [#map9, #map10, #map9], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg4, %expanded_5 : tensor<8x8x1024x128xbf16>, tensor<1x1x1024x128xbf16>) outs(%16 : tensor<8x8x1024x128xbf16>) {
    ^bb0(%in: bf16, %in_19: bf16, %out: bf16):
      %63 = arith.mulf %in, %in_19 : bf16
      linalg.yield %63 : bf16
    } -> tensor<8x8x1024x128xbf16>
    %31 = linalg.generic {indexing_maps = [#map9, #map9, #map9], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%27, %arg5 : tensor<8x8x1024x128xbf16>, tensor<8x8x1024x128xbf16>) outs(%16 : tensor<8x8x1024x128xbf16>) {
    ^bb0(%in: bf16, %in_19: bf16, %out: bf16):
      %63 = arith.addf %in, %in_19 : bf16
      linalg.yield %63 : bf16
    } -> tensor<8x8x1024x128xbf16>
    return
  }
}

