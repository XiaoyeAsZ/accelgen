#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map7 = affine_map<(d0, d1, d2, d3) -> (0, d1, 0, 0)>
#map8 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map9 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map10 = affine_map<(d0, d1, d2, d3) -> (0, 0, 0, d3)>
#map11 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#map12 = affine_map<(d0, d1, d2) -> ()>
#map13 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#map14 = affine_map<(d0, d1) -> (d1, d0)>
#map15 = affine_map<(d0, d1) -> (d0, d1)>
#map16 = affine_map<(d0, d1, d2, d3) -> ()>
module {
  func.func @Cluster_0(%arg0: f32, %arg1: f64, %arg2: f32, %arg3: i64, %arg4: bf16, %arg5: tensor<8x32x1024x1024xbf16>, %arg6: tensor<8x32x1024x1024xf32>) -> tensor<8x32x1024x1024xbf16> {
    %0 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg6 : tensor<8x32x1024x1024xf32>) outs(%arg5 : tensor<8x32x1024x1024xbf16>) attrs =  {inner_order = [0 : ui32, 1 : ui32, 2 : ui32, 3 : ui32], outer_order = [0 : ui32, 1 : ui32, 2 : ui32, 3 : ui32], tiling_size = [1 : ui32, 1 : ui32, 1 : ui32, 1 : ui32], unroll_factor = [1 : ui32, 1 : ui32, 1 : ui32, 1 : ui32]} {
    ^bb0(%in: f32, %out: bf16):
      %1 = arith.truncf %in : f32 to bf16
      linalg.yield %1 : bf16
    } -> tensor<8x32x1024x1024xbf16>
    return %arg5 : tensor<8x32x1024x1024xbf16>
  }
  func.func @Cluster_1(%arg0: tensor<8x32x1024x1024xf32>, %arg1: tensor<8x32x1024x1xf32>, %arg2: f32, %arg3: f64, %arg4: f32, %arg5: i64, %arg6: bf16, %arg7: tensor<8x32x1024x1024xf32>) -> tensor<8x32x1024x1024xf32> {
    %0 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : tensor<8x32x1024x1024xf32>, tensor<8x32x1024x1xf32>) outs(%arg7 : tensor<8x32x1024x1024xf32>) attrs =  {inner_order = [0 : ui32, 1 : ui32, 2 : ui32, 3 : ui32], outer_order = [0 : ui32, 1 : ui32, 2 : ui32, 3 : ui32], tiling_size = [1 : ui32, 1 : ui32, 1 : ui32, 1 : ui32], unroll_factor = [1 : ui32, 1 : ui32, 1 : ui32, 1 : ui32]} {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %1 = arith.divf %in, %in_0 : f32
      linalg.yield %1 : f32
    } -> tensor<8x32x1024x1024xf32>
    return %arg7 : tensor<8x32x1024x1024xf32>
  }
  func.func @Cluster_2(%arg0: tensor<8x32x1024x1024xf32>, %arg1: f32, %arg2: f64, %arg3: f32, %arg4: i64, %arg5: bf16, %arg6: tensor<8x32x1024x1xf32>) -> tensor<8x32x1024x1xf32> {
    %0 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg0 : tensor<8x32x1024x1024xf32>) outs(%arg6 : tensor<8x32x1024x1xf32>) attrs =  {inner_order = [0 : ui32, 1 : ui32, 2 : ui32, 3 : ui32], outer_order = [0 : ui32, 1 : ui32, 2 : ui32, 3 : ui32], tiling_size = [1 : ui32, 1 : ui32, 1 : ui32, 1 : ui32], unroll_factor = [1 : ui32, 1 : ui32, 1 : ui32, 1 : ui32]} {
    ^bb0(%in: f32, %out: f32):
      %1 = arith.addf %in, %out : f32
      linalg.yield %1 : f32
    } -> tensor<8x32x1024x1xf32>
    return %arg6 : tensor<8x32x1024x1xf32>
  }
  func.func @Cluster_3(%arg0: f32, %arg1: f64, %arg2: f32, %arg3: i64, %arg4: bf16, %arg5: tensor<8x32x1024x1024xf32>, %arg6: tensor<8x32x1024x1024xf32>) -> tensor<8x32x1024x1024xf32> {
    %0 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg6 : tensor<8x32x1024x1024xf32>) outs(%arg5 : tensor<8x32x1024x1024xf32>) attrs =  {inner_order = [0 : ui32, 1 : ui32, 2 : ui32, 3 : ui32], outer_order = [0 : ui32, 1 : ui32, 2 : ui32, 3 : ui32], tiling_size = [1 : ui32, 1 : ui32, 1 : ui32, 1 : ui32], unroll_factor = [1 : ui32, 1 : ui32, 1 : ui32, 1 : ui32]} {
    ^bb0(%in: f32, %out: f32):
      %1 = math.exp %in : f32
      linalg.yield %1 : f32
    } -> tensor<8x32x1024x1024xf32>
    return %arg5 : tensor<8x32x1024x1024xf32>
  }
  func.func @Cluster_4(%arg0: tensor<8x32x1024x1024xf32>, %arg1: f32, %arg2: f64, %arg3: tensor<8x32x1024xf32>, %arg4: f32, %arg5: i64, %arg6: bf16, %arg7: tensor<8x32x1024xi64>) -> (tensor<8x32x1024xf32>, tensor<8x32x1024xi64>) {
    %0:2 = linalg.generic {indexing_maps = [#map, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg0 : tensor<8x32x1024x1024xf32>) outs(%arg3, %arg7 : tensor<8x32x1024xf32>, tensor<8x32x1024xi64>) attrs =  {inner_order = [0 : ui32, 1 : ui32, 2 : ui32, 3 : ui32], outer_order = [0 : ui32, 1 : ui32, 2 : ui32, 3 : ui32], tiling_size = [1 : ui32, 1 : ui32, 1 : ui32, 1 : ui32], unroll_factor = [1 : ui32, 1 : ui32, 1 : ui32, 1 : ui32]} {
    ^bb0(%in: f32, %out: f32, %out_0: i64):
      %1 = linalg.index 3 : index
      %2 = arith.index_cast %1 : index to i64
      %3 = arith.maximumf %in, %out : f32
      %4 = arith.cmpf ogt, %in, %out : f32
      %5 = arith.select %4, %2, %out_0 : i64
      linalg.yield %3, %5 : f32, i64
    } -> (tensor<8x32x1024xf32>, tensor<8x32x1024xi64>)
    return %arg3, %arg7 : tensor<8x32x1024xf32>, tensor<8x32x1024xi64>
  }
  func.func @Cluster_5(%arg0: tensor<8x32x1024x1024xf32>, %arg1: tensor<8x32x1024x1xf32>, %arg2: f32, %arg3: f64, %arg4: f32, %arg5: i64, %arg6: bf16, %arg7: tensor<8x32x1024x1024xf32>) -> tensor<8x32x1024x1024xf32> {
    %0 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : tensor<8x32x1024x1024xf32>, tensor<8x32x1024x1xf32>) outs(%arg7 : tensor<8x32x1024x1024xf32>) attrs =  {inner_order = [0 : ui32, 1 : ui32, 2 : ui32, 3 : ui32], outer_order = [0 : ui32, 1 : ui32, 2 : ui32, 3 : ui32], tiling_size = [1 : ui32, 1 : ui32, 1 : ui32, 1 : ui32], unroll_factor = [1 : ui32, 1 : ui32, 1 : ui32, 1 : ui32]} {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %1 = arith.subf %in, %in_0 : f32
      linalg.yield %1 : f32
    } -> tensor<8x32x1024x1024xf32>
    return %arg7 : tensor<8x32x1024x1024xf32>
  }
  func.func @Cluster_6(%arg0: f32, %arg1: tensor<8x8x4x1024x128xbf16>, %arg2: f64, %arg3: tensor<8x8x1024x128xbf16>, %arg4: f32, %arg5: i64, %arg6: bf16) -> tensor<8x8x4x1024x128xbf16> {
    %0 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%arg3 : tensor<8x8x1024x128xbf16>) outs(%arg1 : tensor<8x8x4x1024x128xbf16>) attrs =  {inner_order = [0 : ui32, 1 : ui32, 2 : ui32, 3 : ui32, 4 : ui32], outer_order = [0 : ui32, 1 : ui32, 2 : ui32, 3 : ui32, 4 : ui32], tiling_size = [1 : ui32, 1 : ui32, 1 : ui32, 1 : ui32, 1 : ui32], unroll_factor = [1 : ui32, 1 : ui32, 1 : ui32, 1 : ui32, 1 : ui32]} {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x8x4x1024x128xbf16>
    return %arg1 : tensor<8x8x4x1024x128xbf16>
  }
  func.func @Cluster_7(%arg0: tensor<8x32x1024x1024xbf16>, %arg1: f32, %arg2: f64, %arg3: f32, %arg4: i64, %arg5: bf16, %arg6: tensor<8x32x1024x1024xf32>) -> tensor<8x32x1024x1024xf32> {
    %0 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<8x32x1024x1024xbf16>) outs(%arg6 : tensor<8x32x1024x1024xf32>) attrs =  {inner_order = [0 : ui32, 1 : ui32, 2 : ui32, 3 : ui32], outer_order = [0 : ui32, 1 : ui32, 2 : ui32, 3 : ui32], tiling_size = [1 : ui32, 1 : ui32, 1 : ui32, 1 : ui32], unroll_factor = [1 : ui32, 1 : ui32, 1 : ui32, 1 : ui32]} {
    ^bb0(%in: bf16, %out: f32):
      %1 = arith.extf %in : bf16 to f32
      linalg.yield %1 : f32
    } -> tensor<8x32x1024x1024xf32>
    return %arg6 : tensor<8x32x1024x1024xf32>
  }
  func.func @Cluster_8(%arg0: tensor<8x1024x4096xbf16>, %arg1: f32, %arg2: f64, %arg3: tensor<8x1024x4096xbf16>, %arg4: f32, %arg5: tensor<8x4096x4096xbf16>, %arg6: i64, %arg7: bf16) -> tensor<8x1024x4096xbf16> {
    %0 = linalg.generic {indexing_maps = [#map5, #map6, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg0, %arg5 : tensor<8x1024x4096xbf16>, tensor<8x4096x4096xbf16>) outs(%arg3 : tensor<8x1024x4096xbf16>) attrs =  {inner_order = [0 : ui32, 1 : ui32, 2 : ui32, 3 : ui32], outer_order = [0 : ui32, 1 : ui32, 2 : ui32, 3 : ui32], tiling_size = [1 : ui32, 1 : ui32, 1 : ui32, 1 : ui32], unroll_factor = [1 : ui32, 1 : ui32, 1 : ui32, 1 : ui32]} {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %1 = arith.mulf %in, %in_0 : bf16
      %2 = arith.addf %out, %1 : bf16
      linalg.yield %2 : bf16
    } -> tensor<8x1024x4096xbf16>
    return %arg3 : tensor<8x1024x4096xbf16>
  }
  func.func @Cluster_9(%arg0: f32, %arg1: f64, %arg2: f32, %arg3: i64, %arg4: tensor<8x32x1024x128xbf16>, %arg5: bf16, %arg6: tensor<8x32x1024x128xbf16>, %arg7: tensor<8x32x1024x128xbf16>) -> tensor<8x32x1024x128xbf16> {
    %0 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg7, %arg4 : tensor<8x32x1024x128xbf16>, tensor<8x32x1024x128xbf16>) outs(%arg6 : tensor<8x32x1024x128xbf16>) attrs =  {inner_order = [0 : ui32, 1 : ui32, 2 : ui32, 3 : ui32], outer_order = [0 : ui32, 1 : ui32, 2 : ui32, 3 : ui32], tiling_size = [1 : ui32, 1 : ui32, 1 : ui32, 1 : ui32], unroll_factor = [1 : ui32, 1 : ui32, 1 : ui32, 1 : ui32]} {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %1 = arith.addf %in, %in_0 : bf16
      linalg.yield %1 : bf16
    } -> tensor<8x32x1024x128xbf16>
    return %arg6 : tensor<8x32x1024x128xbf16>
  }
  func.func @Cluster_10(%arg0: tensor<8x1024x4096xbf16>, %arg1: f32, %arg2: f64, %arg3: f32, %arg4: tensor<8x4096x1024xbf16>, %arg5: i64, %arg6: bf16, %arg7: tensor<8x1024x1024xbf16>) -> tensor<8x1024x1024xbf16> {
    %0 = linalg.generic {indexing_maps = [#map5, #map6, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg0, %arg4 : tensor<8x1024x4096xbf16>, tensor<8x4096x1024xbf16>) outs(%arg7 : tensor<8x1024x1024xbf16>) attrs =  {inner_order = [0 : ui32, 1 : ui32, 2 : ui32, 3 : ui32], outer_order = [0 : ui32, 1 : ui32, 2 : ui32, 3 : ui32], tiling_size = [1 : ui32, 1 : ui32, 1 : ui32, 1 : ui32], unroll_factor = [1 : ui32, 1 : ui32, 1 : ui32, 1 : ui32]} {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %1 = arith.mulf %in, %in_0 : bf16
      %2 = arith.addf %out, %1 : bf16
      linalg.yield %2 : bf16
    } -> tensor<8x1024x1024xbf16>
    return %arg7 : tensor<8x1024x1024xbf16>
  }
  func.func @Cluster_11(%arg0: tensor<8x1024x4096xbf16>, %arg1: f32, %arg2: f64, %arg3: f32, %arg4: i64, %arg5: bf16, %arg6: tensor<8x4096x1024xbf16>, %arg7: tensor<8x1024x1024xbf16>) -> tensor<8x1024x1024xbf16> {
    %0 = linalg.generic {indexing_maps = [#map5, #map6, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg0, %arg6 : tensor<8x1024x4096xbf16>, tensor<8x4096x1024xbf16>) outs(%arg7 : tensor<8x1024x1024xbf16>) attrs =  {inner_order = [0 : ui32, 1 : ui32, 2 : ui32, 3 : ui32], outer_order = [0 : ui32, 1 : ui32, 2 : ui32, 3 : ui32], tiling_size = [1 : ui32, 1 : ui32, 1 : ui32, 1 : ui32], unroll_factor = [1 : ui32, 1 : ui32, 1 : ui32, 1 : ui32]} {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %1 = arith.mulf %in, %in_0 : bf16
      %2 = arith.addf %out, %1 : bf16
      linalg.yield %2 : bf16
    } -> tensor<8x1024x1024xbf16>
    return %arg7 : tensor<8x1024x1024xbf16>
  }
  func.func @Cluster_12(%arg0: tensor<8x8x1024x128xbf16>, %arg1: tensor<8x8x1024x128xbf16>, %arg2: f32, %arg3: f64, %arg4: f32, %arg5: i64, %arg6: bf16, %arg7: tensor<8x8x1024x128xbf16>) -> tensor<8x8x1024x128xbf16> {
    %0 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg1, %arg7 : tensor<8x8x1024x128xbf16>, tensor<8x8x1024x128xbf16>) outs(%arg0 : tensor<8x8x1024x128xbf16>) attrs =  {inner_order = [0 : ui32, 1 : ui32, 2 : ui32, 3 : ui32], outer_order = [0 : ui32, 1 : ui32, 2 : ui32, 3 : ui32], tiling_size = [1 : ui32, 1 : ui32, 1 : ui32, 1 : ui32], unroll_factor = [1 : ui32, 1 : ui32, 1 : ui32, 1 : ui32]} {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %1 = arith.addf %in, %in_0 : bf16
      linalg.yield %1 : bf16
    } -> tensor<8x8x1024x128xbf16>
    return %arg0 : tensor<8x8x1024x128xbf16>
  }
  func.func @Cluster_13(%arg0: tensor<8x4096x4096xbf16>, %arg1: f32, %arg2: f64, %arg3: tensor<8x1024x4096xbf16>, %arg4: tensor<8x1024x4096xbf16>, %arg5: f32, %arg6: i64, %arg7: bf16) -> tensor<8x1024x4096xbf16> {
    %0 = linalg.generic {indexing_maps = [#map5, #map6, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg3, %arg0 : tensor<8x1024x4096xbf16>, tensor<8x4096x4096xbf16>) outs(%arg4 : tensor<8x1024x4096xbf16>) attrs =  {inner_order = [0 : ui32, 1 : ui32, 2 : ui32, 3 : ui32], outer_order = [0 : ui32, 1 : ui32, 2 : ui32, 3 : ui32], tiling_size = [1 : ui32, 1 : ui32, 1 : ui32, 1 : ui32], unroll_factor = [1 : ui32, 1 : ui32, 1 : ui32, 1 : ui32]} {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %1 = arith.mulf %in, %in_0 : bf16
      %2 = arith.addf %out, %1 : bf16
      linalg.yield %2 : bf16
    } -> tensor<8x1024x4096xbf16>
    return %arg4 : tensor<8x1024x4096xbf16>
  }
  func.func @Cluster_14(%arg0: f32, %arg1: f64, %arg2: f32, %arg3: i64, %arg4: bf16, %arg5: tensor<8x32x1024x1024xbf16>, %arg6: tensor<8x32x1024x1024xbf16>, %arg7: tensor<1x32x1x1xbf16>) -> tensor<8x32x1024x1024xbf16> {
    %0 = linalg.generic {indexing_maps = [#map, #map7, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg6, %arg7 : tensor<8x32x1024x1024xbf16>, tensor<1x32x1x1xbf16>) outs(%arg5 : tensor<8x32x1024x1024xbf16>) attrs =  {inner_order = [0 : ui32, 1 : ui32, 2 : ui32, 3 : ui32], outer_order = [0 : ui32, 1 : ui32, 2 : ui32, 3 : ui32], tiling_size = [1 : ui32, 1 : ui32, 1 : ui32, 1 : ui32], unroll_factor = [1 : ui32, 1 : ui32, 1 : ui32, 1 : ui32]} {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %1 = arith.addf %in, %in_0 : bf16
      linalg.yield %1 : bf16
    } -> tensor<8x32x1024x1024xbf16>
    return %arg5 : tensor<8x32x1024x1024xbf16>
  }
  func.func @Cluster_15(%arg0: f32, %arg1: tensor<8x8x4x1024x128xbf16>, %arg2: f64, %arg3: f32, %arg4: i64, %arg5: bf16, %arg6: tensor<8x8x1024x128xbf16>) -> tensor<8x8x4x1024x128xbf16> {
    %0 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%arg6 : tensor<8x8x1024x128xbf16>) outs(%arg1 : tensor<8x8x4x1024x128xbf16>) attrs =  {inner_order = [0 : ui32, 1 : ui32, 2 : ui32, 3 : ui32, 4 : ui32], outer_order = [0 : ui32, 1 : ui32, 2 : ui32, 3 : ui32, 4 : ui32], tiling_size = [1 : ui32, 1 : ui32, 1 : ui32, 1 : ui32, 1 : ui32], unroll_factor = [1 : ui32, 1 : ui32, 1 : ui32, 1 : ui32, 1 : ui32]} {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x8x4x1024x128xbf16>
    return %arg1 : tensor<8x8x4x1024x128xbf16>
  }
  func.func @Cluster_16(%arg0: tensor<4096x4096xbf16>, %arg1: f32, %arg2: f64, %arg3: f32, %arg4: i64, %arg5: bf16, %arg6: tensor<8x4096x4096xbf16>) -> tensor<8x4096x4096xbf16> {
    %0 = linalg.generic {indexing_maps = [#map8, #map9], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0 : tensor<4096x4096xbf16>) outs(%arg6 : tensor<8x4096x4096xbf16>) attrs =  {inner_order = [0 : ui32, 1 : ui32, 2 : ui32], outer_order = [0 : ui32, 1 : ui32, 2 : ui32], tiling_size = [1 : ui32, 1 : ui32, 1 : ui32], unroll_factor = [1 : ui32, 1 : ui32, 1 : ui32]} {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x4096x4096xbf16>
    return %arg6 : tensor<8x4096x4096xbf16>
  }
  func.func @Cluster_17(%arg0: tensor<8x32x1024x128xbf16>, %arg1: f32, %arg2: f64, %arg3: f32, %arg4: i64, %arg5: bf16, %arg6: tensor<1x1x1x128xbf16>, %arg7: tensor<8x32x1024x128xbf16>) -> tensor<8x32x1024x128xbf16> {
    %0 = linalg.generic {indexing_maps = [#map, #map10, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg6 : tensor<8x32x1024x128xbf16>, tensor<1x1x1x128xbf16>) outs(%arg7 : tensor<8x32x1024x128xbf16>) attrs =  {inner_order = [0 : ui32, 1 : ui32, 2 : ui32, 3 : ui32], outer_order = [0 : ui32, 1 : ui32, 2 : ui32, 3 : ui32], tiling_size = [1 : ui32, 1 : ui32, 1 : ui32, 1 : ui32], unroll_factor = [1 : ui32, 1 : ui32, 1 : ui32, 1 : ui32]} {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %1 = arith.mulf %in, %in_0 : bf16
      linalg.yield %1 : bf16
    } -> tensor<8x32x1024x128xbf16>
    return %arg7 : tensor<8x32x1024x128xbf16>
  }
  func.func @Cluster_18(%arg0: tensor<4096x1024xbf16>, %arg1: tensor<8x4096x1024xbf16>, %arg2: f32, %arg3: f64, %arg4: f32, %arg5: i64, %arg6: bf16) -> tensor<8x4096x1024xbf16> {
    %0 = linalg.generic {indexing_maps = [#map8, #map9], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0 : tensor<4096x1024xbf16>) outs(%arg1 : tensor<8x4096x1024xbf16>) attrs =  {inner_order = [0 : ui32, 1 : ui32, 2 : ui32], outer_order = [0 : ui32, 1 : ui32, 2 : ui32], tiling_size = [1 : ui32, 1 : ui32, 1 : ui32], unroll_factor = [1 : ui32, 1 : ui32, 1 : ui32]} {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x4096x1024xbf16>
    return %arg1 : tensor<8x4096x1024xbf16>
  }
  func.func @Cluster_19(%arg0: tensor<256x1024x1024xbf16>, %arg1: f32, %arg2: f64, %arg3: f32, %arg4: i64, %arg5: bf16, %arg6: tensor<256x1024x128xbf16>, %arg7: tensor<256x128x1024xbf16>) -> tensor<256x1024x1024xbf16> {
    %0 = linalg.generic {indexing_maps = [#map5, #map6, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg6, %arg7 : tensor<256x1024x128xbf16>, tensor<256x128x1024xbf16>) outs(%arg0 : tensor<256x1024x1024xbf16>) attrs =  {inner_order = [0 : ui32, 1 : ui32, 2 : ui32, 3 : ui32], outer_order = [0 : ui32, 1 : ui32, 2 : ui32, 3 : ui32], tiling_size = [1 : ui32, 1 : ui32, 1 : ui32, 1 : ui32], unroll_factor = [1 : ui32, 1 : ui32, 1 : ui32, 1 : ui32]} {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %1 = arith.mulf %in, %in_0 : bf16
      %2 = arith.addf %out, %1 : bf16
      linalg.yield %2 : bf16
    } -> tensor<256x1024x1024xbf16>
    return %arg0 : tensor<256x1024x1024xbf16>
  }
  func.func @Cluster_20(%arg0: tensor<8x4096x1024xbf16>, %arg1: f32, %arg2: f64, %arg3: tensor<4096x1024xbf16>, %arg4: f32, %arg5: i64, %arg6: bf16) -> tensor<8x4096x1024xbf16> {
    %0 = linalg.generic {indexing_maps = [#map8, #map9], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg3 : tensor<4096x1024xbf16>) outs(%arg0 : tensor<8x4096x1024xbf16>) attrs =  {inner_order = [0 : ui32, 1 : ui32, 2 : ui32], outer_order = [0 : ui32, 1 : ui32, 2 : ui32], tiling_size = [1 : ui32, 1 : ui32, 1 : ui32], unroll_factor = [1 : ui32, 1 : ui32, 1 : ui32]} {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x4096x1024xbf16>
    return %arg0 : tensor<8x4096x1024xbf16>
  }
  func.func @Cluster_21(%arg0: tensor<8x8x1024x128xbf16>, %arg1: f32, %arg2: f64, %arg3: tensor<8x8x1024x128xbf16>, %arg4: f32, %arg5: i64, %arg6: bf16, %arg7: tensor<1x1x1x128xbf16>) -> tensor<8x8x1024x128xbf16> {
    %0 = linalg.generic {indexing_maps = [#map, #map10, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg3, %arg7 : tensor<8x8x1024x128xbf16>, tensor<1x1x1x128xbf16>) outs(%arg0 : tensor<8x8x1024x128xbf16>) attrs =  {inner_order = [0 : ui32, 1 : ui32, 2 : ui32, 3 : ui32], outer_order = [0 : ui32, 1 : ui32, 2 : ui32, 3 : ui32], tiling_size = [1 : ui32, 1 : ui32, 1 : ui32, 1 : ui32], unroll_factor = [1 : ui32, 1 : ui32, 1 : ui32, 1 : ui32]} {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %1 = arith.mulf %in, %in_0 : bf16
      linalg.yield %1 : bf16
    } -> tensor<8x8x1024x128xbf16>
    return %arg0 : tensor<8x8x1024x128xbf16>
  }
  func.func @Cluster_22(%arg0: f32, %arg1: f64, %arg2: tensor<256x1024x128xbf16>, %arg3: tensor<256x1024x1024xbf16>, %arg4: f32, %arg5: i64, %arg6: tensor<256x1024x128xbf16>, %arg7: bf16) -> tensor<256x1024x128xbf16> {
    %0 = linalg.generic {indexing_maps = [#map5, #map6, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg3, %arg6 : tensor<256x1024x1024xbf16>, tensor<256x1024x128xbf16>) outs(%arg2 : tensor<256x1024x128xbf16>) attrs =  {inner_order = [0 : ui32, 1 : ui32, 2 : ui32, 3 : ui32], outer_order = [0 : ui32, 1 : ui32, 2 : ui32, 3 : ui32], tiling_size = [1 : ui32, 1 : ui32, 1 : ui32, 1 : ui32], unroll_factor = [1 : ui32, 1 : ui32, 1 : ui32, 1 : ui32]} {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %1 = arith.mulf %in, %in_0 : bf16
      %2 = arith.addf %out, %1 : bf16
      linalg.yield %2 : bf16
    } -> tensor<256x1024x128xbf16>
    return %arg2 : tensor<256x1024x128xbf16>
  }
  func.func @Cluster_23(%arg0: f32, %arg1: tensor<4096x4096xbf16>, %arg2: f64, %arg3: f32, %arg4: i64, %arg5: bf16, %arg6: tensor<8x4096x4096xbf16>) -> tensor<8x4096x4096xbf16> {
    %0 = linalg.generic {indexing_maps = [#map8, #map9], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg1 : tensor<4096x4096xbf16>) outs(%arg6 : tensor<8x4096x4096xbf16>) attrs =  {inner_order = [0 : ui32, 1 : ui32, 2 : ui32], outer_order = [0 : ui32, 1 : ui32, 2 : ui32], tiling_size = [1 : ui32, 1 : ui32, 1 : ui32], unroll_factor = [1 : ui32, 1 : ui32, 1 : ui32]} {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x4096x4096xbf16>
    return %arg6 : tensor<8x4096x4096xbf16>
  }
  func.func @Cluster_24(%arg0: tensor<8x32x1024x128xbf16>, %arg1: f32, %arg2: tensor<8x32x128x1024xbf16>, %arg3: f64, %arg4: f32, %arg5: i64, %arg6: bf16) -> tensor<8x32x128x1024xbf16> {
    %0 = linalg.generic {indexing_maps = [#map11, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<8x32x1024x128xbf16>) outs(%arg2 : tensor<8x32x128x1024xbf16>) attrs =  {inner_order = [0 : ui32, 1 : ui32, 2 : ui32, 3 : ui32], outer_order = [0 : ui32, 1 : ui32, 2 : ui32, 3 : ui32], tiling_size = [1 : ui32, 1 : ui32, 1 : ui32, 1 : ui32], unroll_factor = [1 : ui32, 1 : ui32, 1 : ui32, 1 : ui32]} {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x32x128x1024xbf16>
    return %arg2 : tensor<8x32x128x1024xbf16>
  }
  func.func @Cluster_25(%arg0: tensor<8x32x1024x128xbf16>, %arg1: f32, %arg2: f64, %arg3: f32, %arg4: i64, %arg5: bf16, %arg6: tensor<1x1x1x128xbf16>, %arg7: tensor<8x32x1024x128xbf16>) -> tensor<8x32x1024x128xbf16> {
    %0 = linalg.generic {indexing_maps = [#map, #map10, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg6 : tensor<8x32x1024x128xbf16>, tensor<1x1x1x128xbf16>) outs(%arg7 : tensor<8x32x1024x128xbf16>) attrs =  {inner_order = [0 : ui32, 1 : ui32, 2 : ui32, 3 : ui32], outer_order = [0 : ui32, 1 : ui32, 2 : ui32, 3 : ui32], tiling_size = [1 : ui32, 1 : ui32, 1 : ui32, 1 : ui32], unroll_factor = [1 : ui32, 1 : ui32, 1 : ui32, 1 : ui32]} {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %1 = arith.mulf %in, %in_0 : bf16
      linalg.yield %1 : bf16
    } -> tensor<8x32x1024x128xbf16>
    return %arg7 : tensor<8x32x1024x128xbf16>
  }
  func.func @Cluster_26(%arg0: f32, %arg1: f64, %arg2: f32, %arg3: i64, %arg4: tensor<8x32x1024x1024xbf16>, %arg5: bf16, %arg6: tensor<8x32x1024x1024xbf16>) -> tensor<8x32x1024x1024xbf16> {
    %0 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg4 : tensor<8x32x1024x1024xbf16>) outs(%arg6 : tensor<8x32x1024x1024xbf16>) attrs =  {inner_order = [0 : ui32, 1 : ui32, 2 : ui32, 3 : ui32], outer_order = [0 : ui32, 1 : ui32, 2 : ui32, 3 : ui32], tiling_size = [1 : ui32, 1 : ui32, 1 : ui32, 1 : ui32], unroll_factor = [1 : ui32, 1 : ui32, 1 : ui32, 1 : ui32]} {
    ^bb0(%in: bf16, %out: bf16):
      %1 = arith.truncf %arg1 : f64 to bf16
      %2 = arith.mulf %in, %1 : bf16
      linalg.yield %2 : bf16
    } -> tensor<8x32x1024x1024xbf16>
    return %arg6 : tensor<8x32x1024x1024xbf16>
  }
  func.func @Cluster_27(%arg0: tensor<8x32x1024x64xbf16>, %arg1: f32, %arg2: f64, %arg3: f32, %arg4: tensor<8x32x1024x64xbf16>, %arg5: i64, %arg6: bf16) -> tensor<8x32x1024x64xbf16> {
    %0 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<8x32x1024x64xbf16>) outs(%arg4 : tensor<8x32x1024x64xbf16>) attrs =  {inner_order = [0 : ui32, 1 : ui32, 2 : ui32, 3 : ui32], outer_order = [0 : ui32, 1 : ui32, 2 : ui32, 3 : ui32], tiling_size = [1 : ui32, 1 : ui32, 1 : ui32, 1 : ui32], unroll_factor = [1 : ui32, 1 : ui32, 1 : ui32, 1 : ui32]} {
    ^bb0(%in: bf16, %out: bf16):
      %1 = arith.negf %in : bf16
      linalg.yield %1 : bf16
    } -> tensor<8x32x1024x64xbf16>
    return %arg4 : tensor<8x32x1024x64xbf16>
  }
  func.func @Cluster_28(%arg0: f32, %arg1: f64, %arg2: tensor<8x8x1024x64xbf16>, %arg3: f32, %arg4: i64, %arg5: bf16, %arg6: tensor<8x8x1024x64xbf16>) -> tensor<8x8x1024x64xbf16> {
    %0 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2 : tensor<8x8x1024x64xbf16>) outs(%arg6 : tensor<8x8x1024x64xbf16>) attrs =  {inner_order = [0 : ui32, 1 : ui32, 2 : ui32, 3 : ui32], outer_order = [0 : ui32, 1 : ui32, 2 : ui32, 3 : ui32], tiling_size = [1 : ui32, 1 : ui32, 1 : ui32, 1 : ui32], unroll_factor = [1 : ui32, 1 : ui32, 1 : ui32, 1 : ui32]} {
    ^bb0(%in: bf16, %out: bf16):
      %1 = arith.negf %in : bf16
      linalg.yield %1 : bf16
    } -> tensor<8x8x1024x64xbf16>
    return %arg6 : tensor<8x8x1024x64xbf16>
  }
  func.func @Cluster_29(%arg0: tensor<8x8x1024x128xbf16>, %arg1: f32, %arg2: f64, %arg3: f32, %arg4: i64, %arg5: tensor<8x8x1024x128xbf16>, %arg6: bf16, %arg7: tensor<1x1x1x128xbf16>) -> tensor<8x8x1024x128xbf16> {
    %0 = linalg.generic {indexing_maps = [#map, #map10, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg5, %arg7 : tensor<8x8x1024x128xbf16>, tensor<1x1x1x128xbf16>) outs(%arg0 : tensor<8x8x1024x128xbf16>) attrs =  {inner_order = [0 : ui32, 1 : ui32, 2 : ui32, 3 : ui32], outer_order = [0 : ui32, 1 : ui32, 2 : ui32, 3 : ui32], tiling_size = [1 : ui32, 1 : ui32, 1 : ui32, 1 : ui32], unroll_factor = [1 : ui32, 1 : ui32, 1 : ui32, 1 : ui32]} {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %1 = arith.mulf %in, %in_0 : bf16
      linalg.yield %1 : bf16
    } -> tensor<8x8x1024x128xbf16>
    return %arg0 : tensor<8x8x1024x128xbf16>
  }
  func.func @Cluster_30(%arg0: tensor<8x32x1024xf32>, %arg1: f32, %arg2: f64, %arg3: f32, %arg4: i64, %arg5: bf16) -> tensor<8x32x1024xf32> {
    %0 = linalg.generic {indexing_maps = [#map12, #map9], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg1 : f32) outs(%arg0 : tensor<8x32x1024xf32>) attrs =  {inner_order = [0 : ui32, 1 : ui32, 2 : ui32], outer_order = [0 : ui32, 1 : ui32, 2 : ui32], tiling_size = [1 : ui32, 1 : ui32, 1 : ui32], unroll_factor = [1 : ui32, 1 : ui32, 1 : ui32]} {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<8x32x1024xf32>
    return %arg0 : tensor<8x32x1024xf32>
  }
  func.func @Cluster_31(%arg0: tensor<8x8x1024x128xbf16>, %arg1: f32, %arg2: f64, %arg3: f32, %arg4: i64, %arg5: bf16, %arg6: tensor<8x1024x8x128xbf16>) -> tensor<8x8x1024x128xbf16> {
    %0 = linalg.generic {indexing_maps = [#map13, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg6 : tensor<8x1024x8x128xbf16>) outs(%arg0 : tensor<8x8x1024x128xbf16>) attrs =  {inner_order = [0 : ui32, 1 : ui32, 2 : ui32, 3 : ui32], outer_order = [0 : ui32, 1 : ui32, 2 : ui32, 3 : ui32], tiling_size = [1 : ui32, 1 : ui32, 1 : ui32, 1 : ui32], unroll_factor = [1 : ui32, 1 : ui32, 1 : ui32, 1 : ui32]} {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x8x1024x128xbf16>
    return %arg0 : tensor<8x8x1024x128xbf16>
  }
  func.func @Cluster_32(%arg0: tensor<4096x4096xbf16>, %arg1: f32, %arg2: f64, %arg3: f32, %arg4: i64, %arg5: bf16, %arg6: tensor<4096x4096xbf16>) -> tensor<4096x4096xbf16> {
    %0 = linalg.generic {indexing_maps = [#map14, #map15], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<4096x4096xbf16>) outs(%arg6 : tensor<4096x4096xbf16>) attrs =  {inner_order = [0 : ui32, 1 : ui32], outer_order = [0 : ui32, 1 : ui32], tiling_size = [1 : ui32, 1 : ui32], unroll_factor = [1 : ui32, 1 : ui32]} {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    return %arg6 : tensor<4096x4096xbf16>
  }
  func.func @Cluster_33(%arg0: f32, %arg1: f64, %arg2: tensor<8x1024x4096xbf16>, %arg3: f32, %arg4: i64, %arg5: bf16) -> tensor<8x1024x4096xbf16> {
    %0 = linalg.generic {indexing_maps = [#map12, #map9], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg5 : bf16) outs(%arg2 : tensor<8x1024x4096xbf16>) attrs =  {inner_order = [0 : ui32, 1 : ui32, 2 : ui32], outer_order = [0 : ui32, 1 : ui32, 2 : ui32], tiling_size = [1 : ui32, 1 : ui32, 1 : ui32], unroll_factor = [1 : ui32, 1 : ui32, 1 : ui32]} {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x1024x4096xbf16>
    return %arg2 : tensor<8x1024x4096xbf16>
  }
  func.func @Cluster_34(%arg0: f32, %arg1: f64, %arg2: tensor<8x1024x32x128xbf16>, %arg3: f32, %arg4: i64, %arg5: bf16, %arg6: tensor<8x32x1024x128xbf16>) -> tensor<8x32x1024x128xbf16> {
    %0 = linalg.generic {indexing_maps = [#map13, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2 : tensor<8x1024x32x128xbf16>) outs(%arg6 : tensor<8x32x1024x128xbf16>) attrs =  {inner_order = [0 : ui32, 1 : ui32, 2 : ui32, 3 : ui32], outer_order = [0 : ui32, 1 : ui32, 2 : ui32, 3 : ui32], tiling_size = [1 : ui32, 1 : ui32, 1 : ui32, 1 : ui32], unroll_factor = [1 : ui32, 1 : ui32, 1 : ui32, 1 : ui32]} {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x32x1024x128xbf16>
    return %arg6 : tensor<8x32x1024x128xbf16>
  }
  func.func @Cluster_35(%arg0: tensor<4096x1024xbf16>, %arg1: f32, %arg2: tensor<1024x4096xbf16>, %arg3: f64, %arg4: f32, %arg5: i64, %arg6: bf16) -> tensor<4096x1024xbf16> {
    %0 = linalg.generic {indexing_maps = [#map14, #map15], iterator_types = ["parallel", "parallel"]} ins(%arg2 : tensor<1024x4096xbf16>) outs(%arg0 : tensor<4096x1024xbf16>) attrs =  {inner_order = [0 : ui32, 1 : ui32], outer_order = [0 : ui32, 1 : ui32], tiling_size = [1 : ui32, 1 : ui32], unroll_factor = [1 : ui32, 1 : ui32]} {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x1024xbf16>
    return %arg0 : tensor<4096x1024xbf16>
  }
  func.func @Cluster_36(%arg0: f32, %arg1: f64, %arg2: f32, %arg3: i64, %arg4: bf16, %arg5: tensor<256x1024x1024xbf16>) -> tensor<256x1024x1024xbf16> {
    %0 = linalg.generic {indexing_maps = [#map12, #map9], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg4 : bf16) outs(%arg5 : tensor<256x1024x1024xbf16>) attrs =  {inner_order = [0 : ui32, 1 : ui32, 2 : ui32], outer_order = [0 : ui32, 1 : ui32, 2 : ui32], tiling_size = [1 : ui32, 1 : ui32, 1 : ui32], unroll_factor = [1 : ui32, 1 : ui32, 1 : ui32]} {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<256x1024x1024xbf16>
    return %arg5 : tensor<256x1024x1024xbf16>
  }
  func.func @Cluster_37(%arg0: f32, %arg1: tensor<8x32x1024x1xf32>, %arg2: f64, %arg3: f32, %arg4: i64, %arg5: bf16) -> tensor<8x32x1024x1xf32> {
    %0 = linalg.generic {indexing_maps = [#map16, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg3 : f32) outs(%arg1 : tensor<8x32x1024x1xf32>) attrs =  {inner_order = [0 : ui32, 1 : ui32, 2 : ui32, 3 : ui32], outer_order = [0 : ui32, 1 : ui32, 2 : ui32, 3 : ui32], tiling_size = [1 : ui32, 1 : ui32, 1 : ui32, 1 : ui32], unroll_factor = [1 : ui32, 1 : ui32, 1 : ui32, 1 : ui32]} {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<8x32x1024x1xf32>
    return %arg1 : tensor<8x32x1024x1xf32>
  }
  func.func @Cluster_38(%arg0: tensor<8x1024x1024xbf16>, %arg1: f32, %arg2: f64, %arg3: f32, %arg4: i64, %arg5: bf16) -> tensor<8x1024x1024xbf16> {
    %0 = linalg.generic {indexing_maps = [#map12, #map9], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg5 : bf16) outs(%arg0 : tensor<8x1024x1024xbf16>) attrs =  {inner_order = [0 : ui32, 1 : ui32, 2 : ui32], outer_order = [0 : ui32, 1 : ui32, 2 : ui32], tiling_size = [1 : ui32, 1 : ui32, 1 : ui32], unroll_factor = [1 : ui32, 1 : ui32, 1 : ui32]} {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x1024x1024xbf16>
    return %arg0 : tensor<8x1024x1024xbf16>
  }
  func.func @Cluster_39(%arg0: tensor<4096x1024xbf16>, %arg1: tensor<1024x4096xbf16>, %arg2: f32, %arg3: f64, %arg4: f32, %arg5: i64, %arg6: bf16) -> tensor<4096x1024xbf16> {
    %0 = linalg.generic {indexing_maps = [#map14, #map15], iterator_types = ["parallel", "parallel"]} ins(%arg1 : tensor<1024x4096xbf16>) outs(%arg0 : tensor<4096x1024xbf16>) attrs =  {inner_order = [0 : ui32, 1 : ui32], outer_order = [0 : ui32, 1 : ui32], tiling_size = [1 : ui32, 1 : ui32], unroll_factor = [1 : ui32, 1 : ui32]} {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x1024xbf16>
    return %arg0 : tensor<4096x1024xbf16>
  }
  func.func @Cluster_40(%arg0: tensor<8x8x1024x128xbf16>, %arg1: tensor<8x1024x8x128xbf16>, %arg2: f32, %arg3: f64, %arg4: f32, %arg5: i64, %arg6: bf16) -> tensor<8x8x1024x128xbf16> {
    %0 = linalg.generic {indexing_maps = [#map13, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg1 : tensor<8x1024x8x128xbf16>) outs(%arg0 : tensor<8x8x1024x128xbf16>) attrs =  {inner_order = [0 : ui32, 1 : ui32, 2 : ui32, 3 : ui32], outer_order = [0 : ui32, 1 : ui32, 2 : ui32, 3 : ui32], tiling_size = [1 : ui32, 1 : ui32, 1 : ui32, 1 : ui32], unroll_factor = [1 : ui32, 1 : ui32, 1 : ui32, 1 : ui32]} {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x8x1024x128xbf16>
    return %arg0 : tensor<8x8x1024x128xbf16>
  }
  func.func @Cluster_41(%arg0: tensor<8x32x1024xi64>, %arg1: f32, %arg2: f64, %arg3: f32, %arg4: i64, %arg5: bf16) -> tensor<8x32x1024xi64> {
    %0 = linalg.generic {indexing_maps = [#map12, #map9], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg4 : i64) outs(%arg0 : tensor<8x32x1024xi64>) attrs =  {inner_order = [0 : ui32, 1 : ui32, 2 : ui32], outer_order = [0 : ui32, 1 : ui32, 2 : ui32], tiling_size = [1 : ui32, 1 : ui32, 1 : ui32], unroll_factor = [1 : ui32, 1 : ui32, 1 : ui32]} {
    ^bb0(%in: i64, %out: i64):
      linalg.yield %in : i64
    } -> tensor<8x32x1024xi64>
    return %arg0 : tensor<8x32x1024xi64>
  }
  func.func @Cluster_42(%arg0: f32, %arg1: f64, %arg2: f32, %arg3: i64, %arg4: bf16, %arg5: tensor<256x1024x128xbf16>) -> tensor<256x1024x128xbf16> {
    %0 = linalg.generic {indexing_maps = [#map12, #map9], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg4 : bf16) outs(%arg5 : tensor<256x1024x128xbf16>) attrs =  {inner_order = [0 : ui32, 1 : ui32, 2 : ui32], outer_order = [0 : ui32, 1 : ui32, 2 : ui32], tiling_size = [1 : ui32, 1 : ui32, 1 : ui32], unroll_factor = [1 : ui32, 1 : ui32, 1 : ui32]} {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<256x1024x128xbf16>
    return %arg5 : tensor<256x1024x128xbf16>
  }
  func.func @Cluster_43(%arg0: f32, %arg1: f64, %arg2: f32, %arg3: i64, %arg4: bf16, %arg5: tensor<8x32x1024x128xbf16>, %arg6: tensor<8x1024x32x128xbf16>) -> tensor<8x1024x32x128xbf16> {
    %0 = linalg.generic {indexing_maps = [#map13, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg5 : tensor<8x32x1024x128xbf16>) outs(%arg6 : tensor<8x1024x32x128xbf16>) attrs =  {inner_order = [0 : ui32, 1 : ui32, 2 : ui32, 3 : ui32], outer_order = [0 : ui32, 1 : ui32, 2 : ui32, 3 : ui32], tiling_size = [1 : ui32, 1 : ui32, 1 : ui32, 1 : ui32], unroll_factor = [1 : ui32, 1 : ui32, 1 : ui32, 1 : ui32]} {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<8x1024x32x128xbf16>
    return %arg6 : tensor<8x1024x32x128xbf16>
  }
  func.func @Cluster_44(%arg0: f32, %arg1: f64, %arg2: f32, %arg3: i64, %arg4: bf16, %arg5: tensor<4096x4096xbf16>, %arg6: tensor<4096x4096xbf16>) -> tensor<4096x4096xbf16> {
    %0 = linalg.generic {indexing_maps = [#map14, #map15], iterator_types = ["parallel", "parallel"]} ins(%arg5 : tensor<4096x4096xbf16>) outs(%arg6 : tensor<4096x4096xbf16>) attrs =  {inner_order = [0 : ui32, 1 : ui32], outer_order = [0 : ui32, 1 : ui32], tiling_size = [1 : ui32, 1 : ui32], unroll_factor = [1 : ui32, 1 : ui32]} {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4096x4096xbf16>
    return %arg6 : tensor<4096x4096xbf16>
  }
}
