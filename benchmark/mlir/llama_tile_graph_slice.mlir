#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>

module {
  func.func @main(%arg0: tensor<256x1024x128xbf16>, %arg1: tensor<256x128x1024xbf16>, %arg2: tensor<256x1024x128xbf16>) -> tensor<256x1024x128xbf16> {
    %0 = "tile_graph.data_path"(%arg0) <{forward_path = "ddr"}> : (tensor<256x1024x128xbf16>) -> tensor<256x1024x128xbf16>
    %1 = "tile_graph.data_path"(%arg1) <{forward_path = "ddr"}> : (tensor<256x128x1024xbf16>) -> tensor<256x128x1024xbf16>
    %2 = "tile_graph.compute_node"(%arg0, %arg1) <{indexing_maps = [#map4, #map5, #map6], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], l1_order = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], l2_order = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], tiling_vector = [1, 6, 6, 7], unroll_vector = [1, 6, 6, 1]}> : (tensor<256x1024x128xbf16>, tensor<256x128x1024xbf16>) -> tensor<256x1024x1024xbf16>
    %3 = "tile_graph.data_path"(%2) <{forward_path = "sram"}> : (tensor<256x1024x1024xbf16>) -> tensor<256x1024x1024xbf16>
    %4 = "tile_graph.data_path"(%arg2) <{forward_path = "ddr"}> : (tensor<256x1024x128xbf16>) -> tensor<256x1024x128xbf16>
    %5 = "tile_graph.compute_node"(%3, %4) <{indexing_maps = [#map4, #map5, #map6], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], l1_order = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], l2_order = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], tiling_vector = [1, 6, 7, 6], unroll_vector = [1, 6, 1, 6]}> : (tensor<256x1024x1024xbf16>, tensor<256x1024x128xbf16>) -> tensor<256x1024x128xbf16>
    return %5 : tensor<8x1024x4096xbf16>
  }
}
