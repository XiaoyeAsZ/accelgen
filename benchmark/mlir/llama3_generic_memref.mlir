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
#map10 = affine_map<(d0, d1, d2, d3) -> (0, 0, 0, d3)>
#map11 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>
#map12 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map13 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#map14 = affine_map<(d0, d1, d2, d3) -> (0, d1, 0, 0)>
#map15 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>
#map16 = affine_map<(d0, d1, d2, d3) -> ()>
"builtin.module"() ({
  "func.func"() <{function_type = (tensor<8x1024x4096xbf16>, tensor<1x1x128xbf16>, tensor<1x1x128xbf16>, tensor<1x32x1x1xbf16>, tensor<4096x4096xbf16>, tensor<1024x4096xbf16>, tensor<1024x4096xbf16>, tensor<4096x4096xbf16>) -> (tensor<8x1024x4096xbf16>, tensor<8x32x1024x1024xbf16>), sym_name = "main"}> ({
  ^bb0(%arg166: tensor<8x1024x4096xbf16>, %arg167: tensor<1x1x128xbf16>, %arg168: tensor<1x1x128xbf16>, %arg169: tensor<1x32x1x1xbf16>, %arg170: tensor<4096x4096xbf16>, %arg171: tensor<1024x4096xbf16>, %arg172: tensor<1024x4096xbf16>, %arg173: tensor<4096x4096xbf16>):
    %79 = "arith.constant"() <{value = 0 : i64}> : () -> i64
    %80 = "arith.constant"() <{value = 0.000000e+00 : bf16}> : () -> bf16
    %81 = "arith.constant"() <{value = 0xFF800000 : f32}> : () -> f32
    %82 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
    %83 = "arith.constant"() <{value = 0.088388347648318447 : f64}> : () -> f64
    %84 = "tensor.empty"() : () -> tensor<4096x4096xbf16>
    %85 = "linalg.generic"(%arg170, %84) <{indexing_maps = [#map, #map1], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg278: bf16, %arg279: bf16):
      "linalg.yield"(%arg278) : (bf16) -> ()
    }) : (tensor<4096x4096xbf16>, tensor<4096x4096xbf16>) -> tensor<4096x4096xbf16>
    %86 = "tensor.empty"() : () -> tensor<8x4096x4096xbf16>
    %87 = "linalg.generic"(%85, %86) <{indexing_maps = [#map2, #map3], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg276: bf16, %arg277: bf16):
      "linalg.yield"(%arg276) : (bf16) -> ()
    }) : (tensor<4096x4096xbf16>, tensor<8x4096x4096xbf16>) -> tensor<8x4096x4096xbf16>
    %88 = "tensor.empty"() : () -> tensor<8x1024x4096xbf16>
    %89 = "linalg.generic"(%80, %88) <{indexing_maps = [#map4, #map3], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg274: bf16, %arg275: bf16):
      "linalg.yield"(%arg274) : (bf16) -> ()
    }) : (bf16, tensor<8x1024x4096xbf16>) -> tensor<8x1024x4096xbf16>
    %90 = "linalg.generic"(%arg166, %87, %89) <{indexing_maps = [#map5, #map6, #map7], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg271: bf16, %arg272: bf16, %arg273: bf16):
      %201 = "arith.mulf"(%arg271, %arg272) <{fastmath = #arith.fastmath<none>}> : (bf16, bf16) -> bf16
      %202 = "arith.addf"(%arg273, %201) <{fastmath = #arith.fastmath<none>}> : (bf16, bf16) -> bf16
      "linalg.yield"(%202) : (bf16) -> ()
    }) : (tensor<8x1024x4096xbf16>, tensor<8x4096x4096xbf16>, tensor<8x1024x4096xbf16>) -> tensor<8x1024x4096xbf16>
    %91 = "tensor.expand_shape"(%90) <{reassociation = [[0], [1], [2, 3]], static_output_shape = array<i64: 8, 1024, 32, 128>}> : (tensor<8x1024x4096xbf16>) -> tensor<8x1024x32x128xbf16>
    %92 = "tensor.empty"() : () -> tensor<8x32x1024x128xbf16>
    %93 = "linalg.generic"(%91, %92) <{indexing_maps = [#map8, #map9], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg269: bf16, %arg270: bf16):
      "linalg.yield"(%arg269) : (bf16) -> ()
    }) : (tensor<8x1024x32x128xbf16>, tensor<8x32x1024x128xbf16>) -> tensor<8x32x1024x128xbf16>
    %94 = "tensor.empty"() : () -> tensor<4096x1024xbf16>
    %95 = "linalg.generic"(%arg171, %94) <{indexing_maps = [#map, #map1], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg267: bf16, %arg268: bf16):
      "linalg.yield"(%arg267) : (bf16) -> ()
    }) : (tensor<1024x4096xbf16>, tensor<4096x1024xbf16>) -> tensor<4096x1024xbf16>
    %96 = "tensor.empty"() : () -> tensor<8x4096x1024xbf16>
    %97 = "linalg.generic"(%95, %96) <{indexing_maps = [#map2, #map3], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg265: bf16, %arg266: bf16):
      "linalg.yield"(%arg265) : (bf16) -> ()
    }) : (tensor<4096x1024xbf16>, tensor<8x4096x1024xbf16>) -> tensor<8x4096x1024xbf16>
    %98 = "tensor.empty"() : () -> tensor<8x1024x1024xbf16>
    %99 = "linalg.generic"(%80, %98) <{indexing_maps = [#map4, #map3], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg263: bf16, %arg264: bf16):
      "linalg.yield"(%arg263) : (bf16) -> ()
    }) : (bf16, tensor<8x1024x1024xbf16>) -> tensor<8x1024x1024xbf16>
    %100 = "linalg.generic"(%arg166, %97, %99) <{indexing_maps = [#map5, #map6, #map7], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg260: bf16, %arg261: bf16, %arg262: bf16):
      %199 = "arith.mulf"(%arg260, %arg261) <{fastmath = #arith.fastmath<none>}> : (bf16, bf16) -> bf16
      %200 = "arith.addf"(%arg262, %199) <{fastmath = #arith.fastmath<none>}> : (bf16, bf16) -> bf16
      "linalg.yield"(%200) : (bf16) -> ()
    }) : (tensor<8x1024x4096xbf16>, tensor<8x4096x1024xbf16>, tensor<8x1024x1024xbf16>) -> tensor<8x1024x1024xbf16>
    %101 = "tensor.expand_shape"(%100) <{reassociation = [[0], [1], [2, 3]], static_output_shape = array<i64: 8, 1024, 8, 128>}> : (tensor<8x1024x1024xbf16>) -> tensor<8x1024x8x128xbf16>
    %102 = "tensor.empty"() : () -> tensor<8x8x1024x128xbf16>
    %103 = "linalg.generic"(%101, %102) <{indexing_maps = [#map8, #map9], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg258: bf16, %arg259: bf16):
      "linalg.yield"(%arg258) : (bf16) -> ()
    }) : (tensor<8x1024x8x128xbf16>, tensor<8x8x1024x128xbf16>) -> tensor<8x8x1024x128xbf16>
    %104 = "linalg.generic"(%arg172, %94) <{indexing_maps = [#map, #map1], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg256: bf16, %arg257: bf16):
      "linalg.yield"(%arg256) : (bf16) -> ()
    }) : (tensor<1024x4096xbf16>, tensor<4096x1024xbf16>) -> tensor<4096x1024xbf16>
    %105 = "linalg.generic"(%104, %96) <{indexing_maps = [#map2, #map3], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg254: bf16, %arg255: bf16):
      "linalg.yield"(%arg254) : (bf16) -> ()
    }) : (tensor<4096x1024xbf16>, tensor<8x4096x1024xbf16>) -> tensor<8x4096x1024xbf16>
    %106 = "linalg.generic"(%arg166, %105, %99) <{indexing_maps = [#map5, #map6, #map7], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg251: bf16, %arg252: bf16, %arg253: bf16):
      %197 = "arith.mulf"(%arg251, %arg252) <{fastmath = #arith.fastmath<none>}> : (bf16, bf16) -> bf16
      %198 = "arith.addf"(%arg253, %197) <{fastmath = #arith.fastmath<none>}> : (bf16, bf16) -> bf16
      "linalg.yield"(%198) : (bf16) -> ()
    }) : (tensor<8x1024x4096xbf16>, tensor<8x4096x1024xbf16>, tensor<8x1024x1024xbf16>) -> tensor<8x1024x1024xbf16>
    %107 = "tensor.expand_shape"(%106) <{reassociation = [[0], [1], [2, 3]], static_output_shape = array<i64: 8, 1024, 8, 128>}> : (tensor<8x1024x1024xbf16>) -> tensor<8x1024x8x128xbf16>
    %108 = "linalg.generic"(%107, %102) <{indexing_maps = [#map8, #map9], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg249: bf16, %arg250: bf16):
      "linalg.yield"(%arg249) : (bf16) -> ()
    }) : (tensor<8x1024x8x128xbf16>, tensor<8x8x1024x128xbf16>) -> tensor<8x8x1024x128xbf16>
    %109 = "tensor.expand_shape"(%arg167) <{reassociation = [[0], [1, 2], [3]], static_output_shape = array<i64: 1, 1, 1, 128>}> : (tensor<1x1x128xbf16>) -> tensor<1x1x1x128xbf16>
    %110 = "tensor.expand_shape"(%arg168) <{reassociation = [[0], [1, 2], [3]], static_output_shape = array<i64: 1, 1, 1, 128>}> : (tensor<1x1x128xbf16>) -> tensor<1x1x1x128xbf16>
    %111 = "linalg.generic"(%93, %109, %92) <{indexing_maps = [#map9, #map10, #map9], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg246: bf16, %arg247: bf16, %arg248: bf16):
      %196 = "arith.mulf"(%arg246, %arg247) <{fastmath = #arith.fastmath<none>}> : (bf16, bf16) -> bf16
      "linalg.yield"(%196) : (bf16) -> ()
    }) : (tensor<8x32x1024x128xbf16>, tensor<1x1x1x128xbf16>, tensor<8x32x1024x128xbf16>) -> tensor<8x32x1024x128xbf16>
    %112 = "tensor.extract_slice"(%93) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 8, 32, 1024, 64>, static_strides = array<i64: 1, 1, 1, 1>}> : (tensor<8x32x1024x128xbf16>) -> tensor<8x32x1024x64xbf16>
    %113 = "tensor.extract_slice"(%93) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0, 0, 64>, static_sizes = array<i64: 8, 32, 1024, 64>, static_strides = array<i64: 1, 1, 1, 1>}> : (tensor<8x32x1024x128xbf16>) -> tensor<8x32x1024x64xbf16>
    %114 = "tensor.empty"() : () -> tensor<8x32x1024x64xbf16>
    %115 = "linalg.generic"(%113, %114) <{indexing_maps = [#map9, #map9], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg244: bf16, %arg245: bf16):
      %195 = "arith.negf"(%arg244) <{fastmath = #arith.fastmath<none>}> : (bf16) -> bf16
      "linalg.yield"(%195) : (bf16) -> ()
    }) : (tensor<8x32x1024x64xbf16>, tensor<8x32x1024x64xbf16>) -> tensor<8x32x1024x64xbf16>
    %116 = "tensor.concat"(%115, %112) <{dim = 3 : i64}> : (tensor<8x32x1024x64xbf16>, tensor<8x32x1024x64xbf16>) -> tensor<8x32x1024x128xbf16>
    %117 = "linalg.generic"(%116, %110, %92) <{indexing_maps = [#map9, #map10, #map9], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg241: bf16, %arg242: bf16, %arg243: bf16):
      %194 = "arith.mulf"(%arg241, %arg242) <{fastmath = #arith.fastmath<none>}> : (bf16, bf16) -> bf16
      "linalg.yield"(%194) : (bf16) -> ()
    }) : (tensor<8x32x1024x128xbf16>, tensor<1x1x1x128xbf16>, tensor<8x32x1024x128xbf16>) -> tensor<8x32x1024x128xbf16>
    %118 = "linalg.generic"(%111, %117, %92) <{indexing_maps = [#map9, #map9, #map9], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg238: bf16, %arg239: bf16, %arg240: bf16):
      %193 = "arith.addf"(%arg238, %arg239) <{fastmath = #arith.fastmath<none>}> : (bf16, bf16) -> bf16
      "linalg.yield"(%193) : (bf16) -> ()
    }) : (tensor<8x32x1024x128xbf16>, tensor<8x32x1024x128xbf16>, tensor<8x32x1024x128xbf16>) -> tensor<8x32x1024x128xbf16>
    %119 = "linalg.generic"(%103, %109, %102) <{indexing_maps = [#map9, #map10, #map9], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg235: bf16, %arg236: bf16, %arg237: bf16):
      %192 = "arith.mulf"(%arg235, %arg236) <{fastmath = #arith.fastmath<none>}> : (bf16, bf16) -> bf16
      "linalg.yield"(%192) : (bf16) -> ()
    }) : (tensor<8x8x1024x128xbf16>, tensor<1x1x1x128xbf16>, tensor<8x8x1024x128xbf16>) -> tensor<8x8x1024x128xbf16>
    %120 = "tensor.extract_slice"(%103) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 8, 8, 1024, 64>, static_strides = array<i64: 1, 1, 1, 1>}> : (tensor<8x8x1024x128xbf16>) -> tensor<8x8x1024x64xbf16>
    %121 = "tensor.extract_slice"(%103) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0, 0, 64>, static_sizes = array<i64: 8, 8, 1024, 64>, static_strides = array<i64: 1, 1, 1, 1>}> : (tensor<8x8x1024x128xbf16>) -> tensor<8x8x1024x64xbf16>
    %122 = "tensor.empty"() : () -> tensor<8x8x1024x64xbf16>
    %123 = "linalg.generic"(%121, %122) <{indexing_maps = [#map9, #map9], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg233: bf16, %arg234: bf16):
      %191 = "arith.negf"(%arg233) <{fastmath = #arith.fastmath<none>}> : (bf16) -> bf16
      "linalg.yield"(%191) : (bf16) -> ()
    }) : (tensor<8x8x1024x64xbf16>, tensor<8x8x1024x64xbf16>) -> tensor<8x8x1024x64xbf16>
    %124 = "tensor.concat"(%123, %120) <{dim = 3 : i64}> : (tensor<8x8x1024x64xbf16>, tensor<8x8x1024x64xbf16>) -> tensor<8x8x1024x128xbf16>
    %125 = "linalg.generic"(%124, %110, %102) <{indexing_maps = [#map9, #map10, #map9], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg230: bf16, %arg231: bf16, %arg232: bf16):
      %190 = "arith.mulf"(%arg230, %arg231) <{fastmath = #arith.fastmath<none>}> : (bf16, bf16) -> bf16
      "linalg.yield"(%190) : (bf16) -> ()
    }) : (tensor<8x8x1024x128xbf16>, tensor<1x1x1x128xbf16>, tensor<8x8x1024x128xbf16>) -> tensor<8x8x1024x128xbf16>
    %126 = "linalg.generic"(%119, %125, %102) <{indexing_maps = [#map9, #map9, #map9], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg227: bf16, %arg228: bf16, %arg229: bf16):
      %189 = "arith.addf"(%arg227, %arg228) <{fastmath = #arith.fastmath<none>}> : (bf16, bf16) -> bf16
      "linalg.yield"(%189) : (bf16) -> ()
    }) : (tensor<8x8x1024x128xbf16>, tensor<8x8x1024x128xbf16>, tensor<8x8x1024x128xbf16>) -> tensor<8x8x1024x128xbf16>
    %127 = "tensor.empty"() : () -> tensor<8x8x4x1024x128xbf16>
    %128 = "linalg.generic"(%126, %127) <{indexing_maps = [#map11, #map12], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg225: bf16, %arg226: bf16):
      "linalg.yield"(%arg225) : (bf16) -> ()
    }) : (tensor<8x8x1024x128xbf16>, tensor<8x8x4x1024x128xbf16>) -> tensor<8x8x4x1024x128xbf16>
    %129 = "tensor.collapse_shape"(%128) <{reassociation = [[0], [1, 2], [3], [4]]}> : (tensor<8x8x4x1024x128xbf16>) -> tensor<8x32x1024x128xbf16>
    %130 = "linalg.generic"(%108, %127) <{indexing_maps = [#map11, #map12], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg223: bf16, %arg224: bf16):
      "linalg.yield"(%arg223) : (bf16) -> ()
    }) : (tensor<8x8x1024x128xbf16>, tensor<8x8x4x1024x128xbf16>) -> tensor<8x8x4x1024x128xbf16>
    %131 = "tensor.empty"() : () -> tensor<8x32x128x1024xbf16>
    %132 = "linalg.generic"(%129, %131) <{indexing_maps = [#map13, #map9], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg221: bf16, %arg222: bf16):
      "linalg.yield"(%arg221) : (bf16) -> ()
    }) : (tensor<8x32x1024x128xbf16>, tensor<8x32x128x1024xbf16>) -> tensor<8x32x128x1024xbf16>
    %133 = "tensor.collapse_shape"(%118) <{reassociation = [[0, 1], [2], [3]]}> : (tensor<8x32x1024x128xbf16>) -> tensor<256x1024x128xbf16>
    %134 = "tensor.collapse_shape"(%132) <{reassociation = [[0, 1], [2], [3]]}> : (tensor<8x32x128x1024xbf16>) -> tensor<256x128x1024xbf16>
    %135 = "tensor.empty"() : () -> tensor<256x1024x1024xbf16>
    %136 = "linalg.generic"(%80, %135) <{indexing_maps = [#map4, #map3], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg219: bf16, %arg220: bf16):
      "linalg.yield"(%arg219) : (bf16) -> ()
    }) : (bf16, tensor<256x1024x1024xbf16>) -> tensor<256x1024x1024xbf16>
    %137 = "linalg.generic"(%133, %134, %136) <{indexing_maps = [#map5, #map6, #map7], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg216: bf16, %arg217: bf16, %arg218: bf16):
      %187 = "arith.mulf"(%arg216, %arg217) <{fastmath = #arith.fastmath<none>}> : (bf16, bf16) -> bf16
      %188 = "arith.addf"(%arg218, %187) <{fastmath = #arith.fastmath<none>}> : (bf16, bf16) -> bf16
      "linalg.yield"(%188) : (bf16) -> ()
    }) : (tensor<256x1024x128xbf16>, tensor<256x128x1024xbf16>, tensor<256x1024x1024xbf16>) -> tensor<256x1024x1024xbf16>
    %138 = "tensor.expand_shape"(%137) <{reassociation = [[0, 1], [2], [3]], static_output_shape = array<i64: 8, 32, 1024, 1024>}> : (tensor<256x1024x1024xbf16>) -> tensor<8x32x1024x1024xbf16>
    %139 = "tensor.empty"() : () -> tensor<8x32x1024x1024xbf16>
    %140 = "linalg.generic"(%138, %139) <{indexing_maps = [#map9, #map9], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg214: bf16, %arg215: bf16):
      %185 = "arith.truncf"(%83) : (f64) -> bf16
      %186 = "arith.mulf"(%arg214, %185) <{fastmath = #arith.fastmath<none>}> : (bf16, bf16) -> bf16
      "linalg.yield"(%186) : (bf16) -> ()
    }) : (tensor<8x32x1024x1024xbf16>, tensor<8x32x1024x1024xbf16>) -> tensor<8x32x1024x1024xbf16>
    %141 = "linalg.generic"(%140, %arg169, %139) <{indexing_maps = [#map9, #map14, #map9], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg211: bf16, %arg212: bf16, %arg213: bf16):
      %184 = "arith.addf"(%arg211, %arg212) <{fastmath = #arith.fastmath<none>}> : (bf16, bf16) -> bf16
      "linalg.yield"(%184) : (bf16) -> ()
    }) : (tensor<8x32x1024x1024xbf16>, tensor<1x32x1x1xbf16>, tensor<8x32x1024x1024xbf16>) -> tensor<8x32x1024x1024xbf16>
    %142 = "tensor.empty"() : () -> tensor<8x32x1024x1024xf32>
    %143 = "linalg.generic"(%141, %142) <{indexing_maps = [#map9, #map9], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg209: bf16, %arg210: f32):
      %183 = "arith.extf"(%arg209) : (bf16) -> f32
      "linalg.yield"(%183) : (f32) -> ()
    }) : (tensor<8x32x1024x1024xbf16>, tensor<8x32x1024x1024xf32>) -> tensor<8x32x1024x1024xf32>
    %144 = "tensor.empty"() : () -> tensor<8x32x1024xi64>
    %145 = "linalg.generic"(%79, %144) <{indexing_maps = [#map4, #map3], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg207: i64, %arg208: i64):
      "linalg.yield"(%arg207) : (i64) -> ()
    }) : (i64, tensor<8x32x1024xi64>) -> tensor<8x32x1024xi64>
    %146 = "tensor.empty"() : () -> tensor<8x32x1024xf32>
    %147 = "linalg.generic"(%81, %146) <{indexing_maps = [#map4, #map3], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg205: f32, %arg206: f32):
      "linalg.yield"(%arg205) : (f32) -> ()
    }) : (f32, tensor<8x32x1024xf32>) -> tensor<8x32x1024xf32>
    %148:2 = "linalg.generic"(%143, %147, %145) <{indexing_maps = [#map9, #map7, #map7], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operandSegmentSizes = array<i32: 1, 2>}> ({
    ^bb0(%arg202: f32, %arg203: f32, %arg204: i64):
      %178 = "linalg.index"() <{dim = 3 : i64}> : () -> index
      %179 = "arith.index_cast"(%178) : (index) -> i64
      %180 = "arith.maximumf"(%arg202, %arg203) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %181 = "arith.cmpf"(%arg202, %arg203) <{fastmath = #arith.fastmath<none>, predicate = 2 : i64}> : (f32, f32) -> i1
      %182 = "arith.select"(%181, %179, %arg204) : (i1, i64, i64) -> i64
      "linalg.yield"(%180, %182) : (f32, i64) -> ()
    }) : (tensor<8x32x1024x1024xf32>, tensor<8x32x1024xf32>, tensor<8x32x1024xi64>) -> (tensor<8x32x1024xf32>, tensor<8x32x1024xi64>)
    %149 = "tensor.expand_shape"(%148#0) <{reassociation = [[0], [1], [2, 3]], static_output_shape = array<i64: 8, 32, 1024, 1>}> : (tensor<8x32x1024xf32>) -> tensor<8x32x1024x1xf32>
    %150 = "linalg.generic"(%143, %149, %142) <{indexing_maps = [#map9, #map15, #map9], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg199: f32, %arg200: f32, %arg201: f32):
      %177 = "arith.subf"(%arg199, %arg200) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%177) : (f32) -> ()
    }) : (tensor<8x32x1024x1024xf32>, tensor<8x32x1024x1xf32>, tensor<8x32x1024x1024xf32>) -> tensor<8x32x1024x1024xf32>
    %151 = "linalg.generic"(%150, %142) <{indexing_maps = [#map9, #map9], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg197: f32, %arg198: f32):
      %176 = "math.exp"(%arg197) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
      "linalg.yield"(%176) : (f32) -> ()
    }) : (tensor<8x32x1024x1024xf32>, tensor<8x32x1024x1024xf32>) -> tensor<8x32x1024x1024xf32>
    %152 = "tensor.empty"() : () -> tensor<8x32x1024x1xf32>
    %153 = "linalg.generic"(%82, %152) <{indexing_maps = [#map16, #map9], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg195: f32, %arg196: f32):
      "linalg.yield"(%arg195) : (f32) -> ()
    }) : (f32, tensor<8x32x1024x1xf32>) -> tensor<8x32x1024x1xf32>
    %154 = "linalg.generic"(%151, %153) <{indexing_maps = [#map9, #map15], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg193: f32, %arg194: f32):
      %175 = "arith.addf"(%arg193, %arg194) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%175) : (f32) -> ()
    }) : (tensor<8x32x1024x1024xf32>, tensor<8x32x1024x1xf32>) -> tensor<8x32x1024x1xf32>
    %155 = "linalg.generic"(%151, %154, %142) <{indexing_maps = [#map9, #map15, #map9], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg190: f32, %arg191: f32, %arg192: f32):
      %174 = "arith.divf"(%arg190, %arg191) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%174) : (f32) -> ()
    }) : (tensor<8x32x1024x1024xf32>, tensor<8x32x1024x1xf32>, tensor<8x32x1024x1024xf32>) -> tensor<8x32x1024x1024xf32>
    %156 = "linalg.generic"(%155, %139) <{indexing_maps = [#map9, #map9], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg188: f32, %arg189: bf16):
      %173 = "arith.truncf"(%arg188) : (f32) -> bf16
      "linalg.yield"(%173) : (bf16) -> ()
    }) : (tensor<8x32x1024x1024xf32>, tensor<8x32x1024x1024xbf16>) -> tensor<8x32x1024x1024xbf16>
    %157 = "tensor.collapse_shape"(%156) <{reassociation = [[0, 1], [2], [3]]}> : (tensor<8x32x1024x1024xbf16>) -> tensor<256x1024x1024xbf16>
    %158 = "tensor.collapse_shape"(%130) <{reassociation = [[0, 1, 2], [3], [4]]}> : (tensor<8x8x4x1024x128xbf16>) -> tensor<256x1024x128xbf16>
    %159 = "tensor.empty"() : () -> tensor<256x1024x128xbf16>
    %160 = "linalg.generic"(%80, %159) <{indexing_maps = [#map4, #map3], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg186: bf16, %arg187: bf16):
      "linalg.yield"(%arg186) : (bf16) -> ()
    }) : (bf16, tensor<256x1024x128xbf16>) -> tensor<256x1024x128xbf16>
    %161 = "linalg.generic"(%157, %158, %160) <{indexing_maps = [#map5, #map6, #map7], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg183: bf16, %arg184: bf16, %arg185: bf16):
      %171 = "arith.mulf"(%arg183, %arg184) <{fastmath = #arith.fastmath<none>}> : (bf16, bf16) -> bf16
      %172 = "arith.addf"(%arg185, %171) <{fastmath = #arith.fastmath<none>}> : (bf16, bf16) -> bf16
      "linalg.yield"(%172) : (bf16) -> ()
    }) : (tensor<256x1024x1024xbf16>, tensor<256x1024x128xbf16>, tensor<256x1024x128xbf16>) -> tensor<256x1024x128xbf16>
    %162 = "tensor.expand_shape"(%161) <{reassociation = [[0, 1], [2], [3]], static_output_shape = array<i64: 8, 32, 1024, 128>}> : (tensor<256x1024x128xbf16>) -> tensor<8x32x1024x128xbf16>
    %163 = "tensor.empty"() : () -> tensor<8x1024x32x128xbf16>
    %164 = "linalg.generic"(%162, %163) <{indexing_maps = [#map8, #map9], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg181: bf16, %arg182: bf16):
      "linalg.yield"(%arg181) : (bf16) -> ()
    }) : (tensor<8x32x1024x128xbf16>, tensor<8x1024x32x128xbf16>) -> tensor<8x1024x32x128xbf16>
    %165 = "tensor.collapse_shape"(%164) <{reassociation = [[0], [1], [2, 3]]}> : (tensor<8x1024x32x128xbf16>) -> tensor<8x1024x4096xbf16>
    %166 = "linalg.generic"(%arg173, %84) <{indexing_maps = [#map, #map1], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg179: bf16, %arg180: bf16):
      "linalg.yield"(%arg179) : (bf16) -> ()
    }) : (tensor<4096x4096xbf16>, tensor<4096x4096xbf16>) -> tensor<4096x4096xbf16>
    %167 = "linalg.generic"(%166, %86) <{indexing_maps = [#map2, #map3], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg177: bf16, %arg178: bf16):
      "linalg.yield"(%arg177) : (bf16) -> ()
    }) : (tensor<4096x4096xbf16>, tensor<8x4096x4096xbf16>) -> tensor<8x4096x4096xbf16>
    %168 = "linalg.generic"(%165, %167, %89) <{indexing_maps = [#map5, #map6, #map7], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg174: bf16, %arg175: bf16, %arg176: bf16):
      %169 = "arith.mulf"(%arg174, %arg175) <{fastmath = #arith.fastmath<none>}> : (bf16, bf16) -> bf16
      %170 = "arith.addf"(%arg176, %169) <{fastmath = #arith.fastmath<none>}> : (bf16, bf16) -> bf16
      "linalg.yield"(%170) : (bf16) -> ()
    }) : (tensor<8x1024x4096xbf16>, tensor<8x4096x4096xbf16>, tensor<8x1024x4096xbf16>) -> tensor<8x1024x4096xbf16>
    "func.return"(%168, %156) : (tensor<8x1024x4096xbf16>, tensor<8x32x1024x1024xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<4096x4096xbf16>) -> (), sym_name = "Cluster_0"}> ({
  ^bb0(%arg163: tensor<4096x4096xbf16>):
    %78 = "linalg.generic"(%arg163, %84) <{indexing_maps = [#map, #map1], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg164: bf16, %arg165: bf16):
      "linalg.yield"(%arg164) : (bf16) -> ()
    }) : (tensor<4096x4096xbf16>, tensor<4096x4096xbf16>) -> tensor<4096x4096xbf16>
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<256x1024x128xbf16>, tensor<256x1024x1024xbf16>) -> (), sym_name = "Cluster_1"}> ({
  ^bb0(%arg158: tensor<256x1024x128xbf16>, %arg159: tensor<256x1024x1024xbf16>):
    %75 = "linalg.generic"(%arg159, %arg158, %160) <{indexing_maps = [#map5, #map6, #map7], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg160: bf16, %arg161: bf16, %arg162: bf16):
      %76 = "arith.mulf"(%arg160, %arg161) <{fastmath = #arith.fastmath<none>}> : (bf16, bf16) -> bf16
      %77 = "arith.addf"(%arg162, %76) <{fastmath = #arith.fastmath<none>}> : (bf16, bf16) -> bf16
      "linalg.yield"(%77) : (bf16) -> ()
    }) : (tensor<256x1024x1024xbf16>, tensor<256x1024x128xbf16>, tensor<256x1024x128xbf16>) -> tensor<256x1024x128xbf16>
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<8x32x1024x1xf32>, tensor<8x32x1024x1024xf32>) -> (), sym_name = "Cluster_2"}> ({
  ^bb0(%arg153: tensor<8x32x1024x1xf32>, %arg154: tensor<8x32x1024x1024xf32>):
    %73 = "linalg.generic"(%arg154, %arg153, %142) <{indexing_maps = [#map9, #map15, #map9], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg155: f32, %arg156: f32, %arg157: f32):
      %74 = "arith.subf"(%arg155, %arg156) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%74) : (f32) -> ()
    }) : (tensor<8x32x1024x1024xf32>, tensor<8x32x1024x1xf32>, tensor<8x32x1024x1024xf32>) -> tensor<8x32x1024x1024xf32>
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<8x32x1024x1024xf32>) -> (), sym_name = "Cluster_3"}> ({
  ^bb0(%arg149: tensor<8x32x1024x1024xf32>):
    %67:2 = "linalg.generic"(%arg149, %147, %145) <{indexing_maps = [#map9, #map7, #map7], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operandSegmentSizes = array<i32: 1, 2>}> ({
    ^bb0(%arg150: f32, %arg151: f32, %arg152: i64):
      %68 = "linalg.index"() <{dim = 3 : i64}> : () -> index
      %69 = "arith.index_cast"(%68) : (index) -> i64
      %70 = "arith.maximumf"(%arg150, %arg151) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %71 = "arith.cmpf"(%arg150, %arg151) <{fastmath = #arith.fastmath<none>, predicate = 2 : i64}> : (f32, f32) -> i1
      %72 = "arith.select"(%71, %69, %arg152) : (i1, i64, i64) -> i64
      "linalg.yield"(%70, %72) : (f32, i64) -> ()
    }) : (tensor<8x32x1024x1024xf32>, tensor<8x32x1024xf32>, tensor<8x32x1024xi64>) -> (tensor<8x32x1024xf32>, tensor<8x32x1024xi64>)
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = (f32) -> (), sym_name = "Cluster_4"}> ({
  ^bb0(%arg146: f32):
    %66 = "linalg.generic"(%arg146, %146) <{indexing_maps = [#map4, #map3], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg147: f32, %arg148: f32):
      "linalg.yield"(%arg147) : (f32) -> ()
    }) : (f32, tensor<8x32x1024xf32>) -> tensor<8x32x1024xf32>
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = (i64) -> (), sym_name = "Cluster_5"}> ({
  ^bb0(%arg143: i64):
    %65 = "linalg.generic"(%arg143, %144) <{indexing_maps = [#map4, #map3], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg144: i64, %arg145: i64):
      "linalg.yield"(%arg144) : (i64) -> ()
    }) : (i64, tensor<8x32x1024xi64>) -> tensor<8x32x1024xi64>
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<8x32x1024x1024xf32>) -> (), sym_name = "Cluster_6"}> ({
  ^bb0(%arg140: tensor<8x32x1024x1024xf32>):
    %63 = "linalg.generic"(%arg140, %139) <{indexing_maps = [#map9, #map9], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg141: f32, %arg142: bf16):
      %64 = "arith.truncf"(%arg141) : (f32) -> bf16
      "linalg.yield"(%64) : (bf16) -> ()
    }) : (tensor<8x32x1024x1024xf32>, tensor<8x32x1024x1024xbf16>) -> tensor<8x32x1024x1024xbf16>
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<8x4096x1024xbf16>, tensor<8x1024x4096xbf16>) -> (), sym_name = "Cluster_7"}> ({
  ^bb0(%arg135: tensor<8x4096x1024xbf16>, %arg136: tensor<8x1024x4096xbf16>):
    %60 = "linalg.generic"(%arg136, %arg135, %99) <{indexing_maps = [#map5, #map6, #map7], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg137: bf16, %arg138: bf16, %arg139: bf16):
      %61 = "arith.mulf"(%arg137, %arg138) <{fastmath = #arith.fastmath<none>}> : (bf16, bf16) -> bf16
      %62 = "arith.addf"(%arg139, %61) <{fastmath = #arith.fastmath<none>}> : (bf16, bf16) -> bf16
      "linalg.yield"(%62) : (bf16) -> ()
    }) : (tensor<8x1024x4096xbf16>, tensor<8x4096x1024xbf16>, tensor<8x1024x1024xbf16>) -> tensor<8x1024x1024xbf16>
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<8x1024x32x128xbf16>) -> (), sym_name = "Cluster_8"}> ({
  ^bb0(%arg132: tensor<8x1024x32x128xbf16>):
    %59 = "linalg.generic"(%arg132, %92) <{indexing_maps = [#map8, #map9], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg133: bf16, %arg134: bf16):
      "linalg.yield"(%arg133) : (bf16) -> ()
    }) : (tensor<8x1024x32x128xbf16>, tensor<8x32x1024x128xbf16>) -> tensor<8x32x1024x128xbf16>
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = (f32) -> (), sym_name = "Cluster_9"}> ({
  ^bb0(%arg129: f32):
    %58 = "linalg.generic"(%arg129, %152) <{indexing_maps = [#map16, #map9], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg130: f32, %arg131: f32):
      "linalg.yield"(%arg130) : (f32) -> ()
    }) : (f32, tensor<8x32x1024x1xf32>) -> tensor<8x32x1024x1xf32>
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<8x1024x8x128xbf16>) -> (), sym_name = "Cluster_10"}> ({
  ^bb0(%arg126: tensor<8x1024x8x128xbf16>):
    %57 = "linalg.generic"(%arg126, %102) <{indexing_maps = [#map8, #map9], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg127: bf16, %arg128: bf16):
      "linalg.yield"(%arg127) : (bf16) -> ()
    }) : (tensor<8x1024x8x128xbf16>, tensor<8x8x1024x128xbf16>) -> tensor<8x8x1024x128xbf16>
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<4096x1024xbf16>) -> (), sym_name = "Cluster_11"}> ({
  ^bb0(%arg123: tensor<4096x1024xbf16>):
    %56 = "linalg.generic"(%arg123, %96) <{indexing_maps = [#map2, #map3], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg124: bf16, %arg125: bf16):
      "linalg.yield"(%arg124) : (bf16) -> ()
    }) : (tensor<4096x1024xbf16>, tensor<8x4096x1024xbf16>) -> tensor<8x4096x1024xbf16>
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<8x32x1024x128xbf16>) -> (), sym_name = "Cluster_12"}> ({
  ^bb0(%arg120: tensor<8x32x1024x128xbf16>):
    %55 = "linalg.generic"(%arg120, %131) <{indexing_maps = [#map13, #map9], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg121: bf16, %arg122: bf16):
      "linalg.yield"(%arg121) : (bf16) -> ()
    }) : (tensor<8x32x1024x128xbf16>, tensor<8x32x128x1024xbf16>) -> tensor<8x32x128x1024xbf16>
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = (bf16) -> (), sym_name = "Cluster_13"}> ({
  ^bb0(%arg117: bf16):
    %54 = "linalg.generic"(%arg117, %135) <{indexing_maps = [#map4, #map3], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg118: bf16, %arg119: bf16):
      "linalg.yield"(%arg118) : (bf16) -> ()
    }) : (bf16, tensor<256x1024x1024xbf16>) -> tensor<256x1024x1024xbf16>
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<8x32x1024x1024xbf16>) -> (), sym_name = "Cluster_14"}> ({
  ^bb0(%arg114: tensor<8x32x1024x1024xbf16>):
    %52 = "linalg.generic"(%arg114, %142) <{indexing_maps = [#map9, #map9], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg115: bf16, %arg116: f32):
      %53 = "arith.extf"(%arg115) : (bf16) -> f32
      "linalg.yield"(%53) : (f32) -> ()
    }) : (tensor<8x32x1024x1024xbf16>, tensor<8x32x1024x1024xf32>) -> tensor<8x32x1024x1024xf32>
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<8x4096x1024xbf16>, tensor<8x1024x4096xbf16>) -> (), sym_name = "Cluster_15"}> ({
  ^bb0(%arg109: tensor<8x4096x1024xbf16>, %arg110: tensor<8x1024x4096xbf16>):
    %49 = "linalg.generic"(%arg110, %arg109, %99) <{indexing_maps = [#map5, #map6, #map7], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg111: bf16, %arg112: bf16, %arg113: bf16):
      %50 = "arith.mulf"(%arg111, %arg112) <{fastmath = #arith.fastmath<none>}> : (bf16, bf16) -> bf16
      %51 = "arith.addf"(%arg113, %50) <{fastmath = #arith.fastmath<none>}> : (bf16, bf16) -> bf16
      "linalg.yield"(%51) : (bf16) -> ()
    }) : (tensor<8x1024x4096xbf16>, tensor<8x4096x1024xbf16>, tensor<8x1024x1024xbf16>) -> tensor<8x1024x1024xbf16>
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = (bf16) -> (), sym_name = "Cluster_16"}> ({
  ^bb0(%arg106: bf16):
    %48 = "linalg.generic"(%arg106, %159) <{indexing_maps = [#map4, #map3], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg107: bf16, %arg108: bf16):
      "linalg.yield"(%arg107) : (bf16) -> ()
    }) : (bf16, tensor<256x1024x128xbf16>) -> tensor<256x1024x128xbf16>
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<8x1024x8x128xbf16>) -> (), sym_name = "Cluster_17"}> ({
  ^bb0(%arg103: tensor<8x1024x8x128xbf16>):
    %47 = "linalg.generic"(%arg103, %102) <{indexing_maps = [#map8, #map9], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg104: bf16, %arg105: bf16):
      "linalg.yield"(%arg104) : (bf16) -> ()
    }) : (tensor<8x1024x8x128xbf16>, tensor<8x8x1024x128xbf16>) -> tensor<8x8x1024x128xbf16>
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<8x32x1024x1024xbf16>, tensor<1x32x1x1xbf16>) -> (), sym_name = "Cluster_18"}> ({
  ^bb0(%arg98: tensor<8x32x1024x1024xbf16>, %arg99: tensor<1x32x1x1xbf16>):
    %45 = "linalg.generic"(%arg98, %arg99, %139) <{indexing_maps = [#map9, #map14, #map9], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg100: bf16, %arg101: bf16, %arg102: bf16):
      %46 = "arith.addf"(%arg100, %arg101) <{fastmath = #arith.fastmath<none>}> : (bf16, bf16) -> bf16
      "linalg.yield"(%46) : (bf16) -> ()
    }) : (tensor<8x32x1024x1024xbf16>, tensor<1x32x1x1xbf16>, tensor<8x32x1024x1024xbf16>) -> tensor<8x32x1024x1024xbf16>
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<8x32x1024x1024xf32>) -> (), sym_name = "Cluster_19"}> ({
  ^bb0(%arg95: tensor<8x32x1024x1024xf32>):
    %43 = "linalg.generic"(%arg95, %153) <{indexing_maps = [#map9, #map15], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg96: f32, %arg97: f32):
      %44 = "arith.addf"(%arg96, %arg97) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%44) : (f32) -> ()
    }) : (tensor<8x32x1024x1024xf32>, tensor<8x32x1024x1xf32>) -> tensor<8x32x1024x1xf32>
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<1024x4096xbf16>) -> (), sym_name = "Cluster_20"}> ({
  ^bb0(%arg92: tensor<1024x4096xbf16>):
    %42 = "linalg.generic"(%arg92, %94) <{indexing_maps = [#map, #map1], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg93: bf16, %arg94: bf16):
      "linalg.yield"(%arg93) : (bf16) -> ()
    }) : (tensor<1024x4096xbf16>, tensor<4096x1024xbf16>) -> tensor<4096x1024xbf16>
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<8x4096x4096xbf16>, tensor<8x1024x4096xbf16>) -> (), sym_name = "Cluster_21"}> ({
  ^bb0(%arg87: tensor<8x4096x4096xbf16>, %arg88: tensor<8x1024x4096xbf16>):
    %39 = "linalg.generic"(%arg88, %arg87, %89) <{indexing_maps = [#map5, #map6, #map7], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg89: bf16, %arg90: bf16, %arg91: bf16):
      %40 = "arith.mulf"(%arg89, %arg90) <{fastmath = #arith.fastmath<none>}> : (bf16, bf16) -> bf16
      %41 = "arith.addf"(%arg91, %40) <{fastmath = #arith.fastmath<none>}> : (bf16, bf16) -> bf16
      "linalg.yield"(%41) : (bf16) -> ()
    }) : (tensor<8x1024x4096xbf16>, tensor<8x4096x4096xbf16>, tensor<8x1024x4096xbf16>) -> tensor<8x1024x4096xbf16>
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<8x32x1024x64xbf16>) -> (), sym_name = "Cluster_22"}> ({
  ^bb0(%arg84: tensor<8x32x1024x64xbf16>):
    %37 = "linalg.generic"(%arg84, %114) <{indexing_maps = [#map9, #map9], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg85: bf16, %arg86: bf16):
      %38 = "arith.negf"(%arg85) <{fastmath = #arith.fastmath<none>}> : (bf16) -> bf16
      "linalg.yield"(%38) : (bf16) -> ()
    }) : (tensor<8x32x1024x64xbf16>, tensor<8x32x1024x64xbf16>) -> tensor<8x32x1024x64xbf16>
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = (bf16) -> (), sym_name = "Cluster_23"}> ({
  ^bb0(%arg81: bf16):
    %36 = "linalg.generic"(%arg81, %98) <{indexing_maps = [#map4, #map3], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg82: bf16, %arg83: bf16):
      "linalg.yield"(%arg82) : (bf16) -> ()
    }) : (bf16, tensor<8x1024x1024xbf16>) -> tensor<8x1024x1024xbf16>
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<8x32x1024x1024xf32>) -> (), sym_name = "Cluster_24"}> ({
  ^bb0(%arg78: tensor<8x32x1024x1024xf32>):
    %34 = "linalg.generic"(%arg78, %142) <{indexing_maps = [#map9, #map9], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg79: f32, %arg80: f32):
      %35 = "math.exp"(%arg79) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
      "linalg.yield"(%35) : (f32) -> ()
    }) : (tensor<8x32x1024x1024xf32>, tensor<8x32x1024x1024xf32>) -> tensor<8x32x1024x1024xf32>
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = (bf16) -> (), sym_name = "Cluster_25"}> ({
  ^bb0(%arg75: bf16):
    %33 = "linalg.generic"(%arg75, %88) <{indexing_maps = [#map4, #map3], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg76: bf16, %arg77: bf16):
      "linalg.yield"(%arg76) : (bf16) -> ()
    }) : (bf16, tensor<8x1024x4096xbf16>) -> tensor<8x1024x4096xbf16>
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<8x32x1024x1xf32>, tensor<8x32x1024x1024xf32>) -> (), sym_name = "Cluster_26"}> ({
  ^bb0(%arg70: tensor<8x32x1024x1xf32>, %arg71: tensor<8x32x1024x1024xf32>):
    %31 = "linalg.generic"(%arg71, %arg70, %142) <{indexing_maps = [#map9, #map15, #map9], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg72: f32, %arg73: f32, %arg74: f32):
      %32 = "arith.divf"(%arg72, %arg73) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%32) : (f32) -> ()
    }) : (tensor<8x32x1024x1024xf32>, tensor<8x32x1024x1xf32>, tensor<8x32x1024x1024xf32>) -> tensor<8x32x1024x1024xf32>
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<4096x1024xbf16>) -> (), sym_name = "Cluster_27"}> ({
  ^bb0(%arg67: tensor<4096x1024xbf16>):
    %30 = "linalg.generic"(%arg67, %96) <{indexing_maps = [#map2, #map3], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg68: bf16, %arg69: bf16):
      "linalg.yield"(%arg68) : (bf16) -> ()
    }) : (tensor<4096x1024xbf16>, tensor<8x4096x1024xbf16>) -> tensor<8x4096x1024xbf16>
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<256x1024x128xbf16>, tensor<256x128x1024xbf16>) -> (), sym_name = "Cluster_28"}> ({
  ^bb0(%arg62: tensor<256x1024x128xbf16>, %arg63: tensor<256x128x1024xbf16>):
    %27 = "linalg.generic"(%arg62, %arg63, %136) <{indexing_maps = [#map5, #map6, #map7], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg64: bf16, %arg65: bf16, %arg66: bf16):
      %28 = "arith.mulf"(%arg64, %arg65) <{fastmath = #arith.fastmath<none>}> : (bf16, bf16) -> bf16
      %29 = "arith.addf"(%arg66, %28) <{fastmath = #arith.fastmath<none>}> : (bf16, bf16) -> bf16
      "linalg.yield"(%29) : (bf16) -> ()
    }) : (tensor<256x1024x128xbf16>, tensor<256x128x1024xbf16>, tensor<256x1024x1024xbf16>) -> tensor<256x1024x1024xbf16>
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<1024x4096xbf16>) -> (), sym_name = "Cluster_29"}> ({
  ^bb0(%arg59: tensor<1024x4096xbf16>):
    %26 = "linalg.generic"(%arg59, %94) <{indexing_maps = [#map, #map1], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg60: bf16, %arg61: bf16):
      "linalg.yield"(%arg60) : (bf16) -> ()
    }) : (tensor<1024x4096xbf16>, tensor<4096x1024xbf16>) -> tensor<4096x1024xbf16>
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<8x32x1024x1024xbf16>) -> (), sym_name = "Cluster_30"}> ({
  ^bb0(%arg56: tensor<8x32x1024x1024xbf16>):
    %23 = "linalg.generic"(%arg56, %139) <{indexing_maps = [#map9, #map9], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg57: bf16, %arg58: bf16):
      %24 = "arith.truncf"(%83) : (f64) -> bf16
      %25 = "arith.mulf"(%arg57, %24) <{fastmath = #arith.fastmath<none>}> : (bf16, bf16) -> bf16
      "linalg.yield"(%25) : (bf16) -> ()
    }) : (tensor<8x32x1024x1024xbf16>, tensor<8x32x1024x1024xbf16>) -> tensor<8x32x1024x1024xbf16>
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<4096x4096xbf16>) -> (), sym_name = "Cluster_31"}> ({
  ^bb0(%arg53: tensor<4096x4096xbf16>):
    %22 = "linalg.generic"(%arg53, %84) <{indexing_maps = [#map, #map1], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg54: bf16, %arg55: bf16):
      "linalg.yield"(%arg54) : (bf16) -> ()
    }) : (tensor<4096x4096xbf16>, tensor<4096x4096xbf16>) -> tensor<4096x4096xbf16>
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<8x32x1024x128xbf16>) -> (), sym_name = "Cluster_32"}> ({
  ^bb0(%arg50: tensor<8x32x1024x128xbf16>):
    %21 = "linalg.generic"(%arg50, %163) <{indexing_maps = [#map8, #map9], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg51: bf16, %arg52: bf16):
      "linalg.yield"(%arg51) : (bf16) -> ()
    }) : (tensor<8x32x1024x128xbf16>, tensor<8x1024x32x128xbf16>) -> tensor<8x1024x32x128xbf16>
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<8x32x1024x128xbf16>, tensor<1x1x1x128xbf16>) -> (), sym_name = "Cluster_33"}> ({
  ^bb0(%arg45: tensor<8x32x1024x128xbf16>, %arg46: tensor<1x1x1x128xbf16>):
    %19 = "linalg.generic"(%arg45, %arg46, %92) <{indexing_maps = [#map9, #map10, #map9], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg47: bf16, %arg48: bf16, %arg49: bf16):
      %20 = "arith.mulf"(%arg47, %arg48) <{fastmath = #arith.fastmath<none>}> : (bf16, bf16) -> bf16
      "linalg.yield"(%20) : (bf16) -> ()
    }) : (tensor<8x32x1024x128xbf16>, tensor<1x1x1x128xbf16>, tensor<8x32x1024x128xbf16>) -> tensor<8x32x1024x128xbf16>
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<1x1x1x128xbf16>, tensor<8x32x1024x128xbf16>) -> (), sym_name = "Cluster_34"}> ({
  ^bb0(%arg40: tensor<1x1x1x128xbf16>, %arg41: tensor<8x32x1024x128xbf16>):
    %17 = "linalg.generic"(%arg41, %arg40, %92) <{indexing_maps = [#map9, #map10, #map9], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg42: bf16, %arg43: bf16, %arg44: bf16):
      %18 = "arith.mulf"(%arg42, %arg43) <{fastmath = #arith.fastmath<none>}> : (bf16, bf16) -> bf16
      "linalg.yield"(%18) : (bf16) -> ()
    }) : (tensor<8x32x1024x128xbf16>, tensor<1x1x1x128xbf16>, tensor<8x32x1024x128xbf16>) -> tensor<8x32x1024x128xbf16>
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<8x32x1024x128xbf16>, tensor<8x32x1024x128xbf16>) -> (), sym_name = "Cluster_35"}> ({
  ^bb0(%arg35: tensor<8x32x1024x128xbf16>, %arg36: tensor<8x32x1024x128xbf16>):
    %15 = "linalg.generic"(%arg36, %arg35, %92) <{indexing_maps = [#map9, #map9, #map9], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg37: bf16, %arg38: bf16, %arg39: bf16):
      %16 = "arith.addf"(%arg37, %arg38) <{fastmath = #arith.fastmath<none>}> : (bf16, bf16) -> bf16
      "linalg.yield"(%16) : (bf16) -> ()
    }) : (tensor<8x32x1024x128xbf16>, tensor<8x32x1024x128xbf16>, tensor<8x32x1024x128xbf16>) -> tensor<8x32x1024x128xbf16>
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<1x1x1x128xbf16>, tensor<8x8x1024x128xbf16>) -> (), sym_name = "Cluster_36"}> ({
  ^bb0(%arg30: tensor<1x1x1x128xbf16>, %arg31: tensor<8x8x1024x128xbf16>):
    %13 = "linalg.generic"(%arg31, %arg30, %102) <{indexing_maps = [#map9, #map10, #map9], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg32: bf16, %arg33: bf16, %arg34: bf16):
      %14 = "arith.mulf"(%arg32, %arg33) <{fastmath = #arith.fastmath<none>}> : (bf16, bf16) -> bf16
      "linalg.yield"(%14) : (bf16) -> ()
    }) : (tensor<8x8x1024x128xbf16>, tensor<1x1x1x128xbf16>, tensor<8x8x1024x128xbf16>) -> tensor<8x8x1024x128xbf16>
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<8x8x1024x128xbf16>, tensor<1x1x1x128xbf16>) -> (), sym_name = "Cluster_37"}> ({
  ^bb0(%arg25: tensor<8x8x1024x128xbf16>, %arg26: tensor<1x1x1x128xbf16>):
    %11 = "linalg.generic"(%arg25, %arg26, %102) <{indexing_maps = [#map9, #map10, #map9], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg27: bf16, %arg28: bf16, %arg29: bf16):
      %12 = "arith.mulf"(%arg27, %arg28) <{fastmath = #arith.fastmath<none>}> : (bf16, bf16) -> bf16
      "linalg.yield"(%12) : (bf16) -> ()
    }) : (tensor<8x8x1024x128xbf16>, tensor<1x1x1x128xbf16>, tensor<8x8x1024x128xbf16>) -> tensor<8x8x1024x128xbf16>
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<8x8x1024x64xbf16>) -> (), sym_name = "Cluster_38"}> ({
  ^bb0(%arg22: tensor<8x8x1024x64xbf16>):
    %9 = "linalg.generic"(%arg22, %122) <{indexing_maps = [#map9, #map9], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg23: bf16, %arg24: bf16):
      %10 = "arith.negf"(%arg23) <{fastmath = #arith.fastmath<none>}> : (bf16) -> bf16
      "linalg.yield"(%10) : (bf16) -> ()
    }) : (tensor<8x8x1024x64xbf16>, tensor<8x8x1024x64xbf16>) -> tensor<8x8x1024x64xbf16>
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<4096x4096xbf16>) -> (), sym_name = "Cluster_39"}> ({
  ^bb0(%arg19: tensor<4096x4096xbf16>):
    %8 = "linalg.generic"(%arg19, %86) <{indexing_maps = [#map2, #map3], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg20: bf16, %arg21: bf16):
      "linalg.yield"(%arg20) : (bf16) -> ()
    }) : (tensor<4096x4096xbf16>, tensor<8x4096x4096xbf16>) -> tensor<8x4096x4096xbf16>
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<4096x4096xbf16>) -> (), sym_name = "Cluster_40"}> ({
  ^bb0(%arg16: tensor<4096x4096xbf16>):
    %7 = "linalg.generic"(%arg16, %86) <{indexing_maps = [#map2, #map3], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg17: bf16, %arg18: bf16):
      "linalg.yield"(%arg17) : (bf16) -> ()
    }) : (tensor<4096x4096xbf16>, tensor<8x4096x4096xbf16>) -> tensor<8x4096x4096xbf16>
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<8x8x1024x128xbf16>, tensor<8x8x1024x128xbf16>) -> (), sym_name = "Cluster_41"}> ({
  ^bb0(%arg11: tensor<8x8x1024x128xbf16>, %arg12: tensor<8x8x1024x128xbf16>):
    %5 = "linalg.generic"(%arg12, %arg11, %102) <{indexing_maps = [#map9, #map9, #map9], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg13: bf16, %arg14: bf16, %arg15: bf16):
      %6 = "arith.addf"(%arg13, %arg14) <{fastmath = #arith.fastmath<none>}> : (bf16, bf16) -> bf16
      "linalg.yield"(%6) : (bf16) -> ()
    }) : (tensor<8x8x1024x128xbf16>, tensor<8x8x1024x128xbf16>, tensor<8x8x1024x128xbf16>) -> tensor<8x8x1024x128xbf16>
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<8x4096x4096xbf16>, tensor<8x1024x4096xbf16>) -> (), sym_name = "Cluster_42"}> ({
  ^bb0(%arg6: tensor<8x4096x4096xbf16>, %arg7: tensor<8x1024x4096xbf16>):
    %2 = "linalg.generic"(%arg7, %arg6, %89) <{indexing_maps = [#map5, #map6, #map7], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg8: bf16, %arg9: bf16, %arg10: bf16):
      %3 = "arith.mulf"(%arg8, %arg9) <{fastmath = #arith.fastmath<none>}> : (bf16, bf16) -> bf16
      %4 = "arith.addf"(%arg10, %3) <{fastmath = #arith.fastmath<none>}> : (bf16, bf16) -> bf16
      "linalg.yield"(%4) : (bf16) -> ()
    }) : (tensor<8x1024x4096xbf16>, tensor<8x4096x4096xbf16>, tensor<8x1024x4096xbf16>) -> tensor<8x1024x4096xbf16>
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<8x8x1024x128xbf16>) -> (), sym_name = "Cluster_43"}> ({
  ^bb0(%arg3: tensor<8x8x1024x128xbf16>):
    %1 = "linalg.generic"(%arg3, %127) <{indexing_maps = [#map11, #map12], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg4: bf16, %arg5: bf16):
      "linalg.yield"(%arg4) : (bf16) -> ()
    }) : (tensor<8x8x1024x128xbf16>, tensor<8x8x4x1024x128xbf16>) -> tensor<8x8x4x1024x128xbf16>
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<8x8x1024x128xbf16>) -> (), sym_name = "Cluster_44"}> ({
  ^bb0(%arg0: tensor<8x8x1024x128xbf16>):
    %0 = "linalg.generic"(%arg0, %127) <{indexing_maps = [#map11, #map12], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg1: bf16, %arg2: bf16):
      "linalg.yield"(%arg1) : (bf16) -> ()
    }) : (tensor<8x8x1024x128xbf16>, tensor<8x8x4x1024x128xbf16>) -> tensor<8x8x4x1024x128xbf16>
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()
