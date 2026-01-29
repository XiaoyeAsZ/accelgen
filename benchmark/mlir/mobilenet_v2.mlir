#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d1, 0, 0)>
#map3 = affine_map<() -> ()>
#map4 = affine_map<(d0, d1, d2, d3) -> ()>
#map5 = affine_map<(d0, d1) -> (d0, d1)>
#map6 = affine_map<(d0, d1) -> (d1)>
module {
  func.func @main(%arg0: tensor<32xf32>, %arg1: tensor<32xf32>, %arg2: tensor<i64>, %arg3: tensor<32xf32>, %arg4: tensor<32xf32>, %arg5: tensor<i64>, %arg6: tensor<16xf32>, %arg7: tensor<16xf32>, %arg8: tensor<i64>, %arg9: tensor<96xf32>, %arg10: tensor<96xf32>, %arg11: tensor<i64>, %arg12: tensor<96xf32>, %arg13: tensor<96xf32>, %arg14: tensor<i64>, %arg15: tensor<24xf32>, %arg16: tensor<24xf32>, %arg17: tensor<i64>, %arg18: tensor<144xf32>, %arg19: tensor<144xf32>, %arg20: tensor<i64>, %arg21: tensor<144xf32>, %arg22: tensor<144xf32>, %arg23: tensor<i64>, %arg24: tensor<24xf32>, %arg25: tensor<24xf32>, %arg26: tensor<i64>, %arg27: tensor<144xf32>, %arg28: tensor<144xf32>, %arg29: tensor<i64>, %arg30: tensor<144xf32>, %arg31: tensor<144xf32>, %arg32: tensor<i64>, %arg33: tensor<32xf32>, %arg34: tensor<32xf32>, %arg35: tensor<i64>, %arg36: tensor<192xf32>, %arg37: tensor<192xf32>, %arg38: tensor<i64>, %arg39: tensor<192xf32>, %arg40: tensor<192xf32>, %arg41: tensor<i64>, %arg42: tensor<32xf32>, %arg43: tensor<32xf32>, %arg44: tensor<i64>, %arg45: tensor<192xf32>, %arg46: tensor<192xf32>, %arg47: tensor<i64>, %arg48: tensor<192xf32>, %arg49: tensor<192xf32>, %arg50: tensor<i64>, %arg51: tensor<32xf32>, %arg52: tensor<32xf32>, %arg53: tensor<i64>, %arg54: tensor<192xf32>, %arg55: tensor<192xf32>, %arg56: tensor<i64>, %arg57: tensor<192xf32>, %arg58: tensor<192xf32>, %arg59: tensor<i64>, %arg60: tensor<64xf32>, %arg61: tensor<64xf32>, %arg62: tensor<i64>, %arg63: tensor<384xf32>, %arg64: tensor<384xf32>, %arg65: tensor<i64>, %arg66: tensor<384xf32>, %arg67: tensor<384xf32>, %arg68: tensor<i64>, %arg69: tensor<64xf32>, %arg70: tensor<64xf32>, %arg71: tensor<i64>, %arg72: tensor<384xf32>, %arg73: tensor<384xf32>, %arg74: tensor<i64>, %arg75: tensor<384xf32>, %arg76: tensor<384xf32>, %arg77: tensor<i64>, %arg78: tensor<64xf32>, %arg79: tensor<64xf32>, %arg80: tensor<i64>, %arg81: tensor<384xf32>, %arg82: tensor<384xf32>, %arg83: tensor<i64>, %arg84: tensor<384xf32>, %arg85: tensor<384xf32>, %arg86: tensor<i64>, %arg87: tensor<64xf32>, %arg88: tensor<64xf32>, %arg89: tensor<i64>, %arg90: tensor<384xf32>, %arg91: tensor<384xf32>, %arg92: tensor<i64>, %arg93: tensor<384xf32>, %arg94: tensor<384xf32>, %arg95: tensor<i64>, %arg96: tensor<96xf32>, %arg97: tensor<96xf32>, %arg98: tensor<i64>, %arg99: tensor<576xf32>, %arg100: tensor<576xf32>, %arg101: tensor<i64>, %arg102: tensor<576xf32>, %arg103: tensor<576xf32>, %arg104: tensor<i64>, %arg105: tensor<96xf32>, %arg106: tensor<96xf32>, %arg107: tensor<i64>, %arg108: tensor<576xf32>, %arg109: tensor<576xf32>, %arg110: tensor<i64>, %arg111: tensor<576xf32>, %arg112: tensor<576xf32>, %arg113: tensor<i64>, %arg114: tensor<96xf32>, %arg115: tensor<96xf32>, %arg116: tensor<i64>, %arg117: tensor<576xf32>, %arg118: tensor<576xf32>, %arg119: tensor<i64>, %arg120: tensor<576xf32>, %arg121: tensor<576xf32>, %arg122: tensor<i64>, %arg123: tensor<160xf32>, %arg124: tensor<160xf32>, %arg125: tensor<i64>, %arg126: tensor<960xf32>, %arg127: tensor<960xf32>, %arg128: tensor<i64>, %arg129: tensor<960xf32>, %arg130: tensor<960xf32>, %arg131: tensor<i64>, %arg132: tensor<160xf32>, %arg133: tensor<160xf32>, %arg134: tensor<i64>, %arg135: tensor<960xf32>, %arg136: tensor<960xf32>, %arg137: tensor<i64>, %arg138: tensor<960xf32>, %arg139: tensor<960xf32>, %arg140: tensor<i64>, %arg141: tensor<160xf32>, %arg142: tensor<160xf32>, %arg143: tensor<i64>, %arg144: tensor<960xf32>, %arg145: tensor<960xf32>, %arg146: tensor<i64>, %arg147: tensor<960xf32>, %arg148: tensor<960xf32>, %arg149: tensor<i64>, %arg150: tensor<320xf32>, %arg151: tensor<320xf32>, %arg152: tensor<i64>, %arg153: tensor<1280xf32>, %arg154: tensor<1280xf32>, %arg155: tensor<i64>, %arg156: tensor<1x3x224x224xf32>, %arg157: tensor<32x3x3x3xf32>, %arg158: tensor<32xf32>, %arg159: tensor<32xf32>, %arg160: tensor<32x1x3x3xf32>, %arg161: tensor<32xf32>, %arg162: tensor<32xf32>, %arg163: tensor<16x32x1x1xf32>, %arg164: tensor<16xf32>, %arg165: tensor<16xf32>, %arg166: tensor<96x16x1x1xf32>, %arg167: tensor<96xf32>, %arg168: tensor<96xf32>, %arg169: tensor<96x1x3x3xf32>, %arg170: tensor<96xf32>, %arg171: tensor<96xf32>, %arg172: tensor<24x96x1x1xf32>, %arg173: tensor<24xf32>, %arg174: tensor<24xf32>, %arg175: tensor<144x24x1x1xf32>, %arg176: tensor<144xf32>, %arg177: tensor<144xf32>, %arg178: tensor<144x1x3x3xf32>, %arg179: tensor<144xf32>, %arg180: tensor<144xf32>, %arg181: tensor<24x144x1x1xf32>, %arg182: tensor<24xf32>, %arg183: tensor<24xf32>, %arg184: tensor<144x24x1x1xf32>, %arg185: tensor<144xf32>, %arg186: tensor<144xf32>, %arg187: tensor<144x1x3x3xf32>, %arg188: tensor<144xf32>, %arg189: tensor<144xf32>, %arg190: tensor<32x144x1x1xf32>, %arg191: tensor<32xf32>, %arg192: tensor<32xf32>, %arg193: tensor<192x32x1x1xf32>, %arg194: tensor<192xf32>, %arg195: tensor<192xf32>, %arg196: tensor<192x1x3x3xf32>, %arg197: tensor<192xf32>, %arg198: tensor<192xf32>, %arg199: tensor<32x192x1x1xf32>, %arg200: tensor<32xf32>, %arg201: tensor<32xf32>, %arg202: tensor<192x32x1x1xf32>, %arg203: tensor<192xf32>, %arg204: tensor<192xf32>, %arg205: tensor<192x1x3x3xf32>, %arg206: tensor<192xf32>, %arg207: tensor<192xf32>, %arg208: tensor<32x192x1x1xf32>, %arg209: tensor<32xf32>, %arg210: tensor<32xf32>, %arg211: tensor<192x32x1x1xf32>, %arg212: tensor<192xf32>, %arg213: tensor<192xf32>, %arg214: tensor<192x1x3x3xf32>, %arg215: tensor<192xf32>, %arg216: tensor<192xf32>, %arg217: tensor<64x192x1x1xf32>, %arg218: tensor<64xf32>, %arg219: tensor<64xf32>, %arg220: tensor<384x64x1x1xf32>, %arg221: tensor<384xf32>, %arg222: tensor<384xf32>, %arg223: tensor<384x1x3x3xf32>, %arg224: tensor<384xf32>, %arg225: tensor<384xf32>, %arg226: tensor<64x384x1x1xf32>, %arg227: tensor<64xf32>, %arg228: tensor<64xf32>, %arg229: tensor<384x64x1x1xf32>, %arg230: tensor<384xf32>, %arg231: tensor<384xf32>, %arg232: tensor<384x1x3x3xf32>, %arg233: tensor<384xf32>, %arg234: tensor<384xf32>, %arg235: tensor<64x384x1x1xf32>, %arg236: tensor<64xf32>, %arg237: tensor<64xf32>, %arg238: tensor<384x64x1x1xf32>, %arg239: tensor<384xf32>, %arg240: tensor<384xf32>, %arg241: tensor<384x1x3x3xf32>, %arg242: tensor<384xf32>, %arg243: tensor<384xf32>, %arg244: tensor<64x384x1x1xf32>, %arg245: tensor<64xf32>, %arg246: tensor<64xf32>, %arg247: tensor<384x64x1x1xf32>, %arg248: tensor<384xf32>, %arg249: tensor<384xf32>, %arg250: tensor<384x1x3x3xf32>, %arg251: tensor<384xf32>, %arg252: tensor<384xf32>, %arg253: tensor<96x384x1x1xf32>, %arg254: tensor<96xf32>, %arg255: tensor<96xf32>, %arg256: tensor<576x96x1x1xf32>, %arg257: tensor<576xf32>, %arg258: tensor<576xf32>, %arg259: tensor<576x1x3x3xf32>, %arg260: tensor<576xf32>, %arg261: tensor<576xf32>, %arg262: tensor<96x576x1x1xf32>, %arg263: tensor<96xf32>, %arg264: tensor<96xf32>, %arg265: tensor<576x96x1x1xf32>, %arg266: tensor<576xf32>, %arg267: tensor<576xf32>, %arg268: tensor<576x1x3x3xf32>, %arg269: tensor<576xf32>, %arg270: tensor<576xf32>, %arg271: tensor<96x576x1x1xf32>, %arg272: tensor<96xf32>, %arg273: tensor<96xf32>, %arg274: tensor<576x96x1x1xf32>, %arg275: tensor<576xf32>, %arg276: tensor<576xf32>, %arg277: tensor<576x1x3x3xf32>, %arg278: tensor<576xf32>, %arg279: tensor<576xf32>, %arg280: tensor<160x576x1x1xf32>, %arg281: tensor<160xf32>, %arg282: tensor<160xf32>, %arg283: tensor<960x160x1x1xf32>, %arg284: tensor<960xf32>, %arg285: tensor<960xf32>, %arg286: tensor<960x1x3x3xf32>, %arg287: tensor<960xf32>, %arg288: tensor<960xf32>, %arg289: tensor<160x960x1x1xf32>, %arg290: tensor<160xf32>, %arg291: tensor<160xf32>, %arg292: tensor<960x160x1x1xf32>, %arg293: tensor<960xf32>, %arg294: tensor<960xf32>, %arg295: tensor<960x1x3x3xf32>, %arg296: tensor<960xf32>, %arg297: tensor<960xf32>, %arg298: tensor<160x960x1x1xf32>, %arg299: tensor<160xf32>, %arg300: tensor<160xf32>, %arg301: tensor<960x160x1x1xf32>, %arg302: tensor<960xf32>, %arg303: tensor<960xf32>, %arg304: tensor<960x1x3x3xf32>, %arg305: tensor<960xf32>, %arg306: tensor<960xf32>, %arg307: tensor<320x960x1x1xf32>, %arg308: tensor<320xf32>, %arg309: tensor<320xf32>, %arg310: tensor<1280x320x1x1xf32>, %arg311: tensor<1280xf32>, %arg312: tensor<1280xf32>, %arg313: tensor<1000x1280xf32>, %arg314: tensor<1000xf32>) -> tensor<1x1000xf32> {
    %cst = arith.constant dense<0.000000e+00> : tensor<f64>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 1.000000e+00 : f32
    %cst_2 = arith.constant 1.000000e-05 : f64
    %cst_3 = arith.constant 4.900000e+01 : f32
    %cst_4 = arith.constant dense<6.000000e+00> : tensor<f64>
    %padded = tensor.pad %arg156 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg315: index, %arg316: index, %arg317: index, %arg318: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x3x224x224xf32> to tensor<1x3x226x226xf32>
    %0 = tensor.empty() : tensor<1x32x112x112xf32>
    %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<1x32x112x112xf32>) -> tensor<1x32x112x112xf32>
    %2 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%padded, %arg157 : tensor<1x3x226x226xf32>, tensor<32x3x3x3xf32>) outs(%1 : tensor<1x32x112x112xf32>) -> tensor<1x32x112x112xf32>
    %3 = tensor.empty() : tensor<32xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg1 : tensor<32xf32>) outs(%3 : tensor<32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.truncf %cst_2 : f64 to f32
      %561 = arith.addf %in, %560 : f32
      linalg.yield %561 : f32
    } -> tensor<32xf32>
    %5 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%4 : tensor<32xf32>) outs(%3 : tensor<32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = math.sqrt %in : f32
      linalg.yield %560 : f32
    } -> tensor<32xf32>
    %6 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%5 : tensor<32xf32>) outs(%3 : tensor<32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.cmpf one, %in, %cst_0 : f32
      cf.assert %560, "unimplemented: tensor with zero element"
      %561 = arith.divf %cst_1, %in : f32
      linalg.yield %561 : f32
    } -> tensor<32xf32>
    %expanded = tensor.expand_shape %arg0 [[0, 1, 2]] output_shape [32, 1, 1] : tensor<32xf32> into tensor<32x1x1xf32>
    %expanded_5 = tensor.expand_shape %6 [[0, 1, 2]] output_shape [32, 1, 1] : tensor<32xf32> into tensor<32x1x1xf32>
    %7 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2, %expanded : tensor<1x32x112x112xf32>, tensor<32x1x1xf32>) outs(%0 : tensor<1x32x112x112xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.subf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x32x112x112xf32>
    %8 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%7, %expanded_5 : tensor<1x32x112x112xf32>, tensor<32x1x1xf32>) outs(%0 : tensor<1x32x112x112xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x32x112x112xf32>
    %expanded_6 = tensor.expand_shape %arg158 [[0, 1, 2]] output_shape [32, 1, 1] : tensor<32xf32> into tensor<32x1x1xf32>
    %9 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%8, %expanded_6 : tensor<1x32x112x112xf32>, tensor<32x1x1xf32>) outs(%0 : tensor<1x32x112x112xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x32x112x112xf32>
    %expanded_7 = tensor.expand_shape %arg159 [[0, 1, 2]] output_shape [32, 1, 1] : tensor<32xf32> into tensor<32x1x1xf32>
    %10 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%9, %expanded_7 : tensor<1x32x112x112xf32>, tensor<32x1x1xf32>) outs(%0 : tensor<1x32x112x112xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.addf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x32x112x112xf32>
    %11 = tensor.empty() : tensor<f32>
    %12 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = []} ins(%cst : tensor<f64>) outs(%11 : tensor<f32>) {
    ^bb0(%in: f64, %out: f32):
      %560 = arith.truncf %in : f64 to f32
      linalg.yield %560 : f32
    } -> tensor<f32>
    %13 = linalg.generic {indexing_maps = [#map1, #map4, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%10, %12 : tensor<1x32x112x112xf32>, tensor<f32>) outs(%0 : tensor<1x32x112x112xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.cmpf ogt, %in, %in_246 : f32
      %561 = arith.select %560, %in, %in_246 : f32
      linalg.yield %561 : f32
    } -> tensor<1x32x112x112xf32>
    %14 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = []} ins(%cst_4 : tensor<f64>) outs(%11 : tensor<f32>) {
    ^bb0(%in: f64, %out: f32):
      %560 = arith.truncf %in : f64 to f32
      linalg.yield %560 : f32
    } -> tensor<f32>
    %15 = linalg.generic {indexing_maps = [#map4, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%14, %13 : tensor<f32>, tensor<1x32x112x112xf32>) outs(%0 : tensor<1x32x112x112xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.cmpf olt, %in, %in_246 : f32
      %561 = arith.select %560, %in, %in_246 : f32
      linalg.yield %561 : f32
    } -> tensor<1x32x112x112xf32>
    %padded_8 = tensor.pad %15 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg315: index, %arg316: index, %arg317: index, %arg318: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x32x112x112xf32> to tensor<1x32x114x114xf32>
    %collapsed = tensor.collapse_shape %arg160 [[0, 1], [2], [3]] : tensor<32x1x3x3xf32> into tensor<32x3x3xf32>
    %16 = linalg.depthwise_conv_2d_nchw_chw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_8, %collapsed : tensor<1x32x114x114xf32>, tensor<32x3x3xf32>) outs(%1 : tensor<1x32x112x112xf32>) -> tensor<1x32x112x112xf32>
    %17 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg4 : tensor<32xf32>) outs(%3 : tensor<32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.truncf %cst_2 : f64 to f32
      %561 = arith.addf %in, %560 : f32
      linalg.yield %561 : f32
    } -> tensor<32xf32>
    %18 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%17 : tensor<32xf32>) outs(%3 : tensor<32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = math.sqrt %in : f32
      linalg.yield %560 : f32
    } -> tensor<32xf32>
    %19 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%18 : tensor<32xf32>) outs(%3 : tensor<32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.cmpf one, %in, %cst_0 : f32
      cf.assert %560, "unimplemented: tensor with zero element"
      %561 = arith.divf %cst_1, %in : f32
      linalg.yield %561 : f32
    } -> tensor<32xf32>
    %expanded_9 = tensor.expand_shape %arg3 [[0, 1, 2]] output_shape [32, 1, 1] : tensor<32xf32> into tensor<32x1x1xf32>
    %expanded_10 = tensor.expand_shape %19 [[0, 1, 2]] output_shape [32, 1, 1] : tensor<32xf32> into tensor<32x1x1xf32>
    %20 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%16, %expanded_9 : tensor<1x32x112x112xf32>, tensor<32x1x1xf32>) outs(%0 : tensor<1x32x112x112xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.subf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x32x112x112xf32>
    %21 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%20, %expanded_10 : tensor<1x32x112x112xf32>, tensor<32x1x1xf32>) outs(%0 : tensor<1x32x112x112xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x32x112x112xf32>
    %expanded_11 = tensor.expand_shape %arg161 [[0, 1, 2]] output_shape [32, 1, 1] : tensor<32xf32> into tensor<32x1x1xf32>
    %22 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%21, %expanded_11 : tensor<1x32x112x112xf32>, tensor<32x1x1xf32>) outs(%0 : tensor<1x32x112x112xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x32x112x112xf32>
    %expanded_12 = tensor.expand_shape %arg162 [[0, 1, 2]] output_shape [32, 1, 1] : tensor<32xf32> into tensor<32x1x1xf32>
    %23 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%22, %expanded_12 : tensor<1x32x112x112xf32>, tensor<32x1x1xf32>) outs(%0 : tensor<1x32x112x112xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.addf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x32x112x112xf32>
    %24 = linalg.generic {indexing_maps = [#map1, #map4, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%23, %12 : tensor<1x32x112x112xf32>, tensor<f32>) outs(%0 : tensor<1x32x112x112xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.cmpf ogt, %in, %in_246 : f32
      %561 = arith.select %560, %in, %in_246 : f32
      linalg.yield %561 : f32
    } -> tensor<1x32x112x112xf32>
    %25 = linalg.generic {indexing_maps = [#map4, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%14, %24 : tensor<f32>, tensor<1x32x112x112xf32>) outs(%0 : tensor<1x32x112x112xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.cmpf olt, %in, %in_246 : f32
      %561 = arith.select %560, %in, %in_246 : f32
      linalg.yield %561 : f32
    } -> tensor<1x32x112x112xf32>
    %26 = tensor.empty() : tensor<1x16x112x112xf32>
    %27 = linalg.fill ins(%cst_0 : f32) outs(%26 : tensor<1x16x112x112xf32>) -> tensor<1x16x112x112xf32>
    %28 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%25, %arg163 : tensor<1x32x112x112xf32>, tensor<16x32x1x1xf32>) outs(%27 : tensor<1x16x112x112xf32>) -> tensor<1x16x112x112xf32>
    %29 = tensor.empty() : tensor<16xf32>
    %30 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg7 : tensor<16xf32>) outs(%29 : tensor<16xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.truncf %cst_2 : f64 to f32
      %561 = arith.addf %in, %560 : f32
      linalg.yield %561 : f32
    } -> tensor<16xf32>
    %31 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%30 : tensor<16xf32>) outs(%29 : tensor<16xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = math.sqrt %in : f32
      linalg.yield %560 : f32
    } -> tensor<16xf32>
    %32 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%31 : tensor<16xf32>) outs(%29 : tensor<16xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.cmpf one, %in, %cst_0 : f32
      cf.assert %560, "unimplemented: tensor with zero element"
      %561 = arith.divf %cst_1, %in : f32
      linalg.yield %561 : f32
    } -> tensor<16xf32>
    %expanded_13 = tensor.expand_shape %arg6 [[0, 1, 2]] output_shape [16, 1, 1] : tensor<16xf32> into tensor<16x1x1xf32>
    %expanded_14 = tensor.expand_shape %32 [[0, 1, 2]] output_shape [16, 1, 1] : tensor<16xf32> into tensor<16x1x1xf32>
    %33 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%28, %expanded_13 : tensor<1x16x112x112xf32>, tensor<16x1x1xf32>) outs(%26 : tensor<1x16x112x112xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.subf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x16x112x112xf32>
    %34 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%33, %expanded_14 : tensor<1x16x112x112xf32>, tensor<16x1x1xf32>) outs(%26 : tensor<1x16x112x112xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x16x112x112xf32>
    %expanded_15 = tensor.expand_shape %arg164 [[0, 1, 2]] output_shape [16, 1, 1] : tensor<16xf32> into tensor<16x1x1xf32>
    %35 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%34, %expanded_15 : tensor<1x16x112x112xf32>, tensor<16x1x1xf32>) outs(%26 : tensor<1x16x112x112xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x16x112x112xf32>
    %expanded_16 = tensor.expand_shape %arg165 [[0, 1, 2]] output_shape [16, 1, 1] : tensor<16xf32> into tensor<16x1x1xf32>
    %36 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%35, %expanded_16 : tensor<1x16x112x112xf32>, tensor<16x1x1xf32>) outs(%26 : tensor<1x16x112x112xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.addf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x16x112x112xf32>
    %37 = tensor.empty() : tensor<1x96x112x112xf32>
    %38 = linalg.fill ins(%cst_0 : f32) outs(%37 : tensor<1x96x112x112xf32>) -> tensor<1x96x112x112xf32>
    %39 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%36, %arg166 : tensor<1x16x112x112xf32>, tensor<96x16x1x1xf32>) outs(%38 : tensor<1x96x112x112xf32>) -> tensor<1x96x112x112xf32>
    %40 = tensor.empty() : tensor<96xf32>
    %41 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg10 : tensor<96xf32>) outs(%40 : tensor<96xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.truncf %cst_2 : f64 to f32
      %561 = arith.addf %in, %560 : f32
      linalg.yield %561 : f32
    } -> tensor<96xf32>
    %42 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%41 : tensor<96xf32>) outs(%40 : tensor<96xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = math.sqrt %in : f32
      linalg.yield %560 : f32
    } -> tensor<96xf32>
    %43 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%42 : tensor<96xf32>) outs(%40 : tensor<96xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.cmpf one, %in, %cst_0 : f32
      cf.assert %560, "unimplemented: tensor with zero element"
      %561 = arith.divf %cst_1, %in : f32
      linalg.yield %561 : f32
    } -> tensor<96xf32>
    %expanded_17 = tensor.expand_shape %arg9 [[0, 1, 2]] output_shape [96, 1, 1] : tensor<96xf32> into tensor<96x1x1xf32>
    %expanded_18 = tensor.expand_shape %43 [[0, 1, 2]] output_shape [96, 1, 1] : tensor<96xf32> into tensor<96x1x1xf32>
    %44 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%39, %expanded_17 : tensor<1x96x112x112xf32>, tensor<96x1x1xf32>) outs(%37 : tensor<1x96x112x112xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.subf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x96x112x112xf32>
    %45 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%44, %expanded_18 : tensor<1x96x112x112xf32>, tensor<96x1x1xf32>) outs(%37 : tensor<1x96x112x112xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x96x112x112xf32>
    %expanded_19 = tensor.expand_shape %arg167 [[0, 1, 2]] output_shape [96, 1, 1] : tensor<96xf32> into tensor<96x1x1xf32>
    %46 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%45, %expanded_19 : tensor<1x96x112x112xf32>, tensor<96x1x1xf32>) outs(%37 : tensor<1x96x112x112xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x96x112x112xf32>
    %expanded_20 = tensor.expand_shape %arg168 [[0, 1, 2]] output_shape [96, 1, 1] : tensor<96xf32> into tensor<96x1x1xf32>
    %47 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%46, %expanded_20 : tensor<1x96x112x112xf32>, tensor<96x1x1xf32>) outs(%37 : tensor<1x96x112x112xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.addf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x96x112x112xf32>
    %48 = linalg.generic {indexing_maps = [#map1, #map4, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%47, %12 : tensor<1x96x112x112xf32>, tensor<f32>) outs(%37 : tensor<1x96x112x112xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.cmpf ogt, %in, %in_246 : f32
      %561 = arith.select %560, %in, %in_246 : f32
      linalg.yield %561 : f32
    } -> tensor<1x96x112x112xf32>
    %49 = linalg.generic {indexing_maps = [#map4, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%14, %48 : tensor<f32>, tensor<1x96x112x112xf32>) outs(%37 : tensor<1x96x112x112xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.cmpf olt, %in, %in_246 : f32
      %561 = arith.select %560, %in, %in_246 : f32
      linalg.yield %561 : f32
    } -> tensor<1x96x112x112xf32>
    %padded_21 = tensor.pad %49 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg315: index, %arg316: index, %arg317: index, %arg318: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x96x112x112xf32> to tensor<1x96x114x114xf32>
    %50 = tensor.empty() : tensor<1x96x56x56xf32>
    %51 = linalg.fill ins(%cst_0 : f32) outs(%50 : tensor<1x96x56x56xf32>) -> tensor<1x96x56x56xf32>
    %collapsed_22 = tensor.collapse_shape %arg169 [[0, 1], [2], [3]] : tensor<96x1x3x3xf32> into tensor<96x3x3xf32>
    %52 = linalg.depthwise_conv_2d_nchw_chw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%padded_21, %collapsed_22 : tensor<1x96x114x114xf32>, tensor<96x3x3xf32>) outs(%51 : tensor<1x96x56x56xf32>) -> tensor<1x96x56x56xf32>
    %53 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg13 : tensor<96xf32>) outs(%40 : tensor<96xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.truncf %cst_2 : f64 to f32
      %561 = arith.addf %in, %560 : f32
      linalg.yield %561 : f32
    } -> tensor<96xf32>
    %54 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%53 : tensor<96xf32>) outs(%40 : tensor<96xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = math.sqrt %in : f32
      linalg.yield %560 : f32
    } -> tensor<96xf32>
    %55 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%54 : tensor<96xf32>) outs(%40 : tensor<96xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.cmpf one, %in, %cst_0 : f32
      cf.assert %560, "unimplemented: tensor with zero element"
      %561 = arith.divf %cst_1, %in : f32
      linalg.yield %561 : f32
    } -> tensor<96xf32>
    %expanded_23 = tensor.expand_shape %arg12 [[0, 1, 2]] output_shape [96, 1, 1] : tensor<96xf32> into tensor<96x1x1xf32>
    %expanded_24 = tensor.expand_shape %55 [[0, 1, 2]] output_shape [96, 1, 1] : tensor<96xf32> into tensor<96x1x1xf32>
    %56 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%52, %expanded_23 : tensor<1x96x56x56xf32>, tensor<96x1x1xf32>) outs(%50 : tensor<1x96x56x56xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.subf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x96x56x56xf32>
    %57 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%56, %expanded_24 : tensor<1x96x56x56xf32>, tensor<96x1x1xf32>) outs(%50 : tensor<1x96x56x56xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x96x56x56xf32>
    %expanded_25 = tensor.expand_shape %arg170 [[0, 1, 2]] output_shape [96, 1, 1] : tensor<96xf32> into tensor<96x1x1xf32>
    %58 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%57, %expanded_25 : tensor<1x96x56x56xf32>, tensor<96x1x1xf32>) outs(%50 : tensor<1x96x56x56xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x96x56x56xf32>
    %expanded_26 = tensor.expand_shape %arg171 [[0, 1, 2]] output_shape [96, 1, 1] : tensor<96xf32> into tensor<96x1x1xf32>
    %59 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%58, %expanded_26 : tensor<1x96x56x56xf32>, tensor<96x1x1xf32>) outs(%50 : tensor<1x96x56x56xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.addf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x96x56x56xf32>
    %60 = linalg.generic {indexing_maps = [#map1, #map4, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%59, %12 : tensor<1x96x56x56xf32>, tensor<f32>) outs(%50 : tensor<1x96x56x56xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.cmpf ogt, %in, %in_246 : f32
      %561 = arith.select %560, %in, %in_246 : f32
      linalg.yield %561 : f32
    } -> tensor<1x96x56x56xf32>
    %61 = linalg.generic {indexing_maps = [#map4, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%14, %60 : tensor<f32>, tensor<1x96x56x56xf32>) outs(%50 : tensor<1x96x56x56xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.cmpf olt, %in, %in_246 : f32
      %561 = arith.select %560, %in, %in_246 : f32
      linalg.yield %561 : f32
    } -> tensor<1x96x56x56xf32>
    %62 = tensor.empty() : tensor<1x24x56x56xf32>
    %63 = linalg.fill ins(%cst_0 : f32) outs(%62 : tensor<1x24x56x56xf32>) -> tensor<1x24x56x56xf32>
    %64 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%61, %arg172 : tensor<1x96x56x56xf32>, tensor<24x96x1x1xf32>) outs(%63 : tensor<1x24x56x56xf32>) -> tensor<1x24x56x56xf32>
    %65 = tensor.empty() : tensor<24xf32>
    %66 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg16 : tensor<24xf32>) outs(%65 : tensor<24xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.truncf %cst_2 : f64 to f32
      %561 = arith.addf %in, %560 : f32
      linalg.yield %561 : f32
    } -> tensor<24xf32>
    %67 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%66 : tensor<24xf32>) outs(%65 : tensor<24xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = math.sqrt %in : f32
      linalg.yield %560 : f32
    } -> tensor<24xf32>
    %68 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%67 : tensor<24xf32>) outs(%65 : tensor<24xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.cmpf one, %in, %cst_0 : f32
      cf.assert %560, "unimplemented: tensor with zero element"
      %561 = arith.divf %cst_1, %in : f32
      linalg.yield %561 : f32
    } -> tensor<24xf32>
    %expanded_27 = tensor.expand_shape %arg15 [[0, 1, 2]] output_shape [24, 1, 1] : tensor<24xf32> into tensor<24x1x1xf32>
    %expanded_28 = tensor.expand_shape %68 [[0, 1, 2]] output_shape [24, 1, 1] : tensor<24xf32> into tensor<24x1x1xf32>
    %69 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%64, %expanded_27 : tensor<1x24x56x56xf32>, tensor<24x1x1xf32>) outs(%62 : tensor<1x24x56x56xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.subf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x24x56x56xf32>
    %70 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%69, %expanded_28 : tensor<1x24x56x56xf32>, tensor<24x1x1xf32>) outs(%62 : tensor<1x24x56x56xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x24x56x56xf32>
    %expanded_29 = tensor.expand_shape %arg173 [[0, 1, 2]] output_shape [24, 1, 1] : tensor<24xf32> into tensor<24x1x1xf32>
    %71 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%70, %expanded_29 : tensor<1x24x56x56xf32>, tensor<24x1x1xf32>) outs(%62 : tensor<1x24x56x56xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x24x56x56xf32>
    %expanded_30 = tensor.expand_shape %arg174 [[0, 1, 2]] output_shape [24, 1, 1] : tensor<24xf32> into tensor<24x1x1xf32>
    %72 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%71, %expanded_30 : tensor<1x24x56x56xf32>, tensor<24x1x1xf32>) outs(%62 : tensor<1x24x56x56xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.addf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x24x56x56xf32>
    %73 = tensor.empty() : tensor<1x144x56x56xf32>
    %74 = linalg.fill ins(%cst_0 : f32) outs(%73 : tensor<1x144x56x56xf32>) -> tensor<1x144x56x56xf32>
    %75 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%72, %arg175 : tensor<1x24x56x56xf32>, tensor<144x24x1x1xf32>) outs(%74 : tensor<1x144x56x56xf32>) -> tensor<1x144x56x56xf32>
    %76 = tensor.empty() : tensor<144xf32>
    %77 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg19 : tensor<144xf32>) outs(%76 : tensor<144xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.truncf %cst_2 : f64 to f32
      %561 = arith.addf %in, %560 : f32
      linalg.yield %561 : f32
    } -> tensor<144xf32>
    %78 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%77 : tensor<144xf32>) outs(%76 : tensor<144xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = math.sqrt %in : f32
      linalg.yield %560 : f32
    } -> tensor<144xf32>
    %79 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%78 : tensor<144xf32>) outs(%76 : tensor<144xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.cmpf one, %in, %cst_0 : f32
      cf.assert %560, "unimplemented: tensor with zero element"
      %561 = arith.divf %cst_1, %in : f32
      linalg.yield %561 : f32
    } -> tensor<144xf32>
    %expanded_31 = tensor.expand_shape %arg18 [[0, 1, 2]] output_shape [144, 1, 1] : tensor<144xf32> into tensor<144x1x1xf32>
    %expanded_32 = tensor.expand_shape %79 [[0, 1, 2]] output_shape [144, 1, 1] : tensor<144xf32> into tensor<144x1x1xf32>
    %80 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%75, %expanded_31 : tensor<1x144x56x56xf32>, tensor<144x1x1xf32>) outs(%73 : tensor<1x144x56x56xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.subf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x144x56x56xf32>
    %81 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%80, %expanded_32 : tensor<1x144x56x56xf32>, tensor<144x1x1xf32>) outs(%73 : tensor<1x144x56x56xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x144x56x56xf32>
    %expanded_33 = tensor.expand_shape %arg176 [[0, 1, 2]] output_shape [144, 1, 1] : tensor<144xf32> into tensor<144x1x1xf32>
    %82 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%81, %expanded_33 : tensor<1x144x56x56xf32>, tensor<144x1x1xf32>) outs(%73 : tensor<1x144x56x56xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x144x56x56xf32>
    %expanded_34 = tensor.expand_shape %arg177 [[0, 1, 2]] output_shape [144, 1, 1] : tensor<144xf32> into tensor<144x1x1xf32>
    %83 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%82, %expanded_34 : tensor<1x144x56x56xf32>, tensor<144x1x1xf32>) outs(%73 : tensor<1x144x56x56xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.addf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x144x56x56xf32>
    %84 = linalg.generic {indexing_maps = [#map1, #map4, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%83, %12 : tensor<1x144x56x56xf32>, tensor<f32>) outs(%73 : tensor<1x144x56x56xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.cmpf ogt, %in, %in_246 : f32
      %561 = arith.select %560, %in, %in_246 : f32
      linalg.yield %561 : f32
    } -> tensor<1x144x56x56xf32>
    %85 = linalg.generic {indexing_maps = [#map4, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%14, %84 : tensor<f32>, tensor<1x144x56x56xf32>) outs(%73 : tensor<1x144x56x56xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.cmpf olt, %in, %in_246 : f32
      %561 = arith.select %560, %in, %in_246 : f32
      linalg.yield %561 : f32
    } -> tensor<1x144x56x56xf32>
    %padded_35 = tensor.pad %85 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg315: index, %arg316: index, %arg317: index, %arg318: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x144x56x56xf32> to tensor<1x144x58x58xf32>
    %collapsed_36 = tensor.collapse_shape %arg178 [[0, 1], [2], [3]] : tensor<144x1x3x3xf32> into tensor<144x3x3xf32>
    %86 = linalg.depthwise_conv_2d_nchw_chw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_35, %collapsed_36 : tensor<1x144x58x58xf32>, tensor<144x3x3xf32>) outs(%74 : tensor<1x144x56x56xf32>) -> tensor<1x144x56x56xf32>
    %87 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg22 : tensor<144xf32>) outs(%76 : tensor<144xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.truncf %cst_2 : f64 to f32
      %561 = arith.addf %in, %560 : f32
      linalg.yield %561 : f32
    } -> tensor<144xf32>
    %88 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%87 : tensor<144xf32>) outs(%76 : tensor<144xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = math.sqrt %in : f32
      linalg.yield %560 : f32
    } -> tensor<144xf32>
    %89 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%88 : tensor<144xf32>) outs(%76 : tensor<144xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.cmpf one, %in, %cst_0 : f32
      cf.assert %560, "unimplemented: tensor with zero element"
      %561 = arith.divf %cst_1, %in : f32
      linalg.yield %561 : f32
    } -> tensor<144xf32>
    %expanded_37 = tensor.expand_shape %arg21 [[0, 1, 2]] output_shape [144, 1, 1] : tensor<144xf32> into tensor<144x1x1xf32>
    %expanded_38 = tensor.expand_shape %89 [[0, 1, 2]] output_shape [144, 1, 1] : tensor<144xf32> into tensor<144x1x1xf32>
    %90 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%86, %expanded_37 : tensor<1x144x56x56xf32>, tensor<144x1x1xf32>) outs(%73 : tensor<1x144x56x56xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.subf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x144x56x56xf32>
    %91 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%90, %expanded_38 : tensor<1x144x56x56xf32>, tensor<144x1x1xf32>) outs(%73 : tensor<1x144x56x56xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x144x56x56xf32>
    %expanded_39 = tensor.expand_shape %arg179 [[0, 1, 2]] output_shape [144, 1, 1] : tensor<144xf32> into tensor<144x1x1xf32>
    %92 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%91, %expanded_39 : tensor<1x144x56x56xf32>, tensor<144x1x1xf32>) outs(%73 : tensor<1x144x56x56xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x144x56x56xf32>
    %expanded_40 = tensor.expand_shape %arg180 [[0, 1, 2]] output_shape [144, 1, 1] : tensor<144xf32> into tensor<144x1x1xf32>
    %93 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%92, %expanded_40 : tensor<1x144x56x56xf32>, tensor<144x1x1xf32>) outs(%73 : tensor<1x144x56x56xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.addf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x144x56x56xf32>
    %94 = linalg.generic {indexing_maps = [#map1, #map4, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%93, %12 : tensor<1x144x56x56xf32>, tensor<f32>) outs(%73 : tensor<1x144x56x56xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.cmpf ogt, %in, %in_246 : f32
      %561 = arith.select %560, %in, %in_246 : f32
      linalg.yield %561 : f32
    } -> tensor<1x144x56x56xf32>
    %95 = linalg.generic {indexing_maps = [#map4, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%14, %94 : tensor<f32>, tensor<1x144x56x56xf32>) outs(%73 : tensor<1x144x56x56xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.cmpf olt, %in, %in_246 : f32
      %561 = arith.select %560, %in, %in_246 : f32
      linalg.yield %561 : f32
    } -> tensor<1x144x56x56xf32>
    %96 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%95, %arg181 : tensor<1x144x56x56xf32>, tensor<24x144x1x1xf32>) outs(%63 : tensor<1x24x56x56xf32>) -> tensor<1x24x56x56xf32>
    %97 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg25 : tensor<24xf32>) outs(%65 : tensor<24xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.truncf %cst_2 : f64 to f32
      %561 = arith.addf %in, %560 : f32
      linalg.yield %561 : f32
    } -> tensor<24xf32>
    %98 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%97 : tensor<24xf32>) outs(%65 : tensor<24xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = math.sqrt %in : f32
      linalg.yield %560 : f32
    } -> tensor<24xf32>
    %99 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%98 : tensor<24xf32>) outs(%65 : tensor<24xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.cmpf one, %in, %cst_0 : f32
      cf.assert %560, "unimplemented: tensor with zero element"
      %561 = arith.divf %cst_1, %in : f32
      linalg.yield %561 : f32
    } -> tensor<24xf32>
    %expanded_41 = tensor.expand_shape %arg24 [[0, 1, 2]] output_shape [24, 1, 1] : tensor<24xf32> into tensor<24x1x1xf32>
    %expanded_42 = tensor.expand_shape %99 [[0, 1, 2]] output_shape [24, 1, 1] : tensor<24xf32> into tensor<24x1x1xf32>
    %100 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%96, %expanded_41 : tensor<1x24x56x56xf32>, tensor<24x1x1xf32>) outs(%62 : tensor<1x24x56x56xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.subf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x24x56x56xf32>
    %101 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%100, %expanded_42 : tensor<1x24x56x56xf32>, tensor<24x1x1xf32>) outs(%62 : tensor<1x24x56x56xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x24x56x56xf32>
    %expanded_43 = tensor.expand_shape %arg182 [[0, 1, 2]] output_shape [24, 1, 1] : tensor<24xf32> into tensor<24x1x1xf32>
    %102 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%101, %expanded_43 : tensor<1x24x56x56xf32>, tensor<24x1x1xf32>) outs(%62 : tensor<1x24x56x56xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x24x56x56xf32>
    %expanded_44 = tensor.expand_shape %arg183 [[0, 1, 2]] output_shape [24, 1, 1] : tensor<24xf32> into tensor<24x1x1xf32>
    %103 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%102, %expanded_44 : tensor<1x24x56x56xf32>, tensor<24x1x1xf32>) outs(%62 : tensor<1x24x56x56xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.addf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x24x56x56xf32>
    %104 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%72, %103 : tensor<1x24x56x56xf32>, tensor<1x24x56x56xf32>) outs(%62 : tensor<1x24x56x56xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.addf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x24x56x56xf32>
    %105 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%104, %arg184 : tensor<1x24x56x56xf32>, tensor<144x24x1x1xf32>) outs(%74 : tensor<1x144x56x56xf32>) -> tensor<1x144x56x56xf32>
    %106 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg28 : tensor<144xf32>) outs(%76 : tensor<144xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.truncf %cst_2 : f64 to f32
      %561 = arith.addf %in, %560 : f32
      linalg.yield %561 : f32
    } -> tensor<144xf32>
    %107 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%106 : tensor<144xf32>) outs(%76 : tensor<144xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = math.sqrt %in : f32
      linalg.yield %560 : f32
    } -> tensor<144xf32>
    %108 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%107 : tensor<144xf32>) outs(%76 : tensor<144xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.cmpf one, %in, %cst_0 : f32
      cf.assert %560, "unimplemented: tensor with zero element"
      %561 = arith.divf %cst_1, %in : f32
      linalg.yield %561 : f32
    } -> tensor<144xf32>
    %expanded_45 = tensor.expand_shape %arg27 [[0, 1, 2]] output_shape [144, 1, 1] : tensor<144xf32> into tensor<144x1x1xf32>
    %expanded_46 = tensor.expand_shape %108 [[0, 1, 2]] output_shape [144, 1, 1] : tensor<144xf32> into tensor<144x1x1xf32>
    %109 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%105, %expanded_45 : tensor<1x144x56x56xf32>, tensor<144x1x1xf32>) outs(%73 : tensor<1x144x56x56xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.subf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x144x56x56xf32>
    %110 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%109, %expanded_46 : tensor<1x144x56x56xf32>, tensor<144x1x1xf32>) outs(%73 : tensor<1x144x56x56xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x144x56x56xf32>
    %expanded_47 = tensor.expand_shape %arg185 [[0, 1, 2]] output_shape [144, 1, 1] : tensor<144xf32> into tensor<144x1x1xf32>
    %111 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%110, %expanded_47 : tensor<1x144x56x56xf32>, tensor<144x1x1xf32>) outs(%73 : tensor<1x144x56x56xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x144x56x56xf32>
    %expanded_48 = tensor.expand_shape %arg186 [[0, 1, 2]] output_shape [144, 1, 1] : tensor<144xf32> into tensor<144x1x1xf32>
    %112 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%111, %expanded_48 : tensor<1x144x56x56xf32>, tensor<144x1x1xf32>) outs(%73 : tensor<1x144x56x56xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.addf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x144x56x56xf32>
    %113 = linalg.generic {indexing_maps = [#map1, #map4, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%112, %12 : tensor<1x144x56x56xf32>, tensor<f32>) outs(%73 : tensor<1x144x56x56xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.cmpf ogt, %in, %in_246 : f32
      %561 = arith.select %560, %in, %in_246 : f32
      linalg.yield %561 : f32
    } -> tensor<1x144x56x56xf32>
    %114 = linalg.generic {indexing_maps = [#map4, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%14, %113 : tensor<f32>, tensor<1x144x56x56xf32>) outs(%73 : tensor<1x144x56x56xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.cmpf olt, %in, %in_246 : f32
      %561 = arith.select %560, %in, %in_246 : f32
      linalg.yield %561 : f32
    } -> tensor<1x144x56x56xf32>
    %padded_49 = tensor.pad %114 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg315: index, %arg316: index, %arg317: index, %arg318: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x144x56x56xf32> to tensor<1x144x58x58xf32>
    %115 = tensor.empty() : tensor<1x144x28x28xf32>
    %116 = linalg.fill ins(%cst_0 : f32) outs(%115 : tensor<1x144x28x28xf32>) -> tensor<1x144x28x28xf32>
    %collapsed_50 = tensor.collapse_shape %arg187 [[0, 1], [2], [3]] : tensor<144x1x3x3xf32> into tensor<144x3x3xf32>
    %117 = linalg.depthwise_conv_2d_nchw_chw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%padded_49, %collapsed_50 : tensor<1x144x58x58xf32>, tensor<144x3x3xf32>) outs(%116 : tensor<1x144x28x28xf32>) -> tensor<1x144x28x28xf32>
    %118 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg31 : tensor<144xf32>) outs(%76 : tensor<144xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.truncf %cst_2 : f64 to f32
      %561 = arith.addf %in, %560 : f32
      linalg.yield %561 : f32
    } -> tensor<144xf32>
    %119 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%118 : tensor<144xf32>) outs(%76 : tensor<144xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = math.sqrt %in : f32
      linalg.yield %560 : f32
    } -> tensor<144xf32>
    %120 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%119 : tensor<144xf32>) outs(%76 : tensor<144xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.cmpf one, %in, %cst_0 : f32
      cf.assert %560, "unimplemented: tensor with zero element"
      %561 = arith.divf %cst_1, %in : f32
      linalg.yield %561 : f32
    } -> tensor<144xf32>
    %expanded_51 = tensor.expand_shape %arg30 [[0, 1, 2]] output_shape [144, 1, 1] : tensor<144xf32> into tensor<144x1x1xf32>
    %expanded_52 = tensor.expand_shape %120 [[0, 1, 2]] output_shape [144, 1, 1] : tensor<144xf32> into tensor<144x1x1xf32>
    %121 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%117, %expanded_51 : tensor<1x144x28x28xf32>, tensor<144x1x1xf32>) outs(%115 : tensor<1x144x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.subf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x144x28x28xf32>
    %122 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%121, %expanded_52 : tensor<1x144x28x28xf32>, tensor<144x1x1xf32>) outs(%115 : tensor<1x144x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x144x28x28xf32>
    %expanded_53 = tensor.expand_shape %arg188 [[0, 1, 2]] output_shape [144, 1, 1] : tensor<144xf32> into tensor<144x1x1xf32>
    %123 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%122, %expanded_53 : tensor<1x144x28x28xf32>, tensor<144x1x1xf32>) outs(%115 : tensor<1x144x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x144x28x28xf32>
    %expanded_54 = tensor.expand_shape %arg189 [[0, 1, 2]] output_shape [144, 1, 1] : tensor<144xf32> into tensor<144x1x1xf32>
    %124 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%123, %expanded_54 : tensor<1x144x28x28xf32>, tensor<144x1x1xf32>) outs(%115 : tensor<1x144x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.addf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x144x28x28xf32>
    %125 = linalg.generic {indexing_maps = [#map1, #map4, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%124, %12 : tensor<1x144x28x28xf32>, tensor<f32>) outs(%115 : tensor<1x144x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.cmpf ogt, %in, %in_246 : f32
      %561 = arith.select %560, %in, %in_246 : f32
      linalg.yield %561 : f32
    } -> tensor<1x144x28x28xf32>
    %126 = linalg.generic {indexing_maps = [#map4, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%14, %125 : tensor<f32>, tensor<1x144x28x28xf32>) outs(%115 : tensor<1x144x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.cmpf olt, %in, %in_246 : f32
      %561 = arith.select %560, %in, %in_246 : f32
      linalg.yield %561 : f32
    } -> tensor<1x144x28x28xf32>
    %127 = tensor.empty() : tensor<1x32x28x28xf32>
    %128 = linalg.fill ins(%cst_0 : f32) outs(%127 : tensor<1x32x28x28xf32>) -> tensor<1x32x28x28xf32>
    %129 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%126, %arg190 : tensor<1x144x28x28xf32>, tensor<32x144x1x1xf32>) outs(%128 : tensor<1x32x28x28xf32>) -> tensor<1x32x28x28xf32>
    %130 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg34 : tensor<32xf32>) outs(%3 : tensor<32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.truncf %cst_2 : f64 to f32
      %561 = arith.addf %in, %560 : f32
      linalg.yield %561 : f32
    } -> tensor<32xf32>
    %131 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%130 : tensor<32xf32>) outs(%3 : tensor<32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = math.sqrt %in : f32
      linalg.yield %560 : f32
    } -> tensor<32xf32>
    %132 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%131 : tensor<32xf32>) outs(%3 : tensor<32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.cmpf one, %in, %cst_0 : f32
      cf.assert %560, "unimplemented: tensor with zero element"
      %561 = arith.divf %cst_1, %in : f32
      linalg.yield %561 : f32
    } -> tensor<32xf32>
    %expanded_55 = tensor.expand_shape %arg33 [[0, 1, 2]] output_shape [32, 1, 1] : tensor<32xf32> into tensor<32x1x1xf32>
    %expanded_56 = tensor.expand_shape %132 [[0, 1, 2]] output_shape [32, 1, 1] : tensor<32xf32> into tensor<32x1x1xf32>
    %133 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%129, %expanded_55 : tensor<1x32x28x28xf32>, tensor<32x1x1xf32>) outs(%127 : tensor<1x32x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.subf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x32x28x28xf32>
    %134 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%133, %expanded_56 : tensor<1x32x28x28xf32>, tensor<32x1x1xf32>) outs(%127 : tensor<1x32x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x32x28x28xf32>
    %expanded_57 = tensor.expand_shape %arg191 [[0, 1, 2]] output_shape [32, 1, 1] : tensor<32xf32> into tensor<32x1x1xf32>
    %135 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%134, %expanded_57 : tensor<1x32x28x28xf32>, tensor<32x1x1xf32>) outs(%127 : tensor<1x32x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x32x28x28xf32>
    %expanded_58 = tensor.expand_shape %arg192 [[0, 1, 2]] output_shape [32, 1, 1] : tensor<32xf32> into tensor<32x1x1xf32>
    %136 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%135, %expanded_58 : tensor<1x32x28x28xf32>, tensor<32x1x1xf32>) outs(%127 : tensor<1x32x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.addf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x32x28x28xf32>
    %137 = tensor.empty() : tensor<1x192x28x28xf32>
    %138 = linalg.fill ins(%cst_0 : f32) outs(%137 : tensor<1x192x28x28xf32>) -> tensor<1x192x28x28xf32>
    %139 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%136, %arg193 : tensor<1x32x28x28xf32>, tensor<192x32x1x1xf32>) outs(%138 : tensor<1x192x28x28xf32>) -> tensor<1x192x28x28xf32>
    %140 = tensor.empty() : tensor<192xf32>
    %141 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg37 : tensor<192xf32>) outs(%140 : tensor<192xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.truncf %cst_2 : f64 to f32
      %561 = arith.addf %in, %560 : f32
      linalg.yield %561 : f32
    } -> tensor<192xf32>
    %142 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%141 : tensor<192xf32>) outs(%140 : tensor<192xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = math.sqrt %in : f32
      linalg.yield %560 : f32
    } -> tensor<192xf32>
    %143 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%142 : tensor<192xf32>) outs(%140 : tensor<192xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.cmpf one, %in, %cst_0 : f32
      cf.assert %560, "unimplemented: tensor with zero element"
      %561 = arith.divf %cst_1, %in : f32
      linalg.yield %561 : f32
    } -> tensor<192xf32>
    %expanded_59 = tensor.expand_shape %arg36 [[0, 1, 2]] output_shape [192, 1, 1] : tensor<192xf32> into tensor<192x1x1xf32>
    %expanded_60 = tensor.expand_shape %143 [[0, 1, 2]] output_shape [192, 1, 1] : tensor<192xf32> into tensor<192x1x1xf32>
    %144 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%139, %expanded_59 : tensor<1x192x28x28xf32>, tensor<192x1x1xf32>) outs(%137 : tensor<1x192x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.subf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x192x28x28xf32>
    %145 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%144, %expanded_60 : tensor<1x192x28x28xf32>, tensor<192x1x1xf32>) outs(%137 : tensor<1x192x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x192x28x28xf32>
    %expanded_61 = tensor.expand_shape %arg194 [[0, 1, 2]] output_shape [192, 1, 1] : tensor<192xf32> into tensor<192x1x1xf32>
    %146 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%145, %expanded_61 : tensor<1x192x28x28xf32>, tensor<192x1x1xf32>) outs(%137 : tensor<1x192x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x192x28x28xf32>
    %expanded_62 = tensor.expand_shape %arg195 [[0, 1, 2]] output_shape [192, 1, 1] : tensor<192xf32> into tensor<192x1x1xf32>
    %147 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%146, %expanded_62 : tensor<1x192x28x28xf32>, tensor<192x1x1xf32>) outs(%137 : tensor<1x192x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.addf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x192x28x28xf32>
    %148 = linalg.generic {indexing_maps = [#map1, #map4, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%147, %12 : tensor<1x192x28x28xf32>, tensor<f32>) outs(%137 : tensor<1x192x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.cmpf ogt, %in, %in_246 : f32
      %561 = arith.select %560, %in, %in_246 : f32
      linalg.yield %561 : f32
    } -> tensor<1x192x28x28xf32>
    %149 = linalg.generic {indexing_maps = [#map4, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%14, %148 : tensor<f32>, tensor<1x192x28x28xf32>) outs(%137 : tensor<1x192x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.cmpf olt, %in, %in_246 : f32
      %561 = arith.select %560, %in, %in_246 : f32
      linalg.yield %561 : f32
    } -> tensor<1x192x28x28xf32>
    %padded_63 = tensor.pad %149 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg315: index, %arg316: index, %arg317: index, %arg318: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x192x28x28xf32> to tensor<1x192x30x30xf32>
    %collapsed_64 = tensor.collapse_shape %arg196 [[0, 1], [2], [3]] : tensor<192x1x3x3xf32> into tensor<192x3x3xf32>
    %150 = linalg.depthwise_conv_2d_nchw_chw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_63, %collapsed_64 : tensor<1x192x30x30xf32>, tensor<192x3x3xf32>) outs(%138 : tensor<1x192x28x28xf32>) -> tensor<1x192x28x28xf32>
    %151 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg40 : tensor<192xf32>) outs(%140 : tensor<192xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.truncf %cst_2 : f64 to f32
      %561 = arith.addf %in, %560 : f32
      linalg.yield %561 : f32
    } -> tensor<192xf32>
    %152 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%151 : tensor<192xf32>) outs(%140 : tensor<192xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = math.sqrt %in : f32
      linalg.yield %560 : f32
    } -> tensor<192xf32>
    %153 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%152 : tensor<192xf32>) outs(%140 : tensor<192xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.cmpf one, %in, %cst_0 : f32
      cf.assert %560, "unimplemented: tensor with zero element"
      %561 = arith.divf %cst_1, %in : f32
      linalg.yield %561 : f32
    } -> tensor<192xf32>
    %expanded_65 = tensor.expand_shape %arg39 [[0, 1, 2]] output_shape [192, 1, 1] : tensor<192xf32> into tensor<192x1x1xf32>
    %expanded_66 = tensor.expand_shape %153 [[0, 1, 2]] output_shape [192, 1, 1] : tensor<192xf32> into tensor<192x1x1xf32>
    %154 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%150, %expanded_65 : tensor<1x192x28x28xf32>, tensor<192x1x1xf32>) outs(%137 : tensor<1x192x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.subf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x192x28x28xf32>
    %155 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%154, %expanded_66 : tensor<1x192x28x28xf32>, tensor<192x1x1xf32>) outs(%137 : tensor<1x192x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x192x28x28xf32>
    %expanded_67 = tensor.expand_shape %arg197 [[0, 1, 2]] output_shape [192, 1, 1] : tensor<192xf32> into tensor<192x1x1xf32>
    %156 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%155, %expanded_67 : tensor<1x192x28x28xf32>, tensor<192x1x1xf32>) outs(%137 : tensor<1x192x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x192x28x28xf32>
    %expanded_68 = tensor.expand_shape %arg198 [[0, 1, 2]] output_shape [192, 1, 1] : tensor<192xf32> into tensor<192x1x1xf32>
    %157 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%156, %expanded_68 : tensor<1x192x28x28xf32>, tensor<192x1x1xf32>) outs(%137 : tensor<1x192x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.addf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x192x28x28xf32>
    %158 = linalg.generic {indexing_maps = [#map1, #map4, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%157, %12 : tensor<1x192x28x28xf32>, tensor<f32>) outs(%137 : tensor<1x192x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.cmpf ogt, %in, %in_246 : f32
      %561 = arith.select %560, %in, %in_246 : f32
      linalg.yield %561 : f32
    } -> tensor<1x192x28x28xf32>
    %159 = linalg.generic {indexing_maps = [#map4, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%14, %158 : tensor<f32>, tensor<1x192x28x28xf32>) outs(%137 : tensor<1x192x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.cmpf olt, %in, %in_246 : f32
      %561 = arith.select %560, %in, %in_246 : f32
      linalg.yield %561 : f32
    } -> tensor<1x192x28x28xf32>
    %160 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%159, %arg199 : tensor<1x192x28x28xf32>, tensor<32x192x1x1xf32>) outs(%128 : tensor<1x32x28x28xf32>) -> tensor<1x32x28x28xf32>
    %161 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg43 : tensor<32xf32>) outs(%3 : tensor<32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.truncf %cst_2 : f64 to f32
      %561 = arith.addf %in, %560 : f32
      linalg.yield %561 : f32
    } -> tensor<32xf32>
    %162 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%161 : tensor<32xf32>) outs(%3 : tensor<32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = math.sqrt %in : f32
      linalg.yield %560 : f32
    } -> tensor<32xf32>
    %163 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%162 : tensor<32xf32>) outs(%3 : tensor<32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.cmpf one, %in, %cst_0 : f32
      cf.assert %560, "unimplemented: tensor with zero element"
      %561 = arith.divf %cst_1, %in : f32
      linalg.yield %561 : f32
    } -> tensor<32xf32>
    %expanded_69 = tensor.expand_shape %arg42 [[0, 1, 2]] output_shape [32, 1, 1] : tensor<32xf32> into tensor<32x1x1xf32>
    %expanded_70 = tensor.expand_shape %163 [[0, 1, 2]] output_shape [32, 1, 1] : tensor<32xf32> into tensor<32x1x1xf32>
    %164 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%160, %expanded_69 : tensor<1x32x28x28xf32>, tensor<32x1x1xf32>) outs(%127 : tensor<1x32x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.subf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x32x28x28xf32>
    %165 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%164, %expanded_70 : tensor<1x32x28x28xf32>, tensor<32x1x1xf32>) outs(%127 : tensor<1x32x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x32x28x28xf32>
    %expanded_71 = tensor.expand_shape %arg200 [[0, 1, 2]] output_shape [32, 1, 1] : tensor<32xf32> into tensor<32x1x1xf32>
    %166 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%165, %expanded_71 : tensor<1x32x28x28xf32>, tensor<32x1x1xf32>) outs(%127 : tensor<1x32x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x32x28x28xf32>
    %expanded_72 = tensor.expand_shape %arg201 [[0, 1, 2]] output_shape [32, 1, 1] : tensor<32xf32> into tensor<32x1x1xf32>
    %167 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%166, %expanded_72 : tensor<1x32x28x28xf32>, tensor<32x1x1xf32>) outs(%127 : tensor<1x32x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.addf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x32x28x28xf32>
    %168 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%136, %167 : tensor<1x32x28x28xf32>, tensor<1x32x28x28xf32>) outs(%127 : tensor<1x32x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.addf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x32x28x28xf32>
    %169 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%168, %arg202 : tensor<1x32x28x28xf32>, tensor<192x32x1x1xf32>) outs(%138 : tensor<1x192x28x28xf32>) -> tensor<1x192x28x28xf32>
    %170 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg46 : tensor<192xf32>) outs(%140 : tensor<192xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.truncf %cst_2 : f64 to f32
      %561 = arith.addf %in, %560 : f32
      linalg.yield %561 : f32
    } -> tensor<192xf32>
    %171 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%170 : tensor<192xf32>) outs(%140 : tensor<192xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = math.sqrt %in : f32
      linalg.yield %560 : f32
    } -> tensor<192xf32>
    %172 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%171 : tensor<192xf32>) outs(%140 : tensor<192xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.cmpf one, %in, %cst_0 : f32
      cf.assert %560, "unimplemented: tensor with zero element"
      %561 = arith.divf %cst_1, %in : f32
      linalg.yield %561 : f32
    } -> tensor<192xf32>
    %expanded_73 = tensor.expand_shape %arg45 [[0, 1, 2]] output_shape [192, 1, 1] : tensor<192xf32> into tensor<192x1x1xf32>
    %expanded_74 = tensor.expand_shape %172 [[0, 1, 2]] output_shape [192, 1, 1] : tensor<192xf32> into tensor<192x1x1xf32>
    %173 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%169, %expanded_73 : tensor<1x192x28x28xf32>, tensor<192x1x1xf32>) outs(%137 : tensor<1x192x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.subf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x192x28x28xf32>
    %174 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%173, %expanded_74 : tensor<1x192x28x28xf32>, tensor<192x1x1xf32>) outs(%137 : tensor<1x192x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x192x28x28xf32>
    %expanded_75 = tensor.expand_shape %arg203 [[0, 1, 2]] output_shape [192, 1, 1] : tensor<192xf32> into tensor<192x1x1xf32>
    %175 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%174, %expanded_75 : tensor<1x192x28x28xf32>, tensor<192x1x1xf32>) outs(%137 : tensor<1x192x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x192x28x28xf32>
    %expanded_76 = tensor.expand_shape %arg204 [[0, 1, 2]] output_shape [192, 1, 1] : tensor<192xf32> into tensor<192x1x1xf32>
    %176 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%175, %expanded_76 : tensor<1x192x28x28xf32>, tensor<192x1x1xf32>) outs(%137 : tensor<1x192x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.addf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x192x28x28xf32>
    %177 = linalg.generic {indexing_maps = [#map1, #map4, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%176, %12 : tensor<1x192x28x28xf32>, tensor<f32>) outs(%137 : tensor<1x192x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.cmpf ogt, %in, %in_246 : f32
      %561 = arith.select %560, %in, %in_246 : f32
      linalg.yield %561 : f32
    } -> tensor<1x192x28x28xf32>
    %178 = linalg.generic {indexing_maps = [#map4, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%14, %177 : tensor<f32>, tensor<1x192x28x28xf32>) outs(%137 : tensor<1x192x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.cmpf olt, %in, %in_246 : f32
      %561 = arith.select %560, %in, %in_246 : f32
      linalg.yield %561 : f32
    } -> tensor<1x192x28x28xf32>
    %padded_77 = tensor.pad %178 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg315: index, %arg316: index, %arg317: index, %arg318: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x192x28x28xf32> to tensor<1x192x30x30xf32>
    %collapsed_78 = tensor.collapse_shape %arg205 [[0, 1], [2], [3]] : tensor<192x1x3x3xf32> into tensor<192x3x3xf32>
    %179 = linalg.depthwise_conv_2d_nchw_chw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_77, %collapsed_78 : tensor<1x192x30x30xf32>, tensor<192x3x3xf32>) outs(%138 : tensor<1x192x28x28xf32>) -> tensor<1x192x28x28xf32>
    %180 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg49 : tensor<192xf32>) outs(%140 : tensor<192xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.truncf %cst_2 : f64 to f32
      %561 = arith.addf %in, %560 : f32
      linalg.yield %561 : f32
    } -> tensor<192xf32>
    %181 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%180 : tensor<192xf32>) outs(%140 : tensor<192xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = math.sqrt %in : f32
      linalg.yield %560 : f32
    } -> tensor<192xf32>
    %182 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%181 : tensor<192xf32>) outs(%140 : tensor<192xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.cmpf one, %in, %cst_0 : f32
      cf.assert %560, "unimplemented: tensor with zero element"
      %561 = arith.divf %cst_1, %in : f32
      linalg.yield %561 : f32
    } -> tensor<192xf32>
    %expanded_79 = tensor.expand_shape %arg48 [[0, 1, 2]] output_shape [192, 1, 1] : tensor<192xf32> into tensor<192x1x1xf32>
    %expanded_80 = tensor.expand_shape %182 [[0, 1, 2]] output_shape [192, 1, 1] : tensor<192xf32> into tensor<192x1x1xf32>
    %183 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%179, %expanded_79 : tensor<1x192x28x28xf32>, tensor<192x1x1xf32>) outs(%137 : tensor<1x192x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.subf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x192x28x28xf32>
    %184 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%183, %expanded_80 : tensor<1x192x28x28xf32>, tensor<192x1x1xf32>) outs(%137 : tensor<1x192x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x192x28x28xf32>
    %expanded_81 = tensor.expand_shape %arg206 [[0, 1, 2]] output_shape [192, 1, 1] : tensor<192xf32> into tensor<192x1x1xf32>
    %185 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%184, %expanded_81 : tensor<1x192x28x28xf32>, tensor<192x1x1xf32>) outs(%137 : tensor<1x192x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x192x28x28xf32>
    %expanded_82 = tensor.expand_shape %arg207 [[0, 1, 2]] output_shape [192, 1, 1] : tensor<192xf32> into tensor<192x1x1xf32>
    %186 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%185, %expanded_82 : tensor<1x192x28x28xf32>, tensor<192x1x1xf32>) outs(%137 : tensor<1x192x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.addf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x192x28x28xf32>
    %187 = linalg.generic {indexing_maps = [#map1, #map4, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%186, %12 : tensor<1x192x28x28xf32>, tensor<f32>) outs(%137 : tensor<1x192x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.cmpf ogt, %in, %in_246 : f32
      %561 = arith.select %560, %in, %in_246 : f32
      linalg.yield %561 : f32
    } -> tensor<1x192x28x28xf32>
    %188 = linalg.generic {indexing_maps = [#map4, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%14, %187 : tensor<f32>, tensor<1x192x28x28xf32>) outs(%137 : tensor<1x192x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.cmpf olt, %in, %in_246 : f32
      %561 = arith.select %560, %in, %in_246 : f32
      linalg.yield %561 : f32
    } -> tensor<1x192x28x28xf32>
    %189 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%188, %arg208 : tensor<1x192x28x28xf32>, tensor<32x192x1x1xf32>) outs(%128 : tensor<1x32x28x28xf32>) -> tensor<1x32x28x28xf32>
    %190 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg52 : tensor<32xf32>) outs(%3 : tensor<32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.truncf %cst_2 : f64 to f32
      %561 = arith.addf %in, %560 : f32
      linalg.yield %561 : f32
    } -> tensor<32xf32>
    %191 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%190 : tensor<32xf32>) outs(%3 : tensor<32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = math.sqrt %in : f32
      linalg.yield %560 : f32
    } -> tensor<32xf32>
    %192 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%191 : tensor<32xf32>) outs(%3 : tensor<32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.cmpf one, %in, %cst_0 : f32
      cf.assert %560, "unimplemented: tensor with zero element"
      %561 = arith.divf %cst_1, %in : f32
      linalg.yield %561 : f32
    } -> tensor<32xf32>
    %expanded_83 = tensor.expand_shape %arg51 [[0, 1, 2]] output_shape [32, 1, 1] : tensor<32xf32> into tensor<32x1x1xf32>
    %expanded_84 = tensor.expand_shape %192 [[0, 1, 2]] output_shape [32, 1, 1] : tensor<32xf32> into tensor<32x1x1xf32>
    %193 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%189, %expanded_83 : tensor<1x32x28x28xf32>, tensor<32x1x1xf32>) outs(%127 : tensor<1x32x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.subf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x32x28x28xf32>
    %194 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%193, %expanded_84 : tensor<1x32x28x28xf32>, tensor<32x1x1xf32>) outs(%127 : tensor<1x32x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x32x28x28xf32>
    %expanded_85 = tensor.expand_shape %arg209 [[0, 1, 2]] output_shape [32, 1, 1] : tensor<32xf32> into tensor<32x1x1xf32>
    %195 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%194, %expanded_85 : tensor<1x32x28x28xf32>, tensor<32x1x1xf32>) outs(%127 : tensor<1x32x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x32x28x28xf32>
    %expanded_86 = tensor.expand_shape %arg210 [[0, 1, 2]] output_shape [32, 1, 1] : tensor<32xf32> into tensor<32x1x1xf32>
    %196 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%195, %expanded_86 : tensor<1x32x28x28xf32>, tensor<32x1x1xf32>) outs(%127 : tensor<1x32x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.addf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x32x28x28xf32>
    %197 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%168, %196 : tensor<1x32x28x28xf32>, tensor<1x32x28x28xf32>) outs(%127 : tensor<1x32x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.addf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x32x28x28xf32>
    %198 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%197, %arg211 : tensor<1x32x28x28xf32>, tensor<192x32x1x1xf32>) outs(%138 : tensor<1x192x28x28xf32>) -> tensor<1x192x28x28xf32>
    %199 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg55 : tensor<192xf32>) outs(%140 : tensor<192xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.truncf %cst_2 : f64 to f32
      %561 = arith.addf %in, %560 : f32
      linalg.yield %561 : f32
    } -> tensor<192xf32>
    %200 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%199 : tensor<192xf32>) outs(%140 : tensor<192xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = math.sqrt %in : f32
      linalg.yield %560 : f32
    } -> tensor<192xf32>
    %201 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%200 : tensor<192xf32>) outs(%140 : tensor<192xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.cmpf one, %in, %cst_0 : f32
      cf.assert %560, "unimplemented: tensor with zero element"
      %561 = arith.divf %cst_1, %in : f32
      linalg.yield %561 : f32
    } -> tensor<192xf32>
    %expanded_87 = tensor.expand_shape %arg54 [[0, 1, 2]] output_shape [192, 1, 1] : tensor<192xf32> into tensor<192x1x1xf32>
    %expanded_88 = tensor.expand_shape %201 [[0, 1, 2]] output_shape [192, 1, 1] : tensor<192xf32> into tensor<192x1x1xf32>
    %202 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%198, %expanded_87 : tensor<1x192x28x28xf32>, tensor<192x1x1xf32>) outs(%137 : tensor<1x192x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.subf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x192x28x28xf32>
    %203 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%202, %expanded_88 : tensor<1x192x28x28xf32>, tensor<192x1x1xf32>) outs(%137 : tensor<1x192x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x192x28x28xf32>
    %expanded_89 = tensor.expand_shape %arg212 [[0, 1, 2]] output_shape [192, 1, 1] : tensor<192xf32> into tensor<192x1x1xf32>
    %204 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%203, %expanded_89 : tensor<1x192x28x28xf32>, tensor<192x1x1xf32>) outs(%137 : tensor<1x192x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x192x28x28xf32>
    %expanded_90 = tensor.expand_shape %arg213 [[0, 1, 2]] output_shape [192, 1, 1] : tensor<192xf32> into tensor<192x1x1xf32>
    %205 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%204, %expanded_90 : tensor<1x192x28x28xf32>, tensor<192x1x1xf32>) outs(%137 : tensor<1x192x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.addf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x192x28x28xf32>
    %206 = linalg.generic {indexing_maps = [#map1, #map4, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%205, %12 : tensor<1x192x28x28xf32>, tensor<f32>) outs(%137 : tensor<1x192x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.cmpf ogt, %in, %in_246 : f32
      %561 = arith.select %560, %in, %in_246 : f32
      linalg.yield %561 : f32
    } -> tensor<1x192x28x28xf32>
    %207 = linalg.generic {indexing_maps = [#map4, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%14, %206 : tensor<f32>, tensor<1x192x28x28xf32>) outs(%137 : tensor<1x192x28x28xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.cmpf olt, %in, %in_246 : f32
      %561 = arith.select %560, %in, %in_246 : f32
      linalg.yield %561 : f32
    } -> tensor<1x192x28x28xf32>
    %padded_91 = tensor.pad %207 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg315: index, %arg316: index, %arg317: index, %arg318: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x192x28x28xf32> to tensor<1x192x30x30xf32>
    %208 = tensor.empty() : tensor<1x192x14x14xf32>
    %209 = linalg.fill ins(%cst_0 : f32) outs(%208 : tensor<1x192x14x14xf32>) -> tensor<1x192x14x14xf32>
    %collapsed_92 = tensor.collapse_shape %arg214 [[0, 1], [2], [3]] : tensor<192x1x3x3xf32> into tensor<192x3x3xf32>
    %210 = linalg.depthwise_conv_2d_nchw_chw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%padded_91, %collapsed_92 : tensor<1x192x30x30xf32>, tensor<192x3x3xf32>) outs(%209 : tensor<1x192x14x14xf32>) -> tensor<1x192x14x14xf32>
    %211 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg58 : tensor<192xf32>) outs(%140 : tensor<192xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.truncf %cst_2 : f64 to f32
      %561 = arith.addf %in, %560 : f32
      linalg.yield %561 : f32
    } -> tensor<192xf32>
    %212 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%211 : tensor<192xf32>) outs(%140 : tensor<192xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = math.sqrt %in : f32
      linalg.yield %560 : f32
    } -> tensor<192xf32>
    %213 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%212 : tensor<192xf32>) outs(%140 : tensor<192xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.cmpf one, %in, %cst_0 : f32
      cf.assert %560, "unimplemented: tensor with zero element"
      %561 = arith.divf %cst_1, %in : f32
      linalg.yield %561 : f32
    } -> tensor<192xf32>
    %expanded_93 = tensor.expand_shape %arg57 [[0, 1, 2]] output_shape [192, 1, 1] : tensor<192xf32> into tensor<192x1x1xf32>
    %expanded_94 = tensor.expand_shape %213 [[0, 1, 2]] output_shape [192, 1, 1] : tensor<192xf32> into tensor<192x1x1xf32>
    %214 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%210, %expanded_93 : tensor<1x192x14x14xf32>, tensor<192x1x1xf32>) outs(%208 : tensor<1x192x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.subf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x192x14x14xf32>
    %215 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%214, %expanded_94 : tensor<1x192x14x14xf32>, tensor<192x1x1xf32>) outs(%208 : tensor<1x192x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x192x14x14xf32>
    %expanded_95 = tensor.expand_shape %arg215 [[0, 1, 2]] output_shape [192, 1, 1] : tensor<192xf32> into tensor<192x1x1xf32>
    %216 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%215, %expanded_95 : tensor<1x192x14x14xf32>, tensor<192x1x1xf32>) outs(%208 : tensor<1x192x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x192x14x14xf32>
    %expanded_96 = tensor.expand_shape %arg216 [[0, 1, 2]] output_shape [192, 1, 1] : tensor<192xf32> into tensor<192x1x1xf32>
    %217 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%216, %expanded_96 : tensor<1x192x14x14xf32>, tensor<192x1x1xf32>) outs(%208 : tensor<1x192x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.addf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x192x14x14xf32>
    %218 = linalg.generic {indexing_maps = [#map1, #map4, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%217, %12 : tensor<1x192x14x14xf32>, tensor<f32>) outs(%208 : tensor<1x192x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.cmpf ogt, %in, %in_246 : f32
      %561 = arith.select %560, %in, %in_246 : f32
      linalg.yield %561 : f32
    } -> tensor<1x192x14x14xf32>
    %219 = linalg.generic {indexing_maps = [#map4, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%14, %218 : tensor<f32>, tensor<1x192x14x14xf32>) outs(%208 : tensor<1x192x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.cmpf olt, %in, %in_246 : f32
      %561 = arith.select %560, %in, %in_246 : f32
      linalg.yield %561 : f32
    } -> tensor<1x192x14x14xf32>
    %220 = tensor.empty() : tensor<1x64x14x14xf32>
    %221 = linalg.fill ins(%cst_0 : f32) outs(%220 : tensor<1x64x14x14xf32>) -> tensor<1x64x14x14xf32>
    %222 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%219, %arg217 : tensor<1x192x14x14xf32>, tensor<64x192x1x1xf32>) outs(%221 : tensor<1x64x14x14xf32>) -> tensor<1x64x14x14xf32>
    %223 = tensor.empty() : tensor<64xf32>
    %224 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg61 : tensor<64xf32>) outs(%223 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.truncf %cst_2 : f64 to f32
      %561 = arith.addf %in, %560 : f32
      linalg.yield %561 : f32
    } -> tensor<64xf32>
    %225 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%224 : tensor<64xf32>) outs(%223 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = math.sqrt %in : f32
      linalg.yield %560 : f32
    } -> tensor<64xf32>
    %226 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%225 : tensor<64xf32>) outs(%223 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.cmpf one, %in, %cst_0 : f32
      cf.assert %560, "unimplemented: tensor with zero element"
      %561 = arith.divf %cst_1, %in : f32
      linalg.yield %561 : f32
    } -> tensor<64xf32>
    %expanded_97 = tensor.expand_shape %arg60 [[0, 1, 2]] output_shape [64, 1, 1] : tensor<64xf32> into tensor<64x1x1xf32>
    %expanded_98 = tensor.expand_shape %226 [[0, 1, 2]] output_shape [64, 1, 1] : tensor<64xf32> into tensor<64x1x1xf32>
    %227 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%222, %expanded_97 : tensor<1x64x14x14xf32>, tensor<64x1x1xf32>) outs(%220 : tensor<1x64x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.subf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x64x14x14xf32>
    %228 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%227, %expanded_98 : tensor<1x64x14x14xf32>, tensor<64x1x1xf32>) outs(%220 : tensor<1x64x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x64x14x14xf32>
    %expanded_99 = tensor.expand_shape %arg218 [[0, 1, 2]] output_shape [64, 1, 1] : tensor<64xf32> into tensor<64x1x1xf32>
    %229 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%228, %expanded_99 : tensor<1x64x14x14xf32>, tensor<64x1x1xf32>) outs(%220 : tensor<1x64x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x64x14x14xf32>
    %expanded_100 = tensor.expand_shape %arg219 [[0, 1, 2]] output_shape [64, 1, 1] : tensor<64xf32> into tensor<64x1x1xf32>
    %230 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%229, %expanded_100 : tensor<1x64x14x14xf32>, tensor<64x1x1xf32>) outs(%220 : tensor<1x64x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.addf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x64x14x14xf32>
    %231 = tensor.empty() : tensor<1x384x14x14xf32>
    %232 = linalg.fill ins(%cst_0 : f32) outs(%231 : tensor<1x384x14x14xf32>) -> tensor<1x384x14x14xf32>
    %233 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%230, %arg220 : tensor<1x64x14x14xf32>, tensor<384x64x1x1xf32>) outs(%232 : tensor<1x384x14x14xf32>) -> tensor<1x384x14x14xf32>
    %234 = tensor.empty() : tensor<384xf32>
    %235 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg64 : tensor<384xf32>) outs(%234 : tensor<384xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.truncf %cst_2 : f64 to f32
      %561 = arith.addf %in, %560 : f32
      linalg.yield %561 : f32
    } -> tensor<384xf32>
    %236 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%235 : tensor<384xf32>) outs(%234 : tensor<384xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = math.sqrt %in : f32
      linalg.yield %560 : f32
    } -> tensor<384xf32>
    %237 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%236 : tensor<384xf32>) outs(%234 : tensor<384xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.cmpf one, %in, %cst_0 : f32
      cf.assert %560, "unimplemented: tensor with zero element"
      %561 = arith.divf %cst_1, %in : f32
      linalg.yield %561 : f32
    } -> tensor<384xf32>
    %expanded_101 = tensor.expand_shape %arg63 [[0, 1, 2]] output_shape [384, 1, 1] : tensor<384xf32> into tensor<384x1x1xf32>
    %expanded_102 = tensor.expand_shape %237 [[0, 1, 2]] output_shape [384, 1, 1] : tensor<384xf32> into tensor<384x1x1xf32>
    %238 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%233, %expanded_101 : tensor<1x384x14x14xf32>, tensor<384x1x1xf32>) outs(%231 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.subf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x384x14x14xf32>
    %239 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%238, %expanded_102 : tensor<1x384x14x14xf32>, tensor<384x1x1xf32>) outs(%231 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x384x14x14xf32>
    %expanded_103 = tensor.expand_shape %arg221 [[0, 1, 2]] output_shape [384, 1, 1] : tensor<384xf32> into tensor<384x1x1xf32>
    %240 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%239, %expanded_103 : tensor<1x384x14x14xf32>, tensor<384x1x1xf32>) outs(%231 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x384x14x14xf32>
    %expanded_104 = tensor.expand_shape %arg222 [[0, 1, 2]] output_shape [384, 1, 1] : tensor<384xf32> into tensor<384x1x1xf32>
    %241 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%240, %expanded_104 : tensor<1x384x14x14xf32>, tensor<384x1x1xf32>) outs(%231 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.addf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x384x14x14xf32>
    %242 = linalg.generic {indexing_maps = [#map1, #map4, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%241, %12 : tensor<1x384x14x14xf32>, tensor<f32>) outs(%231 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.cmpf ogt, %in, %in_246 : f32
      %561 = arith.select %560, %in, %in_246 : f32
      linalg.yield %561 : f32
    } -> tensor<1x384x14x14xf32>
    %243 = linalg.generic {indexing_maps = [#map4, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%14, %242 : tensor<f32>, tensor<1x384x14x14xf32>) outs(%231 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.cmpf olt, %in, %in_246 : f32
      %561 = arith.select %560, %in, %in_246 : f32
      linalg.yield %561 : f32
    } -> tensor<1x384x14x14xf32>
    %padded_105 = tensor.pad %243 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg315: index, %arg316: index, %arg317: index, %arg318: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x384x14x14xf32> to tensor<1x384x16x16xf32>
    %collapsed_106 = tensor.collapse_shape %arg223 [[0, 1], [2], [3]] : tensor<384x1x3x3xf32> into tensor<384x3x3xf32>
    %244 = linalg.depthwise_conv_2d_nchw_chw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_105, %collapsed_106 : tensor<1x384x16x16xf32>, tensor<384x3x3xf32>) outs(%232 : tensor<1x384x14x14xf32>) -> tensor<1x384x14x14xf32>
    %245 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg67 : tensor<384xf32>) outs(%234 : tensor<384xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.truncf %cst_2 : f64 to f32
      %561 = arith.addf %in, %560 : f32
      linalg.yield %561 : f32
    } -> tensor<384xf32>
    %246 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%245 : tensor<384xf32>) outs(%234 : tensor<384xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = math.sqrt %in : f32
      linalg.yield %560 : f32
    } -> tensor<384xf32>
    %247 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%246 : tensor<384xf32>) outs(%234 : tensor<384xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.cmpf one, %in, %cst_0 : f32
      cf.assert %560, "unimplemented: tensor with zero element"
      %561 = arith.divf %cst_1, %in : f32
      linalg.yield %561 : f32
    } -> tensor<384xf32>
    %expanded_107 = tensor.expand_shape %arg66 [[0, 1, 2]] output_shape [384, 1, 1] : tensor<384xf32> into tensor<384x1x1xf32>
    %expanded_108 = tensor.expand_shape %247 [[0, 1, 2]] output_shape [384, 1, 1] : tensor<384xf32> into tensor<384x1x1xf32>
    %248 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%244, %expanded_107 : tensor<1x384x14x14xf32>, tensor<384x1x1xf32>) outs(%231 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.subf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x384x14x14xf32>
    %249 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%248, %expanded_108 : tensor<1x384x14x14xf32>, tensor<384x1x1xf32>) outs(%231 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x384x14x14xf32>
    %expanded_109 = tensor.expand_shape %arg224 [[0, 1, 2]] output_shape [384, 1, 1] : tensor<384xf32> into tensor<384x1x1xf32>
    %250 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%249, %expanded_109 : tensor<1x384x14x14xf32>, tensor<384x1x1xf32>) outs(%231 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x384x14x14xf32>
    %expanded_110 = tensor.expand_shape %arg225 [[0, 1, 2]] output_shape [384, 1, 1] : tensor<384xf32> into tensor<384x1x1xf32>
    %251 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%250, %expanded_110 : tensor<1x384x14x14xf32>, tensor<384x1x1xf32>) outs(%231 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.addf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x384x14x14xf32>
    %252 = linalg.generic {indexing_maps = [#map1, #map4, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%251, %12 : tensor<1x384x14x14xf32>, tensor<f32>) outs(%231 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.cmpf ogt, %in, %in_246 : f32
      %561 = arith.select %560, %in, %in_246 : f32
      linalg.yield %561 : f32
    } -> tensor<1x384x14x14xf32>
    %253 = linalg.generic {indexing_maps = [#map4, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%14, %252 : tensor<f32>, tensor<1x384x14x14xf32>) outs(%231 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.cmpf olt, %in, %in_246 : f32
      %561 = arith.select %560, %in, %in_246 : f32
      linalg.yield %561 : f32
    } -> tensor<1x384x14x14xf32>
    %254 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%253, %arg226 : tensor<1x384x14x14xf32>, tensor<64x384x1x1xf32>) outs(%221 : tensor<1x64x14x14xf32>) -> tensor<1x64x14x14xf32>
    %255 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg70 : tensor<64xf32>) outs(%223 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.truncf %cst_2 : f64 to f32
      %561 = arith.addf %in, %560 : f32
      linalg.yield %561 : f32
    } -> tensor<64xf32>
    %256 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%255 : tensor<64xf32>) outs(%223 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = math.sqrt %in : f32
      linalg.yield %560 : f32
    } -> tensor<64xf32>
    %257 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%256 : tensor<64xf32>) outs(%223 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.cmpf one, %in, %cst_0 : f32
      cf.assert %560, "unimplemented: tensor with zero element"
      %561 = arith.divf %cst_1, %in : f32
      linalg.yield %561 : f32
    } -> tensor<64xf32>
    %expanded_111 = tensor.expand_shape %arg69 [[0, 1, 2]] output_shape [64, 1, 1] : tensor<64xf32> into tensor<64x1x1xf32>
    %expanded_112 = tensor.expand_shape %257 [[0, 1, 2]] output_shape [64, 1, 1] : tensor<64xf32> into tensor<64x1x1xf32>
    %258 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%254, %expanded_111 : tensor<1x64x14x14xf32>, tensor<64x1x1xf32>) outs(%220 : tensor<1x64x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.subf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x64x14x14xf32>
    %259 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%258, %expanded_112 : tensor<1x64x14x14xf32>, tensor<64x1x1xf32>) outs(%220 : tensor<1x64x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x64x14x14xf32>
    %expanded_113 = tensor.expand_shape %arg227 [[0, 1, 2]] output_shape [64, 1, 1] : tensor<64xf32> into tensor<64x1x1xf32>
    %260 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%259, %expanded_113 : tensor<1x64x14x14xf32>, tensor<64x1x1xf32>) outs(%220 : tensor<1x64x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x64x14x14xf32>
    %expanded_114 = tensor.expand_shape %arg228 [[0, 1, 2]] output_shape [64, 1, 1] : tensor<64xf32> into tensor<64x1x1xf32>
    %261 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%260, %expanded_114 : tensor<1x64x14x14xf32>, tensor<64x1x1xf32>) outs(%220 : tensor<1x64x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.addf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x64x14x14xf32>
    %262 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%230, %261 : tensor<1x64x14x14xf32>, tensor<1x64x14x14xf32>) outs(%220 : tensor<1x64x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.addf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x64x14x14xf32>
    %263 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%262, %arg229 : tensor<1x64x14x14xf32>, tensor<384x64x1x1xf32>) outs(%232 : tensor<1x384x14x14xf32>) -> tensor<1x384x14x14xf32>
    %264 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg73 : tensor<384xf32>) outs(%234 : tensor<384xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.truncf %cst_2 : f64 to f32
      %561 = arith.addf %in, %560 : f32
      linalg.yield %561 : f32
    } -> tensor<384xf32>
    %265 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%264 : tensor<384xf32>) outs(%234 : tensor<384xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = math.sqrt %in : f32
      linalg.yield %560 : f32
    } -> tensor<384xf32>
    %266 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%265 : tensor<384xf32>) outs(%234 : tensor<384xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.cmpf one, %in, %cst_0 : f32
      cf.assert %560, "unimplemented: tensor with zero element"
      %561 = arith.divf %cst_1, %in : f32
      linalg.yield %561 : f32
    } -> tensor<384xf32>
    %expanded_115 = tensor.expand_shape %arg72 [[0, 1, 2]] output_shape [384, 1, 1] : tensor<384xf32> into tensor<384x1x1xf32>
    %expanded_116 = tensor.expand_shape %266 [[0, 1, 2]] output_shape [384, 1, 1] : tensor<384xf32> into tensor<384x1x1xf32>
    %267 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%263, %expanded_115 : tensor<1x384x14x14xf32>, tensor<384x1x1xf32>) outs(%231 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.subf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x384x14x14xf32>
    %268 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%267, %expanded_116 : tensor<1x384x14x14xf32>, tensor<384x1x1xf32>) outs(%231 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x384x14x14xf32>
    %expanded_117 = tensor.expand_shape %arg230 [[0, 1, 2]] output_shape [384, 1, 1] : tensor<384xf32> into tensor<384x1x1xf32>
    %269 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%268, %expanded_117 : tensor<1x384x14x14xf32>, tensor<384x1x1xf32>) outs(%231 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x384x14x14xf32>
    %expanded_118 = tensor.expand_shape %arg231 [[0, 1, 2]] output_shape [384, 1, 1] : tensor<384xf32> into tensor<384x1x1xf32>
    %270 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%269, %expanded_118 : tensor<1x384x14x14xf32>, tensor<384x1x1xf32>) outs(%231 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.addf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x384x14x14xf32>
    %271 = linalg.generic {indexing_maps = [#map1, #map4, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%270, %12 : tensor<1x384x14x14xf32>, tensor<f32>) outs(%231 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.cmpf ogt, %in, %in_246 : f32
      %561 = arith.select %560, %in, %in_246 : f32
      linalg.yield %561 : f32
    } -> tensor<1x384x14x14xf32>
    %272 = linalg.generic {indexing_maps = [#map4, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%14, %271 : tensor<f32>, tensor<1x384x14x14xf32>) outs(%231 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.cmpf olt, %in, %in_246 : f32
      %561 = arith.select %560, %in, %in_246 : f32
      linalg.yield %561 : f32
    } -> tensor<1x384x14x14xf32>
    %padded_119 = tensor.pad %272 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg315: index, %arg316: index, %arg317: index, %arg318: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x384x14x14xf32> to tensor<1x384x16x16xf32>
    %collapsed_120 = tensor.collapse_shape %arg232 [[0, 1], [2], [3]] : tensor<384x1x3x3xf32> into tensor<384x3x3xf32>
    %273 = linalg.depthwise_conv_2d_nchw_chw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_119, %collapsed_120 : tensor<1x384x16x16xf32>, tensor<384x3x3xf32>) outs(%232 : tensor<1x384x14x14xf32>) -> tensor<1x384x14x14xf32>
    %274 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg76 : tensor<384xf32>) outs(%234 : tensor<384xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.truncf %cst_2 : f64 to f32
      %561 = arith.addf %in, %560 : f32
      linalg.yield %561 : f32
    } -> tensor<384xf32>
    %275 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%274 : tensor<384xf32>) outs(%234 : tensor<384xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = math.sqrt %in : f32
      linalg.yield %560 : f32
    } -> tensor<384xf32>
    %276 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%275 : tensor<384xf32>) outs(%234 : tensor<384xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.cmpf one, %in, %cst_0 : f32
      cf.assert %560, "unimplemented: tensor with zero element"
      %561 = arith.divf %cst_1, %in : f32
      linalg.yield %561 : f32
    } -> tensor<384xf32>
    %expanded_121 = tensor.expand_shape %arg75 [[0, 1, 2]] output_shape [384, 1, 1] : tensor<384xf32> into tensor<384x1x1xf32>
    %expanded_122 = tensor.expand_shape %276 [[0, 1, 2]] output_shape [384, 1, 1] : tensor<384xf32> into tensor<384x1x1xf32>
    %277 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%273, %expanded_121 : tensor<1x384x14x14xf32>, tensor<384x1x1xf32>) outs(%231 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.subf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x384x14x14xf32>
    %278 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%277, %expanded_122 : tensor<1x384x14x14xf32>, tensor<384x1x1xf32>) outs(%231 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x384x14x14xf32>
    %expanded_123 = tensor.expand_shape %arg233 [[0, 1, 2]] output_shape [384, 1, 1] : tensor<384xf32> into tensor<384x1x1xf32>
    %279 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%278, %expanded_123 : tensor<1x384x14x14xf32>, tensor<384x1x1xf32>) outs(%231 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x384x14x14xf32>
    %expanded_124 = tensor.expand_shape %arg234 [[0, 1, 2]] output_shape [384, 1, 1] : tensor<384xf32> into tensor<384x1x1xf32>
    %280 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%279, %expanded_124 : tensor<1x384x14x14xf32>, tensor<384x1x1xf32>) outs(%231 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.addf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x384x14x14xf32>
    %281 = linalg.generic {indexing_maps = [#map1, #map4, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%280, %12 : tensor<1x384x14x14xf32>, tensor<f32>) outs(%231 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.cmpf ogt, %in, %in_246 : f32
      %561 = arith.select %560, %in, %in_246 : f32
      linalg.yield %561 : f32
    } -> tensor<1x384x14x14xf32>
    %282 = linalg.generic {indexing_maps = [#map4, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%14, %281 : tensor<f32>, tensor<1x384x14x14xf32>) outs(%231 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.cmpf olt, %in, %in_246 : f32
      %561 = arith.select %560, %in, %in_246 : f32
      linalg.yield %561 : f32
    } -> tensor<1x384x14x14xf32>
    %283 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%282, %arg235 : tensor<1x384x14x14xf32>, tensor<64x384x1x1xf32>) outs(%221 : tensor<1x64x14x14xf32>) -> tensor<1x64x14x14xf32>
    %284 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg79 : tensor<64xf32>) outs(%223 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.truncf %cst_2 : f64 to f32
      %561 = arith.addf %in, %560 : f32
      linalg.yield %561 : f32
    } -> tensor<64xf32>
    %285 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%284 : tensor<64xf32>) outs(%223 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = math.sqrt %in : f32
      linalg.yield %560 : f32
    } -> tensor<64xf32>
    %286 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%285 : tensor<64xf32>) outs(%223 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.cmpf one, %in, %cst_0 : f32
      cf.assert %560, "unimplemented: tensor with zero element"
      %561 = arith.divf %cst_1, %in : f32
      linalg.yield %561 : f32
    } -> tensor<64xf32>
    %expanded_125 = tensor.expand_shape %arg78 [[0, 1, 2]] output_shape [64, 1, 1] : tensor<64xf32> into tensor<64x1x1xf32>
    %expanded_126 = tensor.expand_shape %286 [[0, 1, 2]] output_shape [64, 1, 1] : tensor<64xf32> into tensor<64x1x1xf32>
    %287 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%283, %expanded_125 : tensor<1x64x14x14xf32>, tensor<64x1x1xf32>) outs(%220 : tensor<1x64x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.subf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x64x14x14xf32>
    %288 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%287, %expanded_126 : tensor<1x64x14x14xf32>, tensor<64x1x1xf32>) outs(%220 : tensor<1x64x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x64x14x14xf32>
    %expanded_127 = tensor.expand_shape %arg236 [[0, 1, 2]] output_shape [64, 1, 1] : tensor<64xf32> into tensor<64x1x1xf32>
    %289 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%288, %expanded_127 : tensor<1x64x14x14xf32>, tensor<64x1x1xf32>) outs(%220 : tensor<1x64x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x64x14x14xf32>
    %expanded_128 = tensor.expand_shape %arg237 [[0, 1, 2]] output_shape [64, 1, 1] : tensor<64xf32> into tensor<64x1x1xf32>
    %290 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%289, %expanded_128 : tensor<1x64x14x14xf32>, tensor<64x1x1xf32>) outs(%220 : tensor<1x64x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.addf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x64x14x14xf32>
    %291 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%262, %290 : tensor<1x64x14x14xf32>, tensor<1x64x14x14xf32>) outs(%220 : tensor<1x64x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.addf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x64x14x14xf32>
    %292 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%291, %arg238 : tensor<1x64x14x14xf32>, tensor<384x64x1x1xf32>) outs(%232 : tensor<1x384x14x14xf32>) -> tensor<1x384x14x14xf32>
    %293 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg82 : tensor<384xf32>) outs(%234 : tensor<384xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.truncf %cst_2 : f64 to f32
      %561 = arith.addf %in, %560 : f32
      linalg.yield %561 : f32
    } -> tensor<384xf32>
    %294 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%293 : tensor<384xf32>) outs(%234 : tensor<384xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = math.sqrt %in : f32
      linalg.yield %560 : f32
    } -> tensor<384xf32>
    %295 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%294 : tensor<384xf32>) outs(%234 : tensor<384xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.cmpf one, %in, %cst_0 : f32
      cf.assert %560, "unimplemented: tensor with zero element"
      %561 = arith.divf %cst_1, %in : f32
      linalg.yield %561 : f32
    } -> tensor<384xf32>
    %expanded_129 = tensor.expand_shape %arg81 [[0, 1, 2]] output_shape [384, 1, 1] : tensor<384xf32> into tensor<384x1x1xf32>
    %expanded_130 = tensor.expand_shape %295 [[0, 1, 2]] output_shape [384, 1, 1] : tensor<384xf32> into tensor<384x1x1xf32>
    %296 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%292, %expanded_129 : tensor<1x384x14x14xf32>, tensor<384x1x1xf32>) outs(%231 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.subf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x384x14x14xf32>
    %297 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%296, %expanded_130 : tensor<1x384x14x14xf32>, tensor<384x1x1xf32>) outs(%231 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x384x14x14xf32>
    %expanded_131 = tensor.expand_shape %arg239 [[0, 1, 2]] output_shape [384, 1, 1] : tensor<384xf32> into tensor<384x1x1xf32>
    %298 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%297, %expanded_131 : tensor<1x384x14x14xf32>, tensor<384x1x1xf32>) outs(%231 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x384x14x14xf32>
    %expanded_132 = tensor.expand_shape %arg240 [[0, 1, 2]] output_shape [384, 1, 1] : tensor<384xf32> into tensor<384x1x1xf32>
    %299 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%298, %expanded_132 : tensor<1x384x14x14xf32>, tensor<384x1x1xf32>) outs(%231 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.addf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x384x14x14xf32>
    %300 = linalg.generic {indexing_maps = [#map1, #map4, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%299, %12 : tensor<1x384x14x14xf32>, tensor<f32>) outs(%231 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.cmpf ogt, %in, %in_246 : f32
      %561 = arith.select %560, %in, %in_246 : f32
      linalg.yield %561 : f32
    } -> tensor<1x384x14x14xf32>
    %301 = linalg.generic {indexing_maps = [#map4, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%14, %300 : tensor<f32>, tensor<1x384x14x14xf32>) outs(%231 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.cmpf olt, %in, %in_246 : f32
      %561 = arith.select %560, %in, %in_246 : f32
      linalg.yield %561 : f32
    } -> tensor<1x384x14x14xf32>
    %padded_133 = tensor.pad %301 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg315: index, %arg316: index, %arg317: index, %arg318: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x384x14x14xf32> to tensor<1x384x16x16xf32>
    %collapsed_134 = tensor.collapse_shape %arg241 [[0, 1], [2], [3]] : tensor<384x1x3x3xf32> into tensor<384x3x3xf32>
    %302 = linalg.depthwise_conv_2d_nchw_chw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_133, %collapsed_134 : tensor<1x384x16x16xf32>, tensor<384x3x3xf32>) outs(%232 : tensor<1x384x14x14xf32>) -> tensor<1x384x14x14xf32>
    %303 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg85 : tensor<384xf32>) outs(%234 : tensor<384xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.truncf %cst_2 : f64 to f32
      %561 = arith.addf %in, %560 : f32
      linalg.yield %561 : f32
    } -> tensor<384xf32>
    %304 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%303 : tensor<384xf32>) outs(%234 : tensor<384xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = math.sqrt %in : f32
      linalg.yield %560 : f32
    } -> tensor<384xf32>
    %305 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%304 : tensor<384xf32>) outs(%234 : tensor<384xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.cmpf one, %in, %cst_0 : f32
      cf.assert %560, "unimplemented: tensor with zero element"
      %561 = arith.divf %cst_1, %in : f32
      linalg.yield %561 : f32
    } -> tensor<384xf32>
    %expanded_135 = tensor.expand_shape %arg84 [[0, 1, 2]] output_shape [384, 1, 1] : tensor<384xf32> into tensor<384x1x1xf32>
    %expanded_136 = tensor.expand_shape %305 [[0, 1, 2]] output_shape [384, 1, 1] : tensor<384xf32> into tensor<384x1x1xf32>
    %306 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%302, %expanded_135 : tensor<1x384x14x14xf32>, tensor<384x1x1xf32>) outs(%231 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.subf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x384x14x14xf32>
    %307 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%306, %expanded_136 : tensor<1x384x14x14xf32>, tensor<384x1x1xf32>) outs(%231 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x384x14x14xf32>
    %expanded_137 = tensor.expand_shape %arg242 [[0, 1, 2]] output_shape [384, 1, 1] : tensor<384xf32> into tensor<384x1x1xf32>
    %308 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%307, %expanded_137 : tensor<1x384x14x14xf32>, tensor<384x1x1xf32>) outs(%231 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x384x14x14xf32>
    %expanded_138 = tensor.expand_shape %arg243 [[0, 1, 2]] output_shape [384, 1, 1] : tensor<384xf32> into tensor<384x1x1xf32>
    %309 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%308, %expanded_138 : tensor<1x384x14x14xf32>, tensor<384x1x1xf32>) outs(%231 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.addf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x384x14x14xf32>
    %310 = linalg.generic {indexing_maps = [#map1, #map4, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%309, %12 : tensor<1x384x14x14xf32>, tensor<f32>) outs(%231 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.cmpf ogt, %in, %in_246 : f32
      %561 = arith.select %560, %in, %in_246 : f32
      linalg.yield %561 : f32
    } -> tensor<1x384x14x14xf32>
    %311 = linalg.generic {indexing_maps = [#map4, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%14, %310 : tensor<f32>, tensor<1x384x14x14xf32>) outs(%231 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.cmpf olt, %in, %in_246 : f32
      %561 = arith.select %560, %in, %in_246 : f32
      linalg.yield %561 : f32
    } -> tensor<1x384x14x14xf32>
    %312 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%311, %arg244 : tensor<1x384x14x14xf32>, tensor<64x384x1x1xf32>) outs(%221 : tensor<1x64x14x14xf32>) -> tensor<1x64x14x14xf32>
    %313 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg88 : tensor<64xf32>) outs(%223 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.truncf %cst_2 : f64 to f32
      %561 = arith.addf %in, %560 : f32
      linalg.yield %561 : f32
    } -> tensor<64xf32>
    %314 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%313 : tensor<64xf32>) outs(%223 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = math.sqrt %in : f32
      linalg.yield %560 : f32
    } -> tensor<64xf32>
    %315 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%314 : tensor<64xf32>) outs(%223 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.cmpf one, %in, %cst_0 : f32
      cf.assert %560, "unimplemented: tensor with zero element"
      %561 = arith.divf %cst_1, %in : f32
      linalg.yield %561 : f32
    } -> tensor<64xf32>
    %expanded_139 = tensor.expand_shape %arg87 [[0, 1, 2]] output_shape [64, 1, 1] : tensor<64xf32> into tensor<64x1x1xf32>
    %expanded_140 = tensor.expand_shape %315 [[0, 1, 2]] output_shape [64, 1, 1] : tensor<64xf32> into tensor<64x1x1xf32>
    %316 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%312, %expanded_139 : tensor<1x64x14x14xf32>, tensor<64x1x1xf32>) outs(%220 : tensor<1x64x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.subf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x64x14x14xf32>
    %317 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%316, %expanded_140 : tensor<1x64x14x14xf32>, tensor<64x1x1xf32>) outs(%220 : tensor<1x64x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x64x14x14xf32>
    %expanded_141 = tensor.expand_shape %arg245 [[0, 1, 2]] output_shape [64, 1, 1] : tensor<64xf32> into tensor<64x1x1xf32>
    %318 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%317, %expanded_141 : tensor<1x64x14x14xf32>, tensor<64x1x1xf32>) outs(%220 : tensor<1x64x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x64x14x14xf32>
    %expanded_142 = tensor.expand_shape %arg246 [[0, 1, 2]] output_shape [64, 1, 1] : tensor<64xf32> into tensor<64x1x1xf32>
    %319 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%318, %expanded_142 : tensor<1x64x14x14xf32>, tensor<64x1x1xf32>) outs(%220 : tensor<1x64x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.addf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x64x14x14xf32>
    %320 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%291, %319 : tensor<1x64x14x14xf32>, tensor<1x64x14x14xf32>) outs(%220 : tensor<1x64x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.addf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x64x14x14xf32>
    %321 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%320, %arg247 : tensor<1x64x14x14xf32>, tensor<384x64x1x1xf32>) outs(%232 : tensor<1x384x14x14xf32>) -> tensor<1x384x14x14xf32>
    %322 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg91 : tensor<384xf32>) outs(%234 : tensor<384xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.truncf %cst_2 : f64 to f32
      %561 = arith.addf %in, %560 : f32
      linalg.yield %561 : f32
    } -> tensor<384xf32>
    %323 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%322 : tensor<384xf32>) outs(%234 : tensor<384xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = math.sqrt %in : f32
      linalg.yield %560 : f32
    } -> tensor<384xf32>
    %324 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%323 : tensor<384xf32>) outs(%234 : tensor<384xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.cmpf one, %in, %cst_0 : f32
      cf.assert %560, "unimplemented: tensor with zero element"
      %561 = arith.divf %cst_1, %in : f32
      linalg.yield %561 : f32
    } -> tensor<384xf32>
    %expanded_143 = tensor.expand_shape %arg90 [[0, 1, 2]] output_shape [384, 1, 1] : tensor<384xf32> into tensor<384x1x1xf32>
    %expanded_144 = tensor.expand_shape %324 [[0, 1, 2]] output_shape [384, 1, 1] : tensor<384xf32> into tensor<384x1x1xf32>
    %325 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%321, %expanded_143 : tensor<1x384x14x14xf32>, tensor<384x1x1xf32>) outs(%231 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.subf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x384x14x14xf32>
    %326 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%325, %expanded_144 : tensor<1x384x14x14xf32>, tensor<384x1x1xf32>) outs(%231 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x384x14x14xf32>
    %expanded_145 = tensor.expand_shape %arg248 [[0, 1, 2]] output_shape [384, 1, 1] : tensor<384xf32> into tensor<384x1x1xf32>
    %327 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%326, %expanded_145 : tensor<1x384x14x14xf32>, tensor<384x1x1xf32>) outs(%231 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x384x14x14xf32>
    %expanded_146 = tensor.expand_shape %arg249 [[0, 1, 2]] output_shape [384, 1, 1] : tensor<384xf32> into tensor<384x1x1xf32>
    %328 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%327, %expanded_146 : tensor<1x384x14x14xf32>, tensor<384x1x1xf32>) outs(%231 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.addf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x384x14x14xf32>
    %329 = linalg.generic {indexing_maps = [#map1, #map4, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%328, %12 : tensor<1x384x14x14xf32>, tensor<f32>) outs(%231 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.cmpf ogt, %in, %in_246 : f32
      %561 = arith.select %560, %in, %in_246 : f32
      linalg.yield %561 : f32
    } -> tensor<1x384x14x14xf32>
    %330 = linalg.generic {indexing_maps = [#map4, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%14, %329 : tensor<f32>, tensor<1x384x14x14xf32>) outs(%231 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.cmpf olt, %in, %in_246 : f32
      %561 = arith.select %560, %in, %in_246 : f32
      linalg.yield %561 : f32
    } -> tensor<1x384x14x14xf32>
    %padded_147 = tensor.pad %330 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg315: index, %arg316: index, %arg317: index, %arg318: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x384x14x14xf32> to tensor<1x384x16x16xf32>
    %collapsed_148 = tensor.collapse_shape %arg250 [[0, 1], [2], [3]] : tensor<384x1x3x3xf32> into tensor<384x3x3xf32>
    %331 = linalg.depthwise_conv_2d_nchw_chw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_147, %collapsed_148 : tensor<1x384x16x16xf32>, tensor<384x3x3xf32>) outs(%232 : tensor<1x384x14x14xf32>) -> tensor<1x384x14x14xf32>
    %332 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg94 : tensor<384xf32>) outs(%234 : tensor<384xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.truncf %cst_2 : f64 to f32
      %561 = arith.addf %in, %560 : f32
      linalg.yield %561 : f32
    } -> tensor<384xf32>
    %333 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%332 : tensor<384xf32>) outs(%234 : tensor<384xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = math.sqrt %in : f32
      linalg.yield %560 : f32
    } -> tensor<384xf32>
    %334 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%333 : tensor<384xf32>) outs(%234 : tensor<384xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.cmpf one, %in, %cst_0 : f32
      cf.assert %560, "unimplemented: tensor with zero element"
      %561 = arith.divf %cst_1, %in : f32
      linalg.yield %561 : f32
    } -> tensor<384xf32>
    %expanded_149 = tensor.expand_shape %arg93 [[0, 1, 2]] output_shape [384, 1, 1] : tensor<384xf32> into tensor<384x1x1xf32>
    %expanded_150 = tensor.expand_shape %334 [[0, 1, 2]] output_shape [384, 1, 1] : tensor<384xf32> into tensor<384x1x1xf32>
    %335 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%331, %expanded_149 : tensor<1x384x14x14xf32>, tensor<384x1x1xf32>) outs(%231 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.subf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x384x14x14xf32>
    %336 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%335, %expanded_150 : tensor<1x384x14x14xf32>, tensor<384x1x1xf32>) outs(%231 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x384x14x14xf32>
    %expanded_151 = tensor.expand_shape %arg251 [[0, 1, 2]] output_shape [384, 1, 1] : tensor<384xf32> into tensor<384x1x1xf32>
    %337 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%336, %expanded_151 : tensor<1x384x14x14xf32>, tensor<384x1x1xf32>) outs(%231 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x384x14x14xf32>
    %expanded_152 = tensor.expand_shape %arg252 [[0, 1, 2]] output_shape [384, 1, 1] : tensor<384xf32> into tensor<384x1x1xf32>
    %338 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%337, %expanded_152 : tensor<1x384x14x14xf32>, tensor<384x1x1xf32>) outs(%231 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.addf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x384x14x14xf32>
    %339 = linalg.generic {indexing_maps = [#map1, #map4, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%338, %12 : tensor<1x384x14x14xf32>, tensor<f32>) outs(%231 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.cmpf ogt, %in, %in_246 : f32
      %561 = arith.select %560, %in, %in_246 : f32
      linalg.yield %561 : f32
    } -> tensor<1x384x14x14xf32>
    %340 = linalg.generic {indexing_maps = [#map4, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%14, %339 : tensor<f32>, tensor<1x384x14x14xf32>) outs(%231 : tensor<1x384x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.cmpf olt, %in, %in_246 : f32
      %561 = arith.select %560, %in, %in_246 : f32
      linalg.yield %561 : f32
    } -> tensor<1x384x14x14xf32>
    %341 = tensor.empty() : tensor<1x96x14x14xf32>
    %342 = linalg.fill ins(%cst_0 : f32) outs(%341 : tensor<1x96x14x14xf32>) -> tensor<1x96x14x14xf32>
    %343 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%340, %arg253 : tensor<1x384x14x14xf32>, tensor<96x384x1x1xf32>) outs(%342 : tensor<1x96x14x14xf32>) -> tensor<1x96x14x14xf32>
    %344 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg97 : tensor<96xf32>) outs(%40 : tensor<96xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.truncf %cst_2 : f64 to f32
      %561 = arith.addf %in, %560 : f32
      linalg.yield %561 : f32
    } -> tensor<96xf32>
    %345 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%344 : tensor<96xf32>) outs(%40 : tensor<96xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = math.sqrt %in : f32
      linalg.yield %560 : f32
    } -> tensor<96xf32>
    %346 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%345 : tensor<96xf32>) outs(%40 : tensor<96xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.cmpf one, %in, %cst_0 : f32
      cf.assert %560, "unimplemented: tensor with zero element"
      %561 = arith.divf %cst_1, %in : f32
      linalg.yield %561 : f32
    } -> tensor<96xf32>
    %expanded_153 = tensor.expand_shape %arg96 [[0, 1, 2]] output_shape [96, 1, 1] : tensor<96xf32> into tensor<96x1x1xf32>
    %expanded_154 = tensor.expand_shape %346 [[0, 1, 2]] output_shape [96, 1, 1] : tensor<96xf32> into tensor<96x1x1xf32>
    %347 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%343, %expanded_153 : tensor<1x96x14x14xf32>, tensor<96x1x1xf32>) outs(%341 : tensor<1x96x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.subf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x96x14x14xf32>
    %348 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%347, %expanded_154 : tensor<1x96x14x14xf32>, tensor<96x1x1xf32>) outs(%341 : tensor<1x96x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x96x14x14xf32>
    %expanded_155 = tensor.expand_shape %arg254 [[0, 1, 2]] output_shape [96, 1, 1] : tensor<96xf32> into tensor<96x1x1xf32>
    %349 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%348, %expanded_155 : tensor<1x96x14x14xf32>, tensor<96x1x1xf32>) outs(%341 : tensor<1x96x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x96x14x14xf32>
    %expanded_156 = tensor.expand_shape %arg255 [[0, 1, 2]] output_shape [96, 1, 1] : tensor<96xf32> into tensor<96x1x1xf32>
    %350 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%349, %expanded_156 : tensor<1x96x14x14xf32>, tensor<96x1x1xf32>) outs(%341 : tensor<1x96x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.addf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x96x14x14xf32>
    %351 = tensor.empty() : tensor<1x576x14x14xf32>
    %352 = linalg.fill ins(%cst_0 : f32) outs(%351 : tensor<1x576x14x14xf32>) -> tensor<1x576x14x14xf32>
    %353 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%350, %arg256 : tensor<1x96x14x14xf32>, tensor<576x96x1x1xf32>) outs(%352 : tensor<1x576x14x14xf32>) -> tensor<1x576x14x14xf32>
    %354 = tensor.empty() : tensor<576xf32>
    %355 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg100 : tensor<576xf32>) outs(%354 : tensor<576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.truncf %cst_2 : f64 to f32
      %561 = arith.addf %in, %560 : f32
      linalg.yield %561 : f32
    } -> tensor<576xf32>
    %356 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%355 : tensor<576xf32>) outs(%354 : tensor<576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = math.sqrt %in : f32
      linalg.yield %560 : f32
    } -> tensor<576xf32>
    %357 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%356 : tensor<576xf32>) outs(%354 : tensor<576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.cmpf one, %in, %cst_0 : f32
      cf.assert %560, "unimplemented: tensor with zero element"
      %561 = arith.divf %cst_1, %in : f32
      linalg.yield %561 : f32
    } -> tensor<576xf32>
    %expanded_157 = tensor.expand_shape %arg99 [[0, 1, 2]] output_shape [576, 1, 1] : tensor<576xf32> into tensor<576x1x1xf32>
    %expanded_158 = tensor.expand_shape %357 [[0, 1, 2]] output_shape [576, 1, 1] : tensor<576xf32> into tensor<576x1x1xf32>
    %358 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%353, %expanded_157 : tensor<1x576x14x14xf32>, tensor<576x1x1xf32>) outs(%351 : tensor<1x576x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.subf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x576x14x14xf32>
    %359 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%358, %expanded_158 : tensor<1x576x14x14xf32>, tensor<576x1x1xf32>) outs(%351 : tensor<1x576x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x576x14x14xf32>
    %expanded_159 = tensor.expand_shape %arg257 [[0, 1, 2]] output_shape [576, 1, 1] : tensor<576xf32> into tensor<576x1x1xf32>
    %360 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%359, %expanded_159 : tensor<1x576x14x14xf32>, tensor<576x1x1xf32>) outs(%351 : tensor<1x576x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x576x14x14xf32>
    %expanded_160 = tensor.expand_shape %arg258 [[0, 1, 2]] output_shape [576, 1, 1] : tensor<576xf32> into tensor<576x1x1xf32>
    %361 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%360, %expanded_160 : tensor<1x576x14x14xf32>, tensor<576x1x1xf32>) outs(%351 : tensor<1x576x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.addf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x576x14x14xf32>
    %362 = linalg.generic {indexing_maps = [#map1, #map4, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%361, %12 : tensor<1x576x14x14xf32>, tensor<f32>) outs(%351 : tensor<1x576x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.cmpf ogt, %in, %in_246 : f32
      %561 = arith.select %560, %in, %in_246 : f32
      linalg.yield %561 : f32
    } -> tensor<1x576x14x14xf32>
    %363 = linalg.generic {indexing_maps = [#map4, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%14, %362 : tensor<f32>, tensor<1x576x14x14xf32>) outs(%351 : tensor<1x576x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.cmpf olt, %in, %in_246 : f32
      %561 = arith.select %560, %in, %in_246 : f32
      linalg.yield %561 : f32
    } -> tensor<1x576x14x14xf32>
    %padded_161 = tensor.pad %363 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg315: index, %arg316: index, %arg317: index, %arg318: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x576x14x14xf32> to tensor<1x576x16x16xf32>
    %collapsed_162 = tensor.collapse_shape %arg259 [[0, 1], [2], [3]] : tensor<576x1x3x3xf32> into tensor<576x3x3xf32>
    %364 = linalg.depthwise_conv_2d_nchw_chw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_161, %collapsed_162 : tensor<1x576x16x16xf32>, tensor<576x3x3xf32>) outs(%352 : tensor<1x576x14x14xf32>) -> tensor<1x576x14x14xf32>
    %365 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg103 : tensor<576xf32>) outs(%354 : tensor<576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.truncf %cst_2 : f64 to f32
      %561 = arith.addf %in, %560 : f32
      linalg.yield %561 : f32
    } -> tensor<576xf32>
    %366 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%365 : tensor<576xf32>) outs(%354 : tensor<576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = math.sqrt %in : f32
      linalg.yield %560 : f32
    } -> tensor<576xf32>
    %367 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%366 : tensor<576xf32>) outs(%354 : tensor<576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.cmpf one, %in, %cst_0 : f32
      cf.assert %560, "unimplemented: tensor with zero element"
      %561 = arith.divf %cst_1, %in : f32
      linalg.yield %561 : f32
    } -> tensor<576xf32>
    %expanded_163 = tensor.expand_shape %arg102 [[0, 1, 2]] output_shape [576, 1, 1] : tensor<576xf32> into tensor<576x1x1xf32>
    %expanded_164 = tensor.expand_shape %367 [[0, 1, 2]] output_shape [576, 1, 1] : tensor<576xf32> into tensor<576x1x1xf32>
    %368 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%364, %expanded_163 : tensor<1x576x14x14xf32>, tensor<576x1x1xf32>) outs(%351 : tensor<1x576x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.subf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x576x14x14xf32>
    %369 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%368, %expanded_164 : tensor<1x576x14x14xf32>, tensor<576x1x1xf32>) outs(%351 : tensor<1x576x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x576x14x14xf32>
    %expanded_165 = tensor.expand_shape %arg260 [[0, 1, 2]] output_shape [576, 1, 1] : tensor<576xf32> into tensor<576x1x1xf32>
    %370 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%369, %expanded_165 : tensor<1x576x14x14xf32>, tensor<576x1x1xf32>) outs(%351 : tensor<1x576x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x576x14x14xf32>
    %expanded_166 = tensor.expand_shape %arg261 [[0, 1, 2]] output_shape [576, 1, 1] : tensor<576xf32> into tensor<576x1x1xf32>
    %371 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%370, %expanded_166 : tensor<1x576x14x14xf32>, tensor<576x1x1xf32>) outs(%351 : tensor<1x576x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.addf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x576x14x14xf32>
    %372 = linalg.generic {indexing_maps = [#map1, #map4, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%371, %12 : tensor<1x576x14x14xf32>, tensor<f32>) outs(%351 : tensor<1x576x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.cmpf ogt, %in, %in_246 : f32
      %561 = arith.select %560, %in, %in_246 : f32
      linalg.yield %561 : f32
    } -> tensor<1x576x14x14xf32>
    %373 = linalg.generic {indexing_maps = [#map4, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%14, %372 : tensor<f32>, tensor<1x576x14x14xf32>) outs(%351 : tensor<1x576x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.cmpf olt, %in, %in_246 : f32
      %561 = arith.select %560, %in, %in_246 : f32
      linalg.yield %561 : f32
    } -> tensor<1x576x14x14xf32>
    %374 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%373, %arg262 : tensor<1x576x14x14xf32>, tensor<96x576x1x1xf32>) outs(%342 : tensor<1x96x14x14xf32>) -> tensor<1x96x14x14xf32>
    %375 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg106 : tensor<96xf32>) outs(%40 : tensor<96xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.truncf %cst_2 : f64 to f32
      %561 = arith.addf %in, %560 : f32
      linalg.yield %561 : f32
    } -> tensor<96xf32>
    %376 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%375 : tensor<96xf32>) outs(%40 : tensor<96xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = math.sqrt %in : f32
      linalg.yield %560 : f32
    } -> tensor<96xf32>
    %377 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%376 : tensor<96xf32>) outs(%40 : tensor<96xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.cmpf one, %in, %cst_0 : f32
      cf.assert %560, "unimplemented: tensor with zero element"
      %561 = arith.divf %cst_1, %in : f32
      linalg.yield %561 : f32
    } -> tensor<96xf32>
    %expanded_167 = tensor.expand_shape %arg105 [[0, 1, 2]] output_shape [96, 1, 1] : tensor<96xf32> into tensor<96x1x1xf32>
    %expanded_168 = tensor.expand_shape %377 [[0, 1, 2]] output_shape [96, 1, 1] : tensor<96xf32> into tensor<96x1x1xf32>
    %378 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%374, %expanded_167 : tensor<1x96x14x14xf32>, tensor<96x1x1xf32>) outs(%341 : tensor<1x96x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.subf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x96x14x14xf32>
    %379 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%378, %expanded_168 : tensor<1x96x14x14xf32>, tensor<96x1x1xf32>) outs(%341 : tensor<1x96x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x96x14x14xf32>
    %expanded_169 = tensor.expand_shape %arg263 [[0, 1, 2]] output_shape [96, 1, 1] : tensor<96xf32> into tensor<96x1x1xf32>
    %380 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%379, %expanded_169 : tensor<1x96x14x14xf32>, tensor<96x1x1xf32>) outs(%341 : tensor<1x96x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x96x14x14xf32>
    %expanded_170 = tensor.expand_shape %arg264 [[0, 1, 2]] output_shape [96, 1, 1] : tensor<96xf32> into tensor<96x1x1xf32>
    %381 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%380, %expanded_170 : tensor<1x96x14x14xf32>, tensor<96x1x1xf32>) outs(%341 : tensor<1x96x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.addf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x96x14x14xf32>
    %382 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%350, %381 : tensor<1x96x14x14xf32>, tensor<1x96x14x14xf32>) outs(%341 : tensor<1x96x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.addf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x96x14x14xf32>
    %383 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%382, %arg265 : tensor<1x96x14x14xf32>, tensor<576x96x1x1xf32>) outs(%352 : tensor<1x576x14x14xf32>) -> tensor<1x576x14x14xf32>
    %384 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg109 : tensor<576xf32>) outs(%354 : tensor<576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.truncf %cst_2 : f64 to f32
      %561 = arith.addf %in, %560 : f32
      linalg.yield %561 : f32
    } -> tensor<576xf32>
    %385 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%384 : tensor<576xf32>) outs(%354 : tensor<576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = math.sqrt %in : f32
      linalg.yield %560 : f32
    } -> tensor<576xf32>
    %386 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%385 : tensor<576xf32>) outs(%354 : tensor<576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.cmpf one, %in, %cst_0 : f32
      cf.assert %560, "unimplemented: tensor with zero element"
      %561 = arith.divf %cst_1, %in : f32
      linalg.yield %561 : f32
    } -> tensor<576xf32>
    %expanded_171 = tensor.expand_shape %arg108 [[0, 1, 2]] output_shape [576, 1, 1] : tensor<576xf32> into tensor<576x1x1xf32>
    %expanded_172 = tensor.expand_shape %386 [[0, 1, 2]] output_shape [576, 1, 1] : tensor<576xf32> into tensor<576x1x1xf32>
    %387 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%383, %expanded_171 : tensor<1x576x14x14xf32>, tensor<576x1x1xf32>) outs(%351 : tensor<1x576x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.subf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x576x14x14xf32>
    %388 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%387, %expanded_172 : tensor<1x576x14x14xf32>, tensor<576x1x1xf32>) outs(%351 : tensor<1x576x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x576x14x14xf32>
    %expanded_173 = tensor.expand_shape %arg266 [[0, 1, 2]] output_shape [576, 1, 1] : tensor<576xf32> into tensor<576x1x1xf32>
    %389 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%388, %expanded_173 : tensor<1x576x14x14xf32>, tensor<576x1x1xf32>) outs(%351 : tensor<1x576x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x576x14x14xf32>
    %expanded_174 = tensor.expand_shape %arg267 [[0, 1, 2]] output_shape [576, 1, 1] : tensor<576xf32> into tensor<576x1x1xf32>
    %390 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%389, %expanded_174 : tensor<1x576x14x14xf32>, tensor<576x1x1xf32>) outs(%351 : tensor<1x576x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.addf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x576x14x14xf32>
    %391 = linalg.generic {indexing_maps = [#map1, #map4, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%390, %12 : tensor<1x576x14x14xf32>, tensor<f32>) outs(%351 : tensor<1x576x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.cmpf ogt, %in, %in_246 : f32
      %561 = arith.select %560, %in, %in_246 : f32
      linalg.yield %561 : f32
    } -> tensor<1x576x14x14xf32>
    %392 = linalg.generic {indexing_maps = [#map4, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%14, %391 : tensor<f32>, tensor<1x576x14x14xf32>) outs(%351 : tensor<1x576x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.cmpf olt, %in, %in_246 : f32
      %561 = arith.select %560, %in, %in_246 : f32
      linalg.yield %561 : f32
    } -> tensor<1x576x14x14xf32>
    %padded_175 = tensor.pad %392 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg315: index, %arg316: index, %arg317: index, %arg318: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x576x14x14xf32> to tensor<1x576x16x16xf32>
    %collapsed_176 = tensor.collapse_shape %arg268 [[0, 1], [2], [3]] : tensor<576x1x3x3xf32> into tensor<576x3x3xf32>
    %393 = linalg.depthwise_conv_2d_nchw_chw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_175, %collapsed_176 : tensor<1x576x16x16xf32>, tensor<576x3x3xf32>) outs(%352 : tensor<1x576x14x14xf32>) -> tensor<1x576x14x14xf32>
    %394 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg112 : tensor<576xf32>) outs(%354 : tensor<576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.truncf %cst_2 : f64 to f32
      %561 = arith.addf %in, %560 : f32
      linalg.yield %561 : f32
    } -> tensor<576xf32>
    %395 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%394 : tensor<576xf32>) outs(%354 : tensor<576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = math.sqrt %in : f32
      linalg.yield %560 : f32
    } -> tensor<576xf32>
    %396 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%395 : tensor<576xf32>) outs(%354 : tensor<576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.cmpf one, %in, %cst_0 : f32
      cf.assert %560, "unimplemented: tensor with zero element"
      %561 = arith.divf %cst_1, %in : f32
      linalg.yield %561 : f32
    } -> tensor<576xf32>
    %expanded_177 = tensor.expand_shape %arg111 [[0, 1, 2]] output_shape [576, 1, 1] : tensor<576xf32> into tensor<576x1x1xf32>
    %expanded_178 = tensor.expand_shape %396 [[0, 1, 2]] output_shape [576, 1, 1] : tensor<576xf32> into tensor<576x1x1xf32>
    %397 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%393, %expanded_177 : tensor<1x576x14x14xf32>, tensor<576x1x1xf32>) outs(%351 : tensor<1x576x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.subf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x576x14x14xf32>
    %398 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%397, %expanded_178 : tensor<1x576x14x14xf32>, tensor<576x1x1xf32>) outs(%351 : tensor<1x576x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x576x14x14xf32>
    %expanded_179 = tensor.expand_shape %arg269 [[0, 1, 2]] output_shape [576, 1, 1] : tensor<576xf32> into tensor<576x1x1xf32>
    %399 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%398, %expanded_179 : tensor<1x576x14x14xf32>, tensor<576x1x1xf32>) outs(%351 : tensor<1x576x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x576x14x14xf32>
    %expanded_180 = tensor.expand_shape %arg270 [[0, 1, 2]] output_shape [576, 1, 1] : tensor<576xf32> into tensor<576x1x1xf32>
    %400 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%399, %expanded_180 : tensor<1x576x14x14xf32>, tensor<576x1x1xf32>) outs(%351 : tensor<1x576x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.addf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x576x14x14xf32>
    %401 = linalg.generic {indexing_maps = [#map1, #map4, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%400, %12 : tensor<1x576x14x14xf32>, tensor<f32>) outs(%351 : tensor<1x576x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.cmpf ogt, %in, %in_246 : f32
      %561 = arith.select %560, %in, %in_246 : f32
      linalg.yield %561 : f32
    } -> tensor<1x576x14x14xf32>
    %402 = linalg.generic {indexing_maps = [#map4, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%14, %401 : tensor<f32>, tensor<1x576x14x14xf32>) outs(%351 : tensor<1x576x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.cmpf olt, %in, %in_246 : f32
      %561 = arith.select %560, %in, %in_246 : f32
      linalg.yield %561 : f32
    } -> tensor<1x576x14x14xf32>
    %403 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%402, %arg271 : tensor<1x576x14x14xf32>, tensor<96x576x1x1xf32>) outs(%342 : tensor<1x96x14x14xf32>) -> tensor<1x96x14x14xf32>
    %404 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg115 : tensor<96xf32>) outs(%40 : tensor<96xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.truncf %cst_2 : f64 to f32
      %561 = arith.addf %in, %560 : f32
      linalg.yield %561 : f32
    } -> tensor<96xf32>
    %405 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%404 : tensor<96xf32>) outs(%40 : tensor<96xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = math.sqrt %in : f32
      linalg.yield %560 : f32
    } -> tensor<96xf32>
    %406 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%405 : tensor<96xf32>) outs(%40 : tensor<96xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.cmpf one, %in, %cst_0 : f32
      cf.assert %560, "unimplemented: tensor with zero element"
      %561 = arith.divf %cst_1, %in : f32
      linalg.yield %561 : f32
    } -> tensor<96xf32>
    %expanded_181 = tensor.expand_shape %arg114 [[0, 1, 2]] output_shape [96, 1, 1] : tensor<96xf32> into tensor<96x1x1xf32>
    %expanded_182 = tensor.expand_shape %406 [[0, 1, 2]] output_shape [96, 1, 1] : tensor<96xf32> into tensor<96x1x1xf32>
    %407 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%403, %expanded_181 : tensor<1x96x14x14xf32>, tensor<96x1x1xf32>) outs(%341 : tensor<1x96x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.subf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x96x14x14xf32>
    %408 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%407, %expanded_182 : tensor<1x96x14x14xf32>, tensor<96x1x1xf32>) outs(%341 : tensor<1x96x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x96x14x14xf32>
    %expanded_183 = tensor.expand_shape %arg272 [[0, 1, 2]] output_shape [96, 1, 1] : tensor<96xf32> into tensor<96x1x1xf32>
    %409 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%408, %expanded_183 : tensor<1x96x14x14xf32>, tensor<96x1x1xf32>) outs(%341 : tensor<1x96x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x96x14x14xf32>
    %expanded_184 = tensor.expand_shape %arg273 [[0, 1, 2]] output_shape [96, 1, 1] : tensor<96xf32> into tensor<96x1x1xf32>
    %410 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%409, %expanded_184 : tensor<1x96x14x14xf32>, tensor<96x1x1xf32>) outs(%341 : tensor<1x96x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.addf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x96x14x14xf32>
    %411 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%382, %410 : tensor<1x96x14x14xf32>, tensor<1x96x14x14xf32>) outs(%341 : tensor<1x96x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.addf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x96x14x14xf32>
    %412 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%411, %arg274 : tensor<1x96x14x14xf32>, tensor<576x96x1x1xf32>) outs(%352 : tensor<1x576x14x14xf32>) -> tensor<1x576x14x14xf32>
    %413 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg118 : tensor<576xf32>) outs(%354 : tensor<576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.truncf %cst_2 : f64 to f32
      %561 = arith.addf %in, %560 : f32
      linalg.yield %561 : f32
    } -> tensor<576xf32>
    %414 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%413 : tensor<576xf32>) outs(%354 : tensor<576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = math.sqrt %in : f32
      linalg.yield %560 : f32
    } -> tensor<576xf32>
    %415 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%414 : tensor<576xf32>) outs(%354 : tensor<576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.cmpf one, %in, %cst_0 : f32
      cf.assert %560, "unimplemented: tensor with zero element"
      %561 = arith.divf %cst_1, %in : f32
      linalg.yield %561 : f32
    } -> tensor<576xf32>
    %expanded_185 = tensor.expand_shape %arg117 [[0, 1, 2]] output_shape [576, 1, 1] : tensor<576xf32> into tensor<576x1x1xf32>
    %expanded_186 = tensor.expand_shape %415 [[0, 1, 2]] output_shape [576, 1, 1] : tensor<576xf32> into tensor<576x1x1xf32>
    %416 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%412, %expanded_185 : tensor<1x576x14x14xf32>, tensor<576x1x1xf32>) outs(%351 : tensor<1x576x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.subf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x576x14x14xf32>
    %417 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%416, %expanded_186 : tensor<1x576x14x14xf32>, tensor<576x1x1xf32>) outs(%351 : tensor<1x576x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x576x14x14xf32>
    %expanded_187 = tensor.expand_shape %arg275 [[0, 1, 2]] output_shape [576, 1, 1] : tensor<576xf32> into tensor<576x1x1xf32>
    %418 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%417, %expanded_187 : tensor<1x576x14x14xf32>, tensor<576x1x1xf32>) outs(%351 : tensor<1x576x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x576x14x14xf32>
    %expanded_188 = tensor.expand_shape %arg276 [[0, 1, 2]] output_shape [576, 1, 1] : tensor<576xf32> into tensor<576x1x1xf32>
    %419 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%418, %expanded_188 : tensor<1x576x14x14xf32>, tensor<576x1x1xf32>) outs(%351 : tensor<1x576x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.addf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x576x14x14xf32>
    %420 = linalg.generic {indexing_maps = [#map1, #map4, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%419, %12 : tensor<1x576x14x14xf32>, tensor<f32>) outs(%351 : tensor<1x576x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.cmpf ogt, %in, %in_246 : f32
      %561 = arith.select %560, %in, %in_246 : f32
      linalg.yield %561 : f32
    } -> tensor<1x576x14x14xf32>
    %421 = linalg.generic {indexing_maps = [#map4, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%14, %420 : tensor<f32>, tensor<1x576x14x14xf32>) outs(%351 : tensor<1x576x14x14xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.cmpf olt, %in, %in_246 : f32
      %561 = arith.select %560, %in, %in_246 : f32
      linalg.yield %561 : f32
    } -> tensor<1x576x14x14xf32>
    %padded_189 = tensor.pad %421 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg315: index, %arg316: index, %arg317: index, %arg318: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x576x14x14xf32> to tensor<1x576x16x16xf32>
    %422 = tensor.empty() : tensor<1x576x7x7xf32>
    %423 = linalg.fill ins(%cst_0 : f32) outs(%422 : tensor<1x576x7x7xf32>) -> tensor<1x576x7x7xf32>
    %collapsed_190 = tensor.collapse_shape %arg277 [[0, 1], [2], [3]] : tensor<576x1x3x3xf32> into tensor<576x3x3xf32>
    %424 = linalg.depthwise_conv_2d_nchw_chw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%padded_189, %collapsed_190 : tensor<1x576x16x16xf32>, tensor<576x3x3xf32>) outs(%423 : tensor<1x576x7x7xf32>) -> tensor<1x576x7x7xf32>
    %425 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg121 : tensor<576xf32>) outs(%354 : tensor<576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.truncf %cst_2 : f64 to f32
      %561 = arith.addf %in, %560 : f32
      linalg.yield %561 : f32
    } -> tensor<576xf32>
    %426 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%425 : tensor<576xf32>) outs(%354 : tensor<576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = math.sqrt %in : f32
      linalg.yield %560 : f32
    } -> tensor<576xf32>
    %427 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%426 : tensor<576xf32>) outs(%354 : tensor<576xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.cmpf one, %in, %cst_0 : f32
      cf.assert %560, "unimplemented: tensor with zero element"
      %561 = arith.divf %cst_1, %in : f32
      linalg.yield %561 : f32
    } -> tensor<576xf32>
    %expanded_191 = tensor.expand_shape %arg120 [[0, 1, 2]] output_shape [576, 1, 1] : tensor<576xf32> into tensor<576x1x1xf32>
    %expanded_192 = tensor.expand_shape %427 [[0, 1, 2]] output_shape [576, 1, 1] : tensor<576xf32> into tensor<576x1x1xf32>
    %428 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%424, %expanded_191 : tensor<1x576x7x7xf32>, tensor<576x1x1xf32>) outs(%422 : tensor<1x576x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.subf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x576x7x7xf32>
    %429 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%428, %expanded_192 : tensor<1x576x7x7xf32>, tensor<576x1x1xf32>) outs(%422 : tensor<1x576x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x576x7x7xf32>
    %expanded_193 = tensor.expand_shape %arg278 [[0, 1, 2]] output_shape [576, 1, 1] : tensor<576xf32> into tensor<576x1x1xf32>
    %430 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%429, %expanded_193 : tensor<1x576x7x7xf32>, tensor<576x1x1xf32>) outs(%422 : tensor<1x576x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x576x7x7xf32>
    %expanded_194 = tensor.expand_shape %arg279 [[0, 1, 2]] output_shape [576, 1, 1] : tensor<576xf32> into tensor<576x1x1xf32>
    %431 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%430, %expanded_194 : tensor<1x576x7x7xf32>, tensor<576x1x1xf32>) outs(%422 : tensor<1x576x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.addf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x576x7x7xf32>
    %432 = linalg.generic {indexing_maps = [#map1, #map4, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%431, %12 : tensor<1x576x7x7xf32>, tensor<f32>) outs(%422 : tensor<1x576x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.cmpf ogt, %in, %in_246 : f32
      %561 = arith.select %560, %in, %in_246 : f32
      linalg.yield %561 : f32
    } -> tensor<1x576x7x7xf32>
    %433 = linalg.generic {indexing_maps = [#map4, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%14, %432 : tensor<f32>, tensor<1x576x7x7xf32>) outs(%422 : tensor<1x576x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.cmpf olt, %in, %in_246 : f32
      %561 = arith.select %560, %in, %in_246 : f32
      linalg.yield %561 : f32
    } -> tensor<1x576x7x7xf32>
    %434 = tensor.empty() : tensor<1x160x7x7xf32>
    %435 = linalg.fill ins(%cst_0 : f32) outs(%434 : tensor<1x160x7x7xf32>) -> tensor<1x160x7x7xf32>
    %436 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%433, %arg280 : tensor<1x576x7x7xf32>, tensor<160x576x1x1xf32>) outs(%435 : tensor<1x160x7x7xf32>) -> tensor<1x160x7x7xf32>
    %437 = tensor.empty() : tensor<160xf32>
    %438 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg124 : tensor<160xf32>) outs(%437 : tensor<160xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.truncf %cst_2 : f64 to f32
      %561 = arith.addf %in, %560 : f32
      linalg.yield %561 : f32
    } -> tensor<160xf32>
    %439 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%438 : tensor<160xf32>) outs(%437 : tensor<160xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = math.sqrt %in : f32
      linalg.yield %560 : f32
    } -> tensor<160xf32>
    %440 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%439 : tensor<160xf32>) outs(%437 : tensor<160xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.cmpf one, %in, %cst_0 : f32
      cf.assert %560, "unimplemented: tensor with zero element"
      %561 = arith.divf %cst_1, %in : f32
      linalg.yield %561 : f32
    } -> tensor<160xf32>
    %expanded_195 = tensor.expand_shape %arg123 [[0, 1, 2]] output_shape [160, 1, 1] : tensor<160xf32> into tensor<160x1x1xf32>
    %expanded_196 = tensor.expand_shape %440 [[0, 1, 2]] output_shape [160, 1, 1] : tensor<160xf32> into tensor<160x1x1xf32>
    %441 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%436, %expanded_195 : tensor<1x160x7x7xf32>, tensor<160x1x1xf32>) outs(%434 : tensor<1x160x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.subf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x160x7x7xf32>
    %442 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%441, %expanded_196 : tensor<1x160x7x7xf32>, tensor<160x1x1xf32>) outs(%434 : tensor<1x160x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x160x7x7xf32>
    %expanded_197 = tensor.expand_shape %arg281 [[0, 1, 2]] output_shape [160, 1, 1] : tensor<160xf32> into tensor<160x1x1xf32>
    %443 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%442, %expanded_197 : tensor<1x160x7x7xf32>, tensor<160x1x1xf32>) outs(%434 : tensor<1x160x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x160x7x7xf32>
    %expanded_198 = tensor.expand_shape %arg282 [[0, 1, 2]] output_shape [160, 1, 1] : tensor<160xf32> into tensor<160x1x1xf32>
    %444 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%443, %expanded_198 : tensor<1x160x7x7xf32>, tensor<160x1x1xf32>) outs(%434 : tensor<1x160x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.addf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x160x7x7xf32>
    %445 = tensor.empty() : tensor<1x960x7x7xf32>
    %446 = linalg.fill ins(%cst_0 : f32) outs(%445 : tensor<1x960x7x7xf32>) -> tensor<1x960x7x7xf32>
    %447 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%444, %arg283 : tensor<1x160x7x7xf32>, tensor<960x160x1x1xf32>) outs(%446 : tensor<1x960x7x7xf32>) -> tensor<1x960x7x7xf32>
    %448 = tensor.empty() : tensor<960xf32>
    %449 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg127 : tensor<960xf32>) outs(%448 : tensor<960xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.truncf %cst_2 : f64 to f32
      %561 = arith.addf %in, %560 : f32
      linalg.yield %561 : f32
    } -> tensor<960xf32>
    %450 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%449 : tensor<960xf32>) outs(%448 : tensor<960xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = math.sqrt %in : f32
      linalg.yield %560 : f32
    } -> tensor<960xf32>
    %451 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%450 : tensor<960xf32>) outs(%448 : tensor<960xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.cmpf one, %in, %cst_0 : f32
      cf.assert %560, "unimplemented: tensor with zero element"
      %561 = arith.divf %cst_1, %in : f32
      linalg.yield %561 : f32
    } -> tensor<960xf32>
    %expanded_199 = tensor.expand_shape %arg126 [[0, 1, 2]] output_shape [960, 1, 1] : tensor<960xf32> into tensor<960x1x1xf32>
    %expanded_200 = tensor.expand_shape %451 [[0, 1, 2]] output_shape [960, 1, 1] : tensor<960xf32> into tensor<960x1x1xf32>
    %452 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%447, %expanded_199 : tensor<1x960x7x7xf32>, tensor<960x1x1xf32>) outs(%445 : tensor<1x960x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.subf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x960x7x7xf32>
    %453 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%452, %expanded_200 : tensor<1x960x7x7xf32>, tensor<960x1x1xf32>) outs(%445 : tensor<1x960x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x960x7x7xf32>
    %expanded_201 = tensor.expand_shape %arg284 [[0, 1, 2]] output_shape [960, 1, 1] : tensor<960xf32> into tensor<960x1x1xf32>
    %454 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%453, %expanded_201 : tensor<1x960x7x7xf32>, tensor<960x1x1xf32>) outs(%445 : tensor<1x960x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x960x7x7xf32>
    %expanded_202 = tensor.expand_shape %arg285 [[0, 1, 2]] output_shape [960, 1, 1] : tensor<960xf32> into tensor<960x1x1xf32>
    %455 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%454, %expanded_202 : tensor<1x960x7x7xf32>, tensor<960x1x1xf32>) outs(%445 : tensor<1x960x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.addf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x960x7x7xf32>
    %456 = linalg.generic {indexing_maps = [#map1, #map4, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%455, %12 : tensor<1x960x7x7xf32>, tensor<f32>) outs(%445 : tensor<1x960x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.cmpf ogt, %in, %in_246 : f32
      %561 = arith.select %560, %in, %in_246 : f32
      linalg.yield %561 : f32
    } -> tensor<1x960x7x7xf32>
    %457 = linalg.generic {indexing_maps = [#map4, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%14, %456 : tensor<f32>, tensor<1x960x7x7xf32>) outs(%445 : tensor<1x960x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.cmpf olt, %in, %in_246 : f32
      %561 = arith.select %560, %in, %in_246 : f32
      linalg.yield %561 : f32
    } -> tensor<1x960x7x7xf32>
    %padded_203 = tensor.pad %457 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg315: index, %arg316: index, %arg317: index, %arg318: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x960x7x7xf32> to tensor<1x960x9x9xf32>
    %collapsed_204 = tensor.collapse_shape %arg286 [[0, 1], [2], [3]] : tensor<960x1x3x3xf32> into tensor<960x3x3xf32>
    %458 = linalg.depthwise_conv_2d_nchw_chw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_203, %collapsed_204 : tensor<1x960x9x9xf32>, tensor<960x3x3xf32>) outs(%446 : tensor<1x960x7x7xf32>) -> tensor<1x960x7x7xf32>
    %459 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg130 : tensor<960xf32>) outs(%448 : tensor<960xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.truncf %cst_2 : f64 to f32
      %561 = arith.addf %in, %560 : f32
      linalg.yield %561 : f32
    } -> tensor<960xf32>
    %460 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%459 : tensor<960xf32>) outs(%448 : tensor<960xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = math.sqrt %in : f32
      linalg.yield %560 : f32
    } -> tensor<960xf32>
    %461 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%460 : tensor<960xf32>) outs(%448 : tensor<960xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.cmpf one, %in, %cst_0 : f32
      cf.assert %560, "unimplemented: tensor with zero element"
      %561 = arith.divf %cst_1, %in : f32
      linalg.yield %561 : f32
    } -> tensor<960xf32>
    %expanded_205 = tensor.expand_shape %arg129 [[0, 1, 2]] output_shape [960, 1, 1] : tensor<960xf32> into tensor<960x1x1xf32>
    %expanded_206 = tensor.expand_shape %461 [[0, 1, 2]] output_shape [960, 1, 1] : tensor<960xf32> into tensor<960x1x1xf32>
    %462 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%458, %expanded_205 : tensor<1x960x7x7xf32>, tensor<960x1x1xf32>) outs(%445 : tensor<1x960x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.subf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x960x7x7xf32>
    %463 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%462, %expanded_206 : tensor<1x960x7x7xf32>, tensor<960x1x1xf32>) outs(%445 : tensor<1x960x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x960x7x7xf32>
    %expanded_207 = tensor.expand_shape %arg287 [[0, 1, 2]] output_shape [960, 1, 1] : tensor<960xf32> into tensor<960x1x1xf32>
    %464 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%463, %expanded_207 : tensor<1x960x7x7xf32>, tensor<960x1x1xf32>) outs(%445 : tensor<1x960x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x960x7x7xf32>
    %expanded_208 = tensor.expand_shape %arg288 [[0, 1, 2]] output_shape [960, 1, 1] : tensor<960xf32> into tensor<960x1x1xf32>
    %465 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%464, %expanded_208 : tensor<1x960x7x7xf32>, tensor<960x1x1xf32>) outs(%445 : tensor<1x960x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.addf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x960x7x7xf32>
    %466 = linalg.generic {indexing_maps = [#map1, #map4, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%465, %12 : tensor<1x960x7x7xf32>, tensor<f32>) outs(%445 : tensor<1x960x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.cmpf ogt, %in, %in_246 : f32
      %561 = arith.select %560, %in, %in_246 : f32
      linalg.yield %561 : f32
    } -> tensor<1x960x7x7xf32>
    %467 = linalg.generic {indexing_maps = [#map4, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%14, %466 : tensor<f32>, tensor<1x960x7x7xf32>) outs(%445 : tensor<1x960x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.cmpf olt, %in, %in_246 : f32
      %561 = arith.select %560, %in, %in_246 : f32
      linalg.yield %561 : f32
    } -> tensor<1x960x7x7xf32>
    %468 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%467, %arg289 : tensor<1x960x7x7xf32>, tensor<160x960x1x1xf32>) outs(%435 : tensor<1x160x7x7xf32>) -> tensor<1x160x7x7xf32>
    %469 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg133 : tensor<160xf32>) outs(%437 : tensor<160xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.truncf %cst_2 : f64 to f32
      %561 = arith.addf %in, %560 : f32
      linalg.yield %561 : f32
    } -> tensor<160xf32>
    %470 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%469 : tensor<160xf32>) outs(%437 : tensor<160xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = math.sqrt %in : f32
      linalg.yield %560 : f32
    } -> tensor<160xf32>
    %471 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%470 : tensor<160xf32>) outs(%437 : tensor<160xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.cmpf one, %in, %cst_0 : f32
      cf.assert %560, "unimplemented: tensor with zero element"
      %561 = arith.divf %cst_1, %in : f32
      linalg.yield %561 : f32
    } -> tensor<160xf32>
    %expanded_209 = tensor.expand_shape %arg132 [[0, 1, 2]] output_shape [160, 1, 1] : tensor<160xf32> into tensor<160x1x1xf32>
    %expanded_210 = tensor.expand_shape %471 [[0, 1, 2]] output_shape [160, 1, 1] : tensor<160xf32> into tensor<160x1x1xf32>
    %472 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%468, %expanded_209 : tensor<1x160x7x7xf32>, tensor<160x1x1xf32>) outs(%434 : tensor<1x160x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.subf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x160x7x7xf32>
    %473 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%472, %expanded_210 : tensor<1x160x7x7xf32>, tensor<160x1x1xf32>) outs(%434 : tensor<1x160x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x160x7x7xf32>
    %expanded_211 = tensor.expand_shape %arg290 [[0, 1, 2]] output_shape [160, 1, 1] : tensor<160xf32> into tensor<160x1x1xf32>
    %474 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%473, %expanded_211 : tensor<1x160x7x7xf32>, tensor<160x1x1xf32>) outs(%434 : tensor<1x160x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x160x7x7xf32>
    %expanded_212 = tensor.expand_shape %arg291 [[0, 1, 2]] output_shape [160, 1, 1] : tensor<160xf32> into tensor<160x1x1xf32>
    %475 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%474, %expanded_212 : tensor<1x160x7x7xf32>, tensor<160x1x1xf32>) outs(%434 : tensor<1x160x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.addf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x160x7x7xf32>
    %476 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%444, %475 : tensor<1x160x7x7xf32>, tensor<1x160x7x7xf32>) outs(%434 : tensor<1x160x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.addf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x160x7x7xf32>
    %477 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%476, %arg292 : tensor<1x160x7x7xf32>, tensor<960x160x1x1xf32>) outs(%446 : tensor<1x960x7x7xf32>) -> tensor<1x960x7x7xf32>
    %478 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg136 : tensor<960xf32>) outs(%448 : tensor<960xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.truncf %cst_2 : f64 to f32
      %561 = arith.addf %in, %560 : f32
      linalg.yield %561 : f32
    } -> tensor<960xf32>
    %479 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%478 : tensor<960xf32>) outs(%448 : tensor<960xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = math.sqrt %in : f32
      linalg.yield %560 : f32
    } -> tensor<960xf32>
    %480 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%479 : tensor<960xf32>) outs(%448 : tensor<960xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.cmpf one, %in, %cst_0 : f32
      cf.assert %560, "unimplemented: tensor with zero element"
      %561 = arith.divf %cst_1, %in : f32
      linalg.yield %561 : f32
    } -> tensor<960xf32>
    %expanded_213 = tensor.expand_shape %arg135 [[0, 1, 2]] output_shape [960, 1, 1] : tensor<960xf32> into tensor<960x1x1xf32>
    %expanded_214 = tensor.expand_shape %480 [[0, 1, 2]] output_shape [960, 1, 1] : tensor<960xf32> into tensor<960x1x1xf32>
    %481 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%477, %expanded_213 : tensor<1x960x7x7xf32>, tensor<960x1x1xf32>) outs(%445 : tensor<1x960x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.subf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x960x7x7xf32>
    %482 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%481, %expanded_214 : tensor<1x960x7x7xf32>, tensor<960x1x1xf32>) outs(%445 : tensor<1x960x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x960x7x7xf32>
    %expanded_215 = tensor.expand_shape %arg293 [[0, 1, 2]] output_shape [960, 1, 1] : tensor<960xf32> into tensor<960x1x1xf32>
    %483 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%482, %expanded_215 : tensor<1x960x7x7xf32>, tensor<960x1x1xf32>) outs(%445 : tensor<1x960x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x960x7x7xf32>
    %expanded_216 = tensor.expand_shape %arg294 [[0, 1, 2]] output_shape [960, 1, 1] : tensor<960xf32> into tensor<960x1x1xf32>
    %484 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%483, %expanded_216 : tensor<1x960x7x7xf32>, tensor<960x1x1xf32>) outs(%445 : tensor<1x960x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.addf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x960x7x7xf32>
    %485 = linalg.generic {indexing_maps = [#map1, #map4, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%484, %12 : tensor<1x960x7x7xf32>, tensor<f32>) outs(%445 : tensor<1x960x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.cmpf ogt, %in, %in_246 : f32
      %561 = arith.select %560, %in, %in_246 : f32
      linalg.yield %561 : f32
    } -> tensor<1x960x7x7xf32>
    %486 = linalg.generic {indexing_maps = [#map4, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%14, %485 : tensor<f32>, tensor<1x960x7x7xf32>) outs(%445 : tensor<1x960x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.cmpf olt, %in, %in_246 : f32
      %561 = arith.select %560, %in, %in_246 : f32
      linalg.yield %561 : f32
    } -> tensor<1x960x7x7xf32>
    %padded_217 = tensor.pad %486 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg315: index, %arg316: index, %arg317: index, %arg318: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x960x7x7xf32> to tensor<1x960x9x9xf32>
    %collapsed_218 = tensor.collapse_shape %arg295 [[0, 1], [2], [3]] : tensor<960x1x3x3xf32> into tensor<960x3x3xf32>
    %487 = linalg.depthwise_conv_2d_nchw_chw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_217, %collapsed_218 : tensor<1x960x9x9xf32>, tensor<960x3x3xf32>) outs(%446 : tensor<1x960x7x7xf32>) -> tensor<1x960x7x7xf32>
    %488 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg139 : tensor<960xf32>) outs(%448 : tensor<960xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.truncf %cst_2 : f64 to f32
      %561 = arith.addf %in, %560 : f32
      linalg.yield %561 : f32
    } -> tensor<960xf32>
    %489 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%488 : tensor<960xf32>) outs(%448 : tensor<960xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = math.sqrt %in : f32
      linalg.yield %560 : f32
    } -> tensor<960xf32>
    %490 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%489 : tensor<960xf32>) outs(%448 : tensor<960xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.cmpf one, %in, %cst_0 : f32
      cf.assert %560, "unimplemented: tensor with zero element"
      %561 = arith.divf %cst_1, %in : f32
      linalg.yield %561 : f32
    } -> tensor<960xf32>
    %expanded_219 = tensor.expand_shape %arg138 [[0, 1, 2]] output_shape [960, 1, 1] : tensor<960xf32> into tensor<960x1x1xf32>
    %expanded_220 = tensor.expand_shape %490 [[0, 1, 2]] output_shape [960, 1, 1] : tensor<960xf32> into tensor<960x1x1xf32>
    %491 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%487, %expanded_219 : tensor<1x960x7x7xf32>, tensor<960x1x1xf32>) outs(%445 : tensor<1x960x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.subf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x960x7x7xf32>
    %492 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%491, %expanded_220 : tensor<1x960x7x7xf32>, tensor<960x1x1xf32>) outs(%445 : tensor<1x960x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x960x7x7xf32>
    %expanded_221 = tensor.expand_shape %arg296 [[0, 1, 2]] output_shape [960, 1, 1] : tensor<960xf32> into tensor<960x1x1xf32>
    %493 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%492, %expanded_221 : tensor<1x960x7x7xf32>, tensor<960x1x1xf32>) outs(%445 : tensor<1x960x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x960x7x7xf32>
    %expanded_222 = tensor.expand_shape %arg297 [[0, 1, 2]] output_shape [960, 1, 1] : tensor<960xf32> into tensor<960x1x1xf32>
    %494 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%493, %expanded_222 : tensor<1x960x7x7xf32>, tensor<960x1x1xf32>) outs(%445 : tensor<1x960x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.addf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x960x7x7xf32>
    %495 = linalg.generic {indexing_maps = [#map1, #map4, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%494, %12 : tensor<1x960x7x7xf32>, tensor<f32>) outs(%445 : tensor<1x960x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.cmpf ogt, %in, %in_246 : f32
      %561 = arith.select %560, %in, %in_246 : f32
      linalg.yield %561 : f32
    } -> tensor<1x960x7x7xf32>
    %496 = linalg.generic {indexing_maps = [#map4, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%14, %495 : tensor<f32>, tensor<1x960x7x7xf32>) outs(%445 : tensor<1x960x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.cmpf olt, %in, %in_246 : f32
      %561 = arith.select %560, %in, %in_246 : f32
      linalg.yield %561 : f32
    } -> tensor<1x960x7x7xf32>
    %497 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%496, %arg298 : tensor<1x960x7x7xf32>, tensor<160x960x1x1xf32>) outs(%435 : tensor<1x160x7x7xf32>) -> tensor<1x160x7x7xf32>
    %498 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg142 : tensor<160xf32>) outs(%437 : tensor<160xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.truncf %cst_2 : f64 to f32
      %561 = arith.addf %in, %560 : f32
      linalg.yield %561 : f32
    } -> tensor<160xf32>
    %499 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%498 : tensor<160xf32>) outs(%437 : tensor<160xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = math.sqrt %in : f32
      linalg.yield %560 : f32
    } -> tensor<160xf32>
    %500 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%499 : tensor<160xf32>) outs(%437 : tensor<160xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.cmpf one, %in, %cst_0 : f32
      cf.assert %560, "unimplemented: tensor with zero element"
      %561 = arith.divf %cst_1, %in : f32
      linalg.yield %561 : f32
    } -> tensor<160xf32>
    %expanded_223 = tensor.expand_shape %arg141 [[0, 1, 2]] output_shape [160, 1, 1] : tensor<160xf32> into tensor<160x1x1xf32>
    %expanded_224 = tensor.expand_shape %500 [[0, 1, 2]] output_shape [160, 1, 1] : tensor<160xf32> into tensor<160x1x1xf32>
    %501 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%497, %expanded_223 : tensor<1x160x7x7xf32>, tensor<160x1x1xf32>) outs(%434 : tensor<1x160x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.subf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x160x7x7xf32>
    %502 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%501, %expanded_224 : tensor<1x160x7x7xf32>, tensor<160x1x1xf32>) outs(%434 : tensor<1x160x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x160x7x7xf32>
    %expanded_225 = tensor.expand_shape %arg299 [[0, 1, 2]] output_shape [160, 1, 1] : tensor<160xf32> into tensor<160x1x1xf32>
    %503 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%502, %expanded_225 : tensor<1x160x7x7xf32>, tensor<160x1x1xf32>) outs(%434 : tensor<1x160x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x160x7x7xf32>
    %expanded_226 = tensor.expand_shape %arg300 [[0, 1, 2]] output_shape [160, 1, 1] : tensor<160xf32> into tensor<160x1x1xf32>
    %504 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%503, %expanded_226 : tensor<1x160x7x7xf32>, tensor<160x1x1xf32>) outs(%434 : tensor<1x160x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.addf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x160x7x7xf32>
    %505 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%476, %504 : tensor<1x160x7x7xf32>, tensor<1x160x7x7xf32>) outs(%434 : tensor<1x160x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.addf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x160x7x7xf32>
    %506 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%505, %arg301 : tensor<1x160x7x7xf32>, tensor<960x160x1x1xf32>) outs(%446 : tensor<1x960x7x7xf32>) -> tensor<1x960x7x7xf32>
    %507 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg145 : tensor<960xf32>) outs(%448 : tensor<960xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.truncf %cst_2 : f64 to f32
      %561 = arith.addf %in, %560 : f32
      linalg.yield %561 : f32
    } -> tensor<960xf32>
    %508 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%507 : tensor<960xf32>) outs(%448 : tensor<960xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = math.sqrt %in : f32
      linalg.yield %560 : f32
    } -> tensor<960xf32>
    %509 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%508 : tensor<960xf32>) outs(%448 : tensor<960xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.cmpf one, %in, %cst_0 : f32
      cf.assert %560, "unimplemented: tensor with zero element"
      %561 = arith.divf %cst_1, %in : f32
      linalg.yield %561 : f32
    } -> tensor<960xf32>
    %expanded_227 = tensor.expand_shape %arg144 [[0, 1, 2]] output_shape [960, 1, 1] : tensor<960xf32> into tensor<960x1x1xf32>
    %expanded_228 = tensor.expand_shape %509 [[0, 1, 2]] output_shape [960, 1, 1] : tensor<960xf32> into tensor<960x1x1xf32>
    %510 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%506, %expanded_227 : tensor<1x960x7x7xf32>, tensor<960x1x1xf32>) outs(%445 : tensor<1x960x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.subf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x960x7x7xf32>
    %511 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%510, %expanded_228 : tensor<1x960x7x7xf32>, tensor<960x1x1xf32>) outs(%445 : tensor<1x960x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x960x7x7xf32>
    %expanded_229 = tensor.expand_shape %arg302 [[0, 1, 2]] output_shape [960, 1, 1] : tensor<960xf32> into tensor<960x1x1xf32>
    %512 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%511, %expanded_229 : tensor<1x960x7x7xf32>, tensor<960x1x1xf32>) outs(%445 : tensor<1x960x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x960x7x7xf32>
    %expanded_230 = tensor.expand_shape %arg303 [[0, 1, 2]] output_shape [960, 1, 1] : tensor<960xf32> into tensor<960x1x1xf32>
    %513 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%512, %expanded_230 : tensor<1x960x7x7xf32>, tensor<960x1x1xf32>) outs(%445 : tensor<1x960x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.addf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x960x7x7xf32>
    %514 = linalg.generic {indexing_maps = [#map1, #map4, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%513, %12 : tensor<1x960x7x7xf32>, tensor<f32>) outs(%445 : tensor<1x960x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.cmpf ogt, %in, %in_246 : f32
      %561 = arith.select %560, %in, %in_246 : f32
      linalg.yield %561 : f32
    } -> tensor<1x960x7x7xf32>
    %515 = linalg.generic {indexing_maps = [#map4, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%14, %514 : tensor<f32>, tensor<1x960x7x7xf32>) outs(%445 : tensor<1x960x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.cmpf olt, %in, %in_246 : f32
      %561 = arith.select %560, %in, %in_246 : f32
      linalg.yield %561 : f32
    } -> tensor<1x960x7x7xf32>
    %padded_231 = tensor.pad %515 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg315: index, %arg316: index, %arg317: index, %arg318: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x960x7x7xf32> to tensor<1x960x9x9xf32>
    %collapsed_232 = tensor.collapse_shape %arg304 [[0, 1], [2], [3]] : tensor<960x1x3x3xf32> into tensor<960x3x3xf32>
    %516 = linalg.depthwise_conv_2d_nchw_chw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_231, %collapsed_232 : tensor<1x960x9x9xf32>, tensor<960x3x3xf32>) outs(%446 : tensor<1x960x7x7xf32>) -> tensor<1x960x7x7xf32>
    %517 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg148 : tensor<960xf32>) outs(%448 : tensor<960xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.truncf %cst_2 : f64 to f32
      %561 = arith.addf %in, %560 : f32
      linalg.yield %561 : f32
    } -> tensor<960xf32>
    %518 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%517 : tensor<960xf32>) outs(%448 : tensor<960xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = math.sqrt %in : f32
      linalg.yield %560 : f32
    } -> tensor<960xf32>
    %519 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%518 : tensor<960xf32>) outs(%448 : tensor<960xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.cmpf one, %in, %cst_0 : f32
      cf.assert %560, "unimplemented: tensor with zero element"
      %561 = arith.divf %cst_1, %in : f32
      linalg.yield %561 : f32
    } -> tensor<960xf32>
    %expanded_233 = tensor.expand_shape %arg147 [[0, 1, 2]] output_shape [960, 1, 1] : tensor<960xf32> into tensor<960x1x1xf32>
    %expanded_234 = tensor.expand_shape %519 [[0, 1, 2]] output_shape [960, 1, 1] : tensor<960xf32> into tensor<960x1x1xf32>
    %520 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%516, %expanded_233 : tensor<1x960x7x7xf32>, tensor<960x1x1xf32>) outs(%445 : tensor<1x960x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.subf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x960x7x7xf32>
    %521 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%520, %expanded_234 : tensor<1x960x7x7xf32>, tensor<960x1x1xf32>) outs(%445 : tensor<1x960x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x960x7x7xf32>
    %expanded_235 = tensor.expand_shape %arg305 [[0, 1, 2]] output_shape [960, 1, 1] : tensor<960xf32> into tensor<960x1x1xf32>
    %522 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%521, %expanded_235 : tensor<1x960x7x7xf32>, tensor<960x1x1xf32>) outs(%445 : tensor<1x960x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x960x7x7xf32>
    %expanded_236 = tensor.expand_shape %arg306 [[0, 1, 2]] output_shape [960, 1, 1] : tensor<960xf32> into tensor<960x1x1xf32>
    %523 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%522, %expanded_236 : tensor<1x960x7x7xf32>, tensor<960x1x1xf32>) outs(%445 : tensor<1x960x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.addf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x960x7x7xf32>
    %524 = linalg.generic {indexing_maps = [#map1, #map4, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%523, %12 : tensor<1x960x7x7xf32>, tensor<f32>) outs(%445 : tensor<1x960x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.cmpf ogt, %in, %in_246 : f32
      %561 = arith.select %560, %in, %in_246 : f32
      linalg.yield %561 : f32
    } -> tensor<1x960x7x7xf32>
    %525 = linalg.generic {indexing_maps = [#map4, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%14, %524 : tensor<f32>, tensor<1x960x7x7xf32>) outs(%445 : tensor<1x960x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.cmpf olt, %in, %in_246 : f32
      %561 = arith.select %560, %in, %in_246 : f32
      linalg.yield %561 : f32
    } -> tensor<1x960x7x7xf32>
    %526 = tensor.empty() : tensor<1x320x7x7xf32>
    %527 = linalg.fill ins(%cst_0 : f32) outs(%526 : tensor<1x320x7x7xf32>) -> tensor<1x320x7x7xf32>
    %528 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%525, %arg307 : tensor<1x960x7x7xf32>, tensor<320x960x1x1xf32>) outs(%527 : tensor<1x320x7x7xf32>) -> tensor<1x320x7x7xf32>
    %529 = tensor.empty() : tensor<320xf32>
    %530 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg151 : tensor<320xf32>) outs(%529 : tensor<320xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.truncf %cst_2 : f64 to f32
      %561 = arith.addf %in, %560 : f32
      linalg.yield %561 : f32
    } -> tensor<320xf32>
    %531 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%530 : tensor<320xf32>) outs(%529 : tensor<320xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = math.sqrt %in : f32
      linalg.yield %560 : f32
    } -> tensor<320xf32>
    %532 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%531 : tensor<320xf32>) outs(%529 : tensor<320xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.cmpf one, %in, %cst_0 : f32
      cf.assert %560, "unimplemented: tensor with zero element"
      %561 = arith.divf %cst_1, %in : f32
      linalg.yield %561 : f32
    } -> tensor<320xf32>
    %expanded_237 = tensor.expand_shape %arg150 [[0, 1, 2]] output_shape [320, 1, 1] : tensor<320xf32> into tensor<320x1x1xf32>
    %expanded_238 = tensor.expand_shape %532 [[0, 1, 2]] output_shape [320, 1, 1] : tensor<320xf32> into tensor<320x1x1xf32>
    %533 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%528, %expanded_237 : tensor<1x320x7x7xf32>, tensor<320x1x1xf32>) outs(%526 : tensor<1x320x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.subf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x320x7x7xf32>
    %534 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%533, %expanded_238 : tensor<1x320x7x7xf32>, tensor<320x1x1xf32>) outs(%526 : tensor<1x320x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x320x7x7xf32>
    %expanded_239 = tensor.expand_shape %arg308 [[0, 1, 2]] output_shape [320, 1, 1] : tensor<320xf32> into tensor<320x1x1xf32>
    %535 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%534, %expanded_239 : tensor<1x320x7x7xf32>, tensor<320x1x1xf32>) outs(%526 : tensor<1x320x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x320x7x7xf32>
    %expanded_240 = tensor.expand_shape %arg309 [[0, 1, 2]] output_shape [320, 1, 1] : tensor<320xf32> into tensor<320x1x1xf32>
    %536 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%535, %expanded_240 : tensor<1x320x7x7xf32>, tensor<320x1x1xf32>) outs(%526 : tensor<1x320x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.addf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x320x7x7xf32>
    %537 = tensor.empty() : tensor<1x1280x7x7xf32>
    %538 = linalg.fill ins(%cst_0 : f32) outs(%537 : tensor<1x1280x7x7xf32>) -> tensor<1x1280x7x7xf32>
    %539 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%536, %arg310 : tensor<1x320x7x7xf32>, tensor<1280x320x1x1xf32>) outs(%538 : tensor<1x1280x7x7xf32>) -> tensor<1x1280x7x7xf32>
    %540 = tensor.empty() : tensor<1280xf32>
    %541 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg154 : tensor<1280xf32>) outs(%540 : tensor<1280xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.truncf %cst_2 : f64 to f32
      %561 = arith.addf %in, %560 : f32
      linalg.yield %561 : f32
    } -> tensor<1280xf32>
    %542 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%541 : tensor<1280xf32>) outs(%540 : tensor<1280xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = math.sqrt %in : f32
      linalg.yield %560 : f32
    } -> tensor<1280xf32>
    %543 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%542 : tensor<1280xf32>) outs(%540 : tensor<1280xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.cmpf one, %in, %cst_0 : f32
      cf.assert %560, "unimplemented: tensor with zero element"
      %561 = arith.divf %cst_1, %in : f32
      linalg.yield %561 : f32
    } -> tensor<1280xf32>
    %expanded_241 = tensor.expand_shape %arg153 [[0, 1, 2]] output_shape [1280, 1, 1] : tensor<1280xf32> into tensor<1280x1x1xf32>
    %expanded_242 = tensor.expand_shape %543 [[0, 1, 2]] output_shape [1280, 1, 1] : tensor<1280xf32> into tensor<1280x1x1xf32>
    %544 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%539, %expanded_241 : tensor<1x1280x7x7xf32>, tensor<1280x1x1xf32>) outs(%537 : tensor<1x1280x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.subf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x1280x7x7xf32>
    %545 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%544, %expanded_242 : tensor<1x1280x7x7xf32>, tensor<1280x1x1xf32>) outs(%537 : tensor<1x1280x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x1280x7x7xf32>
    %expanded_243 = tensor.expand_shape %arg311 [[0, 1, 2]] output_shape [1280, 1, 1] : tensor<1280xf32> into tensor<1280x1x1xf32>
    %546 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%545, %expanded_243 : tensor<1x1280x7x7xf32>, tensor<1280x1x1xf32>) outs(%537 : tensor<1x1280x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.mulf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x1280x7x7xf32>
    %expanded_244 = tensor.expand_shape %arg312 [[0, 1, 2]] output_shape [1280, 1, 1] : tensor<1280xf32> into tensor<1280x1x1xf32>
    %547 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%546, %expanded_244 : tensor<1x1280x7x7xf32>, tensor<1280x1x1xf32>) outs(%537 : tensor<1x1280x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.addf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x1280x7x7xf32>
    %548 = linalg.generic {indexing_maps = [#map1, #map4, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%547, %12 : tensor<1x1280x7x7xf32>, tensor<f32>) outs(%537 : tensor<1x1280x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.cmpf ogt, %in, %in_246 : f32
      %561 = arith.select %560, %in, %in_246 : f32
      linalg.yield %561 : f32
    } -> tensor<1x1280x7x7xf32>
    %549 = linalg.generic {indexing_maps = [#map4, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%14, %548 : tensor<f32>, tensor<1x1280x7x7xf32>) outs(%537 : tensor<1x1280x7x7xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.cmpf olt, %in, %in_246 : f32
      %561 = arith.select %560, %in, %in_246 : f32
      linalg.yield %561 : f32
    } -> tensor<1x1280x7x7xf32>
    %550 = tensor.empty() : tensor<1x1280x1x1xf32>
    %551 = linalg.fill ins(%cst_0 : f32) outs(%550 : tensor<1x1280x1x1xf32>) -> tensor<1x1280x1x1xf32>
    %552 = tensor.empty() : tensor<7x7xf32>
    %553 = linalg.pooling_nchw_sum {dilations = dense<1> : vector<2xi64>, strides = dense<7> : vector<2xi64>} ins(%549, %552 : tensor<1x1280x7x7xf32>, tensor<7x7xf32>) outs(%551 : tensor<1x1280x1x1xf32>) -> tensor<1x1280x1x1xf32>
    %554 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%553 : tensor<1x1280x1x1xf32>) outs(%550 : tensor<1x1280x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %560 = arith.divf %in, %cst_3 : f32
      linalg.yield %560 : f32
    } -> tensor<1x1280x1x1xf32>
    %collapsed_245 = tensor.collapse_shape %554 [[0], [1, 2, 3]] : tensor<1x1280x1x1xf32> into tensor<1x1280xf32>
    %555 = tensor.empty() : tensor<1280x1000xf32>
    %transposed = linalg.transpose ins(%arg313 : tensor<1000x1280xf32>) outs(%555 : tensor<1280x1000xf32>) permutation = [1, 0] 
    %556 = tensor.empty() : tensor<1x1000xf32>
    %557 = linalg.fill ins(%cst_0 : f32) outs(%556 : tensor<1x1000xf32>) -> tensor<1x1000xf32>
    %558 = linalg.matmul ins(%collapsed_245, %transposed : tensor<1x1280xf32>, tensor<1280x1000xf32>) outs(%557 : tensor<1x1000xf32>) -> tensor<1x1000xf32>
    %559 = linalg.generic {indexing_maps = [#map5, #map6, #map5], iterator_types = ["parallel", "parallel"]} ins(%558, %arg314 : tensor<1x1000xf32>, tensor<1000xf32>) outs(%556 : tensor<1x1000xf32>) {
    ^bb0(%in: f32, %in_246: f32, %out: f32):
      %560 = arith.addf %in, %in_246 : f32
      linalg.yield %560 : f32
    } -> tensor<1x1000xf32>
    return %559 : tensor<1x1000xf32>
  }
}
