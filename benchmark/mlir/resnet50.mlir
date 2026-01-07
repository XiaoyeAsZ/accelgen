#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d1, 0, 0)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1) -> (d1)>
module {
  func.func @main(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<i64>, %arg3: tensor<64xf32>, %arg4: tensor<64xf32>, %arg5: tensor<i64>, %arg6: tensor<64xf32>, %arg7: tensor<64xf32>, %arg8: tensor<i64>, %arg9: tensor<256xf32>, %arg10: tensor<256xf32>, %arg11: tensor<i64>, %arg12: tensor<256xf32>, %arg13: tensor<256xf32>, %arg14: tensor<i64>, %arg15: tensor<64xf32>, %arg16: tensor<64xf32>, %arg17: tensor<i64>, %arg18: tensor<64xf32>, %arg19: tensor<64xf32>, %arg20: tensor<i64>, %arg21: tensor<256xf32>, %arg22: tensor<256xf32>, %arg23: tensor<i64>, %arg24: tensor<64xf32>, %arg25: tensor<64xf32>, %arg26: tensor<i64>, %arg27: tensor<64xf32>, %arg28: tensor<64xf32>, %arg29: tensor<i64>, %arg30: tensor<256xf32>, %arg31: tensor<256xf32>, %arg32: tensor<i64>, %arg33: tensor<128xf32>, %arg34: tensor<128xf32>, %arg35: tensor<i64>, %arg36: tensor<128xf32>, %arg37: tensor<128xf32>, %arg38: tensor<i64>, %arg39: tensor<512xf32>, %arg40: tensor<512xf32>, %arg41: tensor<i64>, %arg42: tensor<512xf32>, %arg43: tensor<512xf32>, %arg44: tensor<i64>, %arg45: tensor<128xf32>, %arg46: tensor<128xf32>, %arg47: tensor<i64>, %arg48: tensor<128xf32>, %arg49: tensor<128xf32>, %arg50: tensor<i64>, %arg51: tensor<512xf32>, %arg52: tensor<512xf32>, %arg53: tensor<i64>, %arg54: tensor<128xf32>, %arg55: tensor<128xf32>, %arg56: tensor<i64>, %arg57: tensor<128xf32>, %arg58: tensor<128xf32>, %arg59: tensor<i64>, %arg60: tensor<512xf32>, %arg61: tensor<512xf32>, %arg62: tensor<i64>, %arg63: tensor<128xf32>, %arg64: tensor<128xf32>, %arg65: tensor<i64>, %arg66: tensor<128xf32>, %arg67: tensor<128xf32>, %arg68: tensor<i64>, %arg69: tensor<512xf32>, %arg70: tensor<512xf32>, %arg71: tensor<i64>, %arg72: tensor<256xf32>, %arg73: tensor<256xf32>, %arg74: tensor<i64>, %arg75: tensor<256xf32>, %arg76: tensor<256xf32>, %arg77: tensor<i64>, %arg78: tensor<1024xf32>, %arg79: tensor<1024xf32>, %arg80: tensor<i64>, %arg81: tensor<1024xf32>, %arg82: tensor<1024xf32>, %arg83: tensor<i64>, %arg84: tensor<256xf32>, %arg85: tensor<256xf32>, %arg86: tensor<i64>, %arg87: tensor<256xf32>, %arg88: tensor<256xf32>, %arg89: tensor<i64>, %arg90: tensor<1024xf32>, %arg91: tensor<1024xf32>, %arg92: tensor<i64>, %arg93: tensor<256xf32>, %arg94: tensor<256xf32>, %arg95: tensor<i64>, %arg96: tensor<256xf32>, %arg97: tensor<256xf32>, %arg98: tensor<i64>, %arg99: tensor<1024xf32>, %arg100: tensor<1024xf32>, %arg101: tensor<i64>, %arg102: tensor<256xf32>, %arg103: tensor<256xf32>, %arg104: tensor<i64>, %arg105: tensor<256xf32>, %arg106: tensor<256xf32>, %arg107: tensor<i64>, %arg108: tensor<1024xf32>, %arg109: tensor<1024xf32>, %arg110: tensor<i64>, %arg111: tensor<256xf32>, %arg112: tensor<256xf32>, %arg113: tensor<i64>, %arg114: tensor<256xf32>, %arg115: tensor<256xf32>, %arg116: tensor<i64>, %arg117: tensor<1024xf32>, %arg118: tensor<1024xf32>, %arg119: tensor<i64>, %arg120: tensor<256xf32>, %arg121: tensor<256xf32>, %arg122: tensor<i64>, %arg123: tensor<256xf32>, %arg124: tensor<256xf32>, %arg125: tensor<i64>, %arg126: tensor<1024xf32>, %arg127: tensor<1024xf32>, %arg128: tensor<i64>, %arg129: tensor<512xf32>, %arg130: tensor<512xf32>, %arg131: tensor<i64>, %arg132: tensor<512xf32>, %arg133: tensor<512xf32>, %arg134: tensor<i64>, %arg135: tensor<2048xf32>, %arg136: tensor<2048xf32>, %arg137: tensor<i64>, %arg138: tensor<2048xf32>, %arg139: tensor<2048xf32>, %arg140: tensor<i64>, %arg141: tensor<512xf32>, %arg142: tensor<512xf32>, %arg143: tensor<i64>, %arg144: tensor<512xf32>, %arg145: tensor<512xf32>, %arg146: tensor<i64>, %arg147: tensor<2048xf32>, %arg148: tensor<2048xf32>, %arg149: tensor<i64>, %arg150: tensor<512xf32>, %arg151: tensor<512xf32>, %arg152: tensor<i64>, %arg153: tensor<512xf32>, %arg154: tensor<512xf32>, %arg155: tensor<i64>, %arg156: tensor<2048xf32>, %arg157: tensor<2048xf32>, %arg158: tensor<i64>, %arg159: tensor<1x3x224x224xf32>, %arg160: tensor<64x3x7x7xf32>, %arg161: tensor<64xf32>, %arg162: tensor<64xf32>, %arg163: tensor<64x64x1x1xf32>, %arg164: tensor<64xf32>, %arg165: tensor<64xf32>, %arg166: tensor<64x64x3x3xf32>, %arg167: tensor<64xf32>, %arg168: tensor<64xf32>, %arg169: tensor<256x64x1x1xf32>, %arg170: tensor<256xf32>, %arg171: tensor<256xf32>, %arg172: tensor<256x64x1x1xf32>, %arg173: tensor<256xf32>, %arg174: tensor<256xf32>, %arg175: tensor<64x256x1x1xf32>, %arg176: tensor<64xf32>, %arg177: tensor<64xf32>, %arg178: tensor<64x64x3x3xf32>, %arg179: tensor<64xf32>, %arg180: tensor<64xf32>, %arg181: tensor<256x64x1x1xf32>, %arg182: tensor<256xf32>, %arg183: tensor<256xf32>, %arg184: tensor<64x256x1x1xf32>, %arg185: tensor<64xf32>, %arg186: tensor<64xf32>, %arg187: tensor<64x64x3x3xf32>, %arg188: tensor<64xf32>, %arg189: tensor<64xf32>, %arg190: tensor<256x64x1x1xf32>, %arg191: tensor<256xf32>, %arg192: tensor<256xf32>, %arg193: tensor<128x256x1x1xf32>, %arg194: tensor<128xf32>, %arg195: tensor<128xf32>, %arg196: tensor<128x128x3x3xf32>, %arg197: tensor<128xf32>, %arg198: tensor<128xf32>, %arg199: tensor<512x128x1x1xf32>, %arg200: tensor<512xf32>, %arg201: tensor<512xf32>, %arg202: tensor<512x256x1x1xf32>, %arg203: tensor<512xf32>, %arg204: tensor<512xf32>, %arg205: tensor<128x512x1x1xf32>, %arg206: tensor<128xf32>, %arg207: tensor<128xf32>, %arg208: tensor<128x128x3x3xf32>, %arg209: tensor<128xf32>, %arg210: tensor<128xf32>, %arg211: tensor<512x128x1x1xf32>, %arg212: tensor<512xf32>, %arg213: tensor<512xf32>, %arg214: tensor<128x512x1x1xf32>, %arg215: tensor<128xf32>, %arg216: tensor<128xf32>, %arg217: tensor<128x128x3x3xf32>, %arg218: tensor<128xf32>, %arg219: tensor<128xf32>, %arg220: tensor<512x128x1x1xf32>, %arg221: tensor<512xf32>, %arg222: tensor<512xf32>, %arg223: tensor<128x512x1x1xf32>, %arg224: tensor<128xf32>, %arg225: tensor<128xf32>, %arg226: tensor<128x128x3x3xf32>, %arg227: tensor<128xf32>, %arg228: tensor<128xf32>, %arg229: tensor<512x128x1x1xf32>, %arg230: tensor<512xf32>, %arg231: tensor<512xf32>, %arg232: tensor<256x512x1x1xf32>, %arg233: tensor<256xf32>, %arg234: tensor<256xf32>, %arg235: tensor<256x256x3x3xf32>, %arg236: tensor<256xf32>, %arg237: tensor<256xf32>, %arg238: tensor<1024x256x1x1xf32>, %arg239: tensor<1024xf32>, %arg240: tensor<1024xf32>, %arg241: tensor<1024x512x1x1xf32>, %arg242: tensor<1024xf32>, %arg243: tensor<1024xf32>, %arg244: tensor<256x1024x1x1xf32>, %arg245: tensor<256xf32>, %arg246: tensor<256xf32>, %arg247: tensor<256x256x3x3xf32>, %arg248: tensor<256xf32>, %arg249: tensor<256xf32>, %arg250: tensor<1024x256x1x1xf32>, %arg251: tensor<1024xf32>, %arg252: tensor<1024xf32>, %arg253: tensor<256x1024x1x1xf32>, %arg254: tensor<256xf32>, %arg255: tensor<256xf32>, %arg256: tensor<256x256x3x3xf32>, %arg257: tensor<256xf32>, %arg258: tensor<256xf32>, %arg259: tensor<1024x256x1x1xf32>, %arg260: tensor<1024xf32>, %arg261: tensor<1024xf32>, %arg262: tensor<256x1024x1x1xf32>, %arg263: tensor<256xf32>, %arg264: tensor<256xf32>, %arg265: tensor<256x256x3x3xf32>, %arg266: tensor<256xf32>, %arg267: tensor<256xf32>, %arg268: tensor<1024x256x1x1xf32>, %arg269: tensor<1024xf32>, %arg270: tensor<1024xf32>, %arg271: tensor<256x1024x1x1xf32>, %arg272: tensor<256xf32>, %arg273: tensor<256xf32>, %arg274: tensor<256x256x3x3xf32>, %arg275: tensor<256xf32>, %arg276: tensor<256xf32>, %arg277: tensor<1024x256x1x1xf32>, %arg278: tensor<1024xf32>, %arg279: tensor<1024xf32>, %arg280: tensor<256x1024x1x1xf32>, %arg281: tensor<256xf32>, %arg282: tensor<256xf32>, %arg283: tensor<256x256x3x3xf32>, %arg284: tensor<256xf32>, %arg285: tensor<256xf32>, %arg286: tensor<1024x256x1x1xf32>, %arg287: tensor<1024xf32>, %arg288: tensor<1024xf32>, %arg289: tensor<512x1024x1x1xf32>, %arg290: tensor<512xf32>, %arg291: tensor<512xf32>, %arg292: tensor<512x512x3x3xf32>, %arg293: tensor<512xf32>, %arg294: tensor<512xf32>, %arg295: tensor<2048x512x1x1xf32>, %arg296: tensor<2048xf32>, %arg297: tensor<2048xf32>, %arg298: tensor<2048x1024x1x1xf32>, %arg299: tensor<2048xf32>, %arg300: tensor<2048xf32>, %arg301: tensor<512x2048x1x1xf32>, %arg302: tensor<512xf32>, %arg303: tensor<512xf32>, %arg304: tensor<512x512x3x3xf32>, %arg305: tensor<512xf32>, %arg306: tensor<512xf32>, %arg307: tensor<2048x512x1x1xf32>, %arg308: tensor<2048xf32>, %arg309: tensor<2048xf32>, %arg310: tensor<512x2048x1x1xf32>, %arg311: tensor<512xf32>, %arg312: tensor<512xf32>, %arg313: tensor<512x512x3x3xf32>, %arg314: tensor<512xf32>, %arg315: tensor<512xf32>, %arg316: tensor<2048x512x1x1xf32>, %arg317: tensor<2048xf32>, %arg318: tensor<2048xf32>, %arg319: tensor<1000x2048xf32>, %arg320: tensor<1000xf32>) -> tensor<1x1000xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %cst_1 = arith.constant 0xFF800000 : f32
    %cst_2 = arith.constant 1.000000e-05 : f64
    %cst_3 = arith.constant 4.900000e+01 : f32
    %padded = tensor.pad %arg159 low[0, 0, 3, 3] high[0, 0, 3, 3] {
    ^bb0(%arg321: index, %arg322: index, %arg323: index, %arg324: index):
      tensor.yield %cst : f32
    } : tensor<1x3x224x224xf32> to tensor<1x3x230x230xf32>
    %0 = tensor.empty() : tensor<1x64x112x112xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
    %2 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%padded, %arg160 : tensor<1x3x230x230xf32>, tensor<64x3x7x7xf32>) outs(%1 : tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
    %3 = tensor.empty() : tensor<64xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg1 : tensor<64xf32>) outs(%3 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.truncf %cst_2 : f64 to f32
      %533 = arith.addf %in, %532 : f32
      linalg.yield %533 : f32
    } -> tensor<64xf32>
    %5 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%4 : tensor<64xf32>) outs(%3 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = math.sqrt %in : f32
      linalg.yield %532 : f32
    } -> tensor<64xf32>
    %6 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%5 : tensor<64xf32>) outs(%3 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf one, %in, %cst : f32
      cf.assert %532, "unimplemented: tensor with zero element"
      %533 = arith.divf %cst_0, %in : f32
      linalg.yield %533 : f32
    } -> tensor<64xf32>
    %expanded = tensor.expand_shape %arg0 [[0, 1, 2]] output_shape [64, 1, 1] : tensor<64xf32> into tensor<64x1x1xf32>
    %expanded_4 = tensor.expand_shape %6 [[0, 1, 2]] output_shape [64, 1, 1] : tensor<64xf32> into tensor<64x1x1xf32>
    %7 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2, %expanded : tensor<1x64x112x112xf32>, tensor<64x1x1xf32>) outs(%0 : tensor<1x64x112x112xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.subf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x64x112x112xf32>
    %8 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%7, %expanded_4 : tensor<1x64x112x112xf32>, tensor<64x1x1xf32>) outs(%0 : tensor<1x64x112x112xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x64x112x112xf32>
    %expanded_5 = tensor.expand_shape %arg161 [[0, 1, 2]] output_shape [64, 1, 1] : tensor<64xf32> into tensor<64x1x1xf32>
    %9 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%8, %expanded_5 : tensor<1x64x112x112xf32>, tensor<64x1x1xf32>) outs(%0 : tensor<1x64x112x112xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x64x112x112xf32>
    %expanded_6 = tensor.expand_shape %arg162 [[0, 1, 2]] output_shape [64, 1, 1] : tensor<64xf32> into tensor<64x1x1xf32>
    %10 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%9, %expanded_6 : tensor<1x64x112x112xf32>, tensor<64x1x1xf32>) outs(%0 : tensor<1x64x112x112xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.addf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x64x112x112xf32>
    %11 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%10 : tensor<1x64x112x112xf32>) outs(%0 : tensor<1x64x112x112xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf ugt, %in, %cst : f32
      %533 = arith.select %532, %in, %cst : f32
      linalg.yield %533 : f32
    } -> tensor<1x64x112x112xf32>
    %padded_7 = tensor.pad %11 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg321: index, %arg322: index, %arg323: index, %arg324: index):
      tensor.yield %cst_1 : f32
    } : tensor<1x64x112x112xf32> to tensor<1x64x114x114xf32>
    %12 = tensor.empty() : tensor<1x64x56x56xf32>
    %13 = linalg.fill ins(%cst_1 : f32) outs(%12 : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %14 = tensor.empty() : tensor<3x3xf32>
    %15 = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%padded_7, %14 : tensor<1x64x114x114xf32>, tensor<3x3xf32>) outs(%13 : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %16 = linalg.fill ins(%cst : f32) outs(%12 : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %17 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%15, %arg163 : tensor<1x64x56x56xf32>, tensor<64x64x1x1xf32>) outs(%16 : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %18 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg4 : tensor<64xf32>) outs(%3 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.truncf %cst_2 : f64 to f32
      %533 = arith.addf %in, %532 : f32
      linalg.yield %533 : f32
    } -> tensor<64xf32>
    %19 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%18 : tensor<64xf32>) outs(%3 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = math.sqrt %in : f32
      linalg.yield %532 : f32
    } -> tensor<64xf32>
    %20 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%19 : tensor<64xf32>) outs(%3 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf one, %in, %cst : f32
      cf.assert %532, "unimplemented: tensor with zero element"
      %533 = arith.divf %cst_0, %in : f32
      linalg.yield %533 : f32
    } -> tensor<64xf32>
    %expanded_8 = tensor.expand_shape %arg3 [[0, 1, 2]] output_shape [64, 1, 1] : tensor<64xf32> into tensor<64x1x1xf32>
    %expanded_9 = tensor.expand_shape %20 [[0, 1, 2]] output_shape [64, 1, 1] : tensor<64xf32> into tensor<64x1x1xf32>
    %21 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%17, %expanded_8 : tensor<1x64x56x56xf32>, tensor<64x1x1xf32>) outs(%12 : tensor<1x64x56x56xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.subf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x64x56x56xf32>
    %22 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%21, %expanded_9 : tensor<1x64x56x56xf32>, tensor<64x1x1xf32>) outs(%12 : tensor<1x64x56x56xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x64x56x56xf32>
    %expanded_10 = tensor.expand_shape %arg164 [[0, 1, 2]] output_shape [64, 1, 1] : tensor<64xf32> into tensor<64x1x1xf32>
    %23 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%22, %expanded_10 : tensor<1x64x56x56xf32>, tensor<64x1x1xf32>) outs(%12 : tensor<1x64x56x56xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x64x56x56xf32>
    %expanded_11 = tensor.expand_shape %arg165 [[0, 1, 2]] output_shape [64, 1, 1] : tensor<64xf32> into tensor<64x1x1xf32>
    %24 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%23, %expanded_11 : tensor<1x64x56x56xf32>, tensor<64x1x1xf32>) outs(%12 : tensor<1x64x56x56xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.addf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x64x56x56xf32>
    %25 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%24 : tensor<1x64x56x56xf32>) outs(%12 : tensor<1x64x56x56xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf ugt, %in, %cst : f32
      %533 = arith.select %532, %in, %cst : f32
      linalg.yield %533 : f32
    } -> tensor<1x64x56x56xf32>
    %padded_12 = tensor.pad %25 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg321: index, %arg322: index, %arg323: index, %arg324: index):
      tensor.yield %cst : f32
    } : tensor<1x64x56x56xf32> to tensor<1x64x58x58xf32>
    %26 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_12, %arg166 : tensor<1x64x58x58xf32>, tensor<64x64x3x3xf32>) outs(%16 : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %27 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg7 : tensor<64xf32>) outs(%3 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.truncf %cst_2 : f64 to f32
      %533 = arith.addf %in, %532 : f32
      linalg.yield %533 : f32
    } -> tensor<64xf32>
    %28 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%27 : tensor<64xf32>) outs(%3 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = math.sqrt %in : f32
      linalg.yield %532 : f32
    } -> tensor<64xf32>
    %29 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%28 : tensor<64xf32>) outs(%3 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf one, %in, %cst : f32
      cf.assert %532, "unimplemented: tensor with zero element"
      %533 = arith.divf %cst_0, %in : f32
      linalg.yield %533 : f32
    } -> tensor<64xf32>
    %expanded_13 = tensor.expand_shape %arg6 [[0, 1, 2]] output_shape [64, 1, 1] : tensor<64xf32> into tensor<64x1x1xf32>
    %expanded_14 = tensor.expand_shape %29 [[0, 1, 2]] output_shape [64, 1, 1] : tensor<64xf32> into tensor<64x1x1xf32>
    %30 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%26, %expanded_13 : tensor<1x64x56x56xf32>, tensor<64x1x1xf32>) outs(%12 : tensor<1x64x56x56xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.subf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x64x56x56xf32>
    %31 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%30, %expanded_14 : tensor<1x64x56x56xf32>, tensor<64x1x1xf32>) outs(%12 : tensor<1x64x56x56xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x64x56x56xf32>
    %expanded_15 = tensor.expand_shape %arg167 [[0, 1, 2]] output_shape [64, 1, 1] : tensor<64xf32> into tensor<64x1x1xf32>
    %32 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%31, %expanded_15 : tensor<1x64x56x56xf32>, tensor<64x1x1xf32>) outs(%12 : tensor<1x64x56x56xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x64x56x56xf32>
    %expanded_16 = tensor.expand_shape %arg168 [[0, 1, 2]] output_shape [64, 1, 1] : tensor<64xf32> into tensor<64x1x1xf32>
    %33 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%32, %expanded_16 : tensor<1x64x56x56xf32>, tensor<64x1x1xf32>) outs(%12 : tensor<1x64x56x56xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.addf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x64x56x56xf32>
    %34 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%33 : tensor<1x64x56x56xf32>) outs(%12 : tensor<1x64x56x56xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf ugt, %in, %cst : f32
      %533 = arith.select %532, %in, %cst : f32
      linalg.yield %533 : f32
    } -> tensor<1x64x56x56xf32>
    %35 = tensor.empty() : tensor<1x256x56x56xf32>
    %36 = linalg.fill ins(%cst : f32) outs(%35 : tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32>
    %37 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%34, %arg169 : tensor<1x64x56x56xf32>, tensor<256x64x1x1xf32>) outs(%36 : tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32>
    %38 = tensor.empty() : tensor<256xf32>
    %39 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg10 : tensor<256xf32>) outs(%38 : tensor<256xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.truncf %cst_2 : f64 to f32
      %533 = arith.addf %in, %532 : f32
      linalg.yield %533 : f32
    } -> tensor<256xf32>
    %40 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%39 : tensor<256xf32>) outs(%38 : tensor<256xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = math.sqrt %in : f32
      linalg.yield %532 : f32
    } -> tensor<256xf32>
    %41 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%40 : tensor<256xf32>) outs(%38 : tensor<256xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf one, %in, %cst : f32
      cf.assert %532, "unimplemented: tensor with zero element"
      %533 = arith.divf %cst_0, %in : f32
      linalg.yield %533 : f32
    } -> tensor<256xf32>
    %expanded_17 = tensor.expand_shape %arg9 [[0, 1, 2]] output_shape [256, 1, 1] : tensor<256xf32> into tensor<256x1x1xf32>
    %expanded_18 = tensor.expand_shape %41 [[0, 1, 2]] output_shape [256, 1, 1] : tensor<256xf32> into tensor<256x1x1xf32>
    %42 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%37, %expanded_17 : tensor<1x256x56x56xf32>, tensor<256x1x1xf32>) outs(%35 : tensor<1x256x56x56xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.subf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x256x56x56xf32>
    %43 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%42, %expanded_18 : tensor<1x256x56x56xf32>, tensor<256x1x1xf32>) outs(%35 : tensor<1x256x56x56xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x256x56x56xf32>
    %expanded_19 = tensor.expand_shape %arg170 [[0, 1, 2]] output_shape [256, 1, 1] : tensor<256xf32> into tensor<256x1x1xf32>
    %44 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%43, %expanded_19 : tensor<1x256x56x56xf32>, tensor<256x1x1xf32>) outs(%35 : tensor<1x256x56x56xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x256x56x56xf32>
    %expanded_20 = tensor.expand_shape %arg171 [[0, 1, 2]] output_shape [256, 1, 1] : tensor<256xf32> into tensor<256x1x1xf32>
    %45 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%44, %expanded_20 : tensor<1x256x56x56xf32>, tensor<256x1x1xf32>) outs(%35 : tensor<1x256x56x56xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.addf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x256x56x56xf32>
    %46 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%15, %arg172 : tensor<1x64x56x56xf32>, tensor<256x64x1x1xf32>) outs(%36 : tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32>
    %47 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg13 : tensor<256xf32>) outs(%38 : tensor<256xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.truncf %cst_2 : f64 to f32
      %533 = arith.addf %in, %532 : f32
      linalg.yield %533 : f32
    } -> tensor<256xf32>
    %48 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%47 : tensor<256xf32>) outs(%38 : tensor<256xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = math.sqrt %in : f32
      linalg.yield %532 : f32
    } -> tensor<256xf32>
    %49 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%48 : tensor<256xf32>) outs(%38 : tensor<256xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf one, %in, %cst : f32
      cf.assert %532, "unimplemented: tensor with zero element"
      %533 = arith.divf %cst_0, %in : f32
      linalg.yield %533 : f32
    } -> tensor<256xf32>
    %expanded_21 = tensor.expand_shape %arg12 [[0, 1, 2]] output_shape [256, 1, 1] : tensor<256xf32> into tensor<256x1x1xf32>
    %expanded_22 = tensor.expand_shape %49 [[0, 1, 2]] output_shape [256, 1, 1] : tensor<256xf32> into tensor<256x1x1xf32>
    %50 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%46, %expanded_21 : tensor<1x256x56x56xf32>, tensor<256x1x1xf32>) outs(%35 : tensor<1x256x56x56xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.subf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x256x56x56xf32>
    %51 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%50, %expanded_22 : tensor<1x256x56x56xf32>, tensor<256x1x1xf32>) outs(%35 : tensor<1x256x56x56xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x256x56x56xf32>
    %expanded_23 = tensor.expand_shape %arg173 [[0, 1, 2]] output_shape [256, 1, 1] : tensor<256xf32> into tensor<256x1x1xf32>
    %52 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%51, %expanded_23 : tensor<1x256x56x56xf32>, tensor<256x1x1xf32>) outs(%35 : tensor<1x256x56x56xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x256x56x56xf32>
    %expanded_24 = tensor.expand_shape %arg174 [[0, 1, 2]] output_shape [256, 1, 1] : tensor<256xf32> into tensor<256x1x1xf32>
    %53 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%52, %expanded_24 : tensor<1x256x56x56xf32>, tensor<256x1x1xf32>) outs(%35 : tensor<1x256x56x56xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.addf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x256x56x56xf32>
    %54 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%45, %53 : tensor<1x256x56x56xf32>, tensor<1x256x56x56xf32>) outs(%35 : tensor<1x256x56x56xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.addf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x256x56x56xf32>
    %55 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%54 : tensor<1x256x56x56xf32>) outs(%35 : tensor<1x256x56x56xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf ugt, %in, %cst : f32
      %533 = arith.select %532, %in, %cst : f32
      linalg.yield %533 : f32
    } -> tensor<1x256x56x56xf32>
    %56 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%55, %arg175 : tensor<1x256x56x56xf32>, tensor<64x256x1x1xf32>) outs(%16 : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %57 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg16 : tensor<64xf32>) outs(%3 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.truncf %cst_2 : f64 to f32
      %533 = arith.addf %in, %532 : f32
      linalg.yield %533 : f32
    } -> tensor<64xf32>
    %58 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%57 : tensor<64xf32>) outs(%3 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = math.sqrt %in : f32
      linalg.yield %532 : f32
    } -> tensor<64xf32>
    %59 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%58 : tensor<64xf32>) outs(%3 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf one, %in, %cst : f32
      cf.assert %532, "unimplemented: tensor with zero element"
      %533 = arith.divf %cst_0, %in : f32
      linalg.yield %533 : f32
    } -> tensor<64xf32>
    %expanded_25 = tensor.expand_shape %arg15 [[0, 1, 2]] output_shape [64, 1, 1] : tensor<64xf32> into tensor<64x1x1xf32>
    %expanded_26 = tensor.expand_shape %59 [[0, 1, 2]] output_shape [64, 1, 1] : tensor<64xf32> into tensor<64x1x1xf32>
    %60 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%56, %expanded_25 : tensor<1x64x56x56xf32>, tensor<64x1x1xf32>) outs(%12 : tensor<1x64x56x56xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.subf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x64x56x56xf32>
    %61 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%60, %expanded_26 : tensor<1x64x56x56xf32>, tensor<64x1x1xf32>) outs(%12 : tensor<1x64x56x56xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x64x56x56xf32>
    %expanded_27 = tensor.expand_shape %arg176 [[0, 1, 2]] output_shape [64, 1, 1] : tensor<64xf32> into tensor<64x1x1xf32>
    %62 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%61, %expanded_27 : tensor<1x64x56x56xf32>, tensor<64x1x1xf32>) outs(%12 : tensor<1x64x56x56xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x64x56x56xf32>
    %expanded_28 = tensor.expand_shape %arg177 [[0, 1, 2]] output_shape [64, 1, 1] : tensor<64xf32> into tensor<64x1x1xf32>
    %63 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%62, %expanded_28 : tensor<1x64x56x56xf32>, tensor<64x1x1xf32>) outs(%12 : tensor<1x64x56x56xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.addf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x64x56x56xf32>
    %64 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%63 : tensor<1x64x56x56xf32>) outs(%12 : tensor<1x64x56x56xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf ugt, %in, %cst : f32
      %533 = arith.select %532, %in, %cst : f32
      linalg.yield %533 : f32
    } -> tensor<1x64x56x56xf32>
    %padded_29 = tensor.pad %64 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg321: index, %arg322: index, %arg323: index, %arg324: index):
      tensor.yield %cst : f32
    } : tensor<1x64x56x56xf32> to tensor<1x64x58x58xf32>
    %65 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_29, %arg178 : tensor<1x64x58x58xf32>, tensor<64x64x3x3xf32>) outs(%16 : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %66 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg19 : tensor<64xf32>) outs(%3 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.truncf %cst_2 : f64 to f32
      %533 = arith.addf %in, %532 : f32
      linalg.yield %533 : f32
    } -> tensor<64xf32>
    %67 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%66 : tensor<64xf32>) outs(%3 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = math.sqrt %in : f32
      linalg.yield %532 : f32
    } -> tensor<64xf32>
    %68 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%67 : tensor<64xf32>) outs(%3 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf one, %in, %cst : f32
      cf.assert %532, "unimplemented: tensor with zero element"
      %533 = arith.divf %cst_0, %in : f32
      linalg.yield %533 : f32
    } -> tensor<64xf32>
    %expanded_30 = tensor.expand_shape %arg18 [[0, 1, 2]] output_shape [64, 1, 1] : tensor<64xf32> into tensor<64x1x1xf32>
    %expanded_31 = tensor.expand_shape %68 [[0, 1, 2]] output_shape [64, 1, 1] : tensor<64xf32> into tensor<64x1x1xf32>
    %69 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%65, %expanded_30 : tensor<1x64x56x56xf32>, tensor<64x1x1xf32>) outs(%12 : tensor<1x64x56x56xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.subf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x64x56x56xf32>
    %70 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%69, %expanded_31 : tensor<1x64x56x56xf32>, tensor<64x1x1xf32>) outs(%12 : tensor<1x64x56x56xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x64x56x56xf32>
    %expanded_32 = tensor.expand_shape %arg179 [[0, 1, 2]] output_shape [64, 1, 1] : tensor<64xf32> into tensor<64x1x1xf32>
    %71 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%70, %expanded_32 : tensor<1x64x56x56xf32>, tensor<64x1x1xf32>) outs(%12 : tensor<1x64x56x56xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x64x56x56xf32>
    %expanded_33 = tensor.expand_shape %arg180 [[0, 1, 2]] output_shape [64, 1, 1] : tensor<64xf32> into tensor<64x1x1xf32>
    %72 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%71, %expanded_33 : tensor<1x64x56x56xf32>, tensor<64x1x1xf32>) outs(%12 : tensor<1x64x56x56xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.addf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x64x56x56xf32>
    %73 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%72 : tensor<1x64x56x56xf32>) outs(%12 : tensor<1x64x56x56xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf ugt, %in, %cst : f32
      %533 = arith.select %532, %in, %cst : f32
      linalg.yield %533 : f32
    } -> tensor<1x64x56x56xf32>
    %74 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%73, %arg181 : tensor<1x64x56x56xf32>, tensor<256x64x1x1xf32>) outs(%36 : tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32>
    %75 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg22 : tensor<256xf32>) outs(%38 : tensor<256xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.truncf %cst_2 : f64 to f32
      %533 = arith.addf %in, %532 : f32
      linalg.yield %533 : f32
    } -> tensor<256xf32>
    %76 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%75 : tensor<256xf32>) outs(%38 : tensor<256xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = math.sqrt %in : f32
      linalg.yield %532 : f32
    } -> tensor<256xf32>
    %77 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%76 : tensor<256xf32>) outs(%38 : tensor<256xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf one, %in, %cst : f32
      cf.assert %532, "unimplemented: tensor with zero element"
      %533 = arith.divf %cst_0, %in : f32
      linalg.yield %533 : f32
    } -> tensor<256xf32>
    %expanded_34 = tensor.expand_shape %arg21 [[0, 1, 2]] output_shape [256, 1, 1] : tensor<256xf32> into tensor<256x1x1xf32>
    %expanded_35 = tensor.expand_shape %77 [[0, 1, 2]] output_shape [256, 1, 1] : tensor<256xf32> into tensor<256x1x1xf32>
    %78 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%74, %expanded_34 : tensor<1x256x56x56xf32>, tensor<256x1x1xf32>) outs(%35 : tensor<1x256x56x56xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.subf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x256x56x56xf32>
    %79 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%78, %expanded_35 : tensor<1x256x56x56xf32>, tensor<256x1x1xf32>) outs(%35 : tensor<1x256x56x56xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x256x56x56xf32>
    %expanded_36 = tensor.expand_shape %arg182 [[0, 1, 2]] output_shape [256, 1, 1] : tensor<256xf32> into tensor<256x1x1xf32>
    %80 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%79, %expanded_36 : tensor<1x256x56x56xf32>, tensor<256x1x1xf32>) outs(%35 : tensor<1x256x56x56xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x256x56x56xf32>
    %expanded_37 = tensor.expand_shape %arg183 [[0, 1, 2]] output_shape [256, 1, 1] : tensor<256xf32> into tensor<256x1x1xf32>
    %81 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%80, %expanded_37 : tensor<1x256x56x56xf32>, tensor<256x1x1xf32>) outs(%35 : tensor<1x256x56x56xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.addf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x256x56x56xf32>
    %82 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%81, %55 : tensor<1x256x56x56xf32>, tensor<1x256x56x56xf32>) outs(%35 : tensor<1x256x56x56xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.addf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x256x56x56xf32>
    %83 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%82 : tensor<1x256x56x56xf32>) outs(%35 : tensor<1x256x56x56xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf ugt, %in, %cst : f32
      %533 = arith.select %532, %in, %cst : f32
      linalg.yield %533 : f32
    } -> tensor<1x256x56x56xf32>
    %84 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%83, %arg184 : tensor<1x256x56x56xf32>, tensor<64x256x1x1xf32>) outs(%16 : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %85 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg25 : tensor<64xf32>) outs(%3 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.truncf %cst_2 : f64 to f32
      %533 = arith.addf %in, %532 : f32
      linalg.yield %533 : f32
    } -> tensor<64xf32>
    %86 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%85 : tensor<64xf32>) outs(%3 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = math.sqrt %in : f32
      linalg.yield %532 : f32
    } -> tensor<64xf32>
    %87 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%86 : tensor<64xf32>) outs(%3 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf one, %in, %cst : f32
      cf.assert %532, "unimplemented: tensor with zero element"
      %533 = arith.divf %cst_0, %in : f32
      linalg.yield %533 : f32
    } -> tensor<64xf32>
    %expanded_38 = tensor.expand_shape %arg24 [[0, 1, 2]] output_shape [64, 1, 1] : tensor<64xf32> into tensor<64x1x1xf32>
    %expanded_39 = tensor.expand_shape %87 [[0, 1, 2]] output_shape [64, 1, 1] : tensor<64xf32> into tensor<64x1x1xf32>
    %88 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%84, %expanded_38 : tensor<1x64x56x56xf32>, tensor<64x1x1xf32>) outs(%12 : tensor<1x64x56x56xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.subf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x64x56x56xf32>
    %89 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%88, %expanded_39 : tensor<1x64x56x56xf32>, tensor<64x1x1xf32>) outs(%12 : tensor<1x64x56x56xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x64x56x56xf32>
    %expanded_40 = tensor.expand_shape %arg185 [[0, 1, 2]] output_shape [64, 1, 1] : tensor<64xf32> into tensor<64x1x1xf32>
    %90 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%89, %expanded_40 : tensor<1x64x56x56xf32>, tensor<64x1x1xf32>) outs(%12 : tensor<1x64x56x56xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x64x56x56xf32>
    %expanded_41 = tensor.expand_shape %arg186 [[0, 1, 2]] output_shape [64, 1, 1] : tensor<64xf32> into tensor<64x1x1xf32>
    %91 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%90, %expanded_41 : tensor<1x64x56x56xf32>, tensor<64x1x1xf32>) outs(%12 : tensor<1x64x56x56xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.addf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x64x56x56xf32>
    %92 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%91 : tensor<1x64x56x56xf32>) outs(%12 : tensor<1x64x56x56xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf ugt, %in, %cst : f32
      %533 = arith.select %532, %in, %cst : f32
      linalg.yield %533 : f32
    } -> tensor<1x64x56x56xf32>
    %padded_42 = tensor.pad %92 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg321: index, %arg322: index, %arg323: index, %arg324: index):
      tensor.yield %cst : f32
    } : tensor<1x64x56x56xf32> to tensor<1x64x58x58xf32>
    %93 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_42, %arg187 : tensor<1x64x58x58xf32>, tensor<64x64x3x3xf32>) outs(%16 : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %94 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg28 : tensor<64xf32>) outs(%3 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.truncf %cst_2 : f64 to f32
      %533 = arith.addf %in, %532 : f32
      linalg.yield %533 : f32
    } -> tensor<64xf32>
    %95 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%94 : tensor<64xf32>) outs(%3 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = math.sqrt %in : f32
      linalg.yield %532 : f32
    } -> tensor<64xf32>
    %96 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%95 : tensor<64xf32>) outs(%3 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf one, %in, %cst : f32
      cf.assert %532, "unimplemented: tensor with zero element"
      %533 = arith.divf %cst_0, %in : f32
      linalg.yield %533 : f32
    } -> tensor<64xf32>
    %expanded_43 = tensor.expand_shape %arg27 [[0, 1, 2]] output_shape [64, 1, 1] : tensor<64xf32> into tensor<64x1x1xf32>
    %expanded_44 = tensor.expand_shape %96 [[0, 1, 2]] output_shape [64, 1, 1] : tensor<64xf32> into tensor<64x1x1xf32>
    %97 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%93, %expanded_43 : tensor<1x64x56x56xf32>, tensor<64x1x1xf32>) outs(%12 : tensor<1x64x56x56xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.subf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x64x56x56xf32>
    %98 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%97, %expanded_44 : tensor<1x64x56x56xf32>, tensor<64x1x1xf32>) outs(%12 : tensor<1x64x56x56xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x64x56x56xf32>
    %expanded_45 = tensor.expand_shape %arg188 [[0, 1, 2]] output_shape [64, 1, 1] : tensor<64xf32> into tensor<64x1x1xf32>
    %99 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%98, %expanded_45 : tensor<1x64x56x56xf32>, tensor<64x1x1xf32>) outs(%12 : tensor<1x64x56x56xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x64x56x56xf32>
    %expanded_46 = tensor.expand_shape %arg189 [[0, 1, 2]] output_shape [64, 1, 1] : tensor<64xf32> into tensor<64x1x1xf32>
    %100 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%99, %expanded_46 : tensor<1x64x56x56xf32>, tensor<64x1x1xf32>) outs(%12 : tensor<1x64x56x56xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.addf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x64x56x56xf32>
    %101 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%100 : tensor<1x64x56x56xf32>) outs(%12 : tensor<1x64x56x56xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf ugt, %in, %cst : f32
      %533 = arith.select %532, %in, %cst : f32
      linalg.yield %533 : f32
    } -> tensor<1x64x56x56xf32>
    %102 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%101, %arg190 : tensor<1x64x56x56xf32>, tensor<256x64x1x1xf32>) outs(%36 : tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32>
    %103 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg31 : tensor<256xf32>) outs(%38 : tensor<256xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.truncf %cst_2 : f64 to f32
      %533 = arith.addf %in, %532 : f32
      linalg.yield %533 : f32
    } -> tensor<256xf32>
    %104 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%103 : tensor<256xf32>) outs(%38 : tensor<256xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = math.sqrt %in : f32
      linalg.yield %532 : f32
    } -> tensor<256xf32>
    %105 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%104 : tensor<256xf32>) outs(%38 : tensor<256xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf one, %in, %cst : f32
      cf.assert %532, "unimplemented: tensor with zero element"
      %533 = arith.divf %cst_0, %in : f32
      linalg.yield %533 : f32
    } -> tensor<256xf32>
    %expanded_47 = tensor.expand_shape %arg30 [[0, 1, 2]] output_shape [256, 1, 1] : tensor<256xf32> into tensor<256x1x1xf32>
    %expanded_48 = tensor.expand_shape %105 [[0, 1, 2]] output_shape [256, 1, 1] : tensor<256xf32> into tensor<256x1x1xf32>
    %106 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%102, %expanded_47 : tensor<1x256x56x56xf32>, tensor<256x1x1xf32>) outs(%35 : tensor<1x256x56x56xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.subf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x256x56x56xf32>
    %107 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%106, %expanded_48 : tensor<1x256x56x56xf32>, tensor<256x1x1xf32>) outs(%35 : tensor<1x256x56x56xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x256x56x56xf32>
    %expanded_49 = tensor.expand_shape %arg191 [[0, 1, 2]] output_shape [256, 1, 1] : tensor<256xf32> into tensor<256x1x1xf32>
    %108 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%107, %expanded_49 : tensor<1x256x56x56xf32>, tensor<256x1x1xf32>) outs(%35 : tensor<1x256x56x56xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x256x56x56xf32>
    %expanded_50 = tensor.expand_shape %arg192 [[0, 1, 2]] output_shape [256, 1, 1] : tensor<256xf32> into tensor<256x1x1xf32>
    %109 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%108, %expanded_50 : tensor<1x256x56x56xf32>, tensor<256x1x1xf32>) outs(%35 : tensor<1x256x56x56xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.addf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x256x56x56xf32>
    %110 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%109, %83 : tensor<1x256x56x56xf32>, tensor<1x256x56x56xf32>) outs(%35 : tensor<1x256x56x56xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.addf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x256x56x56xf32>
    %111 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%110 : tensor<1x256x56x56xf32>) outs(%35 : tensor<1x256x56x56xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf ugt, %in, %cst : f32
      %533 = arith.select %532, %in, %cst : f32
      linalg.yield %533 : f32
    } -> tensor<1x256x56x56xf32>
    %112 = tensor.empty() : tensor<1x128x56x56xf32>
    %113 = linalg.fill ins(%cst : f32) outs(%112 : tensor<1x128x56x56xf32>) -> tensor<1x128x56x56xf32>
    %114 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%111, %arg193 : tensor<1x256x56x56xf32>, tensor<128x256x1x1xf32>) outs(%113 : tensor<1x128x56x56xf32>) -> tensor<1x128x56x56xf32>
    %115 = tensor.empty() : tensor<128xf32>
    %116 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg34 : tensor<128xf32>) outs(%115 : tensor<128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.truncf %cst_2 : f64 to f32
      %533 = arith.addf %in, %532 : f32
      linalg.yield %533 : f32
    } -> tensor<128xf32>
    %117 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%116 : tensor<128xf32>) outs(%115 : tensor<128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = math.sqrt %in : f32
      linalg.yield %532 : f32
    } -> tensor<128xf32>
    %118 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%117 : tensor<128xf32>) outs(%115 : tensor<128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf one, %in, %cst : f32
      cf.assert %532, "unimplemented: tensor with zero element"
      %533 = arith.divf %cst_0, %in : f32
      linalg.yield %533 : f32
    } -> tensor<128xf32>
    %expanded_51 = tensor.expand_shape %arg33 [[0, 1, 2]] output_shape [128, 1, 1] : tensor<128xf32> into tensor<128x1x1xf32>
    %expanded_52 = tensor.expand_shape %118 [[0, 1, 2]] output_shape [128, 1, 1] : tensor<128xf32> into tensor<128x1x1xf32>
    %119 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%114, %expanded_51 : tensor<1x128x56x56xf32>, tensor<128x1x1xf32>) outs(%112 : tensor<1x128x56x56xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.subf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x128x56x56xf32>
    %120 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%119, %expanded_52 : tensor<1x128x56x56xf32>, tensor<128x1x1xf32>) outs(%112 : tensor<1x128x56x56xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x128x56x56xf32>
    %expanded_53 = tensor.expand_shape %arg194 [[0, 1, 2]] output_shape [128, 1, 1] : tensor<128xf32> into tensor<128x1x1xf32>
    %121 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%120, %expanded_53 : tensor<1x128x56x56xf32>, tensor<128x1x1xf32>) outs(%112 : tensor<1x128x56x56xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x128x56x56xf32>
    %expanded_54 = tensor.expand_shape %arg195 [[0, 1, 2]] output_shape [128, 1, 1] : tensor<128xf32> into tensor<128x1x1xf32>
    %122 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%121, %expanded_54 : tensor<1x128x56x56xf32>, tensor<128x1x1xf32>) outs(%112 : tensor<1x128x56x56xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.addf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x128x56x56xf32>
    %123 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%122 : tensor<1x128x56x56xf32>) outs(%112 : tensor<1x128x56x56xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf ugt, %in, %cst : f32
      %533 = arith.select %532, %in, %cst : f32
      linalg.yield %533 : f32
    } -> tensor<1x128x56x56xf32>
    %padded_55 = tensor.pad %123 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg321: index, %arg322: index, %arg323: index, %arg324: index):
      tensor.yield %cst : f32
    } : tensor<1x128x56x56xf32> to tensor<1x128x58x58xf32>
    %124 = tensor.empty() : tensor<1x128x28x28xf32>
    %125 = linalg.fill ins(%cst : f32) outs(%124 : tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %126 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%padded_55, %arg196 : tensor<1x128x58x58xf32>, tensor<128x128x3x3xf32>) outs(%125 : tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %127 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg37 : tensor<128xf32>) outs(%115 : tensor<128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.truncf %cst_2 : f64 to f32
      %533 = arith.addf %in, %532 : f32
      linalg.yield %533 : f32
    } -> tensor<128xf32>
    %128 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%127 : tensor<128xf32>) outs(%115 : tensor<128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = math.sqrt %in : f32
      linalg.yield %532 : f32
    } -> tensor<128xf32>
    %129 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%128 : tensor<128xf32>) outs(%115 : tensor<128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf one, %in, %cst : f32
      cf.assert %532, "unimplemented: tensor with zero element"
      %533 = arith.divf %cst_0, %in : f32
      linalg.yield %533 : f32
    } -> tensor<128xf32>
    %expanded_56 = tensor.expand_shape %arg36 [[0, 1, 2]] output_shape [128, 1, 1] : tensor<128xf32> into tensor<128x1x1xf32>
    %expanded_57 = tensor.expand_shape %129 [[0, 1, 2]] output_shape [128, 1, 1] : tensor<128xf32> into tensor<128x1x1xf32>
    %130 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%126, %expanded_56 : tensor<1x128x28x28xf32>, tensor<128x1x1xf32>) outs(%124 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.subf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x128x28x28xf32>
    %131 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%130, %expanded_57 : tensor<1x128x28x28xf32>, tensor<128x1x1xf32>) outs(%124 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x128x28x28xf32>
    %expanded_58 = tensor.expand_shape %arg197 [[0, 1, 2]] output_shape [128, 1, 1] : tensor<128xf32> into tensor<128x1x1xf32>
    %132 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%131, %expanded_58 : tensor<1x128x28x28xf32>, tensor<128x1x1xf32>) outs(%124 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x128x28x28xf32>
    %expanded_59 = tensor.expand_shape %arg198 [[0, 1, 2]] output_shape [128, 1, 1] : tensor<128xf32> into tensor<128x1x1xf32>
    %133 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%132, %expanded_59 : tensor<1x128x28x28xf32>, tensor<128x1x1xf32>) outs(%124 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.addf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x128x28x28xf32>
    %134 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%133 : tensor<1x128x28x28xf32>) outs(%124 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf ugt, %in, %cst : f32
      %533 = arith.select %532, %in, %cst : f32
      linalg.yield %533 : f32
    } -> tensor<1x128x28x28xf32>
    %135 = tensor.empty() : tensor<1x512x28x28xf32>
    %136 = linalg.fill ins(%cst : f32) outs(%135 : tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %137 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%134, %arg199 : tensor<1x128x28x28xf32>, tensor<512x128x1x1xf32>) outs(%136 : tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %138 = tensor.empty() : tensor<512xf32>
    %139 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg40 : tensor<512xf32>) outs(%138 : tensor<512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.truncf %cst_2 : f64 to f32
      %533 = arith.addf %in, %532 : f32
      linalg.yield %533 : f32
    } -> tensor<512xf32>
    %140 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%139 : tensor<512xf32>) outs(%138 : tensor<512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = math.sqrt %in : f32
      linalg.yield %532 : f32
    } -> tensor<512xf32>
    %141 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%140 : tensor<512xf32>) outs(%138 : tensor<512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf one, %in, %cst : f32
      cf.assert %532, "unimplemented: tensor with zero element"
      %533 = arith.divf %cst_0, %in : f32
      linalg.yield %533 : f32
    } -> tensor<512xf32>
    %expanded_60 = tensor.expand_shape %arg39 [[0, 1, 2]] output_shape [512, 1, 1] : tensor<512xf32> into tensor<512x1x1xf32>
    %expanded_61 = tensor.expand_shape %141 [[0, 1, 2]] output_shape [512, 1, 1] : tensor<512xf32> into tensor<512x1x1xf32>
    %142 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%137, %expanded_60 : tensor<1x512x28x28xf32>, tensor<512x1x1xf32>) outs(%135 : tensor<1x512x28x28xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.subf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x512x28x28xf32>
    %143 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%142, %expanded_61 : tensor<1x512x28x28xf32>, tensor<512x1x1xf32>) outs(%135 : tensor<1x512x28x28xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x512x28x28xf32>
    %expanded_62 = tensor.expand_shape %arg200 [[0, 1, 2]] output_shape [512, 1, 1] : tensor<512xf32> into tensor<512x1x1xf32>
    %144 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%143, %expanded_62 : tensor<1x512x28x28xf32>, tensor<512x1x1xf32>) outs(%135 : tensor<1x512x28x28xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x512x28x28xf32>
    %expanded_63 = tensor.expand_shape %arg201 [[0, 1, 2]] output_shape [512, 1, 1] : tensor<512xf32> into tensor<512x1x1xf32>
    %145 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%144, %expanded_63 : tensor<1x512x28x28xf32>, tensor<512x1x1xf32>) outs(%135 : tensor<1x512x28x28xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.addf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x512x28x28xf32>
    %146 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%111, %arg202 : tensor<1x256x56x56xf32>, tensor<512x256x1x1xf32>) outs(%136 : tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %147 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg43 : tensor<512xf32>) outs(%138 : tensor<512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.truncf %cst_2 : f64 to f32
      %533 = arith.addf %in, %532 : f32
      linalg.yield %533 : f32
    } -> tensor<512xf32>
    %148 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%147 : tensor<512xf32>) outs(%138 : tensor<512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = math.sqrt %in : f32
      linalg.yield %532 : f32
    } -> tensor<512xf32>
    %149 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%148 : tensor<512xf32>) outs(%138 : tensor<512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf one, %in, %cst : f32
      cf.assert %532, "unimplemented: tensor with zero element"
      %533 = arith.divf %cst_0, %in : f32
      linalg.yield %533 : f32
    } -> tensor<512xf32>
    %expanded_64 = tensor.expand_shape %arg42 [[0, 1, 2]] output_shape [512, 1, 1] : tensor<512xf32> into tensor<512x1x1xf32>
    %expanded_65 = tensor.expand_shape %149 [[0, 1, 2]] output_shape [512, 1, 1] : tensor<512xf32> into tensor<512x1x1xf32>
    %150 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%146, %expanded_64 : tensor<1x512x28x28xf32>, tensor<512x1x1xf32>) outs(%135 : tensor<1x512x28x28xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.subf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x512x28x28xf32>
    %151 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%150, %expanded_65 : tensor<1x512x28x28xf32>, tensor<512x1x1xf32>) outs(%135 : tensor<1x512x28x28xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x512x28x28xf32>
    %expanded_66 = tensor.expand_shape %arg203 [[0, 1, 2]] output_shape [512, 1, 1] : tensor<512xf32> into tensor<512x1x1xf32>
    %152 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%151, %expanded_66 : tensor<1x512x28x28xf32>, tensor<512x1x1xf32>) outs(%135 : tensor<1x512x28x28xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x512x28x28xf32>
    %expanded_67 = tensor.expand_shape %arg204 [[0, 1, 2]] output_shape [512, 1, 1] : tensor<512xf32> into tensor<512x1x1xf32>
    %153 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%152, %expanded_67 : tensor<1x512x28x28xf32>, tensor<512x1x1xf32>) outs(%135 : tensor<1x512x28x28xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.addf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x512x28x28xf32>
    %154 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%145, %153 : tensor<1x512x28x28xf32>, tensor<1x512x28x28xf32>) outs(%135 : tensor<1x512x28x28xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.addf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x512x28x28xf32>
    %155 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%154 : tensor<1x512x28x28xf32>) outs(%135 : tensor<1x512x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf ugt, %in, %cst : f32
      %533 = arith.select %532, %in, %cst : f32
      linalg.yield %533 : f32
    } -> tensor<1x512x28x28xf32>
    %156 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%155, %arg205 : tensor<1x512x28x28xf32>, tensor<128x512x1x1xf32>) outs(%125 : tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %157 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg46 : tensor<128xf32>) outs(%115 : tensor<128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.truncf %cst_2 : f64 to f32
      %533 = arith.addf %in, %532 : f32
      linalg.yield %533 : f32
    } -> tensor<128xf32>
    %158 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%157 : tensor<128xf32>) outs(%115 : tensor<128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = math.sqrt %in : f32
      linalg.yield %532 : f32
    } -> tensor<128xf32>
    %159 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%158 : tensor<128xf32>) outs(%115 : tensor<128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf one, %in, %cst : f32
      cf.assert %532, "unimplemented: tensor with zero element"
      %533 = arith.divf %cst_0, %in : f32
      linalg.yield %533 : f32
    } -> tensor<128xf32>
    %expanded_68 = tensor.expand_shape %arg45 [[0, 1, 2]] output_shape [128, 1, 1] : tensor<128xf32> into tensor<128x1x1xf32>
    %expanded_69 = tensor.expand_shape %159 [[0, 1, 2]] output_shape [128, 1, 1] : tensor<128xf32> into tensor<128x1x1xf32>
    %160 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%156, %expanded_68 : tensor<1x128x28x28xf32>, tensor<128x1x1xf32>) outs(%124 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.subf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x128x28x28xf32>
    %161 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%160, %expanded_69 : tensor<1x128x28x28xf32>, tensor<128x1x1xf32>) outs(%124 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x128x28x28xf32>
    %expanded_70 = tensor.expand_shape %arg206 [[0, 1, 2]] output_shape [128, 1, 1] : tensor<128xf32> into tensor<128x1x1xf32>
    %162 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%161, %expanded_70 : tensor<1x128x28x28xf32>, tensor<128x1x1xf32>) outs(%124 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x128x28x28xf32>
    %expanded_71 = tensor.expand_shape %arg207 [[0, 1, 2]] output_shape [128, 1, 1] : tensor<128xf32> into tensor<128x1x1xf32>
    %163 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%162, %expanded_71 : tensor<1x128x28x28xf32>, tensor<128x1x1xf32>) outs(%124 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.addf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x128x28x28xf32>
    %164 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%163 : tensor<1x128x28x28xf32>) outs(%124 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf ugt, %in, %cst : f32
      %533 = arith.select %532, %in, %cst : f32
      linalg.yield %533 : f32
    } -> tensor<1x128x28x28xf32>
    %padded_72 = tensor.pad %164 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg321: index, %arg322: index, %arg323: index, %arg324: index):
      tensor.yield %cst : f32
    } : tensor<1x128x28x28xf32> to tensor<1x128x30x30xf32>
    %165 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_72, %arg208 : tensor<1x128x30x30xf32>, tensor<128x128x3x3xf32>) outs(%125 : tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %166 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg49 : tensor<128xf32>) outs(%115 : tensor<128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.truncf %cst_2 : f64 to f32
      %533 = arith.addf %in, %532 : f32
      linalg.yield %533 : f32
    } -> tensor<128xf32>
    %167 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%166 : tensor<128xf32>) outs(%115 : tensor<128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = math.sqrt %in : f32
      linalg.yield %532 : f32
    } -> tensor<128xf32>
    %168 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%167 : tensor<128xf32>) outs(%115 : tensor<128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf one, %in, %cst : f32
      cf.assert %532, "unimplemented: tensor with zero element"
      %533 = arith.divf %cst_0, %in : f32
      linalg.yield %533 : f32
    } -> tensor<128xf32>
    %expanded_73 = tensor.expand_shape %arg48 [[0, 1, 2]] output_shape [128, 1, 1] : tensor<128xf32> into tensor<128x1x1xf32>
    %expanded_74 = tensor.expand_shape %168 [[0, 1, 2]] output_shape [128, 1, 1] : tensor<128xf32> into tensor<128x1x1xf32>
    %169 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%165, %expanded_73 : tensor<1x128x28x28xf32>, tensor<128x1x1xf32>) outs(%124 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.subf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x128x28x28xf32>
    %170 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%169, %expanded_74 : tensor<1x128x28x28xf32>, tensor<128x1x1xf32>) outs(%124 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x128x28x28xf32>
    %expanded_75 = tensor.expand_shape %arg209 [[0, 1, 2]] output_shape [128, 1, 1] : tensor<128xf32> into tensor<128x1x1xf32>
    %171 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%170, %expanded_75 : tensor<1x128x28x28xf32>, tensor<128x1x1xf32>) outs(%124 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x128x28x28xf32>
    %expanded_76 = tensor.expand_shape %arg210 [[0, 1, 2]] output_shape [128, 1, 1] : tensor<128xf32> into tensor<128x1x1xf32>
    %172 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%171, %expanded_76 : tensor<1x128x28x28xf32>, tensor<128x1x1xf32>) outs(%124 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.addf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x128x28x28xf32>
    %173 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%172 : tensor<1x128x28x28xf32>) outs(%124 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf ugt, %in, %cst : f32
      %533 = arith.select %532, %in, %cst : f32
      linalg.yield %533 : f32
    } -> tensor<1x128x28x28xf32>
    %174 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%173, %arg211 : tensor<1x128x28x28xf32>, tensor<512x128x1x1xf32>) outs(%136 : tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %175 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg52 : tensor<512xf32>) outs(%138 : tensor<512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.truncf %cst_2 : f64 to f32
      %533 = arith.addf %in, %532 : f32
      linalg.yield %533 : f32
    } -> tensor<512xf32>
    %176 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%175 : tensor<512xf32>) outs(%138 : tensor<512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = math.sqrt %in : f32
      linalg.yield %532 : f32
    } -> tensor<512xf32>
    %177 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%176 : tensor<512xf32>) outs(%138 : tensor<512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf one, %in, %cst : f32
      cf.assert %532, "unimplemented: tensor with zero element"
      %533 = arith.divf %cst_0, %in : f32
      linalg.yield %533 : f32
    } -> tensor<512xf32>
    %expanded_77 = tensor.expand_shape %arg51 [[0, 1, 2]] output_shape [512, 1, 1] : tensor<512xf32> into tensor<512x1x1xf32>
    %expanded_78 = tensor.expand_shape %177 [[0, 1, 2]] output_shape [512, 1, 1] : tensor<512xf32> into tensor<512x1x1xf32>
    %178 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%174, %expanded_77 : tensor<1x512x28x28xf32>, tensor<512x1x1xf32>) outs(%135 : tensor<1x512x28x28xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.subf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x512x28x28xf32>
    %179 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%178, %expanded_78 : tensor<1x512x28x28xf32>, tensor<512x1x1xf32>) outs(%135 : tensor<1x512x28x28xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x512x28x28xf32>
    %expanded_79 = tensor.expand_shape %arg212 [[0, 1, 2]] output_shape [512, 1, 1] : tensor<512xf32> into tensor<512x1x1xf32>
    %180 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%179, %expanded_79 : tensor<1x512x28x28xf32>, tensor<512x1x1xf32>) outs(%135 : tensor<1x512x28x28xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x512x28x28xf32>
    %expanded_80 = tensor.expand_shape %arg213 [[0, 1, 2]] output_shape [512, 1, 1] : tensor<512xf32> into tensor<512x1x1xf32>
    %181 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%180, %expanded_80 : tensor<1x512x28x28xf32>, tensor<512x1x1xf32>) outs(%135 : tensor<1x512x28x28xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.addf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x512x28x28xf32>
    %182 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%181, %155 : tensor<1x512x28x28xf32>, tensor<1x512x28x28xf32>) outs(%135 : tensor<1x512x28x28xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.addf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x512x28x28xf32>
    %183 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%182 : tensor<1x512x28x28xf32>) outs(%135 : tensor<1x512x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf ugt, %in, %cst : f32
      %533 = arith.select %532, %in, %cst : f32
      linalg.yield %533 : f32
    } -> tensor<1x512x28x28xf32>
    %184 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%183, %arg214 : tensor<1x512x28x28xf32>, tensor<128x512x1x1xf32>) outs(%125 : tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %185 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg55 : tensor<128xf32>) outs(%115 : tensor<128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.truncf %cst_2 : f64 to f32
      %533 = arith.addf %in, %532 : f32
      linalg.yield %533 : f32
    } -> tensor<128xf32>
    %186 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%185 : tensor<128xf32>) outs(%115 : tensor<128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = math.sqrt %in : f32
      linalg.yield %532 : f32
    } -> tensor<128xf32>
    %187 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%186 : tensor<128xf32>) outs(%115 : tensor<128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf one, %in, %cst : f32
      cf.assert %532, "unimplemented: tensor with zero element"
      %533 = arith.divf %cst_0, %in : f32
      linalg.yield %533 : f32
    } -> tensor<128xf32>
    %expanded_81 = tensor.expand_shape %arg54 [[0, 1, 2]] output_shape [128, 1, 1] : tensor<128xf32> into tensor<128x1x1xf32>
    %expanded_82 = tensor.expand_shape %187 [[0, 1, 2]] output_shape [128, 1, 1] : tensor<128xf32> into tensor<128x1x1xf32>
    %188 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%184, %expanded_81 : tensor<1x128x28x28xf32>, tensor<128x1x1xf32>) outs(%124 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.subf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x128x28x28xf32>
    %189 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%188, %expanded_82 : tensor<1x128x28x28xf32>, tensor<128x1x1xf32>) outs(%124 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x128x28x28xf32>
    %expanded_83 = tensor.expand_shape %arg215 [[0, 1, 2]] output_shape [128, 1, 1] : tensor<128xf32> into tensor<128x1x1xf32>
    %190 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%189, %expanded_83 : tensor<1x128x28x28xf32>, tensor<128x1x1xf32>) outs(%124 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x128x28x28xf32>
    %expanded_84 = tensor.expand_shape %arg216 [[0, 1, 2]] output_shape [128, 1, 1] : tensor<128xf32> into tensor<128x1x1xf32>
    %191 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%190, %expanded_84 : tensor<1x128x28x28xf32>, tensor<128x1x1xf32>) outs(%124 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.addf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x128x28x28xf32>
    %192 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%191 : tensor<1x128x28x28xf32>) outs(%124 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf ugt, %in, %cst : f32
      %533 = arith.select %532, %in, %cst : f32
      linalg.yield %533 : f32
    } -> tensor<1x128x28x28xf32>
    %padded_85 = tensor.pad %192 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg321: index, %arg322: index, %arg323: index, %arg324: index):
      tensor.yield %cst : f32
    } : tensor<1x128x28x28xf32> to tensor<1x128x30x30xf32>
    %193 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_85, %arg217 : tensor<1x128x30x30xf32>, tensor<128x128x3x3xf32>) outs(%125 : tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %194 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg58 : tensor<128xf32>) outs(%115 : tensor<128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.truncf %cst_2 : f64 to f32
      %533 = arith.addf %in, %532 : f32
      linalg.yield %533 : f32
    } -> tensor<128xf32>
    %195 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%194 : tensor<128xf32>) outs(%115 : tensor<128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = math.sqrt %in : f32
      linalg.yield %532 : f32
    } -> tensor<128xf32>
    %196 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%195 : tensor<128xf32>) outs(%115 : tensor<128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf one, %in, %cst : f32
      cf.assert %532, "unimplemented: tensor with zero element"
      %533 = arith.divf %cst_0, %in : f32
      linalg.yield %533 : f32
    } -> tensor<128xf32>
    %expanded_86 = tensor.expand_shape %arg57 [[0, 1, 2]] output_shape [128, 1, 1] : tensor<128xf32> into tensor<128x1x1xf32>
    %expanded_87 = tensor.expand_shape %196 [[0, 1, 2]] output_shape [128, 1, 1] : tensor<128xf32> into tensor<128x1x1xf32>
    %197 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%193, %expanded_86 : tensor<1x128x28x28xf32>, tensor<128x1x1xf32>) outs(%124 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.subf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x128x28x28xf32>
    %198 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%197, %expanded_87 : tensor<1x128x28x28xf32>, tensor<128x1x1xf32>) outs(%124 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x128x28x28xf32>
    %expanded_88 = tensor.expand_shape %arg218 [[0, 1, 2]] output_shape [128, 1, 1] : tensor<128xf32> into tensor<128x1x1xf32>
    %199 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%198, %expanded_88 : tensor<1x128x28x28xf32>, tensor<128x1x1xf32>) outs(%124 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x128x28x28xf32>
    %expanded_89 = tensor.expand_shape %arg219 [[0, 1, 2]] output_shape [128, 1, 1] : tensor<128xf32> into tensor<128x1x1xf32>
    %200 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%199, %expanded_89 : tensor<1x128x28x28xf32>, tensor<128x1x1xf32>) outs(%124 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.addf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x128x28x28xf32>
    %201 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%200 : tensor<1x128x28x28xf32>) outs(%124 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf ugt, %in, %cst : f32
      %533 = arith.select %532, %in, %cst : f32
      linalg.yield %533 : f32
    } -> tensor<1x128x28x28xf32>
    %202 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%201, %arg220 : tensor<1x128x28x28xf32>, tensor<512x128x1x1xf32>) outs(%136 : tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %203 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg61 : tensor<512xf32>) outs(%138 : tensor<512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.truncf %cst_2 : f64 to f32
      %533 = arith.addf %in, %532 : f32
      linalg.yield %533 : f32
    } -> tensor<512xf32>
    %204 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%203 : tensor<512xf32>) outs(%138 : tensor<512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = math.sqrt %in : f32
      linalg.yield %532 : f32
    } -> tensor<512xf32>
    %205 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%204 : tensor<512xf32>) outs(%138 : tensor<512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf one, %in, %cst : f32
      cf.assert %532, "unimplemented: tensor with zero element"
      %533 = arith.divf %cst_0, %in : f32
      linalg.yield %533 : f32
    } -> tensor<512xf32>
    %expanded_90 = tensor.expand_shape %arg60 [[0, 1, 2]] output_shape [512, 1, 1] : tensor<512xf32> into tensor<512x1x1xf32>
    %expanded_91 = tensor.expand_shape %205 [[0, 1, 2]] output_shape [512, 1, 1] : tensor<512xf32> into tensor<512x1x1xf32>
    %206 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%202, %expanded_90 : tensor<1x512x28x28xf32>, tensor<512x1x1xf32>) outs(%135 : tensor<1x512x28x28xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.subf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x512x28x28xf32>
    %207 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%206, %expanded_91 : tensor<1x512x28x28xf32>, tensor<512x1x1xf32>) outs(%135 : tensor<1x512x28x28xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x512x28x28xf32>
    %expanded_92 = tensor.expand_shape %arg221 [[0, 1, 2]] output_shape [512, 1, 1] : tensor<512xf32> into tensor<512x1x1xf32>
    %208 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%207, %expanded_92 : tensor<1x512x28x28xf32>, tensor<512x1x1xf32>) outs(%135 : tensor<1x512x28x28xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x512x28x28xf32>
    %expanded_93 = tensor.expand_shape %arg222 [[0, 1, 2]] output_shape [512, 1, 1] : tensor<512xf32> into tensor<512x1x1xf32>
    %209 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%208, %expanded_93 : tensor<1x512x28x28xf32>, tensor<512x1x1xf32>) outs(%135 : tensor<1x512x28x28xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.addf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x512x28x28xf32>
    %210 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%209, %183 : tensor<1x512x28x28xf32>, tensor<1x512x28x28xf32>) outs(%135 : tensor<1x512x28x28xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.addf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x512x28x28xf32>
    %211 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%210 : tensor<1x512x28x28xf32>) outs(%135 : tensor<1x512x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf ugt, %in, %cst : f32
      %533 = arith.select %532, %in, %cst : f32
      linalg.yield %533 : f32
    } -> tensor<1x512x28x28xf32>
    %212 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%211, %arg223 : tensor<1x512x28x28xf32>, tensor<128x512x1x1xf32>) outs(%125 : tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %213 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg64 : tensor<128xf32>) outs(%115 : tensor<128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.truncf %cst_2 : f64 to f32
      %533 = arith.addf %in, %532 : f32
      linalg.yield %533 : f32
    } -> tensor<128xf32>
    %214 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%213 : tensor<128xf32>) outs(%115 : tensor<128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = math.sqrt %in : f32
      linalg.yield %532 : f32
    } -> tensor<128xf32>
    %215 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%214 : tensor<128xf32>) outs(%115 : tensor<128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf one, %in, %cst : f32
      cf.assert %532, "unimplemented: tensor with zero element"
      %533 = arith.divf %cst_0, %in : f32
      linalg.yield %533 : f32
    } -> tensor<128xf32>
    %expanded_94 = tensor.expand_shape %arg63 [[0, 1, 2]] output_shape [128, 1, 1] : tensor<128xf32> into tensor<128x1x1xf32>
    %expanded_95 = tensor.expand_shape %215 [[0, 1, 2]] output_shape [128, 1, 1] : tensor<128xf32> into tensor<128x1x1xf32>
    %216 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%212, %expanded_94 : tensor<1x128x28x28xf32>, tensor<128x1x1xf32>) outs(%124 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.subf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x128x28x28xf32>
    %217 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%216, %expanded_95 : tensor<1x128x28x28xf32>, tensor<128x1x1xf32>) outs(%124 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x128x28x28xf32>
    %expanded_96 = tensor.expand_shape %arg224 [[0, 1, 2]] output_shape [128, 1, 1] : tensor<128xf32> into tensor<128x1x1xf32>
    %218 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%217, %expanded_96 : tensor<1x128x28x28xf32>, tensor<128x1x1xf32>) outs(%124 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x128x28x28xf32>
    %expanded_97 = tensor.expand_shape %arg225 [[0, 1, 2]] output_shape [128, 1, 1] : tensor<128xf32> into tensor<128x1x1xf32>
    %219 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%218, %expanded_97 : tensor<1x128x28x28xf32>, tensor<128x1x1xf32>) outs(%124 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.addf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x128x28x28xf32>
    %220 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%219 : tensor<1x128x28x28xf32>) outs(%124 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf ugt, %in, %cst : f32
      %533 = arith.select %532, %in, %cst : f32
      linalg.yield %533 : f32
    } -> tensor<1x128x28x28xf32>
    %padded_98 = tensor.pad %220 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg321: index, %arg322: index, %arg323: index, %arg324: index):
      tensor.yield %cst : f32
    } : tensor<1x128x28x28xf32> to tensor<1x128x30x30xf32>
    %221 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_98, %arg226 : tensor<1x128x30x30xf32>, tensor<128x128x3x3xf32>) outs(%125 : tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %222 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg67 : tensor<128xf32>) outs(%115 : tensor<128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.truncf %cst_2 : f64 to f32
      %533 = arith.addf %in, %532 : f32
      linalg.yield %533 : f32
    } -> tensor<128xf32>
    %223 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%222 : tensor<128xf32>) outs(%115 : tensor<128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = math.sqrt %in : f32
      linalg.yield %532 : f32
    } -> tensor<128xf32>
    %224 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%223 : tensor<128xf32>) outs(%115 : tensor<128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf one, %in, %cst : f32
      cf.assert %532, "unimplemented: tensor with zero element"
      %533 = arith.divf %cst_0, %in : f32
      linalg.yield %533 : f32
    } -> tensor<128xf32>
    %expanded_99 = tensor.expand_shape %arg66 [[0, 1, 2]] output_shape [128, 1, 1] : tensor<128xf32> into tensor<128x1x1xf32>
    %expanded_100 = tensor.expand_shape %224 [[0, 1, 2]] output_shape [128, 1, 1] : tensor<128xf32> into tensor<128x1x1xf32>
    %225 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%221, %expanded_99 : tensor<1x128x28x28xf32>, tensor<128x1x1xf32>) outs(%124 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.subf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x128x28x28xf32>
    %226 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%225, %expanded_100 : tensor<1x128x28x28xf32>, tensor<128x1x1xf32>) outs(%124 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x128x28x28xf32>
    %expanded_101 = tensor.expand_shape %arg227 [[0, 1, 2]] output_shape [128, 1, 1] : tensor<128xf32> into tensor<128x1x1xf32>
    %227 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%226, %expanded_101 : tensor<1x128x28x28xf32>, tensor<128x1x1xf32>) outs(%124 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x128x28x28xf32>
    %expanded_102 = tensor.expand_shape %arg228 [[0, 1, 2]] output_shape [128, 1, 1] : tensor<128xf32> into tensor<128x1x1xf32>
    %228 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%227, %expanded_102 : tensor<1x128x28x28xf32>, tensor<128x1x1xf32>) outs(%124 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.addf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x128x28x28xf32>
    %229 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%228 : tensor<1x128x28x28xf32>) outs(%124 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf ugt, %in, %cst : f32
      %533 = arith.select %532, %in, %cst : f32
      linalg.yield %533 : f32
    } -> tensor<1x128x28x28xf32>
    %230 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%229, %arg229 : tensor<1x128x28x28xf32>, tensor<512x128x1x1xf32>) outs(%136 : tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %231 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg70 : tensor<512xf32>) outs(%138 : tensor<512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.truncf %cst_2 : f64 to f32
      %533 = arith.addf %in, %532 : f32
      linalg.yield %533 : f32
    } -> tensor<512xf32>
    %232 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%231 : tensor<512xf32>) outs(%138 : tensor<512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = math.sqrt %in : f32
      linalg.yield %532 : f32
    } -> tensor<512xf32>
    %233 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%232 : tensor<512xf32>) outs(%138 : tensor<512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf one, %in, %cst : f32
      cf.assert %532, "unimplemented: tensor with zero element"
      %533 = arith.divf %cst_0, %in : f32
      linalg.yield %533 : f32
    } -> tensor<512xf32>
    %expanded_103 = tensor.expand_shape %arg69 [[0, 1, 2]] output_shape [512, 1, 1] : tensor<512xf32> into tensor<512x1x1xf32>
    %expanded_104 = tensor.expand_shape %233 [[0, 1, 2]] output_shape [512, 1, 1] : tensor<512xf32> into tensor<512x1x1xf32>
    %234 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%230, %expanded_103 : tensor<1x512x28x28xf32>, tensor<512x1x1xf32>) outs(%135 : tensor<1x512x28x28xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.subf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x512x28x28xf32>
    %235 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%234, %expanded_104 : tensor<1x512x28x28xf32>, tensor<512x1x1xf32>) outs(%135 : tensor<1x512x28x28xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x512x28x28xf32>
    %expanded_105 = tensor.expand_shape %arg230 [[0, 1, 2]] output_shape [512, 1, 1] : tensor<512xf32> into tensor<512x1x1xf32>
    %236 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%235, %expanded_105 : tensor<1x512x28x28xf32>, tensor<512x1x1xf32>) outs(%135 : tensor<1x512x28x28xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x512x28x28xf32>
    %expanded_106 = tensor.expand_shape %arg231 [[0, 1, 2]] output_shape [512, 1, 1] : tensor<512xf32> into tensor<512x1x1xf32>
    %237 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%236, %expanded_106 : tensor<1x512x28x28xf32>, tensor<512x1x1xf32>) outs(%135 : tensor<1x512x28x28xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.addf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x512x28x28xf32>
    %238 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%237, %211 : tensor<1x512x28x28xf32>, tensor<1x512x28x28xf32>) outs(%135 : tensor<1x512x28x28xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.addf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x512x28x28xf32>
    %239 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%238 : tensor<1x512x28x28xf32>) outs(%135 : tensor<1x512x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf ugt, %in, %cst : f32
      %533 = arith.select %532, %in, %cst : f32
      linalg.yield %533 : f32
    } -> tensor<1x512x28x28xf32>
    %240 = tensor.empty() : tensor<1x256x28x28xf32>
    %241 = linalg.fill ins(%cst : f32) outs(%240 : tensor<1x256x28x28xf32>) -> tensor<1x256x28x28xf32>
    %242 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%239, %arg232 : tensor<1x512x28x28xf32>, tensor<256x512x1x1xf32>) outs(%241 : tensor<1x256x28x28xf32>) -> tensor<1x256x28x28xf32>
    %243 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg73 : tensor<256xf32>) outs(%38 : tensor<256xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.truncf %cst_2 : f64 to f32
      %533 = arith.addf %in, %532 : f32
      linalg.yield %533 : f32
    } -> tensor<256xf32>
    %244 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%243 : tensor<256xf32>) outs(%38 : tensor<256xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = math.sqrt %in : f32
      linalg.yield %532 : f32
    } -> tensor<256xf32>
    %245 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%244 : tensor<256xf32>) outs(%38 : tensor<256xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf one, %in, %cst : f32
      cf.assert %532, "unimplemented: tensor with zero element"
      %533 = arith.divf %cst_0, %in : f32
      linalg.yield %533 : f32
    } -> tensor<256xf32>
    %expanded_107 = tensor.expand_shape %arg72 [[0, 1, 2]] output_shape [256, 1, 1] : tensor<256xf32> into tensor<256x1x1xf32>
    %expanded_108 = tensor.expand_shape %245 [[0, 1, 2]] output_shape [256, 1, 1] : tensor<256xf32> into tensor<256x1x1xf32>
    %246 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%242, %expanded_107 : tensor<1x256x28x28xf32>, tensor<256x1x1xf32>) outs(%240 : tensor<1x256x28x28xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.subf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x256x28x28xf32>
    %247 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%246, %expanded_108 : tensor<1x256x28x28xf32>, tensor<256x1x1xf32>) outs(%240 : tensor<1x256x28x28xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x256x28x28xf32>
    %expanded_109 = tensor.expand_shape %arg233 [[0, 1, 2]] output_shape [256, 1, 1] : tensor<256xf32> into tensor<256x1x1xf32>
    %248 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%247, %expanded_109 : tensor<1x256x28x28xf32>, tensor<256x1x1xf32>) outs(%240 : tensor<1x256x28x28xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x256x28x28xf32>
    %expanded_110 = tensor.expand_shape %arg234 [[0, 1, 2]] output_shape [256, 1, 1] : tensor<256xf32> into tensor<256x1x1xf32>
    %249 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%248, %expanded_110 : tensor<1x256x28x28xf32>, tensor<256x1x1xf32>) outs(%240 : tensor<1x256x28x28xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.addf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x256x28x28xf32>
    %250 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%249 : tensor<1x256x28x28xf32>) outs(%240 : tensor<1x256x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf ugt, %in, %cst : f32
      %533 = arith.select %532, %in, %cst : f32
      linalg.yield %533 : f32
    } -> tensor<1x256x28x28xf32>
    %padded_111 = tensor.pad %250 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg321: index, %arg322: index, %arg323: index, %arg324: index):
      tensor.yield %cst : f32
    } : tensor<1x256x28x28xf32> to tensor<1x256x30x30xf32>
    %251 = tensor.empty() : tensor<1x256x14x14xf32>
    %252 = linalg.fill ins(%cst : f32) outs(%251 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %253 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%padded_111, %arg235 : tensor<1x256x30x30xf32>, tensor<256x256x3x3xf32>) outs(%252 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %254 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg76 : tensor<256xf32>) outs(%38 : tensor<256xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.truncf %cst_2 : f64 to f32
      %533 = arith.addf %in, %532 : f32
      linalg.yield %533 : f32
    } -> tensor<256xf32>
    %255 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%254 : tensor<256xf32>) outs(%38 : tensor<256xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = math.sqrt %in : f32
      linalg.yield %532 : f32
    } -> tensor<256xf32>
    %256 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%255 : tensor<256xf32>) outs(%38 : tensor<256xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf one, %in, %cst : f32
      cf.assert %532, "unimplemented: tensor with zero element"
      %533 = arith.divf %cst_0, %in : f32
      linalg.yield %533 : f32
    } -> tensor<256xf32>
    %expanded_112 = tensor.expand_shape %arg75 [[0, 1, 2]] output_shape [256, 1, 1] : tensor<256xf32> into tensor<256x1x1xf32>
    %expanded_113 = tensor.expand_shape %256 [[0, 1, 2]] output_shape [256, 1, 1] : tensor<256xf32> into tensor<256x1x1xf32>
    %257 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%253, %expanded_112 : tensor<1x256x14x14xf32>, tensor<256x1x1xf32>) outs(%251 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.subf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x256x14x14xf32>
    %258 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%257, %expanded_113 : tensor<1x256x14x14xf32>, tensor<256x1x1xf32>) outs(%251 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x256x14x14xf32>
    %expanded_114 = tensor.expand_shape %arg236 [[0, 1, 2]] output_shape [256, 1, 1] : tensor<256xf32> into tensor<256x1x1xf32>
    %259 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%258, %expanded_114 : tensor<1x256x14x14xf32>, tensor<256x1x1xf32>) outs(%251 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x256x14x14xf32>
    %expanded_115 = tensor.expand_shape %arg237 [[0, 1, 2]] output_shape [256, 1, 1] : tensor<256xf32> into tensor<256x1x1xf32>
    %260 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%259, %expanded_115 : tensor<1x256x14x14xf32>, tensor<256x1x1xf32>) outs(%251 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.addf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x256x14x14xf32>
    %261 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%260 : tensor<1x256x14x14xf32>) outs(%251 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf ugt, %in, %cst : f32
      %533 = arith.select %532, %in, %cst : f32
      linalg.yield %533 : f32
    } -> tensor<1x256x14x14xf32>
    %262 = tensor.empty() : tensor<1x1024x14x14xf32>
    %263 = linalg.fill ins(%cst : f32) outs(%262 : tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %264 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%261, %arg238 : tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>) outs(%263 : tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %265 = tensor.empty() : tensor<1024xf32>
    %266 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg79 : tensor<1024xf32>) outs(%265 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.truncf %cst_2 : f64 to f32
      %533 = arith.addf %in, %532 : f32
      linalg.yield %533 : f32
    } -> tensor<1024xf32>
    %267 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%266 : tensor<1024xf32>) outs(%265 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = math.sqrt %in : f32
      linalg.yield %532 : f32
    } -> tensor<1024xf32>
    %268 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%267 : tensor<1024xf32>) outs(%265 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf one, %in, %cst : f32
      cf.assert %532, "unimplemented: tensor with zero element"
      %533 = arith.divf %cst_0, %in : f32
      linalg.yield %533 : f32
    } -> tensor<1024xf32>
    %expanded_116 = tensor.expand_shape %arg78 [[0, 1, 2]] output_shape [1024, 1, 1] : tensor<1024xf32> into tensor<1024x1x1xf32>
    %expanded_117 = tensor.expand_shape %268 [[0, 1, 2]] output_shape [1024, 1, 1] : tensor<1024xf32> into tensor<1024x1x1xf32>
    %269 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%264, %expanded_116 : tensor<1x1024x14x14xf32>, tensor<1024x1x1xf32>) outs(%262 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.subf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x1024x14x14xf32>
    %270 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%269, %expanded_117 : tensor<1x1024x14x14xf32>, tensor<1024x1x1xf32>) outs(%262 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x1024x14x14xf32>
    %expanded_118 = tensor.expand_shape %arg239 [[0, 1, 2]] output_shape [1024, 1, 1] : tensor<1024xf32> into tensor<1024x1x1xf32>
    %271 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%270, %expanded_118 : tensor<1x1024x14x14xf32>, tensor<1024x1x1xf32>) outs(%262 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x1024x14x14xf32>
    %expanded_119 = tensor.expand_shape %arg240 [[0, 1, 2]] output_shape [1024, 1, 1] : tensor<1024xf32> into tensor<1024x1x1xf32>
    %272 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%271, %expanded_119 : tensor<1x1024x14x14xf32>, tensor<1024x1x1xf32>) outs(%262 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.addf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x1024x14x14xf32>
    %273 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%239, %arg241 : tensor<1x512x28x28xf32>, tensor<1024x512x1x1xf32>) outs(%263 : tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %274 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg82 : tensor<1024xf32>) outs(%265 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.truncf %cst_2 : f64 to f32
      %533 = arith.addf %in, %532 : f32
      linalg.yield %533 : f32
    } -> tensor<1024xf32>
    %275 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%274 : tensor<1024xf32>) outs(%265 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = math.sqrt %in : f32
      linalg.yield %532 : f32
    } -> tensor<1024xf32>
    %276 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%275 : tensor<1024xf32>) outs(%265 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf one, %in, %cst : f32
      cf.assert %532, "unimplemented: tensor with zero element"
      %533 = arith.divf %cst_0, %in : f32
      linalg.yield %533 : f32
    } -> tensor<1024xf32>
    %expanded_120 = tensor.expand_shape %arg81 [[0, 1, 2]] output_shape [1024, 1, 1] : tensor<1024xf32> into tensor<1024x1x1xf32>
    %expanded_121 = tensor.expand_shape %276 [[0, 1, 2]] output_shape [1024, 1, 1] : tensor<1024xf32> into tensor<1024x1x1xf32>
    %277 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%273, %expanded_120 : tensor<1x1024x14x14xf32>, tensor<1024x1x1xf32>) outs(%262 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.subf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x1024x14x14xf32>
    %278 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%277, %expanded_121 : tensor<1x1024x14x14xf32>, tensor<1024x1x1xf32>) outs(%262 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x1024x14x14xf32>
    %expanded_122 = tensor.expand_shape %arg242 [[0, 1, 2]] output_shape [1024, 1, 1] : tensor<1024xf32> into tensor<1024x1x1xf32>
    %279 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%278, %expanded_122 : tensor<1x1024x14x14xf32>, tensor<1024x1x1xf32>) outs(%262 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x1024x14x14xf32>
    %expanded_123 = tensor.expand_shape %arg243 [[0, 1, 2]] output_shape [1024, 1, 1] : tensor<1024xf32> into tensor<1024x1x1xf32>
    %280 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%279, %expanded_123 : tensor<1x1024x14x14xf32>, tensor<1024x1x1xf32>) outs(%262 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.addf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x1024x14x14xf32>
    %281 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%272, %280 : tensor<1x1024x14x14xf32>, tensor<1x1024x14x14xf32>) outs(%262 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.addf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x1024x14x14xf32>
    %282 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%281 : tensor<1x1024x14x14xf32>) outs(%262 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf ugt, %in, %cst : f32
      %533 = arith.select %532, %in, %cst : f32
      linalg.yield %533 : f32
    } -> tensor<1x1024x14x14xf32>
    %283 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%282, %arg244 : tensor<1x1024x14x14xf32>, tensor<256x1024x1x1xf32>) outs(%252 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %284 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg85 : tensor<256xf32>) outs(%38 : tensor<256xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.truncf %cst_2 : f64 to f32
      %533 = arith.addf %in, %532 : f32
      linalg.yield %533 : f32
    } -> tensor<256xf32>
    %285 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%284 : tensor<256xf32>) outs(%38 : tensor<256xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = math.sqrt %in : f32
      linalg.yield %532 : f32
    } -> tensor<256xf32>
    %286 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%285 : tensor<256xf32>) outs(%38 : tensor<256xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf one, %in, %cst : f32
      cf.assert %532, "unimplemented: tensor with zero element"
      %533 = arith.divf %cst_0, %in : f32
      linalg.yield %533 : f32
    } -> tensor<256xf32>
    %expanded_124 = tensor.expand_shape %arg84 [[0, 1, 2]] output_shape [256, 1, 1] : tensor<256xf32> into tensor<256x1x1xf32>
    %expanded_125 = tensor.expand_shape %286 [[0, 1, 2]] output_shape [256, 1, 1] : tensor<256xf32> into tensor<256x1x1xf32>
    %287 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%283, %expanded_124 : tensor<1x256x14x14xf32>, tensor<256x1x1xf32>) outs(%251 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.subf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x256x14x14xf32>
    %288 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%287, %expanded_125 : tensor<1x256x14x14xf32>, tensor<256x1x1xf32>) outs(%251 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x256x14x14xf32>
    %expanded_126 = tensor.expand_shape %arg245 [[0, 1, 2]] output_shape [256, 1, 1] : tensor<256xf32> into tensor<256x1x1xf32>
    %289 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%288, %expanded_126 : tensor<1x256x14x14xf32>, tensor<256x1x1xf32>) outs(%251 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x256x14x14xf32>
    %expanded_127 = tensor.expand_shape %arg246 [[0, 1, 2]] output_shape [256, 1, 1] : tensor<256xf32> into tensor<256x1x1xf32>
    %290 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%289, %expanded_127 : tensor<1x256x14x14xf32>, tensor<256x1x1xf32>) outs(%251 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.addf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x256x14x14xf32>
    %291 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%290 : tensor<1x256x14x14xf32>) outs(%251 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf ugt, %in, %cst : f32
      %533 = arith.select %532, %in, %cst : f32
      linalg.yield %533 : f32
    } -> tensor<1x256x14x14xf32>
    %padded_128 = tensor.pad %291 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg321: index, %arg322: index, %arg323: index, %arg324: index):
      tensor.yield %cst : f32
    } : tensor<1x256x14x14xf32> to tensor<1x256x16x16xf32>
    %292 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_128, %arg247 : tensor<1x256x16x16xf32>, tensor<256x256x3x3xf32>) outs(%252 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %293 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg88 : tensor<256xf32>) outs(%38 : tensor<256xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.truncf %cst_2 : f64 to f32
      %533 = arith.addf %in, %532 : f32
      linalg.yield %533 : f32
    } -> tensor<256xf32>
    %294 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%293 : tensor<256xf32>) outs(%38 : tensor<256xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = math.sqrt %in : f32
      linalg.yield %532 : f32
    } -> tensor<256xf32>
    %295 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%294 : tensor<256xf32>) outs(%38 : tensor<256xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf one, %in, %cst : f32
      cf.assert %532, "unimplemented: tensor with zero element"
      %533 = arith.divf %cst_0, %in : f32
      linalg.yield %533 : f32
    } -> tensor<256xf32>
    %expanded_129 = tensor.expand_shape %arg87 [[0, 1, 2]] output_shape [256, 1, 1] : tensor<256xf32> into tensor<256x1x1xf32>
    %expanded_130 = tensor.expand_shape %295 [[0, 1, 2]] output_shape [256, 1, 1] : tensor<256xf32> into tensor<256x1x1xf32>
    %296 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%292, %expanded_129 : tensor<1x256x14x14xf32>, tensor<256x1x1xf32>) outs(%251 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.subf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x256x14x14xf32>
    %297 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%296, %expanded_130 : tensor<1x256x14x14xf32>, tensor<256x1x1xf32>) outs(%251 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x256x14x14xf32>
    %expanded_131 = tensor.expand_shape %arg248 [[0, 1, 2]] output_shape [256, 1, 1] : tensor<256xf32> into tensor<256x1x1xf32>
    %298 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%297, %expanded_131 : tensor<1x256x14x14xf32>, tensor<256x1x1xf32>) outs(%251 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x256x14x14xf32>
    %expanded_132 = tensor.expand_shape %arg249 [[0, 1, 2]] output_shape [256, 1, 1] : tensor<256xf32> into tensor<256x1x1xf32>
    %299 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%298, %expanded_132 : tensor<1x256x14x14xf32>, tensor<256x1x1xf32>) outs(%251 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.addf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x256x14x14xf32>
    %300 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%299 : tensor<1x256x14x14xf32>) outs(%251 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf ugt, %in, %cst : f32
      %533 = arith.select %532, %in, %cst : f32
      linalg.yield %533 : f32
    } -> tensor<1x256x14x14xf32>
    %301 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%300, %arg250 : tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>) outs(%263 : tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %302 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg91 : tensor<1024xf32>) outs(%265 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.truncf %cst_2 : f64 to f32
      %533 = arith.addf %in, %532 : f32
      linalg.yield %533 : f32
    } -> tensor<1024xf32>
    %303 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%302 : tensor<1024xf32>) outs(%265 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = math.sqrt %in : f32
      linalg.yield %532 : f32
    } -> tensor<1024xf32>
    %304 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%303 : tensor<1024xf32>) outs(%265 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf one, %in, %cst : f32
      cf.assert %532, "unimplemented: tensor with zero element"
      %533 = arith.divf %cst_0, %in : f32
      linalg.yield %533 : f32
    } -> tensor<1024xf32>
    %expanded_133 = tensor.expand_shape %arg90 [[0, 1, 2]] output_shape [1024, 1, 1] : tensor<1024xf32> into tensor<1024x1x1xf32>
    %expanded_134 = tensor.expand_shape %304 [[0, 1, 2]] output_shape [1024, 1, 1] : tensor<1024xf32> into tensor<1024x1x1xf32>
    %305 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%301, %expanded_133 : tensor<1x1024x14x14xf32>, tensor<1024x1x1xf32>) outs(%262 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.subf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x1024x14x14xf32>
    %306 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%305, %expanded_134 : tensor<1x1024x14x14xf32>, tensor<1024x1x1xf32>) outs(%262 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x1024x14x14xf32>
    %expanded_135 = tensor.expand_shape %arg251 [[0, 1, 2]] output_shape [1024, 1, 1] : tensor<1024xf32> into tensor<1024x1x1xf32>
    %307 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%306, %expanded_135 : tensor<1x1024x14x14xf32>, tensor<1024x1x1xf32>) outs(%262 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x1024x14x14xf32>
    %expanded_136 = tensor.expand_shape %arg252 [[0, 1, 2]] output_shape [1024, 1, 1] : tensor<1024xf32> into tensor<1024x1x1xf32>
    %308 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%307, %expanded_136 : tensor<1x1024x14x14xf32>, tensor<1024x1x1xf32>) outs(%262 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.addf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x1024x14x14xf32>
    %309 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%308, %282 : tensor<1x1024x14x14xf32>, tensor<1x1024x14x14xf32>) outs(%262 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.addf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x1024x14x14xf32>
    %310 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%309 : tensor<1x1024x14x14xf32>) outs(%262 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf ugt, %in, %cst : f32
      %533 = arith.select %532, %in, %cst : f32
      linalg.yield %533 : f32
    } -> tensor<1x1024x14x14xf32>
    %311 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%310, %arg253 : tensor<1x1024x14x14xf32>, tensor<256x1024x1x1xf32>) outs(%252 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %312 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg94 : tensor<256xf32>) outs(%38 : tensor<256xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.truncf %cst_2 : f64 to f32
      %533 = arith.addf %in, %532 : f32
      linalg.yield %533 : f32
    } -> tensor<256xf32>
    %313 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%312 : tensor<256xf32>) outs(%38 : tensor<256xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = math.sqrt %in : f32
      linalg.yield %532 : f32
    } -> tensor<256xf32>
    %314 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%313 : tensor<256xf32>) outs(%38 : tensor<256xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf one, %in, %cst : f32
      cf.assert %532, "unimplemented: tensor with zero element"
      %533 = arith.divf %cst_0, %in : f32
      linalg.yield %533 : f32
    } -> tensor<256xf32>
    %expanded_137 = tensor.expand_shape %arg93 [[0, 1, 2]] output_shape [256, 1, 1] : tensor<256xf32> into tensor<256x1x1xf32>
    %expanded_138 = tensor.expand_shape %314 [[0, 1, 2]] output_shape [256, 1, 1] : tensor<256xf32> into tensor<256x1x1xf32>
    %315 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%311, %expanded_137 : tensor<1x256x14x14xf32>, tensor<256x1x1xf32>) outs(%251 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.subf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x256x14x14xf32>
    %316 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%315, %expanded_138 : tensor<1x256x14x14xf32>, tensor<256x1x1xf32>) outs(%251 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x256x14x14xf32>
    %expanded_139 = tensor.expand_shape %arg254 [[0, 1, 2]] output_shape [256, 1, 1] : tensor<256xf32> into tensor<256x1x1xf32>
    %317 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%316, %expanded_139 : tensor<1x256x14x14xf32>, tensor<256x1x1xf32>) outs(%251 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x256x14x14xf32>
    %expanded_140 = tensor.expand_shape %arg255 [[0, 1, 2]] output_shape [256, 1, 1] : tensor<256xf32> into tensor<256x1x1xf32>
    %318 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%317, %expanded_140 : tensor<1x256x14x14xf32>, tensor<256x1x1xf32>) outs(%251 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.addf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x256x14x14xf32>
    %319 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%318 : tensor<1x256x14x14xf32>) outs(%251 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf ugt, %in, %cst : f32
      %533 = arith.select %532, %in, %cst : f32
      linalg.yield %533 : f32
    } -> tensor<1x256x14x14xf32>
    %padded_141 = tensor.pad %319 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg321: index, %arg322: index, %arg323: index, %arg324: index):
      tensor.yield %cst : f32
    } : tensor<1x256x14x14xf32> to tensor<1x256x16x16xf32>
    %320 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_141, %arg256 : tensor<1x256x16x16xf32>, tensor<256x256x3x3xf32>) outs(%252 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %321 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg97 : tensor<256xf32>) outs(%38 : tensor<256xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.truncf %cst_2 : f64 to f32
      %533 = arith.addf %in, %532 : f32
      linalg.yield %533 : f32
    } -> tensor<256xf32>
    %322 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%321 : tensor<256xf32>) outs(%38 : tensor<256xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = math.sqrt %in : f32
      linalg.yield %532 : f32
    } -> tensor<256xf32>
    %323 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%322 : tensor<256xf32>) outs(%38 : tensor<256xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf one, %in, %cst : f32
      cf.assert %532, "unimplemented: tensor with zero element"
      %533 = arith.divf %cst_0, %in : f32
      linalg.yield %533 : f32
    } -> tensor<256xf32>
    %expanded_142 = tensor.expand_shape %arg96 [[0, 1, 2]] output_shape [256, 1, 1] : tensor<256xf32> into tensor<256x1x1xf32>
    %expanded_143 = tensor.expand_shape %323 [[0, 1, 2]] output_shape [256, 1, 1] : tensor<256xf32> into tensor<256x1x1xf32>
    %324 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%320, %expanded_142 : tensor<1x256x14x14xf32>, tensor<256x1x1xf32>) outs(%251 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.subf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x256x14x14xf32>
    %325 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%324, %expanded_143 : tensor<1x256x14x14xf32>, tensor<256x1x1xf32>) outs(%251 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x256x14x14xf32>
    %expanded_144 = tensor.expand_shape %arg257 [[0, 1, 2]] output_shape [256, 1, 1] : tensor<256xf32> into tensor<256x1x1xf32>
    %326 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%325, %expanded_144 : tensor<1x256x14x14xf32>, tensor<256x1x1xf32>) outs(%251 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x256x14x14xf32>
    %expanded_145 = tensor.expand_shape %arg258 [[0, 1, 2]] output_shape [256, 1, 1] : tensor<256xf32> into tensor<256x1x1xf32>
    %327 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%326, %expanded_145 : tensor<1x256x14x14xf32>, tensor<256x1x1xf32>) outs(%251 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.addf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x256x14x14xf32>
    %328 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%327 : tensor<1x256x14x14xf32>) outs(%251 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf ugt, %in, %cst : f32
      %533 = arith.select %532, %in, %cst : f32
      linalg.yield %533 : f32
    } -> tensor<1x256x14x14xf32>
    %329 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%328, %arg259 : tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>) outs(%263 : tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %330 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg100 : tensor<1024xf32>) outs(%265 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.truncf %cst_2 : f64 to f32
      %533 = arith.addf %in, %532 : f32
      linalg.yield %533 : f32
    } -> tensor<1024xf32>
    %331 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%330 : tensor<1024xf32>) outs(%265 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = math.sqrt %in : f32
      linalg.yield %532 : f32
    } -> tensor<1024xf32>
    %332 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%331 : tensor<1024xf32>) outs(%265 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf one, %in, %cst : f32
      cf.assert %532, "unimplemented: tensor with zero element"
      %533 = arith.divf %cst_0, %in : f32
      linalg.yield %533 : f32
    } -> tensor<1024xf32>
    %expanded_146 = tensor.expand_shape %arg99 [[0, 1, 2]] output_shape [1024, 1, 1] : tensor<1024xf32> into tensor<1024x1x1xf32>
    %expanded_147 = tensor.expand_shape %332 [[0, 1, 2]] output_shape [1024, 1, 1] : tensor<1024xf32> into tensor<1024x1x1xf32>
    %333 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%329, %expanded_146 : tensor<1x1024x14x14xf32>, tensor<1024x1x1xf32>) outs(%262 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.subf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x1024x14x14xf32>
    %334 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%333, %expanded_147 : tensor<1x1024x14x14xf32>, tensor<1024x1x1xf32>) outs(%262 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x1024x14x14xf32>
    %expanded_148 = tensor.expand_shape %arg260 [[0, 1, 2]] output_shape [1024, 1, 1] : tensor<1024xf32> into tensor<1024x1x1xf32>
    %335 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%334, %expanded_148 : tensor<1x1024x14x14xf32>, tensor<1024x1x1xf32>) outs(%262 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x1024x14x14xf32>
    %expanded_149 = tensor.expand_shape %arg261 [[0, 1, 2]] output_shape [1024, 1, 1] : tensor<1024xf32> into tensor<1024x1x1xf32>
    %336 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%335, %expanded_149 : tensor<1x1024x14x14xf32>, tensor<1024x1x1xf32>) outs(%262 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.addf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x1024x14x14xf32>
    %337 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%336, %310 : tensor<1x1024x14x14xf32>, tensor<1x1024x14x14xf32>) outs(%262 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.addf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x1024x14x14xf32>
    %338 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%337 : tensor<1x1024x14x14xf32>) outs(%262 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf ugt, %in, %cst : f32
      %533 = arith.select %532, %in, %cst : f32
      linalg.yield %533 : f32
    } -> tensor<1x1024x14x14xf32>
    %339 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%338, %arg262 : tensor<1x1024x14x14xf32>, tensor<256x1024x1x1xf32>) outs(%252 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %340 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg103 : tensor<256xf32>) outs(%38 : tensor<256xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.truncf %cst_2 : f64 to f32
      %533 = arith.addf %in, %532 : f32
      linalg.yield %533 : f32
    } -> tensor<256xf32>
    %341 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%340 : tensor<256xf32>) outs(%38 : tensor<256xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = math.sqrt %in : f32
      linalg.yield %532 : f32
    } -> tensor<256xf32>
    %342 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%341 : tensor<256xf32>) outs(%38 : tensor<256xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf one, %in, %cst : f32
      cf.assert %532, "unimplemented: tensor with zero element"
      %533 = arith.divf %cst_0, %in : f32
      linalg.yield %533 : f32
    } -> tensor<256xf32>
    %expanded_150 = tensor.expand_shape %arg102 [[0, 1, 2]] output_shape [256, 1, 1] : tensor<256xf32> into tensor<256x1x1xf32>
    %expanded_151 = tensor.expand_shape %342 [[0, 1, 2]] output_shape [256, 1, 1] : tensor<256xf32> into tensor<256x1x1xf32>
    %343 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%339, %expanded_150 : tensor<1x256x14x14xf32>, tensor<256x1x1xf32>) outs(%251 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.subf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x256x14x14xf32>
    %344 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%343, %expanded_151 : tensor<1x256x14x14xf32>, tensor<256x1x1xf32>) outs(%251 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x256x14x14xf32>
    %expanded_152 = tensor.expand_shape %arg263 [[0, 1, 2]] output_shape [256, 1, 1] : tensor<256xf32> into tensor<256x1x1xf32>
    %345 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%344, %expanded_152 : tensor<1x256x14x14xf32>, tensor<256x1x1xf32>) outs(%251 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x256x14x14xf32>
    %expanded_153 = tensor.expand_shape %arg264 [[0, 1, 2]] output_shape [256, 1, 1] : tensor<256xf32> into tensor<256x1x1xf32>
    %346 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%345, %expanded_153 : tensor<1x256x14x14xf32>, tensor<256x1x1xf32>) outs(%251 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.addf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x256x14x14xf32>
    %347 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%346 : tensor<1x256x14x14xf32>) outs(%251 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf ugt, %in, %cst : f32
      %533 = arith.select %532, %in, %cst : f32
      linalg.yield %533 : f32
    } -> tensor<1x256x14x14xf32>
    %padded_154 = tensor.pad %347 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg321: index, %arg322: index, %arg323: index, %arg324: index):
      tensor.yield %cst : f32
    } : tensor<1x256x14x14xf32> to tensor<1x256x16x16xf32>
    %348 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_154, %arg265 : tensor<1x256x16x16xf32>, tensor<256x256x3x3xf32>) outs(%252 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %349 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg106 : tensor<256xf32>) outs(%38 : tensor<256xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.truncf %cst_2 : f64 to f32
      %533 = arith.addf %in, %532 : f32
      linalg.yield %533 : f32
    } -> tensor<256xf32>
    %350 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%349 : tensor<256xf32>) outs(%38 : tensor<256xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = math.sqrt %in : f32
      linalg.yield %532 : f32
    } -> tensor<256xf32>
    %351 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%350 : tensor<256xf32>) outs(%38 : tensor<256xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf one, %in, %cst : f32
      cf.assert %532, "unimplemented: tensor with zero element"
      %533 = arith.divf %cst_0, %in : f32
      linalg.yield %533 : f32
    } -> tensor<256xf32>
    %expanded_155 = tensor.expand_shape %arg105 [[0, 1, 2]] output_shape [256, 1, 1] : tensor<256xf32> into tensor<256x1x1xf32>
    %expanded_156 = tensor.expand_shape %351 [[0, 1, 2]] output_shape [256, 1, 1] : tensor<256xf32> into tensor<256x1x1xf32>
    %352 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%348, %expanded_155 : tensor<1x256x14x14xf32>, tensor<256x1x1xf32>) outs(%251 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.subf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x256x14x14xf32>
    %353 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%352, %expanded_156 : tensor<1x256x14x14xf32>, tensor<256x1x1xf32>) outs(%251 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x256x14x14xf32>
    %expanded_157 = tensor.expand_shape %arg266 [[0, 1, 2]] output_shape [256, 1, 1] : tensor<256xf32> into tensor<256x1x1xf32>
    %354 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%353, %expanded_157 : tensor<1x256x14x14xf32>, tensor<256x1x1xf32>) outs(%251 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x256x14x14xf32>
    %expanded_158 = tensor.expand_shape %arg267 [[0, 1, 2]] output_shape [256, 1, 1] : tensor<256xf32> into tensor<256x1x1xf32>
    %355 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%354, %expanded_158 : tensor<1x256x14x14xf32>, tensor<256x1x1xf32>) outs(%251 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.addf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x256x14x14xf32>
    %356 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%355 : tensor<1x256x14x14xf32>) outs(%251 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf ugt, %in, %cst : f32
      %533 = arith.select %532, %in, %cst : f32
      linalg.yield %533 : f32
    } -> tensor<1x256x14x14xf32>
    %357 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%356, %arg268 : tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>) outs(%263 : tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %358 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg109 : tensor<1024xf32>) outs(%265 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.truncf %cst_2 : f64 to f32
      %533 = arith.addf %in, %532 : f32
      linalg.yield %533 : f32
    } -> tensor<1024xf32>
    %359 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%358 : tensor<1024xf32>) outs(%265 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = math.sqrt %in : f32
      linalg.yield %532 : f32
    } -> tensor<1024xf32>
    %360 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%359 : tensor<1024xf32>) outs(%265 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf one, %in, %cst : f32
      cf.assert %532, "unimplemented: tensor with zero element"
      %533 = arith.divf %cst_0, %in : f32
      linalg.yield %533 : f32
    } -> tensor<1024xf32>
    %expanded_159 = tensor.expand_shape %arg108 [[0, 1, 2]] output_shape [1024, 1, 1] : tensor<1024xf32> into tensor<1024x1x1xf32>
    %expanded_160 = tensor.expand_shape %360 [[0, 1, 2]] output_shape [1024, 1, 1] : tensor<1024xf32> into tensor<1024x1x1xf32>
    %361 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%357, %expanded_159 : tensor<1x1024x14x14xf32>, tensor<1024x1x1xf32>) outs(%262 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.subf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x1024x14x14xf32>
    %362 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%361, %expanded_160 : tensor<1x1024x14x14xf32>, tensor<1024x1x1xf32>) outs(%262 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x1024x14x14xf32>
    %expanded_161 = tensor.expand_shape %arg269 [[0, 1, 2]] output_shape [1024, 1, 1] : tensor<1024xf32> into tensor<1024x1x1xf32>
    %363 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%362, %expanded_161 : tensor<1x1024x14x14xf32>, tensor<1024x1x1xf32>) outs(%262 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x1024x14x14xf32>
    %expanded_162 = tensor.expand_shape %arg270 [[0, 1, 2]] output_shape [1024, 1, 1] : tensor<1024xf32> into tensor<1024x1x1xf32>
    %364 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%363, %expanded_162 : tensor<1x1024x14x14xf32>, tensor<1024x1x1xf32>) outs(%262 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.addf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x1024x14x14xf32>
    %365 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%364, %338 : tensor<1x1024x14x14xf32>, tensor<1x1024x14x14xf32>) outs(%262 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.addf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x1024x14x14xf32>
    %366 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%365 : tensor<1x1024x14x14xf32>) outs(%262 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf ugt, %in, %cst : f32
      %533 = arith.select %532, %in, %cst : f32
      linalg.yield %533 : f32
    } -> tensor<1x1024x14x14xf32>
    %367 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%366, %arg271 : tensor<1x1024x14x14xf32>, tensor<256x1024x1x1xf32>) outs(%252 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %368 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg112 : tensor<256xf32>) outs(%38 : tensor<256xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.truncf %cst_2 : f64 to f32
      %533 = arith.addf %in, %532 : f32
      linalg.yield %533 : f32
    } -> tensor<256xf32>
    %369 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%368 : tensor<256xf32>) outs(%38 : tensor<256xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = math.sqrt %in : f32
      linalg.yield %532 : f32
    } -> tensor<256xf32>
    %370 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%369 : tensor<256xf32>) outs(%38 : tensor<256xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf one, %in, %cst : f32
      cf.assert %532, "unimplemented: tensor with zero element"
      %533 = arith.divf %cst_0, %in : f32
      linalg.yield %533 : f32
    } -> tensor<256xf32>
    %expanded_163 = tensor.expand_shape %arg111 [[0, 1, 2]] output_shape [256, 1, 1] : tensor<256xf32> into tensor<256x1x1xf32>
    %expanded_164 = tensor.expand_shape %370 [[0, 1, 2]] output_shape [256, 1, 1] : tensor<256xf32> into tensor<256x1x1xf32>
    %371 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%367, %expanded_163 : tensor<1x256x14x14xf32>, tensor<256x1x1xf32>) outs(%251 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.subf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x256x14x14xf32>
    %372 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%371, %expanded_164 : tensor<1x256x14x14xf32>, tensor<256x1x1xf32>) outs(%251 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x256x14x14xf32>
    %expanded_165 = tensor.expand_shape %arg272 [[0, 1, 2]] output_shape [256, 1, 1] : tensor<256xf32> into tensor<256x1x1xf32>
    %373 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%372, %expanded_165 : tensor<1x256x14x14xf32>, tensor<256x1x1xf32>) outs(%251 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x256x14x14xf32>
    %expanded_166 = tensor.expand_shape %arg273 [[0, 1, 2]] output_shape [256, 1, 1] : tensor<256xf32> into tensor<256x1x1xf32>
    %374 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%373, %expanded_166 : tensor<1x256x14x14xf32>, tensor<256x1x1xf32>) outs(%251 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.addf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x256x14x14xf32>
    %375 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%374 : tensor<1x256x14x14xf32>) outs(%251 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf ugt, %in, %cst : f32
      %533 = arith.select %532, %in, %cst : f32
      linalg.yield %533 : f32
    } -> tensor<1x256x14x14xf32>
    %padded_167 = tensor.pad %375 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg321: index, %arg322: index, %arg323: index, %arg324: index):
      tensor.yield %cst : f32
    } : tensor<1x256x14x14xf32> to tensor<1x256x16x16xf32>
    %376 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_167, %arg274 : tensor<1x256x16x16xf32>, tensor<256x256x3x3xf32>) outs(%252 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %377 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg115 : tensor<256xf32>) outs(%38 : tensor<256xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.truncf %cst_2 : f64 to f32
      %533 = arith.addf %in, %532 : f32
      linalg.yield %533 : f32
    } -> tensor<256xf32>
    %378 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%377 : tensor<256xf32>) outs(%38 : tensor<256xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = math.sqrt %in : f32
      linalg.yield %532 : f32
    } -> tensor<256xf32>
    %379 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%378 : tensor<256xf32>) outs(%38 : tensor<256xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf one, %in, %cst : f32
      cf.assert %532, "unimplemented: tensor with zero element"
      %533 = arith.divf %cst_0, %in : f32
      linalg.yield %533 : f32
    } -> tensor<256xf32>
    %expanded_168 = tensor.expand_shape %arg114 [[0, 1, 2]] output_shape [256, 1, 1] : tensor<256xf32> into tensor<256x1x1xf32>
    %expanded_169 = tensor.expand_shape %379 [[0, 1, 2]] output_shape [256, 1, 1] : tensor<256xf32> into tensor<256x1x1xf32>
    %380 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%376, %expanded_168 : tensor<1x256x14x14xf32>, tensor<256x1x1xf32>) outs(%251 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.subf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x256x14x14xf32>
    %381 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%380, %expanded_169 : tensor<1x256x14x14xf32>, tensor<256x1x1xf32>) outs(%251 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x256x14x14xf32>
    %expanded_170 = tensor.expand_shape %arg275 [[0, 1, 2]] output_shape [256, 1, 1] : tensor<256xf32> into tensor<256x1x1xf32>
    %382 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%381, %expanded_170 : tensor<1x256x14x14xf32>, tensor<256x1x1xf32>) outs(%251 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x256x14x14xf32>
    %expanded_171 = tensor.expand_shape %arg276 [[0, 1, 2]] output_shape [256, 1, 1] : tensor<256xf32> into tensor<256x1x1xf32>
    %383 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%382, %expanded_171 : tensor<1x256x14x14xf32>, tensor<256x1x1xf32>) outs(%251 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.addf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x256x14x14xf32>
    %384 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%383 : tensor<1x256x14x14xf32>) outs(%251 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf ugt, %in, %cst : f32
      %533 = arith.select %532, %in, %cst : f32
      linalg.yield %533 : f32
    } -> tensor<1x256x14x14xf32>
    %385 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%384, %arg277 : tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>) outs(%263 : tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %386 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg118 : tensor<1024xf32>) outs(%265 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.truncf %cst_2 : f64 to f32
      %533 = arith.addf %in, %532 : f32
      linalg.yield %533 : f32
    } -> tensor<1024xf32>
    %387 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%386 : tensor<1024xf32>) outs(%265 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = math.sqrt %in : f32
      linalg.yield %532 : f32
    } -> tensor<1024xf32>
    %388 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%387 : tensor<1024xf32>) outs(%265 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf one, %in, %cst : f32
      cf.assert %532, "unimplemented: tensor with zero element"
      %533 = arith.divf %cst_0, %in : f32
      linalg.yield %533 : f32
    } -> tensor<1024xf32>
    %expanded_172 = tensor.expand_shape %arg117 [[0, 1, 2]] output_shape [1024, 1, 1] : tensor<1024xf32> into tensor<1024x1x1xf32>
    %expanded_173 = tensor.expand_shape %388 [[0, 1, 2]] output_shape [1024, 1, 1] : tensor<1024xf32> into tensor<1024x1x1xf32>
    %389 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%385, %expanded_172 : tensor<1x1024x14x14xf32>, tensor<1024x1x1xf32>) outs(%262 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.subf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x1024x14x14xf32>
    %390 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%389, %expanded_173 : tensor<1x1024x14x14xf32>, tensor<1024x1x1xf32>) outs(%262 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x1024x14x14xf32>
    %expanded_174 = tensor.expand_shape %arg278 [[0, 1, 2]] output_shape [1024, 1, 1] : tensor<1024xf32> into tensor<1024x1x1xf32>
    %391 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%390, %expanded_174 : tensor<1x1024x14x14xf32>, tensor<1024x1x1xf32>) outs(%262 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x1024x14x14xf32>
    %expanded_175 = tensor.expand_shape %arg279 [[0, 1, 2]] output_shape [1024, 1, 1] : tensor<1024xf32> into tensor<1024x1x1xf32>
    %392 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%391, %expanded_175 : tensor<1x1024x14x14xf32>, tensor<1024x1x1xf32>) outs(%262 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.addf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x1024x14x14xf32>
    %393 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%392, %366 : tensor<1x1024x14x14xf32>, tensor<1x1024x14x14xf32>) outs(%262 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.addf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x1024x14x14xf32>
    %394 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%393 : tensor<1x1024x14x14xf32>) outs(%262 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf ugt, %in, %cst : f32
      %533 = arith.select %532, %in, %cst : f32
      linalg.yield %533 : f32
    } -> tensor<1x1024x14x14xf32>
    %395 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%394, %arg280 : tensor<1x1024x14x14xf32>, tensor<256x1024x1x1xf32>) outs(%252 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %396 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg121 : tensor<256xf32>) outs(%38 : tensor<256xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.truncf %cst_2 : f64 to f32
      %533 = arith.addf %in, %532 : f32
      linalg.yield %533 : f32
    } -> tensor<256xf32>
    %397 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%396 : tensor<256xf32>) outs(%38 : tensor<256xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = math.sqrt %in : f32
      linalg.yield %532 : f32
    } -> tensor<256xf32>
    %398 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%397 : tensor<256xf32>) outs(%38 : tensor<256xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf one, %in, %cst : f32
      cf.assert %532, "unimplemented: tensor with zero element"
      %533 = arith.divf %cst_0, %in : f32
      linalg.yield %533 : f32
    } -> tensor<256xf32>
    %expanded_176 = tensor.expand_shape %arg120 [[0, 1, 2]] output_shape [256, 1, 1] : tensor<256xf32> into tensor<256x1x1xf32>
    %expanded_177 = tensor.expand_shape %398 [[0, 1, 2]] output_shape [256, 1, 1] : tensor<256xf32> into tensor<256x1x1xf32>
    %399 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%395, %expanded_176 : tensor<1x256x14x14xf32>, tensor<256x1x1xf32>) outs(%251 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.subf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x256x14x14xf32>
    %400 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%399, %expanded_177 : tensor<1x256x14x14xf32>, tensor<256x1x1xf32>) outs(%251 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x256x14x14xf32>
    %expanded_178 = tensor.expand_shape %arg281 [[0, 1, 2]] output_shape [256, 1, 1] : tensor<256xf32> into tensor<256x1x1xf32>
    %401 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%400, %expanded_178 : tensor<1x256x14x14xf32>, tensor<256x1x1xf32>) outs(%251 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x256x14x14xf32>
    %expanded_179 = tensor.expand_shape %arg282 [[0, 1, 2]] output_shape [256, 1, 1] : tensor<256xf32> into tensor<256x1x1xf32>
    %402 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%401, %expanded_179 : tensor<1x256x14x14xf32>, tensor<256x1x1xf32>) outs(%251 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.addf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x256x14x14xf32>
    %403 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%402 : tensor<1x256x14x14xf32>) outs(%251 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf ugt, %in, %cst : f32
      %533 = arith.select %532, %in, %cst : f32
      linalg.yield %533 : f32
    } -> tensor<1x256x14x14xf32>
    %padded_180 = tensor.pad %403 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg321: index, %arg322: index, %arg323: index, %arg324: index):
      tensor.yield %cst : f32
    } : tensor<1x256x14x14xf32> to tensor<1x256x16x16xf32>
    %404 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_180, %arg283 : tensor<1x256x16x16xf32>, tensor<256x256x3x3xf32>) outs(%252 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %405 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg124 : tensor<256xf32>) outs(%38 : tensor<256xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.truncf %cst_2 : f64 to f32
      %533 = arith.addf %in, %532 : f32
      linalg.yield %533 : f32
    } -> tensor<256xf32>
    %406 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%405 : tensor<256xf32>) outs(%38 : tensor<256xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = math.sqrt %in : f32
      linalg.yield %532 : f32
    } -> tensor<256xf32>
    %407 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%406 : tensor<256xf32>) outs(%38 : tensor<256xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf one, %in, %cst : f32
      cf.assert %532, "unimplemented: tensor with zero element"
      %533 = arith.divf %cst_0, %in : f32
      linalg.yield %533 : f32
    } -> tensor<256xf32>
    %expanded_181 = tensor.expand_shape %arg123 [[0, 1, 2]] output_shape [256, 1, 1] : tensor<256xf32> into tensor<256x1x1xf32>
    %expanded_182 = tensor.expand_shape %407 [[0, 1, 2]] output_shape [256, 1, 1] : tensor<256xf32> into tensor<256x1x1xf32>
    %408 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%404, %expanded_181 : tensor<1x256x14x14xf32>, tensor<256x1x1xf32>) outs(%251 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.subf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x256x14x14xf32>
    %409 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%408, %expanded_182 : tensor<1x256x14x14xf32>, tensor<256x1x1xf32>) outs(%251 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x256x14x14xf32>
    %expanded_183 = tensor.expand_shape %arg284 [[0, 1, 2]] output_shape [256, 1, 1] : tensor<256xf32> into tensor<256x1x1xf32>
    %410 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%409, %expanded_183 : tensor<1x256x14x14xf32>, tensor<256x1x1xf32>) outs(%251 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x256x14x14xf32>
    %expanded_184 = tensor.expand_shape %arg285 [[0, 1, 2]] output_shape [256, 1, 1] : tensor<256xf32> into tensor<256x1x1xf32>
    %411 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%410, %expanded_184 : tensor<1x256x14x14xf32>, tensor<256x1x1xf32>) outs(%251 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.addf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x256x14x14xf32>
    %412 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%411 : tensor<1x256x14x14xf32>) outs(%251 : tensor<1x256x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf ugt, %in, %cst : f32
      %533 = arith.select %532, %in, %cst : f32
      linalg.yield %533 : f32
    } -> tensor<1x256x14x14xf32>
    %413 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%412, %arg286 : tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>) outs(%263 : tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %414 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg127 : tensor<1024xf32>) outs(%265 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.truncf %cst_2 : f64 to f32
      %533 = arith.addf %in, %532 : f32
      linalg.yield %533 : f32
    } -> tensor<1024xf32>
    %415 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%414 : tensor<1024xf32>) outs(%265 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = math.sqrt %in : f32
      linalg.yield %532 : f32
    } -> tensor<1024xf32>
    %416 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%415 : tensor<1024xf32>) outs(%265 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf one, %in, %cst : f32
      cf.assert %532, "unimplemented: tensor with zero element"
      %533 = arith.divf %cst_0, %in : f32
      linalg.yield %533 : f32
    } -> tensor<1024xf32>
    %expanded_185 = tensor.expand_shape %arg126 [[0, 1, 2]] output_shape [1024, 1, 1] : tensor<1024xf32> into tensor<1024x1x1xf32>
    %expanded_186 = tensor.expand_shape %416 [[0, 1, 2]] output_shape [1024, 1, 1] : tensor<1024xf32> into tensor<1024x1x1xf32>
    %417 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%413, %expanded_185 : tensor<1x1024x14x14xf32>, tensor<1024x1x1xf32>) outs(%262 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.subf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x1024x14x14xf32>
    %418 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%417, %expanded_186 : tensor<1x1024x14x14xf32>, tensor<1024x1x1xf32>) outs(%262 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x1024x14x14xf32>
    %expanded_187 = tensor.expand_shape %arg287 [[0, 1, 2]] output_shape [1024, 1, 1] : tensor<1024xf32> into tensor<1024x1x1xf32>
    %419 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%418, %expanded_187 : tensor<1x1024x14x14xf32>, tensor<1024x1x1xf32>) outs(%262 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x1024x14x14xf32>
    %expanded_188 = tensor.expand_shape %arg288 [[0, 1, 2]] output_shape [1024, 1, 1] : tensor<1024xf32> into tensor<1024x1x1xf32>
    %420 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%419, %expanded_188 : tensor<1x1024x14x14xf32>, tensor<1024x1x1xf32>) outs(%262 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.addf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x1024x14x14xf32>
    %421 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%420, %394 : tensor<1x1024x14x14xf32>, tensor<1x1024x14x14xf32>) outs(%262 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.addf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x1024x14x14xf32>
    %422 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%421 : tensor<1x1024x14x14xf32>) outs(%262 : tensor<1x1024x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf ugt, %in, %cst : f32
      %533 = arith.select %532, %in, %cst : f32
      linalg.yield %533 : f32
    } -> tensor<1x1024x14x14xf32>
    %423 = tensor.empty() : tensor<1x512x14x14xf32>
    %424 = linalg.fill ins(%cst : f32) outs(%423 : tensor<1x512x14x14xf32>) -> tensor<1x512x14x14xf32>
    %425 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%422, %arg289 : tensor<1x1024x14x14xf32>, tensor<512x1024x1x1xf32>) outs(%424 : tensor<1x512x14x14xf32>) -> tensor<1x512x14x14xf32>
    %426 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg130 : tensor<512xf32>) outs(%138 : tensor<512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.truncf %cst_2 : f64 to f32
      %533 = arith.addf %in, %532 : f32
      linalg.yield %533 : f32
    } -> tensor<512xf32>
    %427 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%426 : tensor<512xf32>) outs(%138 : tensor<512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = math.sqrt %in : f32
      linalg.yield %532 : f32
    } -> tensor<512xf32>
    %428 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%427 : tensor<512xf32>) outs(%138 : tensor<512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf one, %in, %cst : f32
      cf.assert %532, "unimplemented: tensor with zero element"
      %533 = arith.divf %cst_0, %in : f32
      linalg.yield %533 : f32
    } -> tensor<512xf32>
    %expanded_189 = tensor.expand_shape %arg129 [[0, 1, 2]] output_shape [512, 1, 1] : tensor<512xf32> into tensor<512x1x1xf32>
    %expanded_190 = tensor.expand_shape %428 [[0, 1, 2]] output_shape [512, 1, 1] : tensor<512xf32> into tensor<512x1x1xf32>
    %429 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%425, %expanded_189 : tensor<1x512x14x14xf32>, tensor<512x1x1xf32>) outs(%423 : tensor<1x512x14x14xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.subf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x512x14x14xf32>
    %430 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%429, %expanded_190 : tensor<1x512x14x14xf32>, tensor<512x1x1xf32>) outs(%423 : tensor<1x512x14x14xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x512x14x14xf32>
    %expanded_191 = tensor.expand_shape %arg290 [[0, 1, 2]] output_shape [512, 1, 1] : tensor<512xf32> into tensor<512x1x1xf32>
    %431 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%430, %expanded_191 : tensor<1x512x14x14xf32>, tensor<512x1x1xf32>) outs(%423 : tensor<1x512x14x14xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x512x14x14xf32>
    %expanded_192 = tensor.expand_shape %arg291 [[0, 1, 2]] output_shape [512, 1, 1] : tensor<512xf32> into tensor<512x1x1xf32>
    %432 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%431, %expanded_192 : tensor<1x512x14x14xf32>, tensor<512x1x1xf32>) outs(%423 : tensor<1x512x14x14xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.addf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x512x14x14xf32>
    %433 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%432 : tensor<1x512x14x14xf32>) outs(%423 : tensor<1x512x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf ugt, %in, %cst : f32
      %533 = arith.select %532, %in, %cst : f32
      linalg.yield %533 : f32
    } -> tensor<1x512x14x14xf32>
    %padded_193 = tensor.pad %433 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg321: index, %arg322: index, %arg323: index, %arg324: index):
      tensor.yield %cst : f32
    } : tensor<1x512x14x14xf32> to tensor<1x512x16x16xf32>
    %434 = tensor.empty() : tensor<1x512x7x7xf32>
    %435 = linalg.fill ins(%cst : f32) outs(%434 : tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %436 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%padded_193, %arg292 : tensor<1x512x16x16xf32>, tensor<512x512x3x3xf32>) outs(%435 : tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %437 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg133 : tensor<512xf32>) outs(%138 : tensor<512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.truncf %cst_2 : f64 to f32
      %533 = arith.addf %in, %532 : f32
      linalg.yield %533 : f32
    } -> tensor<512xf32>
    %438 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%437 : tensor<512xf32>) outs(%138 : tensor<512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = math.sqrt %in : f32
      linalg.yield %532 : f32
    } -> tensor<512xf32>
    %439 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%438 : tensor<512xf32>) outs(%138 : tensor<512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf one, %in, %cst : f32
      cf.assert %532, "unimplemented: tensor with zero element"
      %533 = arith.divf %cst_0, %in : f32
      linalg.yield %533 : f32
    } -> tensor<512xf32>
    %expanded_194 = tensor.expand_shape %arg132 [[0, 1, 2]] output_shape [512, 1, 1] : tensor<512xf32> into tensor<512x1x1xf32>
    %expanded_195 = tensor.expand_shape %439 [[0, 1, 2]] output_shape [512, 1, 1] : tensor<512xf32> into tensor<512x1x1xf32>
    %440 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%436, %expanded_194 : tensor<1x512x7x7xf32>, tensor<512x1x1xf32>) outs(%434 : tensor<1x512x7x7xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.subf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x512x7x7xf32>
    %441 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%440, %expanded_195 : tensor<1x512x7x7xf32>, tensor<512x1x1xf32>) outs(%434 : tensor<1x512x7x7xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x512x7x7xf32>
    %expanded_196 = tensor.expand_shape %arg293 [[0, 1, 2]] output_shape [512, 1, 1] : tensor<512xf32> into tensor<512x1x1xf32>
    %442 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%441, %expanded_196 : tensor<1x512x7x7xf32>, tensor<512x1x1xf32>) outs(%434 : tensor<1x512x7x7xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x512x7x7xf32>
    %expanded_197 = tensor.expand_shape %arg294 [[0, 1, 2]] output_shape [512, 1, 1] : tensor<512xf32> into tensor<512x1x1xf32>
    %443 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%442, %expanded_197 : tensor<1x512x7x7xf32>, tensor<512x1x1xf32>) outs(%434 : tensor<1x512x7x7xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.addf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x512x7x7xf32>
    %444 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%443 : tensor<1x512x7x7xf32>) outs(%434 : tensor<1x512x7x7xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf ugt, %in, %cst : f32
      %533 = arith.select %532, %in, %cst : f32
      linalg.yield %533 : f32
    } -> tensor<1x512x7x7xf32>
    %445 = tensor.empty() : tensor<1x2048x7x7xf32>
    %446 = linalg.fill ins(%cst : f32) outs(%445 : tensor<1x2048x7x7xf32>) -> tensor<1x2048x7x7xf32>
    %447 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%444, %arg295 : tensor<1x512x7x7xf32>, tensor<2048x512x1x1xf32>) outs(%446 : tensor<1x2048x7x7xf32>) -> tensor<1x2048x7x7xf32>
    %448 = tensor.empty() : tensor<2048xf32>
    %449 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg136 : tensor<2048xf32>) outs(%448 : tensor<2048xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.truncf %cst_2 : f64 to f32
      %533 = arith.addf %in, %532 : f32
      linalg.yield %533 : f32
    } -> tensor<2048xf32>
    %450 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%449 : tensor<2048xf32>) outs(%448 : tensor<2048xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = math.sqrt %in : f32
      linalg.yield %532 : f32
    } -> tensor<2048xf32>
    %451 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%450 : tensor<2048xf32>) outs(%448 : tensor<2048xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf one, %in, %cst : f32
      cf.assert %532, "unimplemented: tensor with zero element"
      %533 = arith.divf %cst_0, %in : f32
      linalg.yield %533 : f32
    } -> tensor<2048xf32>
    %expanded_198 = tensor.expand_shape %arg135 [[0, 1, 2]] output_shape [2048, 1, 1] : tensor<2048xf32> into tensor<2048x1x1xf32>
    %expanded_199 = tensor.expand_shape %451 [[0, 1, 2]] output_shape [2048, 1, 1] : tensor<2048xf32> into tensor<2048x1x1xf32>
    %452 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%447, %expanded_198 : tensor<1x2048x7x7xf32>, tensor<2048x1x1xf32>) outs(%445 : tensor<1x2048x7x7xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.subf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x2048x7x7xf32>
    %453 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%452, %expanded_199 : tensor<1x2048x7x7xf32>, tensor<2048x1x1xf32>) outs(%445 : tensor<1x2048x7x7xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x2048x7x7xf32>
    %expanded_200 = tensor.expand_shape %arg296 [[0, 1, 2]] output_shape [2048, 1, 1] : tensor<2048xf32> into tensor<2048x1x1xf32>
    %454 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%453, %expanded_200 : tensor<1x2048x7x7xf32>, tensor<2048x1x1xf32>) outs(%445 : tensor<1x2048x7x7xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x2048x7x7xf32>
    %expanded_201 = tensor.expand_shape %arg297 [[0, 1, 2]] output_shape [2048, 1, 1] : tensor<2048xf32> into tensor<2048x1x1xf32>
    %455 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%454, %expanded_201 : tensor<1x2048x7x7xf32>, tensor<2048x1x1xf32>) outs(%445 : tensor<1x2048x7x7xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.addf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x2048x7x7xf32>
    %456 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%422, %arg298 : tensor<1x1024x14x14xf32>, tensor<2048x1024x1x1xf32>) outs(%446 : tensor<1x2048x7x7xf32>) -> tensor<1x2048x7x7xf32>
    %457 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg139 : tensor<2048xf32>) outs(%448 : tensor<2048xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.truncf %cst_2 : f64 to f32
      %533 = arith.addf %in, %532 : f32
      linalg.yield %533 : f32
    } -> tensor<2048xf32>
    %458 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%457 : tensor<2048xf32>) outs(%448 : tensor<2048xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = math.sqrt %in : f32
      linalg.yield %532 : f32
    } -> tensor<2048xf32>
    %459 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%458 : tensor<2048xf32>) outs(%448 : tensor<2048xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf one, %in, %cst : f32
      cf.assert %532, "unimplemented: tensor with zero element"
      %533 = arith.divf %cst_0, %in : f32
      linalg.yield %533 : f32
    } -> tensor<2048xf32>
    %expanded_202 = tensor.expand_shape %arg138 [[0, 1, 2]] output_shape [2048, 1, 1] : tensor<2048xf32> into tensor<2048x1x1xf32>
    %expanded_203 = tensor.expand_shape %459 [[0, 1, 2]] output_shape [2048, 1, 1] : tensor<2048xf32> into tensor<2048x1x1xf32>
    %460 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%456, %expanded_202 : tensor<1x2048x7x7xf32>, tensor<2048x1x1xf32>) outs(%445 : tensor<1x2048x7x7xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.subf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x2048x7x7xf32>
    %461 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%460, %expanded_203 : tensor<1x2048x7x7xf32>, tensor<2048x1x1xf32>) outs(%445 : tensor<1x2048x7x7xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x2048x7x7xf32>
    %expanded_204 = tensor.expand_shape %arg299 [[0, 1, 2]] output_shape [2048, 1, 1] : tensor<2048xf32> into tensor<2048x1x1xf32>
    %462 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%461, %expanded_204 : tensor<1x2048x7x7xf32>, tensor<2048x1x1xf32>) outs(%445 : tensor<1x2048x7x7xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x2048x7x7xf32>
    %expanded_205 = tensor.expand_shape %arg300 [[0, 1, 2]] output_shape [2048, 1, 1] : tensor<2048xf32> into tensor<2048x1x1xf32>
    %463 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%462, %expanded_205 : tensor<1x2048x7x7xf32>, tensor<2048x1x1xf32>) outs(%445 : tensor<1x2048x7x7xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.addf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x2048x7x7xf32>
    %464 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%455, %463 : tensor<1x2048x7x7xf32>, tensor<1x2048x7x7xf32>) outs(%445 : tensor<1x2048x7x7xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.addf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x2048x7x7xf32>
    %465 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%464 : tensor<1x2048x7x7xf32>) outs(%445 : tensor<1x2048x7x7xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf ugt, %in, %cst : f32
      %533 = arith.select %532, %in, %cst : f32
      linalg.yield %533 : f32
    } -> tensor<1x2048x7x7xf32>
    %466 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%465, %arg301 : tensor<1x2048x7x7xf32>, tensor<512x2048x1x1xf32>) outs(%435 : tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %467 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg142 : tensor<512xf32>) outs(%138 : tensor<512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.truncf %cst_2 : f64 to f32
      %533 = arith.addf %in, %532 : f32
      linalg.yield %533 : f32
    } -> tensor<512xf32>
    %468 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%467 : tensor<512xf32>) outs(%138 : tensor<512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = math.sqrt %in : f32
      linalg.yield %532 : f32
    } -> tensor<512xf32>
    %469 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%468 : tensor<512xf32>) outs(%138 : tensor<512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf one, %in, %cst : f32
      cf.assert %532, "unimplemented: tensor with zero element"
      %533 = arith.divf %cst_0, %in : f32
      linalg.yield %533 : f32
    } -> tensor<512xf32>
    %expanded_206 = tensor.expand_shape %arg141 [[0, 1, 2]] output_shape [512, 1, 1] : tensor<512xf32> into tensor<512x1x1xf32>
    %expanded_207 = tensor.expand_shape %469 [[0, 1, 2]] output_shape [512, 1, 1] : tensor<512xf32> into tensor<512x1x1xf32>
    %470 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%466, %expanded_206 : tensor<1x512x7x7xf32>, tensor<512x1x1xf32>) outs(%434 : tensor<1x512x7x7xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.subf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x512x7x7xf32>
    %471 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%470, %expanded_207 : tensor<1x512x7x7xf32>, tensor<512x1x1xf32>) outs(%434 : tensor<1x512x7x7xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x512x7x7xf32>
    %expanded_208 = tensor.expand_shape %arg302 [[0, 1, 2]] output_shape [512, 1, 1] : tensor<512xf32> into tensor<512x1x1xf32>
    %472 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%471, %expanded_208 : tensor<1x512x7x7xf32>, tensor<512x1x1xf32>) outs(%434 : tensor<1x512x7x7xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x512x7x7xf32>
    %expanded_209 = tensor.expand_shape %arg303 [[0, 1, 2]] output_shape [512, 1, 1] : tensor<512xf32> into tensor<512x1x1xf32>
    %473 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%472, %expanded_209 : tensor<1x512x7x7xf32>, tensor<512x1x1xf32>) outs(%434 : tensor<1x512x7x7xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.addf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x512x7x7xf32>
    %474 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%473 : tensor<1x512x7x7xf32>) outs(%434 : tensor<1x512x7x7xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf ugt, %in, %cst : f32
      %533 = arith.select %532, %in, %cst : f32
      linalg.yield %533 : f32
    } -> tensor<1x512x7x7xf32>
    %padded_210 = tensor.pad %474 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg321: index, %arg322: index, %arg323: index, %arg324: index):
      tensor.yield %cst : f32
    } : tensor<1x512x7x7xf32> to tensor<1x512x9x9xf32>
    %475 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_210, %arg304 : tensor<1x512x9x9xf32>, tensor<512x512x3x3xf32>) outs(%435 : tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %476 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg145 : tensor<512xf32>) outs(%138 : tensor<512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.truncf %cst_2 : f64 to f32
      %533 = arith.addf %in, %532 : f32
      linalg.yield %533 : f32
    } -> tensor<512xf32>
    %477 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%476 : tensor<512xf32>) outs(%138 : tensor<512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = math.sqrt %in : f32
      linalg.yield %532 : f32
    } -> tensor<512xf32>
    %478 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%477 : tensor<512xf32>) outs(%138 : tensor<512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf one, %in, %cst : f32
      cf.assert %532, "unimplemented: tensor with zero element"
      %533 = arith.divf %cst_0, %in : f32
      linalg.yield %533 : f32
    } -> tensor<512xf32>
    %expanded_211 = tensor.expand_shape %arg144 [[0, 1, 2]] output_shape [512, 1, 1] : tensor<512xf32> into tensor<512x1x1xf32>
    %expanded_212 = tensor.expand_shape %478 [[0, 1, 2]] output_shape [512, 1, 1] : tensor<512xf32> into tensor<512x1x1xf32>
    %479 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%475, %expanded_211 : tensor<1x512x7x7xf32>, tensor<512x1x1xf32>) outs(%434 : tensor<1x512x7x7xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.subf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x512x7x7xf32>
    %480 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%479, %expanded_212 : tensor<1x512x7x7xf32>, tensor<512x1x1xf32>) outs(%434 : tensor<1x512x7x7xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x512x7x7xf32>
    %expanded_213 = tensor.expand_shape %arg305 [[0, 1, 2]] output_shape [512, 1, 1] : tensor<512xf32> into tensor<512x1x1xf32>
    %481 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%480, %expanded_213 : tensor<1x512x7x7xf32>, tensor<512x1x1xf32>) outs(%434 : tensor<1x512x7x7xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x512x7x7xf32>
    %expanded_214 = tensor.expand_shape %arg306 [[0, 1, 2]] output_shape [512, 1, 1] : tensor<512xf32> into tensor<512x1x1xf32>
    %482 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%481, %expanded_214 : tensor<1x512x7x7xf32>, tensor<512x1x1xf32>) outs(%434 : tensor<1x512x7x7xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.addf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x512x7x7xf32>
    %483 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%482 : tensor<1x512x7x7xf32>) outs(%434 : tensor<1x512x7x7xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf ugt, %in, %cst : f32
      %533 = arith.select %532, %in, %cst : f32
      linalg.yield %533 : f32
    } -> tensor<1x512x7x7xf32>
    %484 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%483, %arg307 : tensor<1x512x7x7xf32>, tensor<2048x512x1x1xf32>) outs(%446 : tensor<1x2048x7x7xf32>) -> tensor<1x2048x7x7xf32>
    %485 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg148 : tensor<2048xf32>) outs(%448 : tensor<2048xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.truncf %cst_2 : f64 to f32
      %533 = arith.addf %in, %532 : f32
      linalg.yield %533 : f32
    } -> tensor<2048xf32>
    %486 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%485 : tensor<2048xf32>) outs(%448 : tensor<2048xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = math.sqrt %in : f32
      linalg.yield %532 : f32
    } -> tensor<2048xf32>
    %487 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%486 : tensor<2048xf32>) outs(%448 : tensor<2048xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf one, %in, %cst : f32
      cf.assert %532, "unimplemented: tensor with zero element"
      %533 = arith.divf %cst_0, %in : f32
      linalg.yield %533 : f32
    } -> tensor<2048xf32>
    %expanded_215 = tensor.expand_shape %arg147 [[0, 1, 2]] output_shape [2048, 1, 1] : tensor<2048xf32> into tensor<2048x1x1xf32>
    %expanded_216 = tensor.expand_shape %487 [[0, 1, 2]] output_shape [2048, 1, 1] : tensor<2048xf32> into tensor<2048x1x1xf32>
    %488 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%484, %expanded_215 : tensor<1x2048x7x7xf32>, tensor<2048x1x1xf32>) outs(%445 : tensor<1x2048x7x7xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.subf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x2048x7x7xf32>
    %489 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%488, %expanded_216 : tensor<1x2048x7x7xf32>, tensor<2048x1x1xf32>) outs(%445 : tensor<1x2048x7x7xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x2048x7x7xf32>
    %expanded_217 = tensor.expand_shape %arg308 [[0, 1, 2]] output_shape [2048, 1, 1] : tensor<2048xf32> into tensor<2048x1x1xf32>
    %490 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%489, %expanded_217 : tensor<1x2048x7x7xf32>, tensor<2048x1x1xf32>) outs(%445 : tensor<1x2048x7x7xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x2048x7x7xf32>
    %expanded_218 = tensor.expand_shape %arg309 [[0, 1, 2]] output_shape [2048, 1, 1] : tensor<2048xf32> into tensor<2048x1x1xf32>
    %491 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%490, %expanded_218 : tensor<1x2048x7x7xf32>, tensor<2048x1x1xf32>) outs(%445 : tensor<1x2048x7x7xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.addf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x2048x7x7xf32>
    %492 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%491, %465 : tensor<1x2048x7x7xf32>, tensor<1x2048x7x7xf32>) outs(%445 : tensor<1x2048x7x7xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.addf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x2048x7x7xf32>
    %493 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%492 : tensor<1x2048x7x7xf32>) outs(%445 : tensor<1x2048x7x7xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf ugt, %in, %cst : f32
      %533 = arith.select %532, %in, %cst : f32
      linalg.yield %533 : f32
    } -> tensor<1x2048x7x7xf32>
    %494 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%493, %arg310 : tensor<1x2048x7x7xf32>, tensor<512x2048x1x1xf32>) outs(%435 : tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %495 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg151 : tensor<512xf32>) outs(%138 : tensor<512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.truncf %cst_2 : f64 to f32
      %533 = arith.addf %in, %532 : f32
      linalg.yield %533 : f32
    } -> tensor<512xf32>
    %496 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%495 : tensor<512xf32>) outs(%138 : tensor<512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = math.sqrt %in : f32
      linalg.yield %532 : f32
    } -> tensor<512xf32>
    %497 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%496 : tensor<512xf32>) outs(%138 : tensor<512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf one, %in, %cst : f32
      cf.assert %532, "unimplemented: tensor with zero element"
      %533 = arith.divf %cst_0, %in : f32
      linalg.yield %533 : f32
    } -> tensor<512xf32>
    %expanded_219 = tensor.expand_shape %arg150 [[0, 1, 2]] output_shape [512, 1, 1] : tensor<512xf32> into tensor<512x1x1xf32>
    %expanded_220 = tensor.expand_shape %497 [[0, 1, 2]] output_shape [512, 1, 1] : tensor<512xf32> into tensor<512x1x1xf32>
    %498 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%494, %expanded_219 : tensor<1x512x7x7xf32>, tensor<512x1x1xf32>) outs(%434 : tensor<1x512x7x7xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.subf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x512x7x7xf32>
    %499 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%498, %expanded_220 : tensor<1x512x7x7xf32>, tensor<512x1x1xf32>) outs(%434 : tensor<1x512x7x7xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x512x7x7xf32>
    %expanded_221 = tensor.expand_shape %arg311 [[0, 1, 2]] output_shape [512, 1, 1] : tensor<512xf32> into tensor<512x1x1xf32>
    %500 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%499, %expanded_221 : tensor<1x512x7x7xf32>, tensor<512x1x1xf32>) outs(%434 : tensor<1x512x7x7xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x512x7x7xf32>
    %expanded_222 = tensor.expand_shape %arg312 [[0, 1, 2]] output_shape [512, 1, 1] : tensor<512xf32> into tensor<512x1x1xf32>
    %501 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%500, %expanded_222 : tensor<1x512x7x7xf32>, tensor<512x1x1xf32>) outs(%434 : tensor<1x512x7x7xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.addf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x512x7x7xf32>
    %502 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%501 : tensor<1x512x7x7xf32>) outs(%434 : tensor<1x512x7x7xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf ugt, %in, %cst : f32
      %533 = arith.select %532, %in, %cst : f32
      linalg.yield %533 : f32
    } -> tensor<1x512x7x7xf32>
    %padded_223 = tensor.pad %502 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg321: index, %arg322: index, %arg323: index, %arg324: index):
      tensor.yield %cst : f32
    } : tensor<1x512x7x7xf32> to tensor<1x512x9x9xf32>
    %503 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_223, %arg313 : tensor<1x512x9x9xf32>, tensor<512x512x3x3xf32>) outs(%435 : tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %504 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg154 : tensor<512xf32>) outs(%138 : tensor<512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.truncf %cst_2 : f64 to f32
      %533 = arith.addf %in, %532 : f32
      linalg.yield %533 : f32
    } -> tensor<512xf32>
    %505 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%504 : tensor<512xf32>) outs(%138 : tensor<512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = math.sqrt %in : f32
      linalg.yield %532 : f32
    } -> tensor<512xf32>
    %506 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%505 : tensor<512xf32>) outs(%138 : tensor<512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf one, %in, %cst : f32
      cf.assert %532, "unimplemented: tensor with zero element"
      %533 = arith.divf %cst_0, %in : f32
      linalg.yield %533 : f32
    } -> tensor<512xf32>
    %expanded_224 = tensor.expand_shape %arg153 [[0, 1, 2]] output_shape [512, 1, 1] : tensor<512xf32> into tensor<512x1x1xf32>
    %expanded_225 = tensor.expand_shape %506 [[0, 1, 2]] output_shape [512, 1, 1] : tensor<512xf32> into tensor<512x1x1xf32>
    %507 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%503, %expanded_224 : tensor<1x512x7x7xf32>, tensor<512x1x1xf32>) outs(%434 : tensor<1x512x7x7xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.subf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x512x7x7xf32>
    %508 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%507, %expanded_225 : tensor<1x512x7x7xf32>, tensor<512x1x1xf32>) outs(%434 : tensor<1x512x7x7xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x512x7x7xf32>
    %expanded_226 = tensor.expand_shape %arg314 [[0, 1, 2]] output_shape [512, 1, 1] : tensor<512xf32> into tensor<512x1x1xf32>
    %509 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%508, %expanded_226 : tensor<1x512x7x7xf32>, tensor<512x1x1xf32>) outs(%434 : tensor<1x512x7x7xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x512x7x7xf32>
    %expanded_227 = tensor.expand_shape %arg315 [[0, 1, 2]] output_shape [512, 1, 1] : tensor<512xf32> into tensor<512x1x1xf32>
    %510 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%509, %expanded_227 : tensor<1x512x7x7xf32>, tensor<512x1x1xf32>) outs(%434 : tensor<1x512x7x7xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.addf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x512x7x7xf32>
    %511 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%510 : tensor<1x512x7x7xf32>) outs(%434 : tensor<1x512x7x7xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf ugt, %in, %cst : f32
      %533 = arith.select %532, %in, %cst : f32
      linalg.yield %533 : f32
    } -> tensor<1x512x7x7xf32>
    %512 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%511, %arg316 : tensor<1x512x7x7xf32>, tensor<2048x512x1x1xf32>) outs(%446 : tensor<1x2048x7x7xf32>) -> tensor<1x2048x7x7xf32>
    %513 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg157 : tensor<2048xf32>) outs(%448 : tensor<2048xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.truncf %cst_2 : f64 to f32
      %533 = arith.addf %in, %532 : f32
      linalg.yield %533 : f32
    } -> tensor<2048xf32>
    %514 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%513 : tensor<2048xf32>) outs(%448 : tensor<2048xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = math.sqrt %in : f32
      linalg.yield %532 : f32
    } -> tensor<2048xf32>
    %515 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%514 : tensor<2048xf32>) outs(%448 : tensor<2048xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf one, %in, %cst : f32
      cf.assert %532, "unimplemented: tensor with zero element"
      %533 = arith.divf %cst_0, %in : f32
      linalg.yield %533 : f32
    } -> tensor<2048xf32>
    %expanded_228 = tensor.expand_shape %arg156 [[0, 1, 2]] output_shape [2048, 1, 1] : tensor<2048xf32> into tensor<2048x1x1xf32>
    %expanded_229 = tensor.expand_shape %515 [[0, 1, 2]] output_shape [2048, 1, 1] : tensor<2048xf32> into tensor<2048x1x1xf32>
    %516 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%512, %expanded_228 : tensor<1x2048x7x7xf32>, tensor<2048x1x1xf32>) outs(%445 : tensor<1x2048x7x7xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.subf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x2048x7x7xf32>
    %517 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%516, %expanded_229 : tensor<1x2048x7x7xf32>, tensor<2048x1x1xf32>) outs(%445 : tensor<1x2048x7x7xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x2048x7x7xf32>
    %expanded_230 = tensor.expand_shape %arg317 [[0, 1, 2]] output_shape [2048, 1, 1] : tensor<2048xf32> into tensor<2048x1x1xf32>
    %518 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%517, %expanded_230 : tensor<1x2048x7x7xf32>, tensor<2048x1x1xf32>) outs(%445 : tensor<1x2048x7x7xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.mulf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x2048x7x7xf32>
    %expanded_231 = tensor.expand_shape %arg318 [[0, 1, 2]] output_shape [2048, 1, 1] : tensor<2048xf32> into tensor<2048x1x1xf32>
    %519 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%518, %expanded_231 : tensor<1x2048x7x7xf32>, tensor<2048x1x1xf32>) outs(%445 : tensor<1x2048x7x7xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.addf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x2048x7x7xf32>
    %520 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%519, %493 : tensor<1x2048x7x7xf32>, tensor<1x2048x7x7xf32>) outs(%445 : tensor<1x2048x7x7xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.addf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x2048x7x7xf32>
    %521 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%520 : tensor<1x2048x7x7xf32>) outs(%445 : tensor<1x2048x7x7xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.cmpf ugt, %in, %cst : f32
      %533 = arith.select %532, %in, %cst : f32
      linalg.yield %533 : f32
    } -> tensor<1x2048x7x7xf32>
    %522 = tensor.empty() : tensor<1x2048x1x1xf32>
    %523 = linalg.fill ins(%cst : f32) outs(%522 : tensor<1x2048x1x1xf32>) -> tensor<1x2048x1x1xf32>
    %524 = tensor.empty() : tensor<7x7xf32>
    %525 = linalg.pooling_nchw_sum {dilations = dense<1> : vector<2xi64>, strides = dense<7> : vector<2xi64>} ins(%521, %524 : tensor<1x2048x7x7xf32>, tensor<7x7xf32>) outs(%523 : tensor<1x2048x1x1xf32>) -> tensor<1x2048x1x1xf32>
    %526 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%525 : tensor<1x2048x1x1xf32>) outs(%522 : tensor<1x2048x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %532 = arith.divf %in, %cst_3 : f32
      linalg.yield %532 : f32
    } -> tensor<1x2048x1x1xf32>
    %collapsed = tensor.collapse_shape %526 [[0], [1, 2, 3]] : tensor<1x2048x1x1xf32> into tensor<1x2048xf32>
    %527 = tensor.empty() : tensor<2048x1000xf32>
    %transposed = linalg.transpose ins(%arg319 : tensor<1000x2048xf32>) outs(%527 : tensor<2048x1000xf32>) permutation = [1, 0] 
    %528 = tensor.empty() : tensor<1x1000xf32>
    %529 = linalg.fill ins(%cst : f32) outs(%528 : tensor<1x1000xf32>) -> tensor<1x1000xf32>
    %530 = linalg.matmul ins(%collapsed, %transposed : tensor<1x2048xf32>, tensor<2048x1000xf32>) outs(%529 : tensor<1x1000xf32>) -> tensor<1x1000xf32>
    %531 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel"]} ins(%530, %arg320 : tensor<1x1000xf32>, tensor<1000xf32>) outs(%528 : tensor<1x1000xf32>) {
    ^bb0(%in: f32, %in_232: f32, %out: f32):
      %532 = arith.addf %in, %in_232 : f32
      linalg.yield %532 : f32
    } -> tensor<1x1000xf32>
    return %531 : tensor<1x1000xf32>
  }
}
