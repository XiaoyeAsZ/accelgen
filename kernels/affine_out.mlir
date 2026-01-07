#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  func.func @main(%arg0: memref<1x1x4096xbf16, strided<[?, ?, ?], offset: ?>>, %arg1: memref<14336x4096xbf16, strided<[?, ?], offset: ?>>, %arg2: memref<14336x4096xbf16, strided<[?, ?], offset: ?>>, %arg3: memref<4096x14336xbf16, strided<[?, ?], offset: ?>>) -> memref<1x1x4096xbf16> {
    %cst = arith.constant 1.000000e+00 : bf16
    %cst_0 = arith.constant 0.000000e+00 : bf16
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<4096x14336xbf16>
    linalg.transpose ins(%arg1 : memref<14336x4096xbf16, strided<[?, ?], offset: ?>>) outs(%alloc : memref<4096x14336xbf16>) permutation = [1, 0] 
    %expand_shape = memref.expand_shape %alloc [[0, 1], [2]] output_shape [1, 4096, 14336] : memref<4096x14336xbf16> into memref<1x4096x14336xbf16>
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<1x1x14336xbf16>
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<1x1x14336xbf16>
    linalg.fill ins(%cst_0 : bf16) outs(%alloc_2 : memref<1x1x14336xbf16>)
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<1x1x14336xbf16>
    memref.copy %alloc_2, %alloc_3 : memref<1x1x14336xbf16> to memref<1x1x14336xbf16>
    linalg.batch_matmul ins(%arg0, %expand_shape : memref<1x1x4096xbf16, strided<[?, ?, ?], offset: ?>>, memref<1x4096x14336xbf16>) outs(%alloc_3 : memref<1x1x14336xbf16>)
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%alloc_3 : memref<1x1x14336xbf16>) outs(%alloc_1 : memref<1x1x14336xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %0 = arith.negf %in : bf16
      %1 = math.exp %0 : bf16
      %2 = arith.addf %1, %cst : bf16
      %3 = arith.divf %cst, %2 : bf16
      linalg.yield %3 : bf16
    }
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%alloc_1, %alloc_3 : memref<1x1x14336xbf16>, memref<1x1x14336xbf16>) outs(%alloc_1 : memref<1x1x14336xbf16>) {
    ^bb0(%in: bf16, %in_8: bf16, %out: bf16):
      %0 = arith.mulf %in, %in_8 : bf16
      linalg.yield %0 : bf16
    }
    linalg.transpose ins(%arg2 : memref<14336x4096xbf16, strided<[?, ?], offset: ?>>) outs(%alloc : memref<4096x14336xbf16>) permutation = [1, 0] 
    %expand_shape_4 = memref.expand_shape %alloc [[0, 1], [2]] output_shape [1, 4096, 14336] : memref<4096x14336xbf16> into memref<1x4096x14336xbf16>
    linalg.batch_matmul ins(%arg0, %expand_shape_4 : memref<1x1x4096xbf16, strided<[?, ?, ?], offset: ?>>, memref<1x4096x14336xbf16>) outs(%alloc_2 : memref<1x1x14336xbf16>)
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%alloc_1, %alloc_2 : memref<1x1x14336xbf16>, memref<1x1x14336xbf16>) outs(%alloc_1 : memref<1x1x14336xbf16>) {
    ^bb0(%in: bf16, %in_8: bf16, %out: bf16):
      %0 = arith.mulf %in, %in_8 : bf16
      linalg.yield %0 : bf16
    }
    %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<14336x4096xbf16>
    linalg.transpose ins(%arg3 : memref<4096x14336xbf16, strided<[?, ?], offset: ?>>) outs(%alloc_5 : memref<14336x4096xbf16>) permutation = [1, 0] 
    %expand_shape_6 = memref.expand_shape %alloc_5 [[0, 1], [2]] output_shape [1, 14336, 4096] : memref<14336x4096xbf16> into memref<1x14336x4096xbf16>
    %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<1x1x4096xbf16>
    linalg.fill ins(%cst_0 : bf16) outs(%alloc_7 : memref<1x1x4096xbf16>)
    linalg.batch_matmul ins(%alloc_1, %expand_shape_6 : memref<1x1x14336xbf16>, memref<1x14336x4096xbf16>) outs(%alloc_7 : memref<1x1x4096xbf16>)
    %cast = memref.cast %alloc_7 : memref<1x1x4096xbf16> to memref<1x1x4096xbf16, strided<[?, ?, ?], offset: ?>>
    return %alloc_7 : memref<1x1x4096xbf16>
  }
}

