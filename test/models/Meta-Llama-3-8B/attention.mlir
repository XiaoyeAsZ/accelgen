module {
  func.func @main(%arg0: !torch.vtensor<[1,1,4096],bf16>, %arg1: !torch.vtensor<[1,1,128],bf16>, %arg2: !torch.vtensor<[1,1,128],bf16>, %arg3: !torch.none, %arg4: !torch.vtensor<[4096,4096],f32>, %arg5: !torch.vtensor<[4096,4096],f32>, %arg6: !torch.vtensor<[4096,4096],f32>, %arg7: !torch.vtensor<[4096,4096],f32>) -> (!torch.vtensor<[1,1,4096],bf16>, !torch.none) {
    %float8.838830e-02 = torch.constant.float 0.088388347648318447
    %false = torch.constant.bool false
    %float0.000000e00 = torch.constant.float 0.000000e+00
    %int9223372036854775807 = torch.constant.int 9223372036854775807
    %int64 = torch.constant.int 64
    %int0 = torch.constant.int 0
    %int3 = torch.constant.int 3
    %int2 = torch.constant.int 2
    %int128 = torch.constant.int 128
    %int-1 = torch.constant.int -1
    %int1 = torch.constant.int 1
    %none = torch.constant.none
    %0 = torch.aten.transpose.int %arg4, %int0, %int1 : !torch.vtensor<[4096,4096],f32>, !torch.int, !torch.int -> !torch.vtensor<[4096,4096],f32>
    %1 = torch.aten.matmul %arg0, %0 : !torch.vtensor<[1,1,4096],bf16>, !torch.vtensor<[4096,4096],f32> -> !torch.vtensor<[1,1,4096],bf16>
    %2 = torch.prim.ListConstruct %int1, %int1, %int-1, %int128 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %3 = torch.aten.view %1, %2 : !torch.vtensor<[1,1,4096],bf16>, !torch.list<int> -> !torch.vtensor<[1,1,32,128],bf16>
    %4 = torch.aten.transpose.int %3, %int1, %int2 : !torch.vtensor<[1,1,32,128],bf16>, !torch.int, !torch.int -> !torch.vtensor<[1,32,1,128],bf16>
    %5 = torch.aten.transpose.int %arg5, %int0, %int1 : !torch.vtensor<[4096,4096],f32>, !torch.int, !torch.int -> !torch.vtensor<[4096,4096],f32>
    %6 = torch.aten.matmul %arg0, %5 : !torch.vtensor<[1,1,4096],bf16>, !torch.vtensor<[4096,4096],f32> -> !torch.vtensor<[1,1,4096],bf16>
    %7 = torch.prim.ListConstruct %int1, %int1, %int-1, %int128 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %8 = torch.aten.view %6, %7 : !torch.vtensor<[1,1,4096],bf16>, !torch.list<int> -> !torch.vtensor<[1,1,32,128],bf16>
    %9 = torch.aten.transpose.int %8, %int1, %int2 : !torch.vtensor<[1,1,32,128],bf16>, !torch.int, !torch.int -> !torch.vtensor<[1,32,1,128],bf16>
    %10 = torch.aten.transpose.int %arg6, %int0, %int1 : !torch.vtensor<[4096,4096],f32>, !torch.int, !torch.int -> !torch.vtensor<[4096,4096],f32>
    %11 = torch.aten.matmul %arg0, %10 : !torch.vtensor<[1,1,4096],bf16>, !torch.vtensor<[4096,4096],f32> -> !torch.vtensor<[1,1,4096],bf16>
    %12 = torch.prim.ListConstruct %int1, %int1, %int-1, %int128 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %13 = torch.aten.view %11, %12 : !torch.vtensor<[1,1,4096],bf16>, !torch.list<int> -> !torch.vtensor<[1,1,32,128],bf16>
    %14 = torch.aten.transpose.int %13, %int1, %int2 : !torch.vtensor<[1,1,32,128],bf16>, !torch.int, !torch.int -> !torch.vtensor<[1,32,1,128],bf16>
    %15 = torch.aten.unsqueeze %arg1, %int1 : !torch.vtensor<[1,1,128],bf16>, !torch.int -> !torch.vtensor<[1,1,1,128],bf16>
    %16 = torch.aten.unsqueeze %arg2, %int1 : !torch.vtensor<[1,1,128],bf16>, !torch.int -> !torch.vtensor<[1,1,1,128],bf16>
    %17 = torch.aten.mul.Tensor %4, %15 : !torch.vtensor<[1,32,1,128],bf16>, !torch.vtensor<[1,1,1,128],bf16> -> !torch.vtensor<[1,32,1,128],bf16>
    %18 = torch.aten.slice.Tensor %4, %int3, %int0, %int64, %int1 : !torch.vtensor<[1,32,1,128],bf16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,32,1,64],bf16>
    %19 = torch.aten.slice.Tensor %4, %int3, %int64, %int9223372036854775807, %int1 : !torch.vtensor<[1,32,1,128],bf16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,32,1,64],bf16>
    %20 = torch.aten.neg %19 : !torch.vtensor<[1,32,1,64],bf16> -> !torch.vtensor<[1,32,1,64],bf16>
    %21 = torch.prim.ListConstruct %20, %18 : (!torch.vtensor<[1,32,1,64],bf16>, !torch.vtensor<[1,32,1,64],bf16>) -> !torch.list<vtensor>
    %22 = torch.aten.cat %21, %int-1 : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[1,32,1,128],bf16>
    %23 = torch.aten.mul.Tensor %22, %16 : !torch.vtensor<[1,32,1,128],bf16>, !torch.vtensor<[1,1,1,128],bf16> -> !torch.vtensor<[1,32,1,128],bf16>
    %24 = torch.aten.add.Tensor %17, %23, %int1 : !torch.vtensor<[1,32,1,128],bf16>, !torch.vtensor<[1,32,1,128],bf16>, !torch.int -> !torch.vtensor<[1,32,1,128],bf16>
    %25 = torch.aten.mul.Tensor %9, %15 : !torch.vtensor<[1,32,1,128],bf16>, !torch.vtensor<[1,1,1,128],bf16> -> !torch.vtensor<[1,32,1,128],bf16>
    %26 = torch.aten.slice.Tensor %9, %int3, %int0, %int64, %int1 : !torch.vtensor<[1,32,1,128],bf16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,32,1,64],bf16>
    %27 = torch.aten.slice.Tensor %9, %int3, %int64, %int9223372036854775807, %int1 : !torch.vtensor<[1,32,1,128],bf16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,32,1,64],bf16>
    %28 = torch.aten.neg %27 : !torch.vtensor<[1,32,1,64],bf16> -> !torch.vtensor<[1,32,1,64],bf16>
    %29 = torch.prim.ListConstruct %28, %26 : (!torch.vtensor<[1,32,1,64],bf16>, !torch.vtensor<[1,32,1,64],bf16>) -> !torch.list<vtensor>
    %30 = torch.aten.cat %29, %int-1 : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[1,32,1,128],bf16>
    %31 = torch.aten.mul.Tensor %30, %16 : !torch.vtensor<[1,32,1,128],bf16>, !torch.vtensor<[1,1,1,128],bf16> -> !torch.vtensor<[1,32,1,128],bf16>
    %32 = torch.aten.add.Tensor %25, %31, %int1 : !torch.vtensor<[1,32,1,128],bf16>, !torch.vtensor<[1,32,1,128],bf16>, !torch.int -> !torch.vtensor<[1,32,1,128],bf16>
    %33 = torch.aten.scaled_dot_product_attention %24, %32, %14, %none, %float0.000000e00, %false, %float8.838830e-02, %false : !torch.vtensor<[1,32,1,128],bf16>, !torch.vtensor<[1,32,1,128],bf16>, !torch.vtensor<[1,32,1,128],bf16>, !torch.none, !torch.float, !torch.bool, !torch.float, !torch.bool -> !torch.vtensor<[1,32,1,128],bf16>
    %34 = torch.aten.transpose.int %33, %int1, %int2 : !torch.vtensor<[1,32,1,128],bf16>, !torch.int, !torch.int -> !torch.vtensor<[1,1,32,128],bf16>
    %35 = torch.prim.ListConstruct %int1, %int1, %int-1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %36 = torch.aten.view %34, %35 : !torch.vtensor<[1,1,32,128],bf16>, !torch.list<int> -> !torch.vtensor<[1,1,4096],bf16>
    %37 = torch.aten.transpose.int %arg7, %int0, %int1 : !torch.vtensor<[4096,4096],f32>, !torch.int, !torch.int -> !torch.vtensor<[4096,4096],f32>
    %38 = torch.aten.matmul %36, %37 : !torch.vtensor<[1,1,4096],bf16>, !torch.vtensor<[4096,4096],f32> -> !torch.vtensor<[1,1,4096],bf16>
    return %38, %none : !torch.vtensor<[1,1,4096],bf16>, !torch.none
  }
}
