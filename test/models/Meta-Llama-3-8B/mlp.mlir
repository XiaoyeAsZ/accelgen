module {
  func.func @main(%arg0: !torch.vtensor<[1,1,4096],bf16>, %arg1: !torch.vtensor<[11008,4096],f32>, %arg2: !torch.vtensor<[11008,4096],f32>, %arg3: !torch.vtensor<[4096,11008],f32>) -> !torch.vtensor<[1,1,4096],bf16> {
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %0 = torch.aten.transpose.int %arg1, %int0, %int1 : !torch.vtensor<[11008,4096],f32>, !torch.int, !torch.int -> !torch.vtensor<[4096,11008],f32>
    %1 = torch.aten.matmul %arg0, %0 : !torch.vtensor<[1,1,4096],bf16>, !torch.vtensor<[4096,11008],f32> -> !torch.vtensor<[1,1,11008],bf16>
    %2 = torch.aten.sigmoid %1 : !torch.vtensor<[1,1,11008],bf16> -> !torch.vtensor<[1,1,11008],bf16>
    %3 = torch.aten.mul.Tensor %2, %1 : !torch.vtensor<[1,1,11008],bf16>, !torch.vtensor<[1,1,11008],bf16> -> !torch.vtensor<[1,1,11008],bf16>
    %4 = torch.aten.transpose.int %arg2, %int0, %int1 : !torch.vtensor<[11008,4096],f32>, !torch.int, !torch.int -> !torch.vtensor<[4096,11008],f32>
    %5 = torch.aten.matmul %arg0, %4 : !torch.vtensor<[1,1,4096],bf16>, !torch.vtensor<[4096,11008],f32> -> !torch.vtensor<[1,1,11008],bf16>
    %6 = torch.aten.mul.Tensor %3, %5 : !torch.vtensor<[1,1,11008],bf16>, !torch.vtensor<[1,1,11008],bf16> -> !torch.vtensor<[1,1,11008],bf16>
    %7 = torch.aten.transpose.int %arg3, %int0, %int1 : !torch.vtensor<[4096,11008],f32>, !torch.int, !torch.int -> !torch.vtensor<[11008,4096],f32>
    %8 = torch.aten.matmul %6, %7 : !torch.vtensor<[1,1,11008],bf16>, !torch.vtensor<[11008,4096],f32> -> !torch.vtensor<[1,1,4096],bf16>
    return %8 : !torch.vtensor<[1,1,4096],bf16>
  }
}
