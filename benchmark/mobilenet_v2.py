# import torch
# import torchvision.models as models
# from torch_to_mlir import dump_to_mlir
# from torch._dynamo import mark_dynamic

# # 1. Load pretrained ResNet-50
# model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
# model.eval()

# # 2. Dummy input (batch_size=1, 3×224×224)
# x = torch.randn(2, 3, 224, 224)

# mark_dynamic(x, 0)

# dummy_input = (x,)


# dump_to_mlir("./mlir/mobilenet.mlir", model, dummy_input)


import torch
import torchvision.models as models
from torch_to_mlir import dump_to_mlir

model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
model.eval()

# 2. Dummy input (batch_size=1, 3×224×224)
batchsz = 1
dummy_input = (torch.randn(batchsz, 3, 224, 224),)

dump_to_mlir("./mlir/mobilenet_v2.mlir", model, dummy_input)
