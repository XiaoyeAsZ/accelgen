import torch
import torchvision.models as models
from torch_to_mlir import dump_to_mlir

# 1. Load pretrained ResNet-50
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.eval()

# 2. Dummy input (batch_size=1, 3×224×224)
batchsz = 1
dummy_input = (torch.randn(batchsz, 3, 224, 224),)

dump_to_mlir("./mlir/resnet50.mlir", model, dummy_input)
