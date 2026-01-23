import torch
import torchvision.models as models
from transformers import AutoConfig, AutoModel
from torch_to_mlir import dump_to_mlir

config = AutoConfig.from_pretrained("google/gemma-7b", attn_implementation="eager")
model = AutoModel.from_config(config).layers[0].self_attn
model.eval()

dummy_input = (
    torch.randn((8, 1024, 3072), dtype=torch.bfloat16),
    (
        torch.randn(1, 1, 256, dtype=torch.bfloat16),
        torch.randn(1, 1, 256, dtype=torch.bfloat16),
    ),
    torch.randn(1, 16, 1, 1, dtype=torch.bfloat16),
)

dump_to_mlir("./mlir/gemma-7b.mlir", model, dummy_input)
