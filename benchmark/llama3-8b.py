import torch
import torchvision.models as models
from transformers import AutoConfig, AutoModel
from torch_to_mlir import dump_to_mlir

config = AutoConfig.from_pretrained(
    "meta-llama/Meta-Llama-3-8B", attn_implementation="eager"
)
model = AutoModel.from_config(config).layers[0].self_attn
model.eval()

dummy_input = (
    torch.randn((8, 1024, 4096), dtype=torch.bfloat16),
    (
        torch.randn(1, 1, 128, dtype=torch.bfloat16),
        torch.randn(1, 1, 128, dtype=torch.bfloat16),
    ),
    torch.randn(1, 32, 1, 1, dtype=torch.bfloat16),
)

dump_to_mlir("./mlir/llama3-8b.mlir", model, dummy_input)
