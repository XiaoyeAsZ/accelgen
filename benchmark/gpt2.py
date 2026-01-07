import torch
from transformers import GPT2Model, GPT2Tokenizer
from transformers import AutoConfig, AutoModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
from torch_to_mlir import dump_to_mlir


# model = GPT2Model.from_pretrained("gpt2")
config = AutoConfig.from_pretrained("gpt2", attn_implementation="eager")

config.__dict__["_attn_implementation"] = "eager"
config.use_cache = False
config._use_sdpa = False
model = AutoModel.from_config(config).h[0].attn
model.eval()

print(model.config)

# dummy_input = (
#     torch.randint(low=0, high=config.vocab_size, size=(1, 16), dtype=torch.long),
# )

dummy_input = (torch.randn(1, 16, 768, dtype=torch.bfloat16),)

dump_to_mlir("./mlir/mobilenet_v2.mlir", model, dummy_input)
