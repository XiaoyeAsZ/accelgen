import torch
from torch import nn
from torch.export import export
from torch_mlir.fx import export_and_import
from transformers import AutoConfig, AutoModel
from torch.func import functional_call


class FunctionalModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        param_iter = iter(kwargs.pop("__params"))
        state_dict = {
            name: p for (name, _), p in zip(self.model.named_parameters(), param_iter)
        }

        # Forward with lifted params and original kwargs
        return functional_call(
            self.model,
            state_dict,
            args,
            kwargs,
        )


config = AutoConfig.from_pretrained("meta-llama/Meta-Llama-3-8B",attn_implementation="eager")
model = AutoModel.from_config(config).layers[0].self_attn

func_model = FunctionalModel(model)

x = torch.randn((2, 1024, 4096), dtype=torch.bfloat16)

# Collect parameters as runtime inputs (detach to avoid grad)
params = tuple(p.detach() for _, p in model.named_parameters())

# Export with lifted weights
exported = export(
    func_model,
    (x,(torch.randn(1, 1, 128, dtype=torch.bfloat16),torch.randn(1, 1, 128, dtype=torch.bfloat16)),None,),
    {"__params": params},
)

mlir_module = export_and_import(exported, output_type="tosa")

with open("./attn.mlir", "w") as f:
    f.write(str(mlir_module))


# config = AutoConfig.from_pretrained("meta-llama/Meta-Llama-3-8B")
# model = AutoModel.from_config(config).layers[0].mlp.gate_proj
# print(model)

