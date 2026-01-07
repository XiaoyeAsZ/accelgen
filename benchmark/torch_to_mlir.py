import torch
from torch import nn
from torch.export import export
from torch_mlir.fx import export_and_import
from transformers import AutoConfig, AutoModel
from torch.func import functional_call
from typing import Tuple, Any


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


def dump_to_mlir(path: str, model: torch.nn.Module, placeholder: tuple):
    func_model = FunctionalModel(model)

    params = tuple(p.detach() for _, p in model.named_parameters())
    exported = export(
        func_model,
        placeholder,
        {"__params": params},
    )

    mlir_module = export_and_import(exported, output_type="linalg-on-tensors")
    # mlir_module = export_and_import(exported, output_type="tosa")

    with open(path, "w") as f:
        f.write(str(mlir_module))
