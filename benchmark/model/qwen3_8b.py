import torch
from transformers import AutoConfig, AutoModel
from transformers.models.qwen3 import Qwen3Model
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3MLP,
    apply_rotary_pos_emb,
    Callable,
    eager_attention_forward,
    ALL_ATTENTION_FUNCTIONS,
)


class SimpleQwen3Attention(Qwen3Attention):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)

    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        past_key_values=None,
        cache_position=None,
        **kwargs
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(
            self.q_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)
        key_states = self.k_norm(
            self.k_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_values is not None:
            key_states = torch.cat((past_key_values[0], key_states), dim=-2)
            value_states = torch.cat((past_key_values[1], value_states), dim=-2)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation
            ]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,  # diff with Llama
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output


class SimpleQwen3MLP(Qwen3MLP):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, x):
        return super().forward(x)


def build_model(
    batch: int, length: int, action: str, block: int, layer: str
) -> tuple[torch.nn.Module, torch.Tensor]:
    config = AutoConfig.from_pretrained("Qwen/Qwen3-8B", attn_implementation="eager")
    if layer == "attention":
        model = AutoModel.from_config(config).layers[block].self_attn
        model.__class__ = SimpleQwen3Attention
        model.eval()
        if action == "prefill":
            dummy_input = (
                torch.randn((batch, length, config.hidden_size), dtype=torch.bfloat16),
                (
                    torch.randn(1, length, config.head_dim, dtype=torch.bfloat16),
                    torch.randn(1, length, config.head_dim, dtype=torch.bfloat16),
                ),
                torch.randn(1, config.num_attention_heads, 1, 1, dtype=torch.bfloat16),
            )
        elif action == "decode":
            dummy_input = (
                torch.randn((batch, 1, config.hidden_size), dtype=torch.bfloat16),
                (
                    torch.randn(1, 1, config.head_dim, dtype=torch.bfloat16),
                    torch.randn(1, 1, config.head_dim, dtype=torch.bfloat16),
                ),
                torch.randn(1, config.num_attention_heads, 1, 1, dtype=torch.bfloat16),
                (
                    torch.randn(
                        (
                            batch,
                            config.num_key_value_heads,
                            length - 1,
                            config.head_dim,
                        ),
                        dtype=torch.bfloat16,
                    ),
                    torch.randn(
                        (
                            batch,
                            config.num_key_value_heads,
                            length - 1,
                            config.head_dim,
                        ),
                        dtype=torch.bfloat16,
                    ),
                ),
            )
        else:
            raise NotImplementedError()
    elif layer == "ffn":
        model = AutoModel.from_config(config).layers[block].mlp
        model.__class__ = SimpleQwen3MLP
        model.eval()
        if action == "prefill":
            dummy_input = (
                torch.randn((batch, length, config.hidden_size), dtype=torch.bfloat16),
            )
        elif action == "decode":
            dummy_input = (
                torch.randn((batch, 1, config.hidden_size), dtype=torch.bfloat16),
            )
        else:
            raise NotImplementedError()

    else:
        raise NotImplementedError()

    return (model, dummy_input)
