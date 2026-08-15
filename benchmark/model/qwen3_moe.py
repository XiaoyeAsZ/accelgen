import torch
from transformers.activations import ACT2FN
from torch.nn import functional as F
from transformers import AutoConfig, AutoModel
from transformers.models.qwen3 import Qwen3Model
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeDecoderLayer,
    Qwen3MoeAttention,
    Qwen3MoeMLP,
    Qwen3MoeSparseMoeBlock,
    apply_rotary_pos_emb,
    eager_attention_forward,
    ALL_ATTENTION_FUNCTIONS,
    Callable,
)


class SimpleQwen3MoeAttention(Qwen3MoeAttention):
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


class SimpleQwen3MoeSparseMoeBlock(Qwen3MoeSparseMoeBlock):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, hidden_states: torch.Tensor):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        if self.norm_topk_prob:  # only diff with mixtral sparse moe block!
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.num_experts
        ).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = (
                expert_layer(current_state) * routing_weights[top_x, idx, None]
            )

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(
                0, top_x, current_hidden_states.to(hidden_states.dtype)
            )
        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim
        )
        return final_hidden_states, router_logits


class StaticGroupedQwen3MoeBlock(Qwen3MoeSparseMoeBlock):
    """Static top-k routing with one packed MLP per expert group."""

    def __init__(self, config, device=None, dtype=None, num_packed_groups=None):
        torch.nn.Module.__init__(self)
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        if self.num_experts % self.top_k:
            raise ValueError("static routing requires complete expert groups")

        self.num_expert_groups = self.num_experts // self.top_k
        if num_packed_groups is None:
            num_packed_groups = self.num_expert_groups
        if not 0 < num_packed_groups <= self.num_expert_groups:
            raise ValueError("invalid number of packed expert groups")
        self.num_packed_groups = num_packed_groups
        self.moe_intermediate_size = config.moe_intermediate_size
        self.act_fn = ACT2FN[config.hidden_act]
        self.gate = torch.nn.Linear(
            config.hidden_size,
            self.num_experts,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.group_gate_up_proj_weight = torch.nn.Parameter(
            torch.empty(
                self.num_packed_groups,
                config.hidden_size,
                self.top_k * 2 * self.moe_intermediate_size,
                device=device,
                dtype=dtype,
            ),
            requires_grad=False,
        )
        self.group_down_proj_weight = torch.nn.Parameter(
            torch.empty(
                self.num_packed_groups,
                self.top_k * self.moe_intermediate_size,
                config.hidden_size,
                device=device,
                dtype=dtype,
            ),
            requires_grad=False,
        )
        self.group_selection_mask = torch.nn.Parameter(
            torch.eye(
                self.num_expert_groups,
                device=device,
                dtype=dtype,
            ),
            requires_grad=False,
        )

        with torch.no_grad():
            gate_up_bound = config.hidden_size**-0.5
            down_bound = self.moe_intermediate_size**-0.5
            self.group_gate_up_proj_weight.uniform_(-gate_up_bound, gate_up_bound)
            self.group_down_proj_weight.uniform_(-down_bound, down_bound)

    def forward(self, hidden_states: torch.Tensor):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        tokens = hidden_states.reshape(-1, hidden_dim)
        token_count = batch_size * sequence_length
        active_groups = min(self.num_expert_groups, token_count)
        if active_groups > self.num_packed_groups:
            raise ValueError("input requires more expert groups than were packed")
        if token_count % active_groups:
            raise ValueError(
                "static grouped routing requires token count to divide active groups"
            )

        tokens_per_group = token_count // active_groups
        grouped_tokens = tokens.reshape(active_groups, tokens_per_group, hidden_dim)
        router_logits = self.gate(tokens)
        grouped_router_logits = router_logits.reshape(
            active_groups,
            tokens_per_group,
            self.num_expert_groups,
            self.top_k,
        )
        selection_mask = self.group_selection_mask[
            :active_groups, :active_groups
        ].reshape(active_groups, 1, active_groups, 1)

        group_logits = (
            grouped_router_logits[:, :, :active_groups, :] * selection_mask
        ).sum(dim=2)
        routing_weights = F.softmax(group_logits, dim=-1, dtype=torch.float)

        routing_weights = routing_weights.to(tokens.dtype)

        projected = torch.bmm(
            grouped_tokens,
            self.group_gate_up_proj_weight[:active_groups],
        ).reshape(
            active_groups,
            tokens_per_group,
            self.top_k,
            2 * self.moe_intermediate_size,
        )
        gate_proj = projected[..., : self.moe_intermediate_size]
        up_proj = projected[..., self.moe_intermediate_size :]
        expert_hidden = self.act_fn(gate_proj) * up_proj
        expert_hidden = expert_hidden * routing_weights.unsqueeze(-1)
        output = torch.bmm(
            expert_hidden.reshape(
                active_groups,
                tokens_per_group,
                self.top_k * self.moe_intermediate_size,
            ),
            self.group_down_proj_weight[:active_groups],
        )
        output = output.reshape(batch_size, sequence_length, hidden_dim)
        return output, router_logits


class SimpleQwen3MoeMLP(Qwen3MoeMLP):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, x):
        return super().forward(x)


def build_model(
    batch: int,
    length: int,
    action: str,
    block: int,
    layer: str,
    device: str = None,
    local_path: str = None,
) -> tuple[torch.nn.Module, torch.Tensor]:
    if local_path:
        model_path = local_path
    else:
        model_path = "Qwen/Qwen3-30B-A3B-Thinking-2507"
    config = AutoConfig.from_pretrained(model_path, attn_implementation="eager")
    if layer == "attention":
        model = Qwen3MoeAttention(config, layer_idx=block).to(
            device=device, dtype=torch.bfloat16
        )
        model.__class__ = SimpleQwen3MoeAttention
        model.eval()
        if action == "prefill":
            dummy_input = (
                torch.randn(
                    (batch, length, config.hidden_size),
                    dtype=torch.bfloat16,
                    device=device,
                ),
                (
                    torch.randn(
                        1, length, config.head_dim, dtype=torch.bfloat16, device=device
                    ),
                    torch.randn(
                        1, length, config.head_dim, dtype=torch.bfloat16, device=device
                    ),
                ),
                torch.randn(
                    1,
                    config.num_attention_heads,
                    length,
                    length,
                    dtype=torch.bfloat16,
                    device=device,
                ),
            )
        elif action == "decode":
            dummy_input = (
                torch.randn(
                    (batch, 1, config.hidden_size), dtype=torch.bfloat16, device=device
                ),
                (
                    torch.randn(
                        1, 1, config.head_dim, dtype=torch.bfloat16, device=device
                    ),
                    torch.randn(
                        1, 1, config.head_dim, dtype=torch.bfloat16, device=device
                    ),
                ),
                torch.randn(
                    1,
                    config.num_attention_heads,
                    1,
                    length,
                    dtype=torch.bfloat16,
                    device=device,
                ),
                (
                    torch.randn(
                        (
                            batch,
                            config.num_key_value_heads,
                            length - 1,
                            config.head_dim,
                        ),
                        dtype=torch.bfloat16,
                        device=device,
                    ),
                    torch.randn(
                        (
                            batch,
                            config.num_key_value_heads,
                            length - 1,
                            config.head_dim,
                        ),
                        dtype=torch.bfloat16,
                        device=device,
                    ),
                ),
            )
        else:
            raise NotImplementedError()
    elif layer == "ffn":
        token_count = batch * (length if action == "prefill" else 1)
        model = StaticGroupedQwen3MoeBlock(
            config,
            device=device,
            dtype=torch.bfloat16,
            num_packed_groups=min(
                config.num_experts // config.num_experts_per_tok,
                token_count,
            ),
        )
        model.eval()
        if action == "prefill":
            dummy_input = (
                torch.randn(
                    (batch, length, config.hidden_size),
                    dtype=torch.bfloat16,
                    device=device,
                ),
            )
        elif action == "decode":
            dummy_input = (
                torch.randn(
                    (batch, 1, config.hidden_size), dtype=torch.bfloat16, device=device
                ),
            )
        else:
            raise NotImplementedError()

    else:
        raise NotImplementedError()

    return (model.to(device), dummy_input)
