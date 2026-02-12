import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers.pytorch_utils import Conv1D
from transformers.cache_utils import DynamicCache, EncoderDecoderCache
from torch_to_mlir import dump_to_mlir

class GPT2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size
        intermediate_size = config.n_inner if config.n_inner is not None else 4 * embed_dim
        
        self.c_fc = Conv1D(intermediate_size, embed_dim)
        self.c_proj = Conv1D(embed_dim, intermediate_size)
        self.act = nn.GELU(approximate='tanh')
        self.dropout = nn.Dropout(config.resid_pdrop)
    
    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class GPT2Attention(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        max_positions = config.max_position_embeddings
        embed_dim = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = embed_dim // num_heads
        
        if head_dim * num_heads != embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {embed_dim} and `num_heads`: {num_heads})."
            )
        
    
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
            persistent=False,
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4), persistent=False)
        
    
        self.register_buffer("embed_dim_t", torch.tensor(embed_dim, dtype=torch.int64), persistent=False)
        self.register_buffer("num_heads_t", torch.tensor(num_heads, dtype=torch.int64), persistent=False)
        self.register_buffer("head_dim_t", torch.tensor(head_dim, dtype=torch.int64), persistent=False)
        self.register_buffer("split_size_t", torch.tensor(embed_dim, dtype=torch.int64), persistent=False)
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.split_size = embed_dim
        
        self.scale_attn_weights = config.scale_attn_weights
        self.is_cross_attention = False
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn
        
        self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)
        
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.is_causal = True
    
    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        # 完全遵循官方 eager_attention_forward 的逻辑，但使用 tensor 类型避免 torch.export 问题
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        
        if self.scale_attn_weights:
            # 使用 torch.sqrt + torch.tensor 避免 torch.full 的问题
            attn_weights = attn_weights / torch.sqrt(
                torch.tensor(value.size(-1), dtype=attn_weights.dtype, device=attn_weights.device)
            )
        
        # Layer-wise attention scaling
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1) # type: ignore
        
        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length = query.shape[-2]
            key_length = key.shape[-2]
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length] # type: ignore
            mask_value = torch.finfo(attn_weights.dtype).min
            # 使用 torch.tensor 而不是 torch.full
            mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
        
        if attention_mask is not None:
            # Apply the attention mask
            causal_mask = attention_mask[:, :, :, : key.shape[-2]]
            attn_weights = attn_weights + causal_mask
        
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        
        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)
        
        if head_mask is not None:
            attn_weights = attn_weights * head_mask
        
        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2)  # 与官方库一致，transpose
        
        return attn_output, attn_weights
    
    def _split_heads(self, tensor, num_heads, head_dim):

        batch_size, seq_len, hidden = tensor.shape[0], tensor.shape[1], tensor.shape[2]
        
        tensor = tensor.reshape(batch_size, seq_len, num_heads, head_dim)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_dim)
    
    def _merge_heads(self, tensor, num_heads, head_dim):

        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        batch_size, seq_len = tensor.shape[0], tensor.shape[1]
        return tensor.reshape(batch_size, seq_len, -1)
    
    def forward(self, hidden_states, attention_mask=None, head_mask=None, past_key_values=None, cache_position=None):
        # 处理 KV cache - 与原始 transformers 模型完全一致
        is_cross_attention = False  # 我们只处理 self-attention
        
        if past_key_values is not None:
            if isinstance(past_key_values, EncoderDecoderCache):
                is_updated = past_key_values.is_updated.get(self.layer_idx)
                if is_cross_attention:
                    curr_past_key_values = past_key_values.cross_attention_cache
                else:
                    curr_past_key_values = past_key_values.self_attention_cache
            else:
                curr_past_key_values = past_key_values
        
        # 计算 query, key, value (self-attention)
        # 使用切片方式，与 gpt2_wo_kv.py 一致，避免 torch.export 问题
        qkv = self.c_attn(hidden_states)
        dim_size = qkv.shape[-1] // 3
        query_states = qkv[..., :dim_size]
        key_states = qkv[..., dim_size:2*dim_size]
        value_states = qkv[..., 2*dim_size:]
        
        # Reshape key 和 value - 使用 -1 让 PyTorch 自动推断
        shape_kv = (*key_states.shape[:-1], -1, self.head_dim)
        key_states = key_states.view(shape_kv).transpose(1, 2)
        value_states = value_states.view(shape_kv).transpose(1, 2)
        
        # Reshape query - 使用 -1 让 PyTorch 自动推断
        shape_q = (*query_states.shape[:-1], -1, self.head_dim)
        query_states = query_states.view(shape_q).transpose(1, 2)
       
        # 如果有 past_key_values，更新 cache（与原始逻辑完全一致）
        if (past_key_values is not None and not is_cross_attention) or (
            past_key_values is not None and is_cross_attention and not is_updated
        ):
            # save all key/value_states to cache to be re-used for fast auto-regressive generation
            cache_position_to_use = cache_position if not is_cross_attention else None
            key_states, value_states = curr_past_key_values.update(
                key_states, value_states, self.layer_idx, {"cache_position": cache_position_to_use} # type: ignore
            )
            # set flag that curr layer for cross-attn is already updated so we can re-use in subsequent calls
            if is_cross_attention:
                past_key_values.is_updated[self.layer_idx] = True
        
        # 计算 attention
        attn_output, attn_weights = self._attn(query_states, key_states, value_states, attention_mask, head_mask)
        
        # Reshape 回原始维度 - 与官方库完全一致
        attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        
        # 投影和 dropout
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
      
        return attn_output, attn_weights


class GPT2Block(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        hidden_size = config.hidden_size
        
        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config, layer_idx=layer_idx)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(config)
    
    def forward(self, hidden_states, attention_mask=None, head_mask=None, past_key_values=None, cache_position=None):
        
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output, attn_weights = self.attn(
            hidden_states, 
            attention_mask=attention_mask, 
            head_mask=head_mask,
            past_key_values=past_key_values,
            cache_position=cache_position
        )
        hidden_states = attn_output + residual
        
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)

        mlp_output = self.mlp(hidden_states)
        hidden_states = mlp_output + residual
        
        return hidden_states, attn_weights

config = AutoConfig.from_pretrained("gpt2", attn_implementation="eager")
config.use_cache = False
config._use_sdpa = False

model = GPT2Block(config, layer_idx=0)
model = model.to(dtype=torch.bfloat16) 
model.eval()

print(f"Model config: embed_dim={config.hidden_size}, num_heads={config.num_attention_heads}")

# 测试 1: 不使用 KV Cache
print("\n=== Testing without KV Cache ===")
batch_size, seq_len = 1, 16
hidden_dim = config.hidden_size
dummy_input_no_cache = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.bfloat16)
output_no_cache, _ = model(dummy_input_no_cache)
print(f"Output shape (no cache): {output_no_cache.shape}")

# 测试 2: 使用 KV Cache
print("\n=== Testing with KV Cache ===")
initial_input = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.bfloat16)
past_key_values = DynamicCache()
cache_position = torch.arange(0, seq_len, dtype=torch.long)

output1, _ = model(initial_input, past_key_values=past_key_values, cache_position=cache_position)
print(f"First pass output shape: {output1.shape}")
print(f"Cache length: {past_key_values.get_seq_length(layer_idx=0)}")

new_token = torch.randn(batch_size, 1, hidden_dim, dtype=torch.bfloat16)
new_cache_position = torch.tensor([seq_len], dtype=torch.long)
output2, _ = model(new_token, past_key_values=past_key_values, cache_position=new_cache_position)
print(f"Second pass output shape: {output2.shape}")
print(f"Cache length after update: {past_key_values.get_seq_length(layer_idx=0)}")

# 导出到 MLIR（不使用 cache 的版本，因为 torch_to_mlir 不支持动态 cache 对象）
print("\n=== Exporting to MLIR ===")
dummy_input = (dummy_input_no_cache,)
output_file = "./mlir/gpt2_kv.mlir"
dump_to_mlir(output_file, model, dummy_input)
print(f"MLIR exported to: {output_file}")