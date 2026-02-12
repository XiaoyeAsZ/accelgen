import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers.pytorch_utils import Conv1D
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
    def __init__(self, config):
        super().__init__()
        self.config = config
        
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
        self.layer_idx = None
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn
        
        self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)
        
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.is_causal = True
    
    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        
        if self.scale_attn_weights:
            attn_weights = attn_weights / torch.sqrt(
                torch.tensor(value.size(-1), dtype=attn_weights.dtype, device=attn_weights.device)
            )
        
        if self.scale_attn_by_inverse_layer_idx and self.layer_idx is not None:
            attn_weights = attn_weights / float(self.layer_idx + 1)
        
        if not self.is_cross_attention:
            
            query_length = query.shape[-2]
            key_length = key.shape[-2]
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length] # type: ignore
            mask_value = torch.finfo(attn_weights.dtype).min
            mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)
        
        if head_mask is not None:
            attn_weights = attn_weights * head_mask
        
        attn_output = torch.matmul(attn_weights, value)
        
        return attn_output, attn_weights
    
    def _split_heads(self, tensor, num_heads, head_dim):

        batch_size, seq_len, hidden = tensor.shape[0], tensor.shape[1], tensor.shape[2]
        
        tensor = tensor.reshape(batch_size, seq_len, num_heads, head_dim)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_dim)
    
    def _merge_heads(self, tensor, num_heads, head_dim):

        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        batch_size, seq_len = tensor.shape[0], tensor.shape[1]
        return tensor.reshape(batch_size, seq_len, -1)
    
    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        qkv = self.c_attn(hidden_states)  # [batch, seq_len, 3*embed_dim]
        
        dim_size = qkv.shape[-1] // 3 
        query = qkv[..., :dim_size]
        key = qkv[..., dim_size:2*dim_size]
        value = qkv[..., 2*dim_size:]
        
       
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)
          
        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)
        
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        
        return attn_output, attn_weights


class GPT2Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        
        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(config)
    
    def forward(self, hidden_states, attention_mask=None, head_mask=None):
       
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output, attn_weights = self.attn(hidden_states, attention_mask, head_mask)

        hidden_states = attn_output + residual
        
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        mlp_output = self.mlp(hidden_states)
        
        hidden_states = mlp_output + residual
        
        return hidden_states, attn_weights

config = AutoConfig.from_pretrained("gpt2", attn_implementation="eager")
config.use_cache = False
config._use_sdpa = False

model = GPT2Block(config)
model.eval()

print(f"Model config: embed_dim={config.hidden_size}, num_heads={config.num_attention_heads}")

dummy_input = (torch.randn(1, 16, 768, dtype=torch.bfloat16),)

dump_to_mlir("./mlir/gpt2_wo_cache.mlir", model, dummy_input)