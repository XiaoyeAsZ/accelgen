import torch
import torch.nn as nn
from transformers import AutoConfig
from torch_to_mlir import dump_to_mlir


class Conv1D(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).
    Basically works like a linear layer but the weights are transposed.
    """
    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x


class TensorizedGPT2Attention(nn.Module):
    """GPT2 Attention with all original computations, integers as tensors for MLIR export"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 原始整数值
        max_positions = config.max_position_embeddings
        embed_dim = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = embed_dim // num_heads
        
        if head_dim * num_heads != embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {embed_dim} and `num_heads`: {num_heads})."
            )
        
        # 注册 causal mask buffer（保留原始实现）
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
            persistent=False,
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4), persistent=False)
        
        # 将整数配置注册为张量 buffer
        self.register_buffer("embed_dim_t", torch.tensor(embed_dim, dtype=torch.int64), persistent=False)
        self.register_buffer("num_heads_t", torch.tensor(num_heads, dtype=torch.int64), persistent=False)
        self.register_buffer("head_dim_t", torch.tensor(head_dim, dtype=torch.int64), persistent=False)
        self.register_buffer("split_size_t", torch.tensor(embed_dim, dtype=torch.int64), persistent=False)
        
        # 保留原始值用于初始化
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.split_size = embed_dim
        
        self.scale_attn_weights = config.scale_attn_weights
        self.is_cross_attention = False
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = None
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn
        
        # 使用 Conv1D（与原始 GPT2 一致）
        self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)
        
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.is_causal = True
    
    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        """完整的注意力计算，包含 causal mask"""
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        
        if self.scale_attn_weights:
            attn_weights = attn_weights / torch.sqrt(
                torch.tensor(value.size(-1), dtype=attn_weights.dtype, device=attn_weights.device)
            )
        
        if self.scale_attn_by_inverse_layer_idx and self.layer_idx is not None:
            attn_weights = attn_weights / float(self.layer_idx + 1)
        
        # 应用 causal mask（保留完整逻辑）
        if not self.is_cross_attention:
            # query_length, key_length = query.size(-2), key.size(-2)
            # 直接使用张量操作获取长度
            query_length = query.shape[-2]
            key_length = key.shape[-2]
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
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
        """
        将 hidden_size 拆分为 (num_heads, head_dim)
        避免使用整数参数的 view
        """
        batch_size, seq_len, hidden = tensor.shape[0], tensor.shape[1], tensor.shape[2]
        # 使用 reshape 而不是 view，并直接用 -1 推断
        tensor = tensor.reshape(batch_size, seq_len, num_heads, head_dim)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_dim)
    
    def _merge_heads(self, tensor, num_heads, head_dim):
        """
        合并 attention heads
        避免使用整数参数
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        batch_size, seq_len = tensor.shape[0], tensor.shape[1]
        # 使用 -1 让 PyTorch 自动推断最后一维
        return tensor.reshape(batch_size, seq_len, -1)
    
    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        """
        完整的 forward，保留所有计算但去掉 cache 和 cross-attention
        hidden_states: [batch, seq_len, embed_dim]
        """
        # QKV 投影
        qkv = self.c_attn(hidden_states)  # [batch, seq_len, 3*embed_dim]
        
        # 手动分割 QKV，避免使用 .split() 和整数常量
        # 使用张量切片而不是 split
        dim_size = qkv.shape[-1] // 3  # 这会在编译时计算
        query = qkv[..., :dim_size]
        key = qkv[..., dim_size:2*dim_size]
        value = qkv[..., 2*dim_size:]
        
        # 分割 heads
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)
        
        # 注意力计算（包含 causal mask）
        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)
        
        # 合并 heads
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        
        # 输出投影
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        
        return attn_output, attn_weights


# 配置和模型
config = AutoConfig.from_pretrained("gpt2", attn_implementation="eager")
config.use_cache = False
config._use_sdpa = False

# 使用自定义的 Attention 类
model = TensorizedGPT2Attention(config)
model.eval()

print(f"Model config: embed_dim={config.hidden_size}, num_heads={config.num_attention_heads}")

# 输入: [batch=1, seq_len=16, hidden_size=768]
dummy_input = (torch.randn(1, 16, 768, dtype=torch.bfloat16),)

dump_to_mlir("./mlir/gpt2_attn.mlir", model, dummy_input)