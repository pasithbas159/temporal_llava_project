import torch
import torch.nn as nn
import math

# ====== Helper Functions ======
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

def precompute_rope_freqs(length, dim, dtype=torch.float32):
    theta = 10000 ** (-torch.arange(0, dim, 2, dtype=dtype) / dim)
    idx = torch.arange(length, dtype=dtype)
    freqs = torch.einsum('i,j->ij', idx, theta)
    freqs = torch.cat((freqs, freqs), dim=-1)  # duplicate for cos/sin
    return freqs

def apply_rope(x, freqs):
    sin, cos = freqs.sin(), freqs.cos()
    return x * cos + rotate_half(x) * sin

def apply_dual_rope(q, k, freqs_abs, temporal_ids, freqs_temp):
    # Apply absolute RoPE
    freqs_abs = freqs_abs.to(q.device)[None, None, :q.size(2)]
    q = apply_rope(q, freqs_abs)
    k = apply_rope(k, freqs_abs)
    # Apply temporal RoPE
    freqs_temp = freqs_temp.to(q.device)[temporal_ids]
    freqs_temp = freqs_temp[None, None, :, :]
    q = apply_rope(q, freqs_temp)
    k = apply_rope(k, freqs_temp)
    return q, k

def compute_temporal_ids(seq_len, frame_size, gamma=1.0):
    frame_ids = torch.arange(seq_len) // frame_size
    return torch.floor(frame_ids.float() * gamma).long()

def build_fwbc_mask(seq_len, frame_size):
    mask = torch.full((seq_len, seq_len), float('-inf'))
    for i in range(seq_len):
        for j in range(i + 1):  # causal
            mask[i, j] = 0
        if i // frame_size == j // frame_size:
            mask[i, j] = 0  # same frame
    return mask

def split_heads(x, head_dim):
    B, T, C = x.shape
    H = C // head_dim
    x = x.view(B, T, H, head_dim).transpose(1, 2)  # (B, H, T, dh)
    return x

def combine_heads(x):
    B, H, T, dh = x.shape
    return x.transpose(1, 2).reshape(B, T, H * dh)

class TCAttention(nn.Module):
    def __init__(self, orig_attn, frame_size=576, gamma=0.5, max_seq_len=2048):
        super().__init__()
        self.q_proj = orig_attn.q_proj
        self.k_proj = orig_attn.k_proj
        self.v_proj = orig_attn.v_proj
        self.o_proj = orig_attn.o_proj

        self.embed_dim = self.q_proj.out_features  # total hidden size
        self.frame_size = frame_size
        self.gamma = gamma

        # Infer head_dim from proj shape: (hidden, head_dim * num_heads)
        self.num_heads = getattr(orig_attn, "num_heads", 32)  # fallback if missing
        self.head_dim = self.embed_dim // self.num_heads

        # Rotary embedding precomputation
        self.max_seq_len = max_seq_len
        self.freqs_abs = precompute_rope_freqs(max_seq_len, self.head_dim)
        self.temporal_ids = compute_temporal_ids(max_seq_len, frame_size, gamma)
        self.freqs_temp = precompute_rope_freqs(self.temporal_ids.max().item() + 1, self.head_dim)
        self.mask = build_fwbc_mask(max_seq_len, frame_size)

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        B, T, C = hidden_states.shape
        dtype = hidden_states.dtype
        device = hidden_states.device

        # Project Q, K, V
        q = split_heads(self.q_proj(hidden_states), self.head_dim)
        k = split_heads(self.k_proj(hidden_states), self.head_dim)
        v = split_heads(self.v_proj(hidden_states), self.head_dim)

        # Move and cast freqs
        freqs_abs = self.freqs_abs.to(device=device, dtype=dtype)
        freqs_temp = self.freqs_temp.to(device=device, dtype=dtype)
        temporal_ids = self.temporal_ids[:T].to(device=device)

        # Apply dual RoPE
        q, k = apply_dual_rope(q, k, freqs_abs, temporal_ids, freqs_temp)

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Mask must also match dtype and device
        mask = self.mask[:T, :T].to(device=device, dtype=attn_scores.dtype)
        attn_scores = attn_scores + mask

        attn_weights = torch.softmax(attn_scores, dim=-1).to(dtype=v.dtype)
        out = torch.matmul(attn_weights, v)
        out = combine_heads(out)

        return self.o_proj(out), None