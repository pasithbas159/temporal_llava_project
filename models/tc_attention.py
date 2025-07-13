import torch
import torch.nn as nn
import math

# ====== RoPE Helpers ======
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
    freqs_abs = freqs_abs.to(q.device)[None, None, :q.size(2)]
    q = apply_rope(q, freqs_abs)
    k = apply_rope(k, freqs_abs)

    freqs_temp = freqs_temp.to(q.device)[temporal_ids]
    freqs_temp = freqs_temp[None, None, :, :]
    q = apply_rope(q, freqs_temp)
    k = apply_rope(k, freqs_temp)

    return q, k

# ====== Temporal ID Computation ======
def compute_temporal_ids_corrected(seq_len, v_s, v_e, m, gamma=1.0):
    """
    Compute temporal position ids per paper Eq. (5) and Eq. (6).

    Args:
        seq_len: total sequence length (int)
        v_s: start index of visual tokens (int)
        v_e: end index of visual tokens (int)
        m: number of visual tokens per frame (int)
        gamma: scaling factor (float)

    Returns:
        temporal_ids: tensor of shape (seq_len,)
    """
    n = torch.arange(seq_len)

    # Calculate floor terms
    floor_term = torch.floor((n - v_s).float() / m)
    floor_term_clip = torch.floor(torch.tensor((v_e - v_s) / m, dtype=torch.float32))

    temporal_ids = torch.zeros_like(n, dtype=torch.float)

    # Case 1: n < v_s
    mask1 = n < v_s
    temporal_ids[mask1] = n[mask1].float()

    # Case 2: v_s <= n <= v_e
    mask2 = (n >= v_s) & (n <= v_e)
    temporal_ids[mask2] = v_s + floor_term[mask2]

    # Case 3: n > v_e
    mask3 = n > v_e
    temporal_ids[mask3] = n[mask3].float() - (v_e - v_s + 1 - floor_term_clip)

    # Apply scaling gamma as per Eq. (6)
    temporal_ids = n.float() + gamma * temporal_ids

    return temporal_ids.float()

# ====== Attention Mask Helper ======
def build_fwbc_mask(seq_len, frame_size):
    mask = torch.full((seq_len, seq_len), float('-inf'))
    for i in range(seq_len):
        for j in range(i + 1):  # causal
            mask[i, j] = 0
        if i // frame_size == j // frame_size:
            mask[i, j] = 0  # same frame
    return mask

# ====== Head split/merge ======
def split_heads(x, head_dim):
    B, T, C = x.shape
    H = C // head_dim
    x = x.view(B, T, H, head_dim).transpose(1, 2)
    return x

def combine_heads(x):
    B, H, T, dh = x.shape
    return x.transpose(1, 2).reshape(B, T, H * dh)

# ====== TCAttention Module with Dynamic Temporal IDs ======
class TCAttention(nn.Module):
    def __init__(self, orig_attn, frame_size=576, gamma=1.0, max_seq_len=2048):
        super().__init__()
        self.q_proj = orig_attn.q_proj
        self.k_proj = orig_attn.k_proj
        self.v_proj = orig_attn.v_proj
        self.o_proj = orig_attn.o_proj

        self.embed_dim = self.q_proj.out_features
        self.frame_size = frame_size
        self.gamma = gamma
        self.num_heads = getattr(orig_attn, "num_heads", 32)
        self.head_dim = self.embed_dim // self.num_heads
        self.max_seq_len = max_seq_len

        self.freqs_abs = precompute_rope_freqs(max_seq_len, self.head_dim)
        self.freqs_temp = precompute_rope_freqs(max_seq_len, self.head_dim)
        self.mask = build_fwbc_mask(max_seq_len, frame_size)

    def forward(self, hidden_states, attention_mask=None, visual_token_mask=None, **kwargs):
        B, T, C = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        assert visual_token_mask is not None, "You must pass visual_token_mask"
        temporal_ids_batch = []

        for b in range(B):
            visual_mask = visual_token_mask[b]
            visual_indices = torch.where(visual_mask)[0]
            if len(visual_indices) == 0:
                temporal_ids = torch.arange(T).long()
            else:
                v_s = visual_indices[0].item()
                v_e = visual_indices[-1].item()
                temporal_ids = compute_temporal_ids_corrected(
                    seq_len=T,
                    v_s=v_s,
                    v_e=v_e,
                    m=self.frame_size,
                    gamma=self.gamma
                )
            temporal_ids_batch.append(temporal_ids)

        temporal_ids = torch.stack(temporal_ids_batch).to(device=device)

        q = split_heads(self.q_proj(hidden_states), self.head_dim)
        k = split_heads(self.k_proj(hidden_states), self.head_dim)
        v = split_heads(self.v_proj(hidden_states), self.head_dim)

        freqs_abs = self.freqs_abs[:T].to(device=device, dtype=dtype)
        freqs_temp = self.freqs_temp.to(device=device, dtype=dtype)

        q_out, k_out = [], []
        for b in range(B):
            q_b, k_b = apply_dual_rope(
                q[b:b+1], k[b:b+1], freqs_abs, temporal_ids[b], freqs_temp
            )
            q_out.append(q_b)
            k_out.append(k_b)

        q = torch.cat(q_out, dim=0)
        k = torch.cat(k_out, dim=0)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = self.mask[:T, :T].to(device=device, dtype=attn_scores.dtype)
        attn_scores = attn_scores + mask

        attn_weights = torch.softmax(attn_scores, dim=-1).to(dtype=v.dtype)
        out = torch.matmul(attn_weights, v)
        out = combine_heads(out)

        return self.o_proj(out), None