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