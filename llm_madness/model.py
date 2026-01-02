from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float = 0.1
    use_rmsnorm: bool = False
    use_swiglu: bool = False
    use_rope: bool = False
    use_sdpa: bool = False
    use_kv_cache: bool = False
    rope_theta: float = 10000.0
    rope_cache: tuple[torch.Tensor, torch.Tensor] | None = None


class RMSNorm(nn.Module):
    def __init__(self, n_embd: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embd))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(rms + self.eps)
        return x * self.weight


def build_rope_cache(config: GPTConfig) -> tuple[torch.Tensor, torch.Tensor]:
    if config.rope_cache is not None:
        return config.rope_cache
    head_dim = config.n_embd // config.n_head
    if head_dim % 2 != 0:
        raise ValueError("RoPE requires head_dim to be even")
    inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(config.block_size, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
    config.rope_cache = (cos, sin)
    return config.rope_cache


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.use_rope = bool(config.use_rope)
        self.use_sdpa = bool(config.use_sdpa)

        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)

        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))
        if self.use_rope:
            cos, sin = build_rope_cache(config)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)

    def _apply_rope(
        self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos = self.rope_cos.index_select(0, positions)
        sin = self.rope_sin.index_select(0, positions)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        half = self.head_dim // 2
        q1, q2 = q[..., :half], q[..., half:]
        k1, k2 = k[..., :half], k[..., half:]
        q = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
        k = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)
        return q, k

    def _manual_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        t: int,
        k_len: int,
        positions: torch.Tensor | None,
    ) -> torch.Tensor:
        att = (q @ k.transpose(-2, -1)) * self.scale
        if positions is None:
            att = att.masked_fill(self.mask[:, :, :t, :k_len] == 0, float("-inf"))
        else:
            k_pos = torch.arange(k_len, device=att.device)
            causal = k_pos[None, :] <= positions[:, None]
            att = att.masked_fill(causal.unsqueeze(0).unsqueeze(0) == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        return att

    def forward(self, x: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        b, t, c = x.size()
        qkv = self.qkv(x)
        q, k, v = qkv.split(c, dim=2)
        q = q.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_head, self.head_dim).transpose(1, 2)

        if self.use_rope:
            if positions is None:
                positions = torch.arange(t, device=x.device)
            q, k = self._apply_rope(q, k, positions)

        if self.use_sdpa and x.is_cuda:
            dropout_p = self.attn_drop.p if self.training else 0.0
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=True)
        else:
            att = self._manual_attention(q, k, t, t, None)
            att = self.attn_drop(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        y = self.resid_drop(self.proj(y))
        return y

    def forward_with_attn(
        self, x: torch.Tensor, positions: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, c = x.size()
        qkv = self.qkv(x)
        q, k, v = qkv.split(c, dim=2)
        q = q.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_head, self.head_dim).transpose(1, 2)

        if self.use_rope:
            if positions is None:
                positions = torch.arange(t, device=x.device)
            q, k = self._apply_rope(q, k, positions)

        att = self._manual_attention(q, k, t, t, None)
        att_drop = self.attn_drop(att)
        y = att_drop @ v
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        y = self.resid_drop(self.proj(y))
        return y, att

    def forward_with_kv(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        b, t, c = x.size()
        qkv = self.qkv(x)
        q, k, v = qkv.split(c, dim=2)
        q = q.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_head, self.head_dim).transpose(1, 2)

        if self.use_rope:
            q, k = self._apply_rope(q, k, positions)

        if kv_cache is not None:
            k = torch.cat([kv_cache["k"], k], dim=2)
            v = torch.cat([kv_cache["v"], v], dim=2)

        k_len = k.size(2)
        att = self._manual_attention(q, k, t, k_len, positions)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        y = self.resid_drop(self.proj(y))
        return y, {"k": k, "v": v}


class MLP(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = F.gelu(x)
        x = self.proj(x)
        return self.drop(x)


class SwiGLU(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        hidden_dim = 4 * config.n_embd
        self.fc = nn.Linear(config.n_embd, 2 * hidden_dim)
        self.proj = nn.Linear(hidden_dim, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x, gate = x.chunk(2, dim=-1)
        x = F.silu(gate) * x
        x = self.proj(x)
        return self.drop(x)


class Block(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        norm = RMSNorm if config.use_rmsnorm else nn.LayerNorm
        self.ln1 = norm(config.n_embd)
        self.ln2 = norm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = SwiGLU(config) if config.use_swiglu else MLP(config)

    def forward(self, x: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), positions=positions)
        x = x + self.mlp(self.ln2(x))
        return x

    def forward_with_trace(
        self, x: torch.Tensor, positions: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attn_out, attn = self.attn.forward_with_attn(self.ln1(x), positions=positions)
        x = x + attn_out
        mlp_out = self.mlp(self.ln2(x))
        x = x + mlp_out
        return x, attn, mlp_out

    def forward_with_kv(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        attn_out, new_cache = self.attn.forward_with_kv(self.ln1(x), positions, kv_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, new_cache


class GPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        if config.use_rope:
            build_rope_cache(config)
            self.pos_emb = None
        else:
            self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = RMSNorm(config.n_embd) if config.use_rmsnorm else nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        b, t = idx.size()
        if t > self.config.block_size:
            raise ValueError("sequence length exceeds block size")

        positions = torch.arange(0, t, device=idx.device) if self.config.use_rope else None
        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            pos = torch.arange(0, t, device=idx.device)
            x = x + self.pos_emb(pos)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x, positions=positions)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def forward_with_trace(self, idx: torch.Tensor):
        b, t = idx.size()
        if t > self.config.block_size:
            raise ValueError("sequence length exceeds block size")

        positions = torch.arange(0, t, device=idx.device) if self.config.use_rope else None
        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            pos = torch.arange(0, t, device=idx.device)
            x = x + self.pos_emb(pos)
        x = self.drop(x)

        attn_maps: list[torch.Tensor] = []
        mlp_outputs: list[torch.Tensor] = []
        for block in self.blocks:
            x, attn, mlp_out = block.forward_with_trace(x, positions=positions)
            attn_maps.append(attn)
            mlp_outputs.append(mlp_out)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits, {"attn": attn_maps, "mlp": mlp_outputs}

    def forward_with_hidden_states(self, idx: torch.Tensor):
        b, t = idx.size()
        if t > self.config.block_size:
            raise ValueError("sequence length exceeds block size")

        positions = torch.arange(0, t, device=idx.device) if self.config.use_rope else None
        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            pos = torch.arange(0, t, device=idx.device)
            x = x + self.pos_emb(pos)
        x = self.drop(x)

        hidden_states: list[torch.Tensor] = []
        for block in self.blocks:
            x = block(x, positions=positions)
            hidden_states.append(x)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits, hidden_states

    def _forward_with_kv(
        self, idx: torch.Tensor, kv_cache: list[dict[str, torch.Tensor] | None]
    ) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        b, t = idx.size()
        if t > self.config.block_size:
            raise ValueError("sequence length exceeds block size")

        past_len = 0
        if kv_cache and kv_cache[0] is not None:
            past_len = kv_cache[0]["k"].size(2)
        positions = torch.arange(past_len, past_len + t, device=idx.device)

        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            x = x + self.pos_emb(positions)
        x = self.drop(x)

        new_cache: list[dict[str, torch.Tensor]] = []
        for block, layer_cache in zip(self.blocks, kv_cache):
            x, next_cache = block.forward_with_kv(x, positions, layer_cache)
            new_cache.append(next_cache)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits, new_cache

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        if not self.config.use_kv_cache:
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -self.config.block_size :]
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :] / max(temperature, 1e-5)
                if top_k is not None:
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[:, [-1]]] = float("-inf")
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, next_id), dim=1)
            return idx

        kv_cache: list[dict[str, torch.Tensor] | None] = [None] * len(self.blocks)
        for _ in range(max_new_tokens):
            if idx.size(1) > self.config.block_size:
                idx_cond = idx[:, -self.config.block_size :]
                kv_cache = [None] * len(self.blocks)
            else:
                idx_cond = idx

            if kv_cache[0] is not None and kv_cache[0]["k"].size(2) >= self.config.block_size:
                idx_cond = idx[:, -self.config.block_size :]
                kv_cache = [None] * len(self.blocks)

            if kv_cache[0] is None:
                logits, kv_cache = self._forward_with_kv(idx_cond, kv_cache)
            else:
                logits, kv_cache = self._forward_with_kv(idx_cond[:, -1:], kv_cache)

            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx
