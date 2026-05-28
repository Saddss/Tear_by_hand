import torch
import torch.nn as nn

from scaled_dot_product_attention import scaled_dot_product_attention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0, bias=False):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model, self.num_heads, self.d_k = d_model, num_heads, d_model // num_heads

        # 合并 H 个 W^(h) 成一个 [D, D] 矩阵
        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_k = nn.Linear(d_model, d_model, bias=bias)
        self.W_v = nn.Linear(d_model, d_model, bias=bias)
        self.W_o = nn.Linear(d_model, d_model, bias=bias)
        self.dropout_p = dropout

    def _split(self, x):  # [B, L, D] → [B, H, L, d_k]
        B, L, _ = x.shape
        return x.view(B, L, self.num_heads, self.d_k).transpose(1, 2)

    def _merge(self, x):  # [B, H, L, d_k] → [B, L, D]
        B, _, L, _ = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, self.d_model)

    def forward(self, query, key, value, mask=None):
        Q = self._split(self.W_q(query))
        K = self._split(self.W_k(key))
        V = self._split(self.W_v(value))

        if mask is not None:
            if mask.dim() == 2: mask = mask.unsqueeze(0).unsqueeze(0)   # [1,1,L_q,L_k]
            elif mask.dim() == 3: mask = mask.unsqueeze(1)              # [B,1,L_q,L_k]
            # dim=4: 已对齐

        out, w = scaled_dot_product_attention(Q, K, V, mask=mask, dropout_p=self.dropout_p, training=self.training)
        return self.W_o(self._merge(out)), w


def main():
    torch.manual_seed(42)

    mha = MultiHeadAttention(d_model=8, num_heads=2)
    mha.eval()
    x = torch.randn(1, 4, 8)

    out, w = mha(x, x, x)
    assert out.shape == (1, 4, 8)
    assert w.shape == (1, 2, 4, 4)
    assert torch.allclose(w.sum(-1), torch.ones(1, 2, 4), atol=1e-5)

    causal = torch.tril(torch.ones(4, 4, dtype=torch.bool))
    _, w = mha(x, x, x, mask=causal)
    assert w[0, 0, 1, 2] == 0

    print("OK")


if __name__ == "__main__":
    main() 