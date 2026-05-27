import math

import torch
import torch.nn.functional as F


def scaled_dot_product_attention(Q, K, V, mask=None, dropout_p=0.0, training=True):
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1)                 # [..., L_q, L_k]
    scores = scores / math.sqrt(d_k)                 # ← 关键 scale

    if mask is not None:
        all_masked = (~mask).all(dim=-1, keepdim=True)
        safe_mask = mask | all_masked
        scores = scores.masked_fill(~safe_mask, float("-inf"))
    else:
        all_masked = None

    weights = F.softmax(scores, dim=-1)

    if all_masked is not None:
        weights = weights.masked_fill(all_masked, 0.0)   # NaN 防护

    if dropout_p > 0.0 and training:
        weights = F.dropout(weights, p=dropout_p)

    return weights @ V, weights                       # output, weights


def main():
    torch.manual_seed(42)

    Q = torch.randn(1, 1, 4, 8)
    K = torch.randn(1, 1, 4, 8)
    V = torch.randn(1, 1, 4, 8)

    out, weights = scaled_dot_product_attention(Q, K, V)
    assert out.shape == (1, 1, 4, 8)
    assert torch.allclose(weights.sum(-1), torch.ones(1, 1, 4))

    causal = torch.tril(torch.ones(4, 4, dtype=torch.bool))[None, None]
    _, w = scaled_dot_product_attention(Q, K, V, mask=causal)
    assert w[0, 0, 1, 2] == 0  # query1 不能看 key2

    ref = F.scaled_dot_product_attention(Q, K, V)
    assert torch.allclose(out, ref, atol=1e-5)
    print("OK")


if __name__ == "__main__":
    main()
