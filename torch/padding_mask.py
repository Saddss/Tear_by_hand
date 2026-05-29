import torch


def padding_mask(lengths, max_len=None):
    if max_len is None:
        max_len = int(lengths.max())
    idx = torch.arange(max_len, device=lengths.device).unsqueeze(0).expand(len(lengths), -1)
    return idx < lengths.unsqueeze(1)  # [B, L]  True=valid, False=padding


def main():
    lengths = torch.tensor([3, 2])
    mask = padding_mask(lengths, max_len=4)

    assert mask.shape == (2, 4)
    assert mask.tolist() == [[True, True, True, False], [True, True, False, False]]
    print("OK")


if __name__ == "__main__":
    main()
