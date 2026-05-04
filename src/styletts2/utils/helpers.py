import numpy as np
import torch

try:
    from monotonic_align.core import maximum_path_c  # type: ignore
except ImportError:
    # Fallback or placeholder if monotonic_align is not installed
    def maximum_path_c(*args, **kwargs):
        raise ImportError(
            "monotonic_align.core is not installed. Please install it to use maximum_path."
        )


def maximum_path(neg_cent, mask):
    """Cython optimized version.
    neg_cent: [b, t_t, t_s]
    mask: [b, t_t, t_s]
    """
    device = neg_cent.device
    dtype = neg_cent.dtype
    neg_cent = np.ascontiguousarray(neg_cent.data.cpu().numpy().astype(np.float32))
    path = np.ascontiguousarray(np.zeros(neg_cent.shape, dtype=np.int32))

    t_t_max = np.ascontiguousarray(
        mask.sum(1)[:, 0].data.cpu().numpy().astype(np.int32)
    )
    t_s_max = np.ascontiguousarray(
        mask.sum(2)[:, 0].data.cpu().numpy().astype(np.int32)
    )
    maximum_path_c(path, neg_cent, t_t_max, t_s_max)
    return torch.from_numpy(path).to(device=device, dtype=dtype)


def length_to_mask(lengths):
    mask = (
        torch.arange(lengths.max())
        .unsqueeze(0)
        .expand(lengths.shape[0], -1)
        .type_as(lengths)
    )
    mask = torch.gt(mask + 1, lengths.unsqueeze(1))
    return mask


def get_data_path_list(train_path, val_path):
    with open(train_path, encoding="utf-8", errors="ignore") as f:
        train_list = f.readlines()
    with open(val_path, encoding="utf-8", errors="ignore") as f:
        val_list = f.readlines()
    return train_list, val_list
