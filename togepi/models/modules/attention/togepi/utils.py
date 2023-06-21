import numpy as np
import torch

from togepi.models.modules.attention.togepi.sparse import TogepiSparse


def prune_sparse(transformer, sparse_dens, use_spectral_norm=True, return_masks=False):
    # https://github.com/ArvindSubramaniam/Explicit-Connection-Sensitivity-Pruning-and-Quantization
    masks = []

    for module in transformer.modules():
        if isinstance(module, TogepiSparse):
            # https://discuss.pytorch.org/t/is-any-case-to-prefer-detach-cpu-to-cpu-detach-or-vice-versa/93712/2
            if use_spectral_norm:
                grad = abs(module.sparse_mat_orig.grad.detach().cpu().numpy())
                weight = abs(module.sparse_mat_orig.detach().cpu().numpy())
            else:
                grad = abs(module.sparse_mat.grad.detach().cpu().numpy())
                weight = abs(module.sparse_mat.detach().cpu().numpy())

            grad_threshold = np.percentile(grad, 100 - sparse_dens * 100)
            weight_threshold = np.percentile(weight, 100 - sparse_dens * 100)
            grad_mask = np.where(grad >= grad_threshold, 1, 0)
            weight_mask = np.where(weight >= weight_threshold, 1, 0)
            weight_or_grad_mask = np.logical_or(grad_mask, weight_mask).astype(float)
            if return_masks:
                masks.append(torch.from_numpy(weight_or_grad_mask).to(module.sparse_mat.device))
            del grad, weight, grad_mask, weight_mask  # clear out memory

            # prune the weight
            if use_spectral_norm:
                module.sparse_mat_orig.data[weight_or_grad_mask == 0.0] = 0.0
            else:
                module.sparse_mat.data[weight_or_grad_mask == 0.0] = 0.0
            del weight_or_grad_mask  # clear out memory
    if return_masks:
        return transformer, masks
    return transformer
