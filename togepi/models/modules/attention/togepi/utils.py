import numpy as np
import torch

from togepi.models.modules.attention.togepi.sparse import TogepiSparse


def prune_sparse(transformer, sparse_dens, use_spectral_norm=True, return_masks=False):
    # https://github.com/ArvindSubramaniam/Explicit-Connection-Sensitivity-Pruning-and-Quantization
    masks = []

    for module in transformer.modules():
        if isinstance(module, TogepiSparse):
            if use_spectral_norm:
                sparse = module.sparse_mat_orig
            else:
                sparse = module.sparse_mat
            grad = abs(sparse.grad.cpu().detach().numpy())
            weight = abs(sparse.cpu().detach().numpy())

            grad_threshold = np.percentile(grad, 100 - sparse_dens * 100)
            weight_threshold = np.percentile(weight, 100 - sparse_dens * 100)
            grad_mask = np.where(grad >= grad_threshold, 1, 0)
            weight_mask = np.where(weight >= weight_threshold, 1, 0)
            weight_or_grad_mask = np.logical_or(grad_mask, weight_mask).astype(float)
            if return_masks:
                masks.append(torch.from_numpy(weight_or_grad_mask).to(module.sparse_mat.device))

            # prune the weight
            sparse.data[weight_or_grad_mask == 0.0] = 0.0
    if return_masks:
        return masks
