"""Truncated Geman penalty used by the structured low-rank regularizer."""

import torch


def _resolve_svd_dtype(mat: torch.Tensor) -> torch.dtype:
    return torch.float32 if mat.dtype in (torch.float16, torch.bfloat16) else mat.dtype


def svd_safe(mat: torch.Tensor, full_matrices: bool = False):
    """Compute a numerically safer SVD for batched matrices."""
    if mat.numel() == 0:
        raise ValueError("SVD is undefined for empty tensors.")

    return torch.linalg.svd(
        mat.to(dtype=_resolve_svd_dtype(mat)),
        full_matrices=full_matrices,
    )


def svdvals_safe(mat: torch.Tensor) -> torch.Tensor:
    """Compute singular values in a dtype that is stable for SVD."""
    if mat.numel() == 0:
        return mat.new_zeros(*mat.shape[:-2], 0)

    compute_dtype = _resolve_svd_dtype(mat)
    return torch.linalg.svdvals(mat.to(dtype=compute_dtype))


def truncated_geman_tail(singular_vals: torch.Tensor, rank_r: int) -> torch.Tensor:
    """Return g(sigma_i)=sigma_i/(1+sigma_i) for the penalized tail singular values."""
    rank_r = max(int(rank_r), 0)
    if rank_r >= singular_vals.shape[-1]:
        return singular_vals.new_zeros(*singular_vals.shape[:-1], 0)

    tail = singular_vals[..., rank_r:]
    return tail / (1.0 + tail)


def truncated_geman_penalty(mat: torch.Tensor, rank_r: int) -> torch.Tensor:
    """Apply phi_r(M) = sum_{i>r} sigma_i / (1 + sigma_i)."""
    singular_vals = svdvals_safe(mat)
    return truncated_geman_tail(singular_vals, rank_r).sum(dim=-1)


def truncated_geman_spectral_profile(
    mat: torch.Tensor,
    rank_r: int,
    reduction: str = 'symmetric',
) -> torch.Tensor:
    """
    Decompose the truncated Geman tail into row/column attributions.

    For M = U diag(sigma) V^T and tail weights g_i = sigma_i / (1 + sigma_i),
    the row contribution is sum_i g_i * U[:, i]^2 and the column contribution is
    sum_i g_i * V[:, i]^2. The symmetric profile averages both sides.
    """
    if mat.numel() == 0:
        if reduction in ('row', 'symmetric'):
            return mat.new_zeros(*mat.shape[:-2], mat.shape[-2])
        if reduction == 'col':
            return mat.new_zeros(*mat.shape[:-2], mat.shape[-1])
        raise ValueError(f'Unsupported reduction: {reduction}')

    if reduction not in ('row', 'col', 'symmetric'):
        raise ValueError(f'Unsupported reduction: {reduction}')

    u, singular_vals, vh = svd_safe(mat, full_matrices=False)
    tail = truncated_geman_tail(singular_vals, rank_r)
    if tail.shape[-1] == 0:
        if reduction == 'row':
            return u.new_zeros(*u.shape[:-2], u.shape[-2]).to(dtype=mat.dtype)
        if reduction == 'col':
            return vh.new_zeros(*vh.shape[:-2], vh.shape[-1]).to(dtype=mat.dtype)
        return u.new_zeros(*u.shape[:-2], u.shape[-2]).to(dtype=mat.dtype)

    tail_u = u[..., :, rank_r:]
    tail_v = vh.transpose(-2, -1)[..., :, rank_r:]

    row_profile = (tail_u.square() * tail.unsqueeze(-2)).sum(dim=-1)
    col_profile = (tail_v.square() * tail.unsqueeze(-2)).sum(dim=-1)

    if reduction == 'row':
        return row_profile.to(dtype=mat.dtype)
    if reduction == 'col':
        return col_profile.to(dtype=mat.dtype)
    return (0.5 * (row_profile + col_profile)).to(dtype=mat.dtype)
