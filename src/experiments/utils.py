import torch
def project_nonneg_l1_ball(v: torch.Tensor, r: float) -> torch.Tensor:
    """
    Euclidean projection of a non-negative vector v onto {x >= 0, ||x||_1 <= r}.
    
    Parameters
    ----------
    v : torch.Tensor
        Input tensor of shape (n,). Must be non-negative.
    r : float
        L1 radius (must be >= 0).

    Returns
    -------
    torch.Tensor
        Projection of v onto the non-negative L1 ball of radius r.
    """
    if r < 0:
        raise ValueError("r must be non-negative.")
    if (v < 0).any():
        raise ValueError("v must be elementwise non-negative.")
        
    if v.sum() <= r:
        return v.clone()

    # Sort v in descending order
    u, _ = torch.sort(v, descending=True)
    cssv = torch.cumsum(u, dim=0)
    j = torch.arange(1, v.numel() + 1, dtype=v.dtype, device=v.device)

    # Find rho = max { i : u_i - (cssv_i - r)/i > 0 }
    cond = u - (cssv - r) / j > 0
    rho = torch.nonzero(cond, as_tuple=False)[-1, 0]
    theta = (cssv[rho] - r) / (rho + 1)

    return torch.clamp(v - theta, min=0.0)

