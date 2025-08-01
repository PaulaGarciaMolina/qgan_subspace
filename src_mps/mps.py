import torch
from typing import List
import tensorkrowch as tk
import opt_einsum as oe
from seemps.truncate.simplify_mpo import simplify_mpo, SIMPLIFICATION_STRATEGY
from seemps.mpo import MPOList, MPO
from seemps.qft import qft_mpo
import numpy as np

FT_strategy = SIMPLIFICATION_STRATEGY.replace(normalize=False)

def get_tensors(x):
    return [n.tensor for n in x.mats_env]
def mps2vector(data: List[torch.Tensor]) -> torch.Tensor:
    """
    Reconstruct a vector from a list of MPS tensors with support for multiple batch dimensions.

    Args:
        data: List of MPS tensors with shape:
              - (α, d, β) for unbatched, or
              - (..., α, d, β) for batched with arbitrary number of batch dimensions

    Returns:
        torch.Tensor: Reconstructed vector (1D for unbatched, or N-D with leading batch dims).
    """
    # Check for batching
    if isinstance(data, tk.models.MPSData):
        data = get_tensors(data)
    n = len(data)
    batch_dims = data[0].shape[:-2]  # All dims before (α, d, β)
    is_batched = len(batch_dims) > 0
    batch_size = int(torch.prod(torch.tensor(batch_dims))) if is_batched else 1
    # Normalize shapes: ensure each tensor is (..., α, d, β)
    processed = []
    for i, A in enumerate(data):
        A_shape = A.shape
        if i == 0 and len(A_shape) == 4:
            A = A.reshape(batch_size, 1, A_shape[-2], A_shape[-1])
        elif i == n - 1  and len(A_shape) == 4:
            A = A.reshape(batch_size, A_shape[-2], A_shape[-1], 1)
        else:
            A = A.reshape(batch_size, A_shape[-3], A_shape[-2], A_shape[-1])
        processed.append(A)

    # Contraction
    device = data[0].device
    dtype = data[0].dtype

    B =  batch_size
    Ψ = torch.ones(size=(B, 1, 1), device=device, dtype=dtype)

    for A in (processed):
        # A: [B, α, d, β]
        B, α, d, β = A.shape
        #Ψ = torch.einsum("bDa,bakc->bDkc", Ψ, A)
        Ψ = oe.contract("bDa,bakc->bDkc", Ψ, A)

        # Merge physical dims into one dimension for next iteration:
        B, D_prev, d, β = Ψ.shape
        Ψ = Ψ.reshape(B, D_prev * d, β)


    # Final reshape: [B, D], where D = product of all dᵢ
    return Ψ.view(B, -1)

def factor_powers_of_two(shape, n_leading_dims=2):
    leading = list(shape[:n_leading_dims])
    trailing = shape[n_leading_dims:]

    factors = []
    for dim in trailing:
        if dim & (dim - 1) != 0:
            raise ValueError(f"Dimension {dim} is not a power of 2.")
        while dim > 1:
            factors.append(2)
            dim //= 2
    return leading + factors

def mpo_seemps_to_tk(array_list):
    """
    Converts a list of NumPy arrays to PyTorch tensors and permutes them based on their position.
    Handles reshaping for the first and last tensors.

    Parameters:
        array_list (list of np.ndarray): Input arrays with varying shapes.

    Returns:
        list of torch.Tensor: List of permuted PyTorch tensors.
    """
    result = []
    n = len(array_list)

    for i, arr in enumerate(array_list):
        tensor = torch.from_numpy(arr).to(torch.complex128)
        if i == 0:
            # First tensor: [1, O, I, D2] → [O, I, D2] → [I, D2, O]
            if tensor.shape[0] != 1:
                raise ValueError(f"Expected first dim to be 1 at index 0, got shape {arr.shape}")
            tensor = tensor.squeeze(0)              # [O, I, D2]
            tensor = tensor.permute(1, 2, 0)        # [I, D2, O]

        elif i == n - 1:
            # Last tensor: [D1, O, I, 1] → [D1, O, I] → [D1, I, O]
            if tensor.shape[-1] != 1:
                raise ValueError(f"Expected last dim to be 1 at last index, got shape {arr.shape}")
            tensor = tensor.squeeze(-1)             # [D1, O, I]
            tensor = tensor.permute(0, 2, 1)        # [D1, I, O]

        else:
            # Middle: [D1, O, I, D2] → [D1, I, D2, O]
            if tensor.dim() != 4:
                raise ValueError(f"Expected 4D tensor in middle index {i}, got shape {arr.shape}")
            tensor = tensor.permute(0, 2, 3, 1)
        result.append(tensor)

    return result


def create_2D_QFT(n, extra_dims):
    FT = simplify_mpo(qft_mpo(n), strategy=FT_strategy)
    dims = [2]*n
    FT1 = FT.extend(L=2*n, sites=np.arange(n), dimensions=dims)
    FT2 = FT.extend(L=2*n, sites=np.arange(n,2*n), dimensions=dims)
    FT = MPOList([FT1, FT2]).join()
    B = len(extra_dims)
    size = B + 2*n
    FT = FT.extend(L=size, sites=np.arange(B,size), dimensions=extra_dims)
    tk_tensors = mpo_seemps_to_tk(FT._data)
    return tk.models.MPO(tensors=tk_tensors, n_batches=0)

def contract_sites(A,B):
    A = A.to(dtype=torch.complex128)
    B = B.to(dtype=torch.complex128)
    T = oe.contract('acb,dcef->adfbe', A, B)
    return T.reshape(T.shape[0]*T.shape[1], T.shape[2], T.shape[3]*T.shape[4])

def contract_mps_mpo(A,B):
    A_tensors = A.tensors
    B_tensors = B.tensors
    n = len(A_tensors)
    new_tensors = []
    for i, t_A in enumerate(A_tensors):
        t_B = B_tensors[i]
        if i == 0:
            t_A = t_A.reshape(1,t_A.shape[0], t_A.shape[1])
            t_B = t_B.reshape(1,t_B.shape[0], t_B.shape[1], t_B.shape[2])
        elif i == n - 1:
            t_A = t_A.reshape(t_A.shape[0], t_A.shape[1], 1)
            t_B = t_B.reshape(t_B.shape[0], t_B.shape[1], 1, t_B.shape[2])
        new_tensors.append(contract_sites(t_A,t_B))
    new_tensors[0] = new_tensors[0].squeeze(0)
    new_tensors[-1] = new_tensors[-1].squeeze(-1)
    new_mps = tk.models.MPSData(n_features=n,
                                          phys_dim=A.phys_dim,
                                          bond_dim=1,
                                          n_batches=0,
                                          boundary='obc')
    new_mps.add_data(new_tensors)
    return new_mps

def canonicalize(mps, cum_percentage=1.0):
    if isinstance(mps, tk.models.MPSData):
        mps = get_tensors(mps)
    L = len(mps)
    for i in range(L - 1):
        A = mps[i]
        B_next = mps[i + 1]

        Dl, d, Dr = A.shape
        A_reshaped = A.reshape(Dl * d, Dr)

        U, S, Vh = torch.linalg.svd(A_reshaped, full_matrices=False)

        S_abs = S.abs()
        S_total = S_abs.sum()
        S_cumsum = torch.cumsum(S_abs, dim=0)

        mask = (S_cumsum / S_total >= cum_percentage).nonzero(as_tuple=True)
        keep_dim = int(mask[0][0]) + 1 if len(mask[0]) > 0 else 1

        U = U[:, :keep_dim]
        S = S[:keep_dim]
        Vh = Vh[:keep_dim, :]

        new_A = U.reshape(Dl, d, keep_dim)

        S_complex = S.to(Vh.dtype)
        B_contract = torch.matmul(torch.diag(S_complex), Vh)
        new_B_next = torch.matmul(B_contract, B_next.reshape(Dr, -1))
        i_next = B_next.shape[-2]
        Dr_next = B_next.shape[-1]
        new_B_next = new_B_next.reshape(keep_dim, i_next, Dr_next)

        mps[i] = new_A
        mps[i + 1] = new_B_next

    # Remove trivial bond dims if needed
    mps[0] = mps[0].squeeze(0)
    mps[-1] = mps[-1].squeeze(-1)

    phys_dims = [t.shape[1] for t in mps]
    x_tk = tk.models.MPSData(
        n_features=len(phys_dims),
        phys_dim=phys_dims,
        bond_dim=1,
        n_batches=0,
        boundary='obc'
    )
    x_tk.add_data(mps)
    return x_tk
