import torch 
from seemps.state import MPS
def from_vector_qr(state: torch.Tensor, dims, max_bond_dim: int = None, normalize=True):
        """
        Construct an MPS from a state vector using QR decomposition (instead of SVD).
        Keeps the process differentiable and allows truncation.

        Args:
            state (torch.Tensor): The full state vector (1D).
            dims (List[int]): Physical dimension at each site (e.g., [2]*n).
            max_bond_dim (int, optional): If set, truncate bond dims to this maximum.
            normalize (bool): If True, normalize the state.

        Returns:
            MPS: The compressed MPS object.
        """
        assert state.ndim == 1, "State must be 1D"
        n_sites = len(dims)

        psi = state
        if normalize:
            psi = psi / psi.norm()

        tensors = []
        bond_dim = 1

        for i in range(n_sites - 1):
            d = dims[i]
            psi = psi.reshape(bond_dim * d, -1)

            # QR decomposition
            q, r = torch.linalg.qr(psi)

            # Optional truncation
            if max_bond_dim is not None and q.shape[1] > max_bond_dim:
                q = q[:, :max_bond_dim]
                r = r[:max_bond_dim, :]

            tensor = q.reshape(bond_dim, d, -1)
            tensors.append(tensor)
            psi = r
            bond_dim = tensor.shape[2]  # next left dim

        # Last tensor
        tensors.append(psi.reshape(bond_dim, dims[-1], 1))
        return MPS(tensors)