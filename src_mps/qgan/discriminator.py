# Copyright 2025 GIQ, Universitat Autònoma de Barcelona
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Discriminator module implemented in PyTorch."""

import torch
import torch.nn as nn
import numpy as np 
from seemps.operators import MPO, MPOSum, MPOList
#from seemps.truncate.simplify_mpo import simplify_mpo
from seemps.truncate.simplify import SIMPLIFICATION_STRATEGY
import scipy
from config import CFG
from tools.qobjects.qgates import I, X, Y, Z, device, COMPLEX_TYPE

strategy = SIMPLIFICATION_STRATEGY.replace(normalize=False)

class Discriminator(nn.Module):
    """
    Discriminator class for the Quantum GAN, implemented as a PyTorch Module.
    The parameters alpha and beta are learned automatically via AutoGrad.
    """
    def __init__(self, config = CFG):
        super().__init__()
        self.config = config
        # Determine the size of the Hilbert space the discriminator acts on
        self.size: int = self.config.system_size * 2 + (1 if self.config.extra_ancilla and self.config.ancilla_mode == "pass" else 0)
        
        # Store a list of Hermitian operators (Pauli matrices)
        self.herm: list[torch.Tensor] = [I, X, Y, Z]

        # Define alpha and beta as trainable parameters
        # They are initialized with random values between -1 and 1
        alpha_init = torch.empty(self.size, len(self.herm)).uniform_(-1, 1)
        beta_init = torch.empty(self.size, len(self.herm)).uniform_(-1, 1)
        self.alpha = nn.Parameter(alpha_init)
        self.beta = nn.Parameter(beta_init)
        
        # Store config for compatibility checks during model loading
        self.target_size: int = self.config.system_size
        self.target_hamiltonian: str = self.config.target_hamiltonian
        self.ancilla_mode: str = self.config.ancilla_mode

    def forward_old(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Constructs the psi and phi matrices from the alpha and beta parameters.
        This defines the forward pass of the discriminator model.
        """
        device = self.alpha.device
        psi = torch.tensor([[1.+0.j]], dtype=COMPLEX_TYPE, device=device)
        phi = torch.tensor([[1.+0.j]], dtype=COMPLEX_TYPE, device=device)

        for i in range(self.size):
            # Sum of Pauli matrices scaled by alpha/beta coefficients for the i-th qubit
            psi_i = sum(self.alpha[i, j] * self.herm[j] for j in range(len(self.herm)))
            phi_i = sum(self.beta[i, j] * self.herm[j] for j in range(len(self.herm)))
            
            # Build the full psi and phi matrices via Kronecker product
            psi = torch.kron(psi, psi_i)
            phi = torch.kron(phi, phi_i)
            
        return psi, phi
    
    def kron_all(self, op_list):
        result = op_list[0]
        for op in op_list[1:]:
            result = torch.kron(result, op)
        return result
    
    def lift_to_full(self, op, site, size, I):
        ops = [I] * size
        ops[site] = op
        return self.kron_all(ops)
    
    def op_to_tensor_list(self, op, site, size, I):
        ops = [I.reshape(1,2,2,1)] * size
        ops[site] = op.reshape(1,2,2,1)
        return ops

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Constructs the psi and phi matrices from the alpha and beta parameters.
        This defines the forward pass of the discriminator model.
        """
        device = self.alpha.device
        I = self.herm[0].to(device)
        lamb = self.config.lamb

        # Build lifted local ops and sum them
        psi_tensors = []
        phi_tensors = []
        psi_exp_tensors = []
        phi_exp_tensors = []

        for i in range(self.size):
            # Local operator on site i
            psi_i = sum(self.alpha[i, j] * self.herm[j].to(device) for j in range(len(self.herm)))
            phi_i = sum(self.beta[i, j] * self.herm[j].to(device) for j in range(len(self.herm)))

            psi_tensors.append(MPO(self.op_to_tensor_list(psi_i, i, self.size, I)))
            psi_exp_tensors.append(MPO(self.op_to_tensor_list(torch.matrix_exp((1.0 / lamb) * psi_i), i, self.size, I)))
            phi_tensors.append(MPO(self.op_to_tensor_list(phi_i, i, self.size, I)))
            phi_exp_tensors.append(MPO(self.op_to_tensor_list(torch.matrix_exp((-1.0 / lamb) * phi_i), i, self.size, I)))
       
        return MPOSum(psi_tensors).join(), MPOSum(phi_tensors).join(), psi_exp_tensors, phi_exp_tensors
    
    def get_dis_matrices_rep(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the matrices A and B from psi and phi. These are used in the cost function.
        """
        psi, phi, psi_exp_tensors, phi_exp_tensors = self.forward()
        
        # lamb is a hyperparameter from the config
        
        # A = expm(-1/lamb * phi)
        A = MPOList(psi_exp_tensors).join()
        #A = simplify_mpo(MPOList(psi_exp_tensors).join(), strategy=strategy)
        # B = expm(1/lamb * psi)
        B = MPOList(phi_exp_tensors).join()
        #B = simplify_mpo(MPOList(phi_exp_tensors).join(), strategy=strategy)
        A.strategy = A.strategy.replace(simplify=0, method=0)
        B.strategy = B.strategy.replace(simplify=0, method=0)
        phi.strategy = phi.strategy.replace(simplify=0, method=0)
        psi.strategy = psi.strategy.replace(simplify=0, method=0)
        return A, B, psi, phi
    
    def get_dis_matrices_rep_old(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the matrices A and B from psi and phi. These are used in the cost function.
        """
        psi, phi = self.forward()
        
        # lamb is a hyperparameter from the config
        lamb = self.config.lamb
        
        # A = expm(-1/lamb * phi)
        A = torch.matrix_exp((-1.0 / lamb) * phi)
        # B = expm(1/lamb * psi)
        B = torch.matrix_exp((1.0 / lamb) * psi)

        return A, B, psi, phi
        
    def load_model_params(self, file_path: str):
        """Loads discriminator parameters from a saved state_dict."""
        try:
            # Load the entire saved dictionary, which includes config
            saved_data = torch.load(file_path, map_location=device)
            saved_config = saved_data.get('config', {})
            
            # --- Perform compatibility checks ---
            if saved_config.get('target_size') != self.target_size:
                raise ValueError("Incompatible target size.")
            if saved_config.get('target_hamiltonian') != self.target_hamiltonian:
                raise ValueError("Incompatible target Hamiltonian.")
            if saved_config.get('ancilla_mode') != self.ancilla_mode:
                raise ValueError("Incompatible ancilla mode.")

            self.load_state_dict(saved_data['model_state_dict'])
            print(f"Discriminator parameters loaded successfully from {file_path}")

        except Exception as e:
            print(f"ERROR: Could not load discriminator model: {e}")

    def save_model_params(self, file_path: str):
        """Saves discriminator parameters and config to a file."""
        save_data = {
            'model_state_dict': self.state_dict(),
            'config': {
                'target_size': self.target_size,
                'target_hamiltonian': self.target_hamiltonian,
                'ancilla_mode': self.ancilla_mode,
            }
        }
        torch.save(save_data, file_path)
        print(f"Discriminator model saved to {file_path}")