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
"""Cost and Fidelity Functions"""

import torch 
from seemps.state import MPS, scprod
import numpy as np
from config import CFG


def compute_cost(dis, final_target_state: torch.Tensor, final_gen_state: torch.Tensor, config=CFG) -> float:
    """Calculate the cost function. Which is basically equivalent to the Wasserstein distance.

    Args:
        dis (Discriminator): the discriminator.
        final_target_state (np.ndarray): the target state to input into the Discriminator.
        final_gen_state (np.ndarray): the gen state to input into the Discriminator.
        config (Config): training configuration (defaults to CFG).
    Returns:
        float: the cost function.
    """
    A, B, psi, phi = dis.get_dis_matrices_rep()   
    # fmt: off
    A_final_gen_state = A @ final_gen_state
    B_final_gen_state = B @ final_gen_state

    term1 = scprod(final_gen_state, A_final_gen_state)
    term2 = scprod(final_target_state, B @ final_target_state)

    term3 = scprod(B_final_gen_state, final_target_state)
    term4 = scprod(final_target_state, A_final_gen_state)

    term5 = scprod(A_final_gen_state, final_target_state)
    term6 = scprod(final_target_state, B_final_gen_state)

    term7 = scprod(B_final_gen_state, final_gen_state)
    term8 = scprod(final_target_state, A @ final_target_state)

    psiterm = scprod(final_target_state, psi @ final_target_state)
    phiterm = scprod(final_gen_state, phi @ final_gen_state)

    regterm = (config.lamb / torch.e) * (config.cst1 * term1 * term2 - config.cst2 * (term3 * term4 + term5 * term6) + config.cst3 * term7 * term8)
    # fmt: on

    # The final loss must be a real-valued scalar tensor
    loss = (psiterm - phiterm - regterm).real
    return loss


def compute_fidelity(final_target_state: torch.Tensor, final_gen_state: torch.Tensor) -> float:
    """Calculate the fidelity between target state and gen state

    Args:
        final_target_state (np.ndarray): The final target state of the system.
        final_gen_state (np.ndarray): The final gen state of the system.

    Returns:
        float: the fidelity between the target state and the gen state.
    """
    braket_result = torch.vdot(final_target_state.flatten(), final_gen_state.flatten())
    # .item() extracts the scalar value from the tensor
    return torch.abs(braket_result).pow(2).item()


def compute_fidelity_and_cost(dis, final_target_state: torch.Tensor, final_gen_state: torch.Tensor,
                              final_target_state_mps, final_gen_state_mps) -> tuple[float, float]:
    """Calculate the fidelity and cost function

    Args:
        dis (Discriminator): the discriminator.
        final_target_state (np.ndarray): the target state.
        final_gen_state (np.ndarray): the gen state.

    Returns:
        tuple[float, float]: the fidelity and cost function.
    """
    fidelity = compute_fidelity(final_target_state, final_gen_state)
    cost = compute_cost(dis, final_target_state_mps, final_gen_state_mps)

    return fidelity, cost