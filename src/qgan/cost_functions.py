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

import numpy as np

from config import CFG

np.random.seed()

def compute_cost(dis, final_target_state: np.ndarray, final_gen_state: np.ndarray) -> float:
    """Calculate the cost function. Which is basically equivalent to the Wasserstein distance.

    Args:
        dis (Discriminator): the discriminator.
        final_target_state (np.ndarray): the target state to input into the Discriminator.
        final_gen_state (np.ndarray): the gen state to input into the Discriminator.

    Returns:
        float: the cost function.
    """
    A, B, psi, phi = dis.get_dis_matrices_rep()

    # fmt: off
    A_final_gen_state = A @ final_gen_state
    B_final_gen_state = B @ final_gen_state

    term1 = np.vdot(final_gen_state, A_final_gen_state)
    term2 = np.vdot(final_target_state, B @ final_target_state)

    term3 = np.vdot(B_final_gen_state, final_target_state)
    term4 = np.vdot(final_target_state, A_final_gen_state)

    term5 = np.vdot(A_final_gen_state, final_target_state)
    term6 = np.vdot(final_target_state, B_final_gen_state)

    term7 = np.vdot(B_final_gen_state, final_gen_state)
    term8 = np.vdot(final_target_state, A @ final_target_state)

    psiterm = np.trace(np.outer(final_target_state, final_target_state.conj().T) @ psi)
    phiterm = np.trace(np.outer(final_gen_state, final_gen_state.conj().T) @ phi)

    regterm = CFG.lamb / np.e * (CFG.cst1 * term1 * term2 - CFG.cst2 * (term3 * term4 + term5 * term6) + CFG.cst3 * term7 * term8)
    # fmt: on

    loss = np.real(psiterm - phiterm - regterm)

    return loss


def compute_fidelity(final_target_state: np.ndarray, final_gen_state: np.ndarray) -> float:
    """Calculate the fidelity between target state and gen state

    Args:
        final_target_state (np.ndarray): The final target state of the system.
        final_gen_state (np.ndarray): The final gen state of the system.

    Returns:
        float: the fidelity between the target state and the gen state.
    """
    braket_result = np.vdot(final_target_state, final_gen_state)
    return np.abs(braket_result) ** 2
    # return np.abs(np.asscalar(np.matmul(target_state.conj().T, total_final_state))) ** 2


def compute_fidelity_and_cost(dis, final_target_state: np.ndarray, final_gen_state: np.ndarray) -> tuple[float, float]:
    """Calculate the fidelity and cost function

    Args:
        dis (Discriminator): the discriminator.
        final_target_state (np.ndarray): the target state.
        final_gen_state (np.ndarray): the gen state.

    Returns:
        tuple[float, float]: the fidelity and cost function.
    """
    fidelity = compute_fidelity(final_target_state, final_gen_state)
    cost = compute_cost(dis, final_target_state, final_gen_state)

    return fidelity, cost
