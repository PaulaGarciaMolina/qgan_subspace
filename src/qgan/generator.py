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
"""Generator module implemented in PyTorch."""

import torch
import torch.nn as nn
import itertools
from config import CFG
from tools.qobjects.qcircuit import QuantumCircuit
from tools.qobjects.qgates import QuantumGate, Identity
import os
import pickle
from qgan.ansatz import ZZ_X_Z_circuit, XX_YY_ZZ_Z_circuit

class Generator(nn.Module):
    """Generator class for the Quantum GAN, implemented as a PyTorch Module."""

    def __init__(self, config = CFG):
        super().__init__()
        self.config = config
        self.target_size: int = self.config.system_size
        
        # Store config for loading/saving compatibility checks
        self.ancilla: bool = self.config.extra_ancilla
        self.ancilla_topology: str = self.config.ancilla_topology
        self.ansatz_type: str = self.config.gen_ansatz
        self.layers: int = self.config.gen_layers
        self.target_hamiltonian: str = self.config.target_hamiltonian

        # The circuit is now a submodule of the Generator
        if config.gen_ansatz == 'ZZ_X_Z':
            self.ansatz = ZZ_X_Z_circuit(config=self.config)
        elif config.gen_ansatz == 'XX_YY_ZZ_Z':
            self.ansatz = XX_YY_ZZ_Z_circuit(config=self.config)

    def forward(self, total_input_state: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the generator. It applies the quantum circuit
        to the input state.
        """
        return self.ansatz(total_input_state.flatten()).to(total_input_state.device)

    def load_model_params(self, file_path: str):
        """Load generator parameters from a torch file."""
        try:
            saved_data = torch.load(file_path, map_location="cpu")

            saved_config = saved_data.get("config", {})
            theta = saved_data.get("theta", None)
            if theta is None:
                raise ValueError("No theta found in checkpoint.")

            # --- Compatibility checks ---
            if saved_config.get("target_size") != self.target_size:
                raise ValueError("Incompatible target size.")
            if saved_config.get("target_hamiltonian") != self.target_hamiltonian:
                raise ValueError("Incompatible target Hamiltonian.")
            if saved_config.get("ansatz_type") != self.ansatz_type:
                raise ValueError("Incompatible ansatz type.")
            if saved_config.get("layers") != self.layers:
                raise ValueError("Incompatible number of layers.")

            with torch.no_grad():
                current_n = self.ansatz.n_params
                saved_n = len(theta)

                if saved_n < current_n:
                    print(f"Expanding θ: {saved_n} → {current_n}")
                    padded = torch.zeros(current_n, dtype=theta.dtype)
                    padded[:saved_n] = theta
                    self.ansatz.theta.copy_(padded)

                elif saved_n > current_n:
                    print(f"Trimming θ: {saved_n} → {current_n}")
                    self.ansatz.theta.copy_(theta[:current_n])

                else:
                    self.ansatz.theta.copy_(theta)

                    print(f"Generator parameters loaded successfully from {file_path}")

        except Exception as e:
            print(f"ERROR: Could not load generator model: {e}")

    def save_model_params(self, file_path: str):
        """Save generator parameters and config safely."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        save_data = {
            "theta": self.ansatz.theta.detach().cpu(),
            "config": {
                "target_size": self.target_size,
                "target_hamiltonian": self.target_hamiltonian,
                "ansatz_type": self.ansatz_type,
                "layers": self.layers,
            },
        }

        torch.save(save_data, file_path)
        print(f"Generator parameters saved to {file_path}")