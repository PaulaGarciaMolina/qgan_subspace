from config import CFG
import pennylane as qml
import torch.nn as nn
import torch
import itertools
import numpy as np


class ZZ_X_Z_circuit(nn.Module):
    def __init__(self, config = CFG):
        super().__init__()
        self.config = config
        self.size = self.config.system_size
        self.n_qubits = 2 * self.size + (1 if self.config.extra_ancilla else 0)
        self.layer = self.config.gen_layers 
        self.str_device = config.device
        self.device = qml.device(self.str_device, wires=self.n_qubits)
        self.n_params = self.count_n_params()
        self.theta = nn.Parameter(torch.randn(self.n_params, dtype=torch.float32))
        self.qnode = qml.QNode(self.circuit, self.device, interface="torch")

    def count_n_params(self):
        n_params = 0
        base_size = self.size # - 1 if self.config.extra_ancilla else self.size

        # 1-qubit RX and RZ
        n_params += self.size * 2

        if self.config.extra_ancilla and self.config.do_ancilla_1q_gates:
            n_params += 2  # for ancilla

        # 2-qubit ZZ between neighbors
        n_params += base_size - 1

        if self.config.extra_ancilla:
            if self.config.ancilla_topology == "total":
                n_params += base_size
            if self.config.ancilla_topology == "bridge":
                n_params += 1
            if self.config.ancilla_topology in ["bridge", "ansatz"]:
                n_params += 1

        return n_params * self.layer
    
    def circuit(self, theta, initial_state):
        qml.StatePrep(initial_state, wires=range(self.n_qubits))  # initialize custom state
        shift = self.size
        # If extra ancilla is used, different than ansatz, we reduce the size by 1,
        # to implement the ancilla logic separately.
        size = self.size
        #if self.config.extra_ancilla:
        #    size -= 1            

        idx = 0
        for _ in range(self.layer):
            # First 1 qubit gates
            for i in range(size):
                qml.RX(theta[idx], wires=shift + i)
                idx += 1
                qml.RZ(theta[idx], wires=shift + i)
                idx += 1

            # # Ancilla 1q gates for: total, bridge and disconnected:
            if self.config.extra_ancilla and self.config.do_ancilla_1q_gates:
                qml.RX(theta[idx], wires=shift + size)
                idx += 1
                qml.RZ(theta[idx], wires=shift + size)
                idx += 1

                # Then 2 qubit gates
            for i in range(size - 1):
                qml.IsingZZ(-theta[idx], wires=[shift + i, shift + i+1])
                idx += 1

            # Ancilla ancilla coupling (2q) logic for: total and bridge
            if self.config.extra_ancilla:
                if self.config.ancilla_topology == "total":
                    for i in range(size):
                        qml.IsingZZ(-theta[idx], wires=[shift + i, shift + size])
                        idx += 1

                if self.config.ancilla_topology == "bridge":
                    qml.IsingZZ(-theta[idx], wires=[shift + 0, shift + size])
                    idx += 1

                if self.config.ancilla_topology in ["bridge", "ansatz"]:
                    qubit_to_connect_to = self.config.ancilla_connect_to if self.config.ancilla_connect_to is not None else size - 1
                    qml.IsingZZ(-theta[idx], wires = [shift + qubit_to_connect_to, shift + size])
                    idx += 1
        return qml.state()
    
    def forward(self, initial_state):
        return self.qnode(self.theta, initial_state)