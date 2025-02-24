# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Functions to convert generic matrix product states to quantum circuits."""

import numpy as np
from qiskit import QuantumCircuit

from .utils import (
    _gram_schmidt,
    _pad_tensor,
    _env_unitary,
    _right_env
)


def _imps_to_circuit_exact(A: np.ndarray, *, shape: str = "lpr", n: int) -> QuantumCircuit:
    """
    Convert an infinite, translationally-invariant matrix product state to a quantum circuit.

    TODO: Make work for infinite-MPS with unit cell of >1 sites (Do we need this?)

    Args:
        A: The tensor representing the translationally invariant MPS in left-canonical form.

        shape: The ordering of the dimensions of A. 'left', 'physical', 'right' by default.

        n: The number of physical sites to represent. NOTE: requires n + 2 * ⌈log2(D)⌉ qubits, where
        D is the bond dimension.

    Returns:
        A quantum circuit representing n sites of the infinite MPS.
    """
    # Sort indices as vL, p, vR and pad virtual dimensions to nearest power of 2.
    A = np.transpose(A, (shape.find("l"), shape.find("p"), shape.find("r")))
    A = _pad_tensor(A)
    vL, p, vR = A.shape
    assert p == 2
    assert vL == vR
    z = p

    # Reshape to vL * p, vR
    A = A.reshape(vL * p, vR)

    # Create unitary from isometry, indices p * vL, z * vR. Original isometry columns make up the
    # least-significant bits
    matrix = np.zeros((A.shape[0], A.shape[0]), dtype=A.dtype)
    matrix[:, : A.shape[1]] = A
    # U has shape vL * p, z * vR, required for circuit U
    U = _gram_schmidt(matrix)

    # Calculate the right-environment tensor, for this U needs to have shape z * vR, p * vL
    U1 = U.reshape(vL, p, z, vR)
    U1 = U1.transpose(2, 3, 1, 0)
    U1 = U1.reshape(z * vR, p * vL)

    V = _env_unitary(_right_env(U1, d=2, D=vL))

    # Gate sizes
    U_size = int(np.ceil(np.log2(U.shape[0])))
    V_size = int(np.ceil(np.log2(V.shape[0])))

    # Number of qubits required in the circuit
    N = n + V_size

    qc = QuantumCircuit(N)

    # Reverse the order of qubits for consistency with Qiskit's little-endian ordering.
    qc.unitary(V, qubits=list(reversed(range(N - V_size, N))), label="V")
    for i in list(reversed(range(n))):
        qc.unitary(U, qubits=(list(reversed(range(i, i + U_size)))), label="U")

    return qc
