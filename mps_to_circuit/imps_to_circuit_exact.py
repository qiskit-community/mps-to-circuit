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

from .utils import _gram_schmidt, _pad_tensor, _env_unitary, _right_env


def _imps_to_circuit_exact(
    mps: np.ndarray, *, shape: str = "lpr", num_sites: int
) -> QuantumCircuit:
    """
    Convert an infinite, translationally-invariant matrix product state to a quantum circuit.

    TODO: Make work for infinite-MPS with unit cell of >1 sites (Do we need this?)

    :param mps: The tensor representing the translationally invariant MPS in left-canonical form.
    :param shape: The ordering of the dimensions of mps. 'left', 'physical', 'right' by default.
    :param num_sites: The number of physical sites to represent.
        NOTE: requires num_sites + 2 * ⌈log2(D)⌉ qubits, where D is the bond dimension.

    :return: mps quantum circuit representing num_sites sites of the infinite MPS.
    """
    # Sort indices as vL, p, vR and pad virtual dimensions to nearest power of 2.
    mps = np.transpose(mps, (shape.find("l"), shape.find("p"), shape.find("r")))
    mps = _pad_tensor(mps)
    d_left, d, d_right = mps.shape
    assert d == 2
    assert d_left == d_right
    z = d

    # Reshape to vL * p, vR
    mps = mps.reshape(d_left * d, d_right)

    # Create unitary from isometry, indices p * vL, z * vR. Original isometry columns make up the
    # least-significant bits
    matrix = np.zeros((mps.shape[0], mps.shape[0]), dtype=mps.dtype)
    matrix[:, : mps.shape[1]] = mps

    # unitary has shape vL * p, z * vR, required for circuit unitary
    unitary = _gram_schmidt(matrix)

    # Calculate the right-environment tensor, for this unitary needs to have shape z * vR, p * vL
    U1 = unitary.reshape(d_left, d, z, d_right)
    U1 = U1.transpose(2, 3, 1, 0)
    U1 = U1.reshape(z * d_right, d * d_left)

    env_unitary = _env_unitary(_right_env(U1, d=2, D=d_left))

    # Gate sizes
    u_size = int(np.ceil(np.log2(unitary.shape[0])))
    v_size = int(np.ceil(np.log2(env_unitary.shape[0])))

    # Number of qubits required in the circuit
    num_qubits = num_sites + v_size

    qc = QuantumCircuit(num_qubits)

    # Reverse the order of qubits for consistency with Qiskit's little-endian ordering.
    qc.unitary(
        env_unitary,
        qubits=list(reversed(range(num_qubits - v_size, num_qubits))),
        label="V",
    )
    for i in list(reversed(range(num_sites))):
        qc.unitary(unitary, qubits=(list(reversed(range(i, i + u_size)))), label="U")

    return qc
