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
    _env_unitary,
    _env_unitary_cholesky,
    _gram_schmidt,
    _pad_tensor,
    _right_env,
)


def _imps_to_circuit_exact(
    mps: np.ndarray,
    *,
    shape: str = "lpr",
    num_sites: int,
    cholesky: bool = False,
) -> QuantumCircuit:
    """
    Convert an infinite, translationally-invariant matrix product state to a quantum circuit.

    :param mps: The tensor representing the translationally invariant MPS in left-canonical form.
    :param shape: The ordering of the dimensions of mps. 'left', 'physical', 'right' by default.
    :param num_sites: The number of physical sites to represent.
        NOTE: requires num_sites + 2 * ⌈log2(D)⌉ qubits, where D is the bond dimension.
    :param cholesky: Use Cholesky decomposition to create the right environment tensor.

    :return: mps quantum circuit representing num_sites sites of the infinite MPS.
    """
    # Sort indices as vL, p, vR and pad virtual dimensions to nearest power of 2.
    mps = np.transpose(mps, (shape.find("l"), shape.find("p"), shape.find("r")))
    mps = _pad_tensor(mps)
    d_left, d, d_right = mps.shape
    assert d == 2, "The physical dimension must be equal to two for qubits."
    assert (
        d_left == d_right
    ), "The left and right virtual dimensions should be the same."
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
    u_right = unitary.reshape(d_left, d, z, d_right)
    u_right = u_right.transpose(2, 3, 1, 0)
    u_right = u_right.reshape(z * d_right, d * d_left)

    if cholesky:
        env_unitary = _env_unitary_cholesky(_right_env(u_right, d=2, d_left=d_left))
    else:
        env_unitary = _env_unitary(_right_env(u_right, d=2, d_left=d_left))

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
