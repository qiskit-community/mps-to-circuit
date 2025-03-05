# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Exact MPS to circuit mapping function."""

import numpy as np
from qiskit import QuantumCircuit

from .utils import (
    _gram_schmidt,
    _has_orthonormal_columns,
    _is_unitary,
    _pad_tensor,
    _prepare_mps,
)


def _mps_to_circuit_exact(
    mps: list[np.ndarray],
    *,
    shape: str = "lrp",
) -> QuantumCircuit:
    """Convert a matrix product state to a quantum circuit.

    Convert a generic matrix product state (MPS) with arbitrary bond dimensions to a Qiskit quantum
    circuit. See Appendix A of S-H Lin, PRX Quantum 2, 010342 (2021).

    :param mps: A matrix product state (MPS) representation of a quantum state.
    :param shape: The ordering of the dimensions of each MPS tensor. 'left', 'right', 'physical' by
        default.
    :param shape: Encodes which index each tensor dimension corresponds to, where `l` is the left
        virtual index, `r` is the right virtual index and `p` is the physical index.

    :return: A quantum circuit consisting of multi-qubit isometries that represents the input MPS.
    """
    _mps = _prepare_mps(mps, shape=shape)
    num_sites = _mps._L
    circuit = QuantumCircuit(num_sites)

    for i, tensor in reversed(list(enumerate(_mps.arrays))):
        # Convert to a dense NumPy array and pad until virtual dimensions are powers of 2.
        # Quimb defines the indices of the MPS tensors as (d_left, d_right, d), but the left and
        # right end tensors do not have a left and right dimension, respectively.
        if i == 0:
            d_right, d = tensor.shape
            tensor = tensor.reshape((1, d_right, d))
        if i == num_sites - 1:
            d_left, d = tensor.shape
            tensor = tensor.reshape((d_left, 1, d))

        tensor = np.swapaxes(tensor, 1, 2)
        padded_tensor = _pad_tensor(tensor)

        # Combine the physical index and right-virtual index of the tensor to construct an isometry
        # matrix. Check the isometry has the required properties.
        d_left, d, d_right = padded_tensor.shape
        assert d == 2, "The physical dimension must be equal to two for qubits."
        isometry = padded_tensor.reshape((d * d_left, d_right))
        assert _has_orthonormal_columns(isometry)

        # Reverse the order of qubits for consistency with Qiskit's little-endian ordering.
        qubits = list(reversed(range(i - int(np.ceil(np.log2(d_left))), i + 1)))

        # Create all-zero matrix and add the isometry columns.
        matrix = np.zeros((isometry.shape[0], isometry.shape[0]), dtype=isometry.dtype)

        # Keep columns for which all ancillas are in the zero state.
        matrix[:, : isometry.shape[1]] = isometry

        # Apply Gram-Schmidt process.
        unitary = _gram_schmidt(matrix)

        # Check that the columns of the isometry have been preserved.
        for j in range(isometry.shape[1]):
            if not np.all(isometry[:, j] == 0.0):
                assert np.allclose(unitary[:, j], isometry[:, j])

        # Check that the final matrix operator is unitary.
        assert _is_unitary(unitary)

        # Apply unitary to the circuit.
        circuit.unitary(unitary, qubits)

    return circuit
