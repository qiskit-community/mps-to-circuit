# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Approximate MPS to circuit mapping function."""

import numpy as np
from qiskit import QuantumCircuit

from .mps_to_circuit_exact import _mps_to_circuit_exact
from .utils import _prepare_mps


def _mps_to_circuit_approx(
    mps: list[np.ndarray],
    *,
    shape: str = "lrp",
    num_layers: int = 1,
    compress: bool = False,
    cutoff: float = 1e-3,
    chi_max: int | None = None,
    history: dict | None = None,
) -> QuantumCircuit:
    """Convert a matrix product state to a quantum circuit.

    Approximately convert a generic matrix product state (MPS) with arbitrary bond dimensions to a
    Qiskit quantum circuit. See Section III of "Encoding of matrix product states into quantum
    circuits of one- and two-qubit gates", S J Ran, Phys. Rev. A, 2020.

    :param mps: A matrix product state (MPS) representation of a quantum state.
    :param shape: The ordering of the dimensions of each MPS tensor. 'left', 'right', 'physical' by
        default.
    :param num_layers: The number of layers to add to the circuit.
    :param compress: Set to `True` to compress the MPS after each layer to a maximum bond dimension
        of `chi_max`.
    :param cutoff: Cutoff threshold used when `compress` is `True`. Defaults to 0.001.
    :param chi_max: See description for compress. `chi_max` will be ignored if compress is `False`.
    :param history: Dictionary to store intermediate data from algorithm.

    :return: A quantum circuit consisting of `num_layers` of two-qubit isometries that approximately
        represents the input MPS.
    """
    # Prepare Quimb MPS in the correct form.
    _mps = _prepare_mps(mps, shape)
    disentangled_mps = _mps.copy(deep=True)

    # Check chi_max.
    if not compress and chi_max is not None:
        print("Warning: `chi_max` is ignored when compress is `False`.")

    if chi_max is None:
        chi_max = _mps.max_bond()

    assert chi_max > 0, "`chi_max` must be an integer greater than 0."

    final_circuit = None

    layer = 0
    while layer < num_layers:
        # Compress the MPS from the previous layer to a maximum bond dimension of 2,
        # |ψ_k> -> |ψ'_k>.
        compressed_mps = disentangled_mps.copy(deep=True)
        compressed_mps.compress(form="left", max_bond=2)
        compressed_mps.normalize()

        # Find the unitary U_k such that |ψ'_k> = U_k @ |0>.
        circuit = _mps_to_circuit_exact(list(compressed_mps.arrays), shape="lrp")

        unitaries = [instruction.operation.to_matrix() for instruction in circuit.data]

        # inv(U_k) @ ... @ inv(U_1) @ inv(U_0) @ |0>.
        final_circuit = (
            circuit.compose(final_circuit) if final_circuit is not None else circuit
        )

        if history is not None:
            history["circuits"].append(final_circuit)

        # Apply the inverse of U_k to disentangle |ψ_k>,
        # |ψ_(k+1)> = inv(U_k) @ |ψ_k>.
        for i, _ in enumerate(unitaries):
            inverse = unitaries[-(i + 1)].conj().T
            if inverse.shape[0] == 4:
                # Keep the residual update exact.
                disentangled_mps.gate_split(
                    inverse, (i - 1, i), inplace=True, cutoff=0.0
                )
            else:
                disentangled_mps.gate(inverse, (i), inplace=True, contract=True)

        if compress:
            # Compress |ψ_(k+1)> to have maximum bond dimension chi_max.
            disentangled_mps.compress(form="left", max_bond=chi_max, cutoff=cutoff)

        layer += 1

    # Return final circuit.
    return final_circuit
