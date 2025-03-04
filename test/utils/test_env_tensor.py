# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import numpy as np
import pytest
from scipy.sparse.linalg import eigs

from mps_to_circuit.utils.env_tensor import _construct_transfer_matrix


def _generate_unitary_matrix(d: int, d_left: int) -> np.ndarray:
    """
    Generates a random unitary matrix of appropriate dimensions.

    :param d: Physical dimension.
    :param d_left: Bond dimension.

    :return: A random unitary matrix of shape (d * d_left, d * d_left).
    """
    size = d * d_left
    matrix = np.random.randn(size, size) + 1j * np.random.randn(size, size)
    q, _ = np.linalg.qr(matrix)
    return q


@pytest.mark.parametrize("d, d_left", [(2, 2), (2, 4)])
def test_construct_transfer_matrix(d: int, d_left: int):
    """
    :param d: Physical dimension.
    :param d_left: Bond dimension.
    """
    tolerance = 1e-8
    u = _generate_unitary_matrix(d, d_left)

    transfer_matrix = _construct_transfer_matrix(u, d, d_left)

    transfer = transfer_matrix.reshape((d_left, d_left, d_left, d_left))
    contraction = np.einsum("iijk->jk", transfer)
    assert np.allclose(
        contraction, np.eye(d_left), atol=tolerance
    ), "Transfer tensor contraction not close to identity matrix."

    _, left_env = eigs(transfer_matrix.T, k=1, which="LM")
    left_env = left_env.reshape(d_left, d_left)
    assert np.allclose(
        np.eye(d_left), left_env / np.trace(left_env) * d_left, atol=tolerance
    )
