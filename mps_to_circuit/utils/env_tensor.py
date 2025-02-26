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
from ncon import ncon
from scipy.sparse.linalg import eigs


def _right_env(u: np.ndarray, d: int, D: int, tolerance: float = 1e-08) -> np.ndarray:
    """
    Calculates the right environment of a given translationally invariant MPS.
    Assumes u indices are (zero * vR, p * vL)

    Args:
        u: A unitary representing the translationally invariant MPS site.

        d: The physical dimension of the MPS site.

        D: The bond dimension of the MPS site.

    Returns:
        A matrix representing the normalised right environment.
    """
    zero = np.zeros(d)
    zero[0] = 1

    u = u.reshape([d, D, d, D])
    transfer = ncon(
        [zero, u, np.conj(u), zero], ([1], [1, -3, 2, -1], [3, -4, 2, -2], [3])
    )
    assert np.allclose(np.eye(D), ncon(transfer, [1, 1, -1, -2]), atol=tolerance)
    trans_matrix = transfer.reshape(D * D, D * D)

    _, left_env = eigs(trans_matrix.T, k=1, which="LM")
    left_env = left_env.reshape(D, D)
    assert np.allclose(np.eye(D), left_env / np.trace(left_env) * D, atol=tolerance)

    _, right_env = eigs(trans_matrix, k=1, which="LM")
    right_env = right_env.reshape(D, D)

    norm = np.trace(right_env)
    return right_env / norm


def _env_unitary(right_env: np.ndarray) -> np.ndarray:
    """
    Convert full right environment R into single unitary form V.

    Args:
        right_env: A 2D NumPy array representing the right environment.

    Returns:
        A unitary matrix representing the single environment tensor V.
    """
    u, s, _ = np.linalg.svd(right_env, hermitian=True)
    s_sqrt = np.sqrt(np.diag(s))
    v = u @ s_sqrt
    return _vect_to_unitary(v.reshape(v.shape[0] * v.shape[1], 1))


def _env_unitary_cholesky(right_env: np.ndarray) -> np.ndarray:
    """
    Convert full right environment R into single unitary form V using the Cholesky decomposition.

    Args:
        right_env: A 2D NumPy array representing the right environment.

    Returns:
        A unitary matrix representing the single environment tensor V.
    """
    L = np.linalg.cholesky(right_env)
    return _vect_to_unitary(L.reshape(L.shape[0] * L.shape[0], 1))


# vect must be 2D array with 1 column
def _vect_to_unitary(vect: np.ndarray) -> np.ndarray:
    u, _, _ = np.linalg.svd(vect)
    assert np.allclose(vect[:, 0], u[:, 0])
    return u
