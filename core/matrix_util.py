import numpy as np


def resize(matrix: np.ndarray, new_shape: tuple) -> np.ndarray:
    """Resize a matrix, truncating and filling with zeros"""

    mat_rows, mat_cols = matrix.shape
    new_rows, new_cols = new_shape
    new_mat = np.zeros(new_shape, np.uint8)
    new_mat[: min(mat_rows, new_rows), : min(mat_cols, new_cols)] = matrix[
        : min(mat_rows, new_rows), : min(mat_cols, new_cols)
    ]
    return new_mat


def reduced_row_echelon_form(
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, list[int]]:
    """Convert a matrix to reduced row echelon form"""

    rows, columns = matrix.shape
    reduced_form = np.copy(matrix)
    inverse_form = np.identity(rows, np.uint8)
    rank = 0
    pivots = []

    for j in range(columns):
        for i in range(rank, rows):
            if reduced_form[i, j]:
                for k in range(rows):
                    if (k != i) and reduced_form[k, j]:
                        reduced_form[k] ^= reduced_form[i]
                        inverse_form[k] ^= inverse_form[i]
                reduced_form[[i, rank]] = reduced_form[[rank, i]]
                inverse_form[[i, rank]] = inverse_form[[rank, i]]
                pivots.append(j)
                rank += 1
                break
    return reduced_form, inverse_form, rank, pivots


def generalized_inverse(matrix: np.ndarray) -> np.ndarray:
    """Compute the generalized inverse of a matrix"""

    _, inverse_form, rank, pivots = reduced_row_echelon_form(matrix)
    inverse_form = resize(inverse_form, (matrix.shape[1], matrix.shape[0]))
    for i in range(rank - 1, -1, -1):
        column_index = pivots[i]
        inverse_form[[i, column_index]] = inverse_form[[column_index, i]]
    return inverse_form


# TODO: smarter computation
def co_kernel_basis(matrix: np.ndarray) -> np.ndarray:
    """Compute the basis for the cokernel of a matrix"""

    matrix_inverse = generalized_inverse(matrix)
    basis = (matrix @ matrix_inverse) % 2
    basis = (basis + np.identity(basis.shape[0], np.uint8)) % 2
    basis, _, rank, _ = reduced_row_echelon_form(basis)
    return basis[:rank]


def int_to_bit_vector(integer: int, bit_length: int) -> np.ndarray:
    """Convert an integer to its bit vector representation"""

    bit_vec = np.empty(bit_length, np.uint8)
    bit_vec[:] = list(((int(integer) >> i) & 1 for i in range(bit_length)))

    return bit_vec


def bit_vector_to_int(bit_vector: np.ndarray) -> int:
    """Convert a bit vector to its integer representation"""

    integer = 0
    for i, bit in enumerate(bit_vector):
        integer |= int(bit) << i
    return integer
