# Create row_size x column_size matrix with initial values of 0
def create_matrix(row_size, column_size):
    matrix = [[0 for _ in range(column_size)] for _ in range(row_size)]
    return matrix


# Check if matrix consists of non integer or non float values
def check_matrix(matrix, is_square=False):
    if is_square:
        if len(matrix) == 0 or len(matrix) != len(matrix[0]):
            return False

    for index, row in enumerate(matrix):
        for val in row:
            if type(val) != int and type(val) != float:
                return False
        # Check if all rows have same length
        if index == 0:
            continue
        if len(row) != len(matrix[index - 1]):
            return False

    return True


# Return the column of the given matrix at the given index as a row
def matrix_col_to_row(matrix, index):
    if len(matrix) > 0 and len(matrix[0]) >= index:
        pass

    new_row = [row[index] for row in matrix]
    return new_row


def change_rows_matrix(matrix, index_first, index_second):
    if index_first > len(matrix) or index_second > len(matrix):
        raise ValueError('Index is out of range')

    temp = matrix[index_first]
    matrix[index_first] = matrix[index_second]
    matrix[index_second] = temp

    return matrix


# Convert all the values in a matrix to floats
def convert_float(matrix):
    for i, row in enumerate(matrix):
        for j, val in enumerate(row):
            matrix[i][j] = float(val)
    return matrix


# Check multiplication condition (column size of first matrix must be equal to row size of second matrix)
def check_multiplication(*argv):
    # len argv must be at least 2
    if len(argv) < 2:
        raise Exception('At least two matrices are needed for matrix multiplication')

    # Store each result matrix's row size and column size
    result_row, result_col = 0, 0
    for index, matrix in enumerate(argv):
        if check_matrix(matrix) is False:
            raise Exception('At least one of the matrices is invalid')

        # Specific for multiplication of first two matrices
        if index < 2:
            # second matrix is unnecessary in this operation
            # because result_row and result_col are saved in first repeat of the loop.
            if index == 1:
                continue

            r_size_first, c_size_first = len(matrix), len(matrix[0])
            r_size_second, c_size_second = len(argv[1]), len(argv[1][0])
            # Multiplication condition
            if c_size_first == r_size_second:
                result_row, result_col = r_size_first, c_size_second
            else:
                return False

        else:
            # result_row and result_col are always size of first matrix
            second_row, second_col = len(matrix), len(matrix[0])
            if result_col == second_row:
                result_col = second_col
            else:
                return False

    return True


# Helper function for two_matrix_multiplication
def inner_product(vector1, vector2):
    # Normally there are two possible inputs for inner product
    # first type: [1, 2, 3] and second type: [[1], [2], [3]]
    # this function will always get first type vectors as inputs inside two_matrix_multiplication

    if len(vector1) != len(vector2):
        raise Exception("These vectors' sizes are not equal to each other")

    total = 0
    for index in range(len(vector1)):
        total += vector1[index] * vector2[index]

    return total


# C[i][j] = ith row of A * jth column of B, where AB = C
def two_matrix_multiplication(first_matrix, second_matrix):
    rows_first, cols_second = len(first_matrix), len(second_matrix[0])

    result = create_matrix(rows_first, cols_second)
    for i in range(rows_first):
        for j in range(cols_second):
            col_as_row = matrix_col_to_row(second_matrix, j)  # for inner_product
            result[i][j] = inner_product(first_matrix[i], col_as_row)

    return result


# Old version
# Multiply 2 matrices (helper function for Matrix.multiply)
# def two_matrix_multiplication(first_matrix, second_matrix):
#     number_rows = len(first_matrix)  # number of rows in the first matrix
#     number_cols = len(second_matrix[0])  # number of columns in the second matrix
#
#     # C[i][j] = ith row of A * jth column of B, where AB = C
#     values = create_matrix(number_rows, number_cols)
#     for i in range(number_cols):  # i equals the number of columns in the second matrix
#         for row_index, row in enumerate(first_matrix):
#             total = 0
#             for val_index, value in enumerate(row):
#                 total += value * second_matrix[val_index][i]
#             values[row_index][i] = total
#
#     return values


class Matrix:
    @staticmethod
    def multiply(*argv):
        if check_multiplication(*argv) is False:
            raise Exception('These matrices cannot be multiplied')

        result_matrix = []
        for index, matrix in enumerate(argv):
            # Specific for the first two matrices
            if index < 2:
                # second matrix is unnecessary because result matrix is already saved in first repeat
                if index == 1:
                    continue
                result_matrix = two_matrix_multiplication(matrix, argv[1])
            else:
                result_matrix = two_matrix_multiplication(result_matrix, matrix)

        return result_matrix

    @staticmethod
    def transpose(matrix):
        if check_matrix(matrix) is False:
            raise Exception('This matrix is not valid')

        rows, columns = len(matrix), len(matrix[0])
        new_matrix = create_matrix(columns, rows)  # reverse for the transpose
        for row_index, row in enumerate(matrix):
            for col_index, col in enumerate(row):
                new_matrix[col_index][row_index] = matrix[row_index][col_index]

        return new_matrix

    @staticmethod
    # TODO: complete this function
    def inverse(matrix):
        pass

    # Creates a size x size diagonal matrix with specified value in main diagonal and 0 all the others
    # with initial value of 1 it is also an Identity Matrix
    @staticmethod
    def create_diagonal_matrix(size, value=1):
        diagonal_matrix = create_matrix(size, size)
        for i in range(size):
            diagonal_matrix[i][i] = value

        return diagonal_matrix

    # Apply Gaussian elimination for square matrices
    @staticmethod
    def gaussian_elimination(matrix, augmenting_matrix):
        if check_matrix(matrix, is_square=True) is False:
            raise Exception('Invalid Matrix')

        for row_index, row in enumerate(matrix):
            if row_index == len(matrix) - 1:
                # elementary operations for the last row is not needed
                return matrix, augmenting_matrix

            pivot = row[row_index]

            # Fix temporary fails (change rows if pivot equals to 0)
            if pivot == 0:
                col_as_row = matrix_col_to_row(matrix, row_index)
                for ind, val in enumerate(col_as_row[row_index:]):
                    if val != 0:
                        change_index = ind + row_index
                        change_rows_matrix(matrix, change_index, row_index)
                        break
                continue

            for index_below, row_below in enumerate(matrix[row_index + 1:]):
                current_ind = row_index + index_below + 1
                alpha = -1 * row_below[row_index] / pivot
                # If alpha equals to 0, then no operation is needed. Just pass that row.
                if alpha == 0:
                    continue
                matrix[current_ind] = [val * alpha + row_below[ind]
                                       for ind, val in enumerate(row)]
                augmenting_matrix[current_ind] = [val * alpha + augmenting_matrix[current_ind][ind]
                                            for ind, val in enumerate(augmenting_matrix[row_index])]

    @staticmethod
    def gauss_jordan(matrix):
        if check_matrix(matrix, is_square=True) is False:
            raise Exception('Invalid Matrix')

        # Create an identity matrix for augmented matrix
        identity_matrix = Matrix.create_diagonal_matrix(len(matrix), value=1)
        upper_matrix, augmenting_matrix = Matrix.gaussian_elimination(matrix, identity_matrix)
        for i, row in enumerate(reversed(upper_matrix)):
            pivot_index = (-1 * i) - 1
            if i == 0:
                pivot_index = -1

            pivot = row[pivot_index]

            # No temporary fails because we fixed them in gaussian elimination
            if pivot == 0:
                continue

            for index_upper, row_upper in enumerate(matrix[:pivot_index]):
                alpha = -1 * row_upper[pivot_index] / pivot
                matrix[index_upper] = [val * alpha + row_upper[ind]
                                       for ind, val in enumerate(row)]

                row_index = len(matrix) - i - 1  # row is started as reversed
                augmenting_matrix[index_upper] = [val * alpha + augmenting_matrix[index_upper][ind]
                                            for ind, val in enumerate(augmenting_matrix[row_index])]
                # As for equality in indexes
                # augmenting_matrix[index_upper] = matrix[index_upper]
                # augmenting_matrix[row_index] = row
                # Those are same indexes respectively

        return matrix, augmenting_matrix
