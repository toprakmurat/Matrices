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


def change_vectors(vector_list):
    # Normally there are two possible inputs for inner product
    # first type: [1, 2, 3] and second type: [[1], [2], [3]]
    # this function will always change second type to first type and consider only these two cases
    # if something different is given, it will be considered as Invalid

    for index, vector in enumerate(vector_list):
        if index == 0:
            continue
        if len(vector) != len(vector_list[index - 1]):
            raise Exception('Invalid vector(s) given')

    vectors = []
    try:
        for vector in vector_list:
            # second type
            if type(vector[0]) == list:
                tmp = []
                for lst in vector:
                    if len(lst) != 1:  # check length
                        raise Exception('Invalid vector(s) given')
                    tmp.append(lst[0])
                vectors.append(tmp)

            # first type
            else:
                vectors.append(vector)
        return vectors
    except:
        raise Exception('Invalid vector(s) given')



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


# # Convert all the values in a matrix to floats
# def convert_float(matrix):
#     for i, row in enumerate(matrix):
#         for j, val in enumerate(row):
#             if type(val) != float:
#                 matrix[i][j] = float(val)
#     return matrix


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


# Helper function for vector multiplication
def inner_product(vector1, vector2):
    vectors = change_vectors([vector1, vector2])
    total = 0
    for index in range(len(vectors[0])):
        total += vectors[0][index] * vectors[1][index]

    return total


def subtract_vector(vector1, vector2):
    vectors = change_vectors([vector1, vector2])
    new_vector = []
    for index in range(len(vectors[0])):
        new_vector.append(vectors[0][index] - vectors[1][index])
    return new_vector


def scalar_multiply_vector(vector, scalar):
    vector = change_vectors([vector])[0]
    new_vector = []
    for val in vector:
        new_vector.append(scalar * val)
    return new_vector


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
    # TODO: complete inverse function
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
    def gaussian_elimination(matrix):
        """
        Apply Gaussian Elimination to a square matrix using elementary matrix operations
        Input: matrix: matrix
        Output: matrix: matrix
        """
        if check_matrix(matrix, is_square=True) is False:
            raise Exception('Invalid Matrix')

        for row_index, row in enumerate(matrix):
            pivot_index = row_index  # index of the pivot is equal to current row's index
            pivot = row[pivot_index]
            # Fix temporary fails if any
            if pivot == 0:
                for i in range(row_index + 1, len(matrix)):  # i = other indexes below
                    if matrix[i][pivot_index] != 0:
                        changing_index = row_index + i + 1
                        row = matrix[changing_index]
                        change_rows_matrix(matrix, row_index, changing_index)
                        break
                pivot = row[pivot_index]  # update pivot
                if pivot == 0:
                    continue  # temporary fail becomes permanent fail

            for below_index, below_row in enumerate(matrix[row_index + 1:]):  # start from below the current row
                current_index = row_index + below_index + 1
                if below_row[pivot_index] == 0:  # no need for elementary operations
                    continue
                alpha = -1 * (below_row[pivot_index] / pivot)  # find multiplier
                new_row = [number*alpha + below_row[ind] for ind, number in enumerate(row)]
                matrix[current_index] = new_row

        return matrix

    # Return bases of the column space of a given matrix
    @staticmethod
    def column_space(matrix):
        upper_matrix = Matrix.gaussian_elimination(matrix)
        pivot_indexes = []
        for num in range(len(upper_matrix)):
            for index, val in enumerate(upper_matrix[num][num:]):
                if val != 0:
                    pivot_indexes.append(num + index)
                    break

        columns = []
        for index in pivot_indexes:
            columns.append(matrix_col_to_row(matrix, index))

        return columns

    # Return bases of the row space of a given matrix
    @staticmethod
    def row_space(matrix):
        upper_matrix = Matrix.gaussian_elimination(matrix)
        pivot_indexes = []
        for num in range(len(upper_matrix)):
            for index, val in enumerate(upper_matrix[num][num:]):
                if val != 0:
                    pivot_indexes.append(num + index)
                    break

        rows = []
        for index in pivot_indexes:
            rows.append(matrix[index])

        return rows

    @staticmethod
    def gram_schmidt(*argv):
        pass