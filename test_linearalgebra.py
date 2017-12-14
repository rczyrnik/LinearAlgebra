
from linearalgebra import *
import unittest as unittest
import linearalgebra as la
import numpy as np

class TestLinearAlgebra(unittest.TestCase):
    def setUp(self):
        self.A = [[1, 2, 3, 4], [5, 6, 7, 8]]
        # self.B = [1, 2, 3, 4]	# not a matrix
        self.C = [[1, 5], [2, 6], [3, 7], [4, 8]]
        self.D = [[5, 2], [3, 4]]
        self.E = [[1, 4, 5, 6],[7, 2, 4, 9],[2, 4, 4, 3],[1, 9, 7, 8]]
        self.F = [[1,2,1,2],[3,8,1,12],[0,4,1,2]]
        self.G = [[3,3,3],[4,4,5],[1,2,2]]		# this matrix can't upper triangle :(
        self.H = [[1, 2, 3],[2, 3, 4],[3, 4, 5]]
        self.I = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        self.J = [[1, 2, 3, 4],[0, 0, 3, 7],[0, 6, 7, 2],[0, 0, 7, 2]]
        self.K = [[0, 2, 3],[1, 3, 5],[9, 3, 1]]
        self.L = [[5, 2],[3, 4]]
        self.M = [[0, 2],[5, 3]]
        self.all_lol = [self.A, self.C, self.D, self.E,
                            self.F,self.H, self.I, self.J,
                            self.K, self.L, self.M]

# ------- SETUP ----------
    def test_legit(self):
        '''NEED TO DO THIS '''
        A = Matrix([[1, 2, 3, 4], [5, 6, 7, 8]])
        B = Matrix([1, 2, 3, 4])	# not a matrix
        # with self.assertRaises(ValueError) as ctx:
        #     Matrix(self.B)
        #     expected_msg = "This is not a list of lists"
        # self.assertEquals(ctx.exception.message, expected_msg)
        self.assertRaises(TypeError, B, "This is not a list of lists")

    def test_square(self):
        M = Matrix(self.M)
        C = Matrix(self.C)
        self.assertEqual(M.square(), True)
        self.assertEqual(C.square(), False)

    def test_det(self):
        for lol in self.all_lol:
            if Matrix(lol).square():
                self.assertAlmostEqual(Matrix(lol).det(), np.linalg.det(np.array(lol)))

        D = Matrix([[5, 2], [3, 4]])
        self.assertAlmostEqual(D.det(), np.linalg.det(np.array([[5, 2], [3, 4]])))

    def test_copy(self):
        Q = Matrix([[0, 2],[5, 3]])
        new_matrix = Q._copy()
        self.assertEqual(new_matrix, Q)
        self.assertEqual(new_matrix.height, 2)
        self.assertEqual(new_matrix.width, 2)

# ------- LECTURE 2: ELIMINATION WITH MATRICES ----------
    def test_row_echelon(self):
        # tuples with matrix and its upper triangle
        A = (Matrix([[0, 1],[2,2]]), Matrix([[2, 2],[0, 1]]))
        B = (Matrix([[2, 1],[0,2]]), Matrix([[2, 1],[0, 2]]))
        C = (Matrix([[0,1],[0,2],[0,3],[0,4],[9,5]]), Matrix([[9,5],[0,2],[0,0],[0,0],[0,0]]))
        D = (Matrix([[0,1],[0,2],[9,3]]), Matrix([[9,3],[0,2],[0,0]]))
        E = (Matrix([[3,3,3],[4,4,5],[1,2,2]]), Matrix([[3,3,3],[0,1,1],[0,0,1]]))
        F = (Matrix([[5, 2],[3, 4]]), Matrix([[5, 2],[0, 2.8]]))
        letters = [A, B, C, D, E, F]
        for letter in letters:
            matrix, answer = letter
            matrix_ut = matrix.row_echelon()
            self.assertAlmostEqual(matrix_ut, answer)
            self.assertEqual(matrix_ut.height, answer.height)
            self.assertEqual(matrix_ut.width, answer.width)

    def test_backsubsitituion(self):
        N = (Matrix([[3, 4, 11],[6, 4, 14]]),
                ([1.0, 2.0], Matrix([[1, 0.0, 1.0], [0.0, 1, 2.0]])) )
        M = (Matrix([[1,2,1,2],[3,8,1,12],[0,4,1,2]]),
                ([2, 1, -2], Matrix([[1, 0, 0, 2],[0, 1, 0, 1],[0, 0, 1, -2]])) )
        letters = [N, M]
        for letter in letters:
            matrix, answer = letter
            self.assertEqual(matrix.back_substitution(), answer)

# ------- LECTURE 3: MULTIPLICATION AND INVERSE MATRICS -------------

    def test_multiply(self):
        O = Matrix([[1, 2, 3],[4, 5, 6]])
        P = Matrix([[7],[8],[9]])
        new_matrix = O.multiply(P)
        self.assertEqual(new_matrix, [[50], [122]])
        self.assertEqual(new_matrix.height, 2)
        self.assertEqual(new_matrix.width, 1)

    def test_augment(self):
        O = Matrix([[1, 2, 3],[4, 5, 6]])
        new_matrix = O.augment()
        self.assertEqual(new_matrix, [[1, 2, 3, 1, 0, 0], [4, 5, 6, 0, 1, 0]])
        self.assertEqual(new_matrix.height, 2)
        self.assertEqual(new_matrix.width, 6)

    def test_get_pivot(self):
        R = Matrix([[5, 2],[3, 4]])
        _, pivot = R._get_pivot(1)
        self.assertEqual(pivot, 4)

    def test_row_op(self):
        R = Matrix([[5, 2],[3, 4]])
        new_matrix = R._row_op(0, 1, 2)
        self.assertEqual(new_matrix, Matrix([[-1, -6],[3, 4]]))
        self.assertEqual(new_matrix.height, 2)
        self.assertEqual(new_matrix.width, 2)

    def test_divide(self):
        R = Matrix([[5, 2],[3, 4]])
        new_matrix = R._divide(1, 2)
        self.assertEqual(new_matrix, Matrix([[5, 2],[1.5, 2]]))
        self.assertEqual(new_matrix.height, 2)
        self.assertEqual(new_matrix.width, 2)

    def test_swap_rows(self):
        R = Matrix([[5, 2],[3, 4]])
        new_matrix = R._swap_rows(0, 1)
        self.assertEqual(new_matrix._list, Matrix([[3, 4],[5, 2]])._list)
        self.assertEqual(new_matrix.height, 2)
        self.assertEqual(new_matrix.width, 2)

    def test_inverse(self):
        # tuples of matrices and their inverses
        D = (Matrix([[5, 3], [3, 2]]), Matrix([[2, -3],[-3, 5]]))
        G = (Matrix([[3,3,3],[4,4,5],[1,2,2]]), Matrix([[ 0.66666667,  0., -1.],[ 1., -1.,  1.],[-1.33333333,  1., -0.]]))
        K = (Matrix([[0, 2, 3],[1, 3, 5],[9, 3, 1]]), Matrix([[-0.75  ,  0.4375,  0.0625],[ 2.75  , -1.6875,  0.1875],[-1.5   ,  1.125 , -0.125 ]]))
        L = (Matrix([[2,1],[0,3]]), Matrix([[.5,-1/6],[0,1/3]]))
        matrices = [D,G,K,L]
        for matrix in matrices:
            m, m_inv = matrix
            self.assertAlmostEqual(m.inverse()-m_inv, 0)

    def test_identity(self):
        D = Matrix([[5, 2], [3, 4]])
        new_matrix = D.identity()
        self.assertEqual(new_matrix, Matrix([[1, 0],[0, 1]]))
        self.assertEqual(new_matrix.height, 2)
        self.assertEqual(new_matrix.width, 2)

# -------- LECTURE 4: FACTORIZATION INTO A=LU --------------------
    def test_factor(self):
        A = Matrix([[2, 1],[8,7]])
        l = Matrix([[1, 0],[4,1]])
        d = Matrix([[2, 0],[0,3]])
        u = Matrix([[1, .5],[0,1]])
        L, D, U = A.factor()
        self.assertAlmostEqual(l-L,0)
        self.assertAlmostEqual(d-D,0)
        self.assertAlmostEqual(u-U,0)
        self.assertAlmostEqual(L.multiply(D).multiply(U)-A,0)

# -------- LECTURE 5: TRANSPOSES, PERMUTATIONS, SPACES R^n --------------------
    def test_transpose(self):
        E = (Matrix([[5, 4], [3, 2]]), Matrix([[5, 3], [4, 2]]))
        D = (Matrix([[5, 3], [3, 2]]), Matrix([[5, 3], [3, 2]]))
        letters = [E, D]
        for letter in letters:
            matrix, answer = letter
            self.assertEqual(matrix.transpose(), answer)

    def test_symmetric(self):
        E = Matrix([[5, 4], [3, 2]])
        D = Matrix([[5, 3], [3, 2]])
        self.assertFalse(E.symmetric())
        self.assertTrue(D.symmetric())

    def test_symmetric_transpose(self):
        D = Matrix([[5, 2], [3, 4]])
        A = Matrix([[1, 2, 3, 4], [5, 6, 7, 8]])
        self.assertTrue(D.multiply(D.transpose()).symmetric())
        self.assertTrue(A.multiply(A.transpose()).symmetric())


if __name__ == '__main__':
    unittest.main()
