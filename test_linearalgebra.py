
from linearalgebra import *
import unittest as unittest
import linearalgebra as la
import numpy as np

class TestLinearAlgebra(unittest.TestCase):

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
        M = Matrix([[0, 2],[5, 3]])
        C = Matrix([[1, 5], [2, 6], [3, 7], [4, 8]])
        self.assertEqual(M.square(), True)
        self.assertEqual(C.square(), False)

    def test_det(self):
        D = Matrix([[5, 2], [3, 4]])
        self.assertAlmostEqual(D.det(), np.linalg.det(np.array([[5, 2], [3, 4]])))

    def test_copy(self):
        Q = Matrix([[0, 2],[5, 3]])
        new_matrix = Q._copy()
        self.assertEqual(new_matrix, Q)
        self.assertEqual(new_matrix.height, 2)
        self.assertEqual(new_matrix.width, 2)

# ------- LECTURE 2: ELIMINATION WITH MATRICES ----------
    def test_uppertriangle(self):
        L = [[5, 2],[3, 4]]
        new_matrix = Matrix(L).upper_triangle()
        self.assertAlmostEqual(new_matrix, np.triu(np.array(L)).tolist())
        self.assertEqual(new_matrix.height, 2)
        self.assertEqual(new_matrix.width, 2)

    def test_backsubsitituion(self):
        N = [[3, 4, 11],[6,4, 14]]
        my_matrix = Matrix(N)
        self.assertAlmostEqual(my_matrix.back_substitution(), ([1.0, 2.0], Matrix([[1, 0.0, 1.0], [0.0, 1, 2.0]])))

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
        self.assertEqual(R._get_pivot(1), 4)

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
        D = Matrix([[5, 3], [3, 2]])
        new_matrix = D.inverse()
        self.assertAlmostEqual(new_matrix- Matrix([[2, -3],[-3, 5]]), 0)
        self.assertEqual(new_matrix.height, 2)
        self.assertEqual(new_matrix.width, 2)

    def test_identity(self):
        D = Matrix([[5, 2], [3, 4]])
        new_matrix = D.identity()
        self.assertEqual(new_matrix, Matrix([[1, 0],[0, 1]]))
        self.assertEqual(new_matrix.height, 2)
        self.assertEqual(new_matrix.width, 2)

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

# def setUp(self):
    # self.A = [[1, 2, 3, 4], [5, 6, 7, 8]]
    # self.B = [1, 2, 3, 4]	# not a matrix
    # self.C = [[1, 5], [2, 6], [3, 7], [4, 8]]
    # self.D = [[5, 2], [3, 4]]
    # self.E = [[1, 4, 5, 6],[7, 2, 4, 9],[2, 4, 4, 3],[1, 9, 7, 8]]
    # self.F = [[1,2,1,2],[3,8,1,12],[0,4,1,2]]
    # self.G = [[3,3,3],[4,4,5],[1,2,2]]		# this matrix can't upper triangle :(
    # self.H = [[1, 2, 3],[2, 3, 4],[3, 4, 5]]
    # self.I = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    # self.J = [[1, 2, 3, 4],[0, 0, 3, 7],[0, 6, 7, 2],[0, 0, 7, 2]]
    # self.K = [[0, 2, 3],[1, 3, 5],[9, 3, 1]]
    # self.L = [[5, 2],[3, 4]]
    # self.M = [[0, 2],[5, 3]]
