'''
scratch paper
'''

from linearalgebra import *
import linearalgebra as la
import numpy as np
#
A = Matrix([[2, 1],[8,7]])
#
# U = A.row_echelon()
# A = Matrix([[2, 1],[8,7]])
# L = A.multiply(U.inverse())
#
# print("A:")
# print(A)
# print("\nU:")
# print(U)
#
# print("\nU-1:")
# print(U.inverse())
# print("\nL:")
# print(L)
# print("\nLU:")
# print(L.multiply(U))
# A.factor()

U, D, L = A.factor()
print("\nU:")
print(U)
print("\nD:")
print(D)
print("\nL:")
print(L)
print("\nA:")
print(A)
print("\nU*D*L:")
print(L.multiply(D).multiply(U))
#
# D = Matrix([[5, 3], [3, 2]])
# print()
# D_inv = D.inverse()
# print("\nWrong answer?")
# print(D_inv)
# print("\nRight Answer.")
# print(Matrix([[2, -3],[-3, 5]]))

# AA = Matrix([[0, 1],[2,2]])
# AB = Matrix([[2, 1],[0,2]])
# AC = Matrix([[0,1],[0,2],[0,3],[0,4],[9,5]])
# AD = Matrix([[0,1],[0,2],[9,3]])
# G = [[3,3,3],[4,4,5],[1,2,2]]
# GG = Matrix(G)
# # pivot = AC._get_pivot(0)
# # # print(AA._get_pivot(0))
# print(GG.upper_triangle())
# print(np.triu(np.array(G)))
#

# print(AA._swap_rows(0,1))

# O = [[1, 2, 3],[4, 5, 6]]
# O = Matrix(O)
# a = O.augment()
# # print(a)
#
#
# E = Matrix([[5, 4], [3, 2]])
# D = Matrix([[5, 3], [3, 2]])
# # E = Matrix([[5.5, 3.5],[3.2,3.3]])
# # print(D-E)
# print(E)
# print()
# print(E.symmetric())
# print(D.symmetric())
# print(D.multiply(D.inverse()))
# P = [[7],[8],[9]]
# P = Matrix(P)
#
#
# L = Matrix([[5, 2],[3, 4]])
#
# Q = Matrix([[5, 2],[10,1]])
# # print('\n')
# Q.inverse()

#
# inverse of D = Matrix([[5, 2], [3, 4]])
# 	B1	B2
# 1	2/7	-1/7
# 2	-3/14	5/14
#
#
# Result:
# 5	3
# 3	2
#
# B1	B2
# 1	2	-3
# 2	-3	5
