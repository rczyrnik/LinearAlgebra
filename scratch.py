'''
scratch paper
'''

from linearalgebra import *
import linearalgebra as la
import numpy as np


# O = [[1, 2, 3],[4, 5, 6]]
# O = Matrix(O)
# a = O.augment()
# # print(a)


E = Matrix([[5, 4], [3, 2]])
D = Matrix([[5, 3], [3, 2]])
# E = Matrix([[5.5, 3.5],[3.2,3.3]])
# print(D-E)
print(E)
print()
print(E.symmetric())
print(D.symmetric())
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
