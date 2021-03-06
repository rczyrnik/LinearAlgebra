'''
Designed for the MIT OpenCourse Linear Algebra Class
'''

import collections
class Matrix(collections.MutableSequence):
# -------- INITIALIZE ---------------------------------------------------------
	# initialize matrix
	def __init__(self,lst=[]):
		self._list = lst
		# self.legit()		# check its a 2D array
		#self.f()				# switch to floating points
		try:
			self.height = len(self)
		except:
			self.height = 0
		try:
			self.width = len(self[0])
		except:
			self.width = 0

	# magic methods
	def __delitem__(self, i): del self._list[i]
	def __getitem__(self, i): return self._list[i]
	def __len__(self): return len(self._list)
	def __setitem__(self, i, v): self._list.insert(i,v)
	def __iter__(self): return iter(self._list)
	def __eq__(self, other):
	  if self._list == other:
	    return True
	  return False
	def __sub__(self, other):
		'''
		not a real function
		needed to use assertAlmostEqual in testing
		'''
		A = self._list
		B = other._list
		s = 0
		for x in range(len(A)):
			for y in range(len(A[0])):
				 s += (A[x][y]-B[x][y])
		return s
	def __repr__(self):	   	   # show me this matrix!
		string = []
		for x in range(0,self.height):
			row = ''
			for y in range(0,self.width):
				row += "{0:6.2f}".format(self[x][y])
			string.append(row)
		return '\n'.join(string)

	def insert(self, i, v): self._list.insert(i,v)
	def append(self, v): self._list.append(v)

	# general useful matrix stuff
	def legit(self):	       # is it legit?
		if self._list == []: return True
		if type(self._list) != list or type(self._list[0]) != list:
			raise TypeError("This is not a list of lists")
			return False
		for i in range(1, len(self._list)):
			if len(self._list[i]) != len(self._list[i-1]):
				raise TypeError("This is not a rectangle")
				return False
		return True

	def square(self):		   # is this a square matrix? cause those are cool
		if self.height == self.width: return True
		else: return False

	def det(self):	           # to find the determinant
		if not self.square():
			raise TypeError("No determinant of a non-square matrix")
			return None

		if self.width == 2:
			s = self._list[0][0]*self._list[1][1] - self._list[1][0]*self._list[0][1]

		else:
			s = 0
			for a in range(self.width):
				t = []
				for y in range(1,self.height):
					t.append(self._list[y][0:a]+self._list[y][a+1:self.width])
				s += self._list[0][a] * Matrix(t).det() * (-1)**a

		return s

	def _copy(self):	       # returns a copy of the matrix
		new_matrix = Matrix([])
		for row in range(self.height):
			r = []
			for col in range(self.width):
				r.append(self[row][col])
			new_matrix.append(r)
		new_matrix.height = self.height
		new_matrix.width = self.width
		return new_matrix

	def zeros(self):		   # returns a zero matrix the size of self
		'''
		returns a zero matriz the same shape as the self matrix
		'''
		zero = self._copy()
		for row in range(self.height):
			for col in range(self.width):
				zero[row][col] = 0
		return zero

# ------- LECTURE 1: THE GEOMETRY OF LINEAR SPACES ------------------------

# ------- LECTURE 2: ELIMINATION WITH MATRICES ----------------------------
	def row_echelon(self):		# upper triangle
		'''
		Finds the uppertriangular matrix
		and saves it as self.ut
		'''
		ut = self._copy()
		size = min(self.height, self.width)
		for col in range(0, size):
			ut, pivot = ut._get_pivot(col)
			for row in range(col+1, self.height):
				multiplier = ut[row][col]/pivot
				ut = ut._row_op(row, col, multiplier)
		return ut

	def back_substitution(self):
		'''
		INPUT:
		only works on matrics with m rows, n columns: n = m+1
		OUTPUT:
		list with variable values
		'''

		if self.width-self.height != 1: return False

		variables = []
		solved = self.row_echelon()._copy()
		for rowA in range(self.height-1,-1,-1):		# start from bottom, work to top
			''' STEP 1: FIND THE VARIABLE '''
			variable = solved[rowA][-1]/solved[rowA][rowA]
			solved[rowA][-1] = variable
			solved[rowA][rowA] = 1
			variables.insert(0, variable)
			''' STEP 2: CLEAR ROWS ABOVE '''
			for rowB in range(0, rowA):				# start at first row, go until current
				multiplier = -1*solved[rowB][rowA]/solved[rowA][rowA] # find mult to remove variable
				for col in range(0, self.width):			# multiply every member of the row
					solved[rowB][col] += solved[rowA][col]*multiplier
		return variables, solved

# -------- LECTURE 3: MULTIPLICATION AND INVERSE MATRICS ------------------
	def multiply(self, b):
		'''
		finds the product of two matrices
		'''

		if self.width != b.height:
			return False
		c = []
		for y in range(0, self.height):		# cycle through columns of a
			t = []
			for x in range(0,b.width):
				sum = 0
				for i in range(0, self.width):
					sum += self[y][i]*b[i][x]
				t.append(sum)
			c.append(t)
		return Matrix(c)

	def augment(self):
		augmented = Matrix([])
		i = []

		# n = self.width
		# I forgot why I needed an augmented matrix. Add square or rect?
		for a in range(0,self.width):
			i.append([0]*self.width)
			i[a][a] = 1
		# for x in range(0,self.height):
		# 	i[x][x] = 1

		for y in range(0,self.height):
			augmented.append(self[y]+i[y])

		augmented.width = len(augmented[0])
		augmented.height = len(augmented)
		return augmented

	def _get_pivot(self, row):
		'''
		for now returns the pivot
		in the future will swap rows if necessary
		'''
		col = row
		for r in range(row, self.height):
			if self[r][col] != 0:
				self = self._swap_rows(row,r)
				return self, self[row][col]
		return False

	def _row_op(self, rowA, rowB, mult):
		'''
		INPUT:
		self MATRIX a matrix
		rowA INT    the row to be changed
		rowB INT    the row subtracted from rowA
		mult FLOAT  what to multiply rowB by before subtracting

		creates a new matrix (could be bad timewise)
		'''
		row_op_matrix = self._copy()
		for col in range(0, self.width):
			row_op_matrix[rowA][col] -= row_op_matrix[rowB][col]*mult
		return row_op_matrix

	def _divide(self, row, mult):
		'''
		INPUT:
		self MATRIX  a matrix
		row  INT     the row to divide
		mult FLOAT   what to divide by

		creates a new matrix (could be bad timewise)
		'''
		divide = self._copy()
		for col in range(0, divide.width):
			divide[row][col] /= mult
		return divide

	def _swap_rows(self, rowA, rowB):
		'''
		INPUT:
		self           MATRIX a matrix
		rowA and rowB  INT    rows to swap
		mult           FLOAT  what to divide by

		modifies in place
		'''
		temp = self._copy()._list
		temp[rowB] = self[rowA]
		temp[rowA] = self[rowB]
		# temp.height = self.height
		# temp.width = self.width
		self = Matrix(temp)
		return self

	def inverse(self):
		'''
		from top to bottom:
		1. make pivot 1
		2. make zeros below pivot
		then, from bottom to top
		3. make zeros above pivot
		'''

		augmented = self.augment()		# add an identity matrix

		# go through making each pivot 1 and clearing rows below
		for col in range(0, self.height):
			augmented, pivot = augmented._get_pivot(col)
			augmented = augmented._divide(col, pivot)

			# make spots below it 0
			for row in range(col+1, self.height):
				augmented = augmented._row_op(row, col, augmented[row][col])

		# go back through clearing rows above
		for col in range(self.height-1, 0, -1):	        # work left to right
			for row in range(col-1, -1, -1):		    # work bottom to top
				augmented = augmented._row_op(row, col, augmented[row][col])

		# cut off identity
		inverse_matrix = Matrix([])
		inverse_matrix.height = self.height
		inverse_matrix.width = self.width
		for row in range(inverse_matrix.height):
			inverse_matrix.append(augmented[row][self.width:])

		return inverse_matrix



		# creates an identity matrix with the same width as another matrix

	def identity(self, size = 0):
		'''
		returns an identity matrix the same size as the
		given matrix
		'''
		if size == 0: size = self.width
		i = []
		for a in range(0,size):
			i.append([0]*size)
			i[a][a] = 1
		return Matrix(i)

# -------- LECTURE 4: FACTORIZATION INTO A=LU --------------------
	def factor(self):
		'''
		INPUT
		square matrix

		OUTPUT
		L, D, U
		'''

		U = self.row_echelon()
		D = self.zeros()
		L = self.multiply(U.inverse())
		for row in range(self.height):
			pivot = U[row][row]
			D[row][row] = pivot
			U = U._divide(row, pivot)
		return L, D, U

# -------- LECTURE 5: TRANSPOSES, PERMUTATIONS, SPACES R^n --------------------
	def transpose(self):
		return Matrix([[self[y][x] for y in range(0, self.height)] for x in range(0, self.width)])

	def symmetric(self):
		return self == self.transpose()

# -------- LECTURE 6: COLUMN SPACES AND NULL SPACES ---------------------------

	def rank(self):			# rank of matrix, might not be lecture 6


# -------- LECTURE 7: SOLVING AX=0, PIVOT VARIABLES, SPECIAL SOLUTIONS --------
