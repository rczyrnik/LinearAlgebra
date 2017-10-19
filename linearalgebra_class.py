'''
Designed for the MIT OpenCourse Linear Algebra Class
'''

class matrix:
		def __init__(self,m):
			self.m = m
			self.legit()		# check its a 2D array
			#self.f()				# switch to floating points
			self.height = len(m)
			self.width = len(m[0])

		def legit(self):	# is it legit?
			if type(self.m) != list or type(self.m[0]) != list:
				raise TypeError("This is not a list of lists")
			for i in range(1, len(self.m)):
				if len(self.m[i]) != len(self.m[i-1]):
					raise TypeError("This is not a rectangle")

		def f(self):			# converts values to floating points
			for x in range(0, self.width):
				for y in range(0, self.height):
					self.m[y][x] = float(self.m[y][x])
			return(self)

		def square(self):		# is this a square matrix? cause those are cool
			if self.height == self.width: return True
			else: return False

		def display(self):		# show me this matrix!
			for x in range(0,self.height):
				for y in range(0,self.width):
					print("{0:6.2f}".format(self.m[x][y]), end = '')
				print()
			print()

		def determinant(self):	# to find the determinant
			if not self.square(): return False
			if self.width == 2:
				sum = self.m[0][0]*self.m[1][1] - self.m[1][0]*self.m[0][1]
			else:
				sum = 0
				for a in range(0,self.width):
					t = []
					for y in range(1,self.height):
						t.append(self.m[y][0:a]+self.m[y][a+1:self.width])
						tt = matrix(t)
					sum += self.m[0][a] * tt.determinant() * (-1)**a
			return sum

# ------- LECTURE 2: ELIMINATION WITH MATRICES ----------

		def uppertriangle(self):
			multipliers = []
			ut = self.m
			for x in range(0, self.width-1):
															# alter cols 1 to penult

				for y in range(x+1, self.height):
															# alter rows x+1 to the bottom
					multiplier = self.m[y][x]/self.m[x][x]
															# divide the 1st in yth row
															#		by the 1st in the 1st row
					multipliers.append(multiplier)
					for n in range(0,self.width):
															# D all the values in the yth row
						ut[y][n] = self.m[y][n]-self.m[x][n]*multiplier
			return matrix(ut)


		def backsubstitution(self):
			variables = []
			ut = self.m
			for y in range(self.height-1,-1,-1):				# start from bottom
				variables.insert(0,self.m[y][self.width-1]/self.m[y][y])

				# remove that variable from all the equations above
				for yy in range(0, y):		# start at first row, go until current
					multiplier = self.m[yy][y]/self.m[y][y]
																	# find mult to remove variable

					ut[yy][y] = self.m[yy][y] - self.m[y][y]*multiplier
																	# make zero
					ut[yy][-1] = self.m[yy][-1] - self.m[y][-1]*multiplier
																	# fix rt side
			return(variables)

		def solvesystem(self):
			if self.width-self.height != 1: return False
			answers = F.uppertriangle().backsubstitution()
			varnames = ['x', 'y', 'z']
			for n in range(0, len(answers)):
				print("{} = {}".format(varnames[n], answers[n]))


# -------- LECTURE 3: MULTIPLICATION AND INVERSE MATRICS -------------

		def multiply(self, b):
			if self.width != b.height:
				return False

			c = []
			for y in range(0, self.height):		# cycle through columns of a
				t = []
				for x in range(0,b.width):
					sum = 0
					for i in range(0, self.width):
						sum += self.m[y][i]*b.m[i][x]
					t.append(sum)
				c.append(t)

			return matrix(c)

		def augmented(self):
			new = []
			i = []
			n = self.width
			for a in range(0,n):
				i.append([0]*n)
			for x in range(0,n):
				i[x][x] = 1

			for y in range(0,self.height):
				new.append(self.m[y]+i[y])
			print()

			return matrix(new)

		def inverse(self):
			new = self.augmented()
			print("Initial matrix")
			new.display()

			# deal with bits in the lower triangle
			for x in range(0,self.width):

				# make pivot 1
				pivot = new.m[x][x]						# pivots along the diagonals
				if pivot == 0.0:							# dealing with zero pivots
					new.zeropivot(x)
					pivot = new.m[x][x]
				# print("pivot is: {}".format(pivot))
				new.divide(x, pivot)					# make the pivot 1

				# make spots below it 0
				for y in range(x+1, self.height):
					# print("Col is {}, row is {}".format(x, y))
					# print("Multiplier is: {}".format(new.m[y][x]))
					new.rowopp(y, x, new.m[y][x])

			print("Upper trianle")
			new.display()
			# deal with bits in the upper triangle
			for x in range(self.width-1, 0, -1):		# work left to right
				for y in range(x-1, -1, -1):						# work bottom to top
					new.rowopp(y, x, new.m[y][x])

			print("Final matrix")
			new.display()
			return new

		def rowopp(self, rowA, rowB, mult):
			for x in range(0, self.width):
				self.m[rowA][x] -= self.m[rowB][x]*mult
			return self

		def divide(self, row, mult):
			for x in range(0, self.width):
				self.m[row][x] /= mult
			return self

		def swaprows(self, rowA, rowB):
			temp = self.m[rowA]
			self.m[rowA] = self.m[rowB]
			self.m[rowB] = temp
			return self

		def zeropivot(self, pivot):
			for yy in range(pivot+1, self.height):
				if self.m[yy][pivot] != 0:
					self.swaprows(yy, pivot)
					return self
				else:
					return False








		def inverse2(self):
			new = self.copy()
			i = self.identity()

			# deal with bits in the lower triangle
			for x in range(0,self.width):
				pivot = new.m[x][x]						# pivots along the diagonals
				if pivot == 0.0:							# dealing with zero pivots
					new.zeropivot(x)
					pivot = new.m[x][x]
				new.divide(x, pivot)					# make the pivot 1
				i.divide(x, pivot)

				# make spots below it 0
				for y in range(x+1, self.height):
					m = new.m[y][x]
					new.rowopp(y, x, m)
					i.rowopp(y, x, m)

			# deal with bits in the upper triangle
			for x in range(self.width-1, 0, -1):		# work left to right
				for y in range(x-1, -1, -1):						# work bottom to top
					m = new.m[y][x]
					new.rowopp(y, x, m)
					i.rowopp(y, x, m)

			i.display()
			return new








		# creates an identity matrix with the same width as another matrix
		def identity(self):
			i = []
			for a in range(0,self.width):
				i.append([0]*self.width)
				i[a][a] = 1
			return matrix(i)

		def copy(self):
			new = []
			for y in range(self.height):
				temp = []
				for x in range(self.width):
					temp.append(self.m[y][x])
				new.append(temp)
			return matrix(new)

# -------- LECTURE 5: TRANSPOSES, PERMUTATIONS, SPACES R^n -----------

		def transpose(self):
			t = []
			for x in range(0, self.width):
				temp = []
				for y in range(0, self.height):
					temp.append(self.m[y][x])
				t.append(temp)
			return matrix(t)

		def symmetric(self):
			if not self.square() or self.f() != self.transpose().f():
				return False
			else:
				return True



# -------- MY FAVORITE MATRICES ---------
if __name__ == "__main__":
	A = matrix([[1, 2, 3, 4],
			 				[5, 6, 7, 8]])

	B = [1, 2, 3, 4]	# not a matrix

	B = matrix([[1, 5],
							[2, 6],
							[3, 7],
							[4, 8]])

	C = matrix([[5, 2],
			 				[3, 4]])

	D = matrix([[1, 2, 3],
							[4, 5, 6],
							[7, 8, 9]])

	E = matrix([[1, 4, 5, 6],
						  [7, 2, 4, 9],
			 				[2, 4, 4, 3],
			 				[1, 9, 7, 8]])

	F = matrix([[1,2,1,2],
			 				[3,8,1,12],
			 				[0,4,1,2]])

	G = matrix([[3,3,3],		# this matrix can't upper triangle :(
			 				[4,4,5],
			 				[1,2,2]])

	H = matrix([[1, 2, 3],
							[2, 3, 4],
							[3, 4, 5]])

	J = matrix([[1, 2, 3, 4],
							[0, 0, 3, 7],
							[0, 6, 7, 2],
							[0, 0, 7, 2]])
	K = matrix([[0, 2, 3],
							[1, 3, 5],
							[9, 3, 1]])
	L = matrix([[0,2],
							[5, 3]])
	var = ['w', 'x', 'y', 'z']
