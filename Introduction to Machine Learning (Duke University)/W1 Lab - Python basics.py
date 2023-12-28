import matplotlib.pyplot as plt
import numpy as np

# create a rank 1 array
x = np.array([2, 4, 6])
# create a rank 2 array
A = np.array([[1, 3, 5], [2, 4, 6]])
B = np.array([[1, 2, 3], [4, 5, 6]])
print("Matrix A: \n")
print(A)
print("\nMatrix B: \n")
print(B)

## indexing
# index the first row and all columns
print(A[0, :])
# index the second row and third column
print(A[1, 2])
# index the second column and all rows
print(A[:, 1])

## multiplication
# multiply every element of A by 2
C = A * 2
# perform element-wise rather than matrix multiplication
D = A * B

E = np.transpose(B)
# perform matrix multiplication
F = np.matmul(A, E)
F2 = np.dot(A, E)
G = np.matmul(A, x)

print("\n Matrix E (the transpose of B): \n")
print(E)
print("\n Matrix F (result of matrix multiplication A x E): \n")
print(F)
print("\n Matrix G (result of matrix-vector multiplication A*x): \n")
print(G)

## broadcasting
# broadcast x for element-wise multiplication with the rows of A
H = A * x
print(H)
# broadcast x for addition across the rows of A
J = B + x
print(J)

# maximum value operations
X = np.array([[3, 9, 4], [10, 2, 7], [5, 11, 8]])
# get the maximum value of the matrix
all_max = np.max(X)
# get the maximum in each column; returns a rank-1 array
col_max = np.max(X, axis=0)
# get the maximum in each row; returns a rank-1 array
row_max = np.max(X, axis=1)

# get the index of the maximum value in each colum
col_argmax = np.argmax(X, axis=0)
print("Matrix X: \n")
print(X)
print("\n Maximum value in X: \n")
print(all_max)
print("\n Column-wise max of X: \n")
print(col_max)
print("\n Indices of column max: \n")
print(col_argmax)
print("\n Row-wise max of X: \n")
print(row_max)

# can also perform other operations like getting the minimum or summing over rows or columns

## reshaping
# make a rank-1 array of integers from 0 to 15
X = np.arange(16)
# reshape into a 4x4 matrix
X_sq = np.reshape(X, (4, 4))
# reshape into a 2x2x4 rank-3 array
X_rank3 = np.reshape(X, (2, 2, 4))
print("Rank-1 array X: \n")
print(X)
print("\n Reshaped into a square matrix: \n")
print(X_sq)
print("\n Reshaped into a rank-3 array with dimensions 2 x 2 x 4: \n")
print(X_rank3)

## plotting
# check the backend and change if required
import matplotlib as mpl

mpl_backend = mpl.get_backend()
if mpl_backend != "Qt5Agg":
    mpl.use("Qt5Agg")
else:
    pass

X = np.identity(10)
imat_img = plt.imshow(X, cmap='Greys_r')

A = np.random.randn(10, 10)
rmat_img = plt.imshow(A)

