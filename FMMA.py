import numpy as np
import time
def FMMA1(A, B):
    "Receive 2 matrices with size m x n and n x p and return the matrix product"
    m = len(A)
    n = len(A[0])
    p = len(B[0])
    if n != len(B):
        print("Error: incompatible dimensions")
        return
    C = []
    for i in range(m):
        C.append([0]*p)
        for j in range(p):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C

# Define the dimensions of the matrices
m = 500
n = 500
p = 500

# Generate random matrices

A = np.random.randint(0, 10, size=(m, n))
B = np.random.randint(0, 10, size=(n, p))

start = time.time()
C = FMMA1(A, B)
end = time.time()
print("Time elapsed: ", end - start)

start = time.time()
D = np.dot(A, B)
end = time.time()
print("Time elapsed: ", end - start)

if np.array_equal(C, D):
    print("Correct")
else:
    print("Incorrect")

import numpy as np
import time

def add(A, B):
    A = np.array(A)
    B = np.array(B)
    if A.shape != B.shape:
        raise ValueError("Matrices must be of the same dimension")
    C = np.zeros(A.shape)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            C[i,j] = A[i,j] + B[i,j]
    return C

def sub(A, B):
    A = np.array(A)
    B = np.array(B)
    if A.shape != B.shape:
        raise ValueError("Matrices must be of the same dimension")
    C = np.zeros(A.shape)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            C[i,j] = A[i,j] - B[i,j]
    return C

def split(A):
    A = np.array(A)
    m, n = A.shape
    top_left = A[0:m//2, 0:n//2]
    top_right = A[0:m//2, n//2:n]
    bottom_left = A[m//2:m, 0:n//2]
    bottom_right = A[m//2:m, n//2:n]
    return top_left, top_right, bottom_left, bottom_right

def concat(top_left, top_right, bottom_left, bottom_right):
    top = np.concat((top_left, top_right), axis=1)
    bottom = np.concat((bottom_left, bottom_right), axis=1)
    return np.concat((top, bottom), axis=0)

def mm(A, B):
    A = np.array(A)
    B = np.array(B)
    if A.shape[1] != B.shape[0]:
        raise ValueError("Matrices must be of compatible dimensions")
    C = np.zeros((A.shape[0], B.shape[1]))
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            for k in range(A.shape[1]):
                C[i,j] += A[i,k] * B[k,j]
    return C

def strassen(A, B):
    "Require A and B to be small enough"
    A = np.array(A)
    B = np.array(B)
    m, n = A.shape
    p, q = B.shape
    if n != p:
        raise ValueError("Matrices must be of compatible dimensions")
    elif m == 2 and n == 2 and p == 2 and q == 2:
        M1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
        M2 = (A[1,0] + A[1,1]) * B[0,0]
        M3 = A[0,0] * (B[0,1] - B[1,1])
        M4 = A[1,1] * (B[1,0] - B[0,0])
        M5 = (A[0,0] + A[0,1]) * B[1,1]
        M6 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
        M7 = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
        C = np.zeros((2,2))
        C[0,0] = M1 + M4 - M5 + M7
        C[0,1] = M3 + M5
        C[1,0] = M2 + M4
        C[1,1] = M1 - M2 + M3 + M6
        return C
    else:
        return smm(A, B)

def smm(A, B):
    A = np.array(A)
    B = np.array(B)
    m, n = A.shape
    p, q = B.shape
    if n != p:
        raise ValueError("Matrices must be of compatible dimensions")
    elif m == 1 and n == 1 and p == 1 and q == 1:
        return A * B
    else:
        A11, A12, A21, A22 = split(A)
        B11, B12, B21, B22 = split(B)
        M1 = smm(add(A11, A12), add(B11, B22))
        M2 = smm(add(A21, A22), B11)
        M3 = smm(A11, sub(B12, B22))
        M4 = smm(A22, sub(B21, B11))
        M5 = smm(add(A11, A22), add(B11, B12))
        M6 = smm(sub(A12, A22), add(B21, B22))
        M7 = smm(sub(A11, A21), add(B11, B12))
        C11 = add(sub(add(M1, M4), M5), M7)
        C12 = add(M3, M5)
        C21 = add(M2, M4)
        C22 = add(add(sub(M1, M2), M3), M6)
        return concat(C11, C12, C21, C22)

A = np.random.randint(-10000, 10000, (500,500))
B = np.random.randint(-10000, 10000, (500,500))

start = time.time()
C = mm(A, B)
end = time.time()
print("time taken for mm: ", end - start)

start = time.time()
E = np.matmul(A, B)
end = time.time()
print("time taken for np.matmul: ", end - start)

if np.array_equal(C, E):
    print("mm and np.matmul are equal")
else:
    print("mm and np.matmul are not equal")
