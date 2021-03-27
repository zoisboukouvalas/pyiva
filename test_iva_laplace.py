import numpy as np
from pyiva.iva_laplace import *

N = 8
K = 4
T = 10000

S = np.zeros((K, N, T))

for n in range(0, N):
   Z = randmv_laplace(K, T)
   S[:, n, :] = Z

A = np.random.rand(K,N,N)
X = S

for k in range(0,K):
    A[k,:,:]= np.transpose(vecnorm(A[k,:,:])[0])
    X[k,:,:]= np.matmul(A[k,:,:], S[k,:,:])

W = iva_laplace(X, A=A)

print("Done")
