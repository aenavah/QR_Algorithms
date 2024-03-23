import numpy as np 
import scipy 

def norm(matrix):
  f = np.sqrt(np.sum(matrix**2))
  return f

#Q1
'''Hessenberg page 115'''
def reduction(A):
  m, n = np.shape(A)
  H = np.copy(A)
  for j in range(0, m-1):
    s_j = np.sign(A[j+1, j]) * np.sqrt(sum(A[j+1:, j]**2))
    vj = np.zeros(m)
    vj[j+1] = A[j+1, j] + s_j
    for i in range(j+1, m-1):
      vj[i+1] = A[i+1, j]
    vj = vj.reshape((-1, 1))
    vj = vj / np.sqrt((np.sum(vj**2)))
    H = H - 2 * vj @ vj.T @ H
    H = H - 2* H @ vj @ vj.T
  return H

'''Householder pg 147'''
def qr_decomposition(A):
  m, n = np.shape(A)
  Q = np.eye(m)
  for j in range(0, n):
    vj = np.zeros(m)
    sign_sj = np.sign(A[j, j]) 
    norm = np.sqrt(sum(A[j:, j]**2))  #[j,m) 
    s_j = sign_sj * norm
    vj[j] = A[j, j] + s_j


    for i in range(j+1, m):
      vj[i] = A[i, j]
    vj = vj.reshape((-1, 1))
    vj_norm = np.sqrt((np.sum(vj**2)))
    vj = vj / vj_norm
    vj_T = np.transpose(vj)
    A = A - (2 * vj @ vj_T @ A)
    Q =  Q @ (np.eye(m) - 2 *(vj @ vj_T)) #@ Q
  R = np.triu(A)
  return Q, R


def qr_algorithm_sinshift(A):  
    m,n = np.shape(A)
    V = np.eye(m)
    count = 0
    while count <= 1000:
        count += 1
        Q, R = qr_decomposition(A)
        A = R @ Q
        V = V @ Q
        off_diagonal = A - np.diag(np.diagonal(A))
        lambdas = np.diag(A)
        if np.all(np.abs(off_diagonal) < 10**(-10)):
            print("QR algorithm without shift took " + str(count) + " iterations...")
            return lambdas, V
    

def qr_algorithm_conshift(A):
    m, n = np.shape(A)
    count = 0
    while count <= 1000:
        count += 1
        mu = A[m-1, m-1]
        A_shift = A - mu*np.identity(m)
        Q, R = np.linalg.qr(A_shift)
        A = R @ Q + mu * np.identity(m)

        off_diagonal = A - np.diag(np.diagonal(A))
        lambdas = np.diag(A)
        if np.all(np.abs(off_diagonal) < 10**(-10)):
          print("QR algorithm with shift took " + str(count) + " iterations...")
          return lambdas
    
def inverse_iteration(A, mu):
    m, n = np.shape(A)  # init mu
    x = np.ones(m)
    x = x / norm(x)  # normalize x
    B = np.linalg.inv(A - mu * np.eye(n)) 
    
    for i in range(0, 1000):
        Bx = B @ x 
        y = Bx / norm(Bx)  
        r = y - x 
        x = y  
        if norm(r) < 10**(-10): 
            return x
    return x  
if __name__ == "__main__":
  matrix1 = np.array([[5, 4, 1, 1], [4, 5, 1, 1],[1, 1, 4, 2], [1, 1, 2, 4]])
  
#   #1. 
  H = reduction(matrix1)
  H_scipy = scipy.linalg.hessenberg(matrix1)
  print(H)
  print(H_scipy)
  print("||H_scipy - H_mine||")
  print(norm(H_scipy - H)) # -> approx zero <3