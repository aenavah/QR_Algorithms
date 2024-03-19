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
  for j in range(0, m - 2):
    s_j = np.sign(A[j+1, j]) * np.sqrt(sum(A[j+1:, j]**2))
    vj = np.zeros(m)
    vj[j+1] = A[j+1, j] + s_j
    for i in range(j+2, m-1):
      vj[i] = A[i+1, j]
    vj = vj.reshape((-1, 1))
    vj = vj / np.sqrt((np.sum(vj**2)))
    H = H - 2 * vj @ vj.T @ H
    H = H - 2 * H @ vj @ vj.T
  return H

'''Householder pg 147'''
def qr_decomposition(A):
  m, n = np.shape(A)
  Q = np.eye(m)
  for j in range(0, n):
    vj = np.zeros(m)
    #sign_sj = np.sign(A[j, j]) 
    print("ajj", str(A[j,j]))
    if A[j,j] < 0:
      sign_sj = -1
    if A[j,j] > 0:
      sign_sj = 1
    print("sign" + str(sign_sj))
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
  #R = A
  #Q[:, -1] *= -1
  #R[-1] *= -1
  #last row or R, all signs flipped, last column of Q all signs flipped???
  return Q, R

def qr_algorithm_sinshift(Q, R, threshold):
  while True:
    A = R@Q
    Q_next , R_next = qr_decomposition(A)
    A_next = Q_next@R_next
    error = norm(A - A_next)
    if error < threshold:
      return A_next
    Q, R = Q_next, R_next
  return A
  

# if __name__ == "__main__":
#   matrix1 = np.array([[5, 4, 1, 1], [4, 5, 1, 1],[1, 1, 4, 2], [1, 1, 2, 4]])
  
#   #1. 
#   H = reduction(matrix1)
#   H_scipy = scipy.linalg.hessenberg(matrix1)
#   print(H)
#   print(H_scipy)
#   print("||H_scipy - H_mine||")
#   print(norm(H_scipy - H)) # -> approx zero <3

