import numpy as np
import scipy 

import Hessenberg 
hessenberg_reduction = Hessenberg.reduction
norm = Hessenberg.norm
qr_decomposition = Hessenberg.qr_decomposition
qr_algorithm_sinshift = Hessenberg.qr_algorithm_sinshift

if __name__ == "__main__":
  #Q1
  matrix1 = np.array([[5, 4, 1, 1], [4, 5, 1, 1],[1, 1, 4, 2], [1, 1, 2, 4]])
  H = hessenberg_reduction(matrix1)

  #Q2
  matrix2 = np.array([[3.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 1.0]])
  error = 1.0**(-10)
  Q, R = qr_decomposition(matrix2)
  Q_sci, R_sci = np.linalg.qr(matrix2)
  #eigvals = qr_algorithm_sinshift(Q, R, error)
  #print(eigvals)
  #print(Q)
  #print(Q_sci)
  print("---")
  #print(R)
  #print(R_sci)
  print(norm(matrix2 - Q@R))
