import numpy as np
import scipy 

import Hessenberg 
hessenberg_reduction = Hessenberg.reduction
norm = Hessenberg.norm
qr_decomposition = Hessenberg.qr_decomposition
qr_algorithm_sinshift = Hessenberg.qr_algorithm_sinshift
qr_algorithm_conshift = Hessenberg.qr_algorithm_conshift
inverse_iteration = Hessenberg.inverse_iteration

if __name__ == "__main__":
  #Q1
  '''Hessenberg form'''
  print("Q1---------------")
  matrix1 = np.array([[5, 4, 1, 1], 
                      [4, 5, 1, 1],
                      [1, 1, 4, 2], 
                      [1, 1, 2, 4]])
  matrix1 = matrix1.copy()
  print("Hessenberg form of A = ")
  print(matrix1)
  H = hessenberg_reduction(matrix1)
  H_sci = scipy.linalg.hessenberg(matrix1)
  print("Difference between scipy Hessenberg and my Hessenberg:")
  print(norm(H_sci - H))

  #Q2
  '''QR algorithms with and without shift'''
  matrix2 = np.array([[3.0, 1.0, 0.0], 
                      [1.0, 2.0, 1.0], 
                      [0.0, 1.0, 1.0]])
  matrix2 = np.copy(matrix2)
  # (i) without shift 
  print("Q2i---------------")
  eigenvals_wo, eigenvecs_wo = qr_algorithm_sinshift(matrix2)
  print("eigenvalues and eigenvectors for QR algorithm without shift:")
  print(eigenvals_wo)
  print(eigenvecs_wo)

  # (ii) with shift
  print("Q2ii---------------")
  eigenvals_w = qr_algorithm_conshift(matrix2)
  print("eigenvalues and eigenvectors for QR algorithm with shift:")
  print(eigenvals_w)

  #print(np.linalg.eig(matrix2))

  #Q3
  print("Q3---------------")
  '''inverse iteration'''
  matrix3 = np.array([[2, 1, 3, 4],
                     [1, -3, 1, 5],
                     [3, 1, 6, -2],
                     [4, 5, -2, -1]])
  
  eigenvals_invit = [-8.0286, 7.9329, 5.6689, -1.5732]
  print("inverse iteration algorithm:")
  for lambdas in eigenvals_invit:
    v = inverse_iteration(matrix3, lambdas)
    print("lambda and v: ")
    print(lambdas, v)


  #Hayley helped me :) 

