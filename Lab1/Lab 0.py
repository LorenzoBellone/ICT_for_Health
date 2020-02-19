#Generate a matrix A and a vector y
#In this fake case we know already the values of w, and calculate y

import numpy as np
Nr=5
Nc=4
A=np.random.randn(Nr, Nc)
w_id=np.random.randn(Nc)
y=np.dot(A, w_id)
hes=nd.Gradient(np.dot(A, w_id)-y)
print(hes)

w=np.dot(np.linalg.pinv(A), y) #We obtained w from the matrix A and the vector y, assuming the error = 0
                      #There is a function that allows you to calculate

