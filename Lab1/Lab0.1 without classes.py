import numpy as np
import matplotlib.pyplot as plt

def grad_algo(A, y, w_id):
    print("Gradient Algorithm: ")
    Nit = 3000
    gamma = 1e-2
    err = np.ones((Nit, 2), dtype=float)
    for it in range(Nit):
        grad = 2 * np.dot(A.T, np.dot(A, w_id) - y)
        w_id = w_id - gamma * grad
        err[it, 0] = it
        err[it, 1] = np.linalg.norm(np.dot(A, w_id) - y) ** 2

    plt.figure()
    plt.semilogy(err[:, 0], err[:, 1])
    plt.xlabel("Nit")
    plt.ylabel("e(n)")
    plt.title("Gradient Algorithm")
    plt.margins(0.01, 0.1)
    plt.grid()
    plt.show()

    print("The optimum weight vector is ", w_id)


def steep_algo(A, y, w_id):
    print("Steepest Descent Algorithm: ")
    Nit = 1000
    H = 4 * np.dot(A.T, A)
    err = np.ones((Nit, 2), dtype=float)
    for it in range(Nit):
        grad = 2 * np.dot(A.T, np.dot(A, w_id) - y)
        w_id = w_id - np.linalg.norm(grad) ** 2 / np.dot(np.dot(grad.T, H), grad) * grad
        err[it, 0] = it
        err[it, 1] = np.linalg.norm(np.dot(A, w_id) - y) ** 2

    plt.figure()
    plt.semilogy(err[:, 0], err[:, 1])
    plt.xlabel("Nit")
    plt.ylabel("e(n)")
    plt.title("Steepest Descent Algorithm")
    plt.margins(0.01, 0.1)
    plt.grid()
    plt.show()

    print("The optimum weight vector is ", w_id)

def LLS(A, y):
    print("LLS Solution: ")
    w_id=np.dot(np.dot(np.linalg.inv(np.dot(A.T, A)), A.T), y)
    print("The best weight vector is ", w_id)

Np=4
Nf=4
A=np.random.randn(Np, Nf)
y=np.random.randn(Np, 1)
w=np.random.rand(Nf, 1)
grad_algo(A, y, w)
steep_algo(A, y, w)
LLS(A, y)



