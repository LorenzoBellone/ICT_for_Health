#Use of classes, 1: initialization part, we tìdefine the input of parameters, for axample the matrix A or the vector y.

import numpy as np
import matplotlib.pyplot as plt

class SolveMinProbl:
    def __init__(self, y=np.ones((6, 1)), A=np.eye(5)): #initialization
        self.matr=A
        self.Np=y.shape[0]
        self.Nf=A.shape[1]
        self.vect=y
        self.sol=np.zeros((self.Nf, 1), dtype=float)
        return
    def plot_w(self, title='Solution'):
        w=self.sol
        n=np.arange(self.Nf)
        plt.figure()
        plt.plot(n, w)
        plt.xlabel('n')
        plt.ylabel('w(n)')
        plt.title(title)
        plt.grid()
        plt.show()
        return
    def print_result(self, title):
        print(title, '_:')
        print("The optimum weight vector is:_")
        print(self.sol)
        return

    def plot_err(self, title='Square_error', logy=0, logx=0):
        err=self.err
        plt.figure()
        if (logy==0) and (logx==0):
            plt.plot(err[:, 0], err[:, 1])
        if (logy == 1) and (logx == 0):
            plt.plot(err[:, 0], err[:, 1])
        if (logy == 0) and (logx == 1):
            plt.semilogy(err[:, 0], err[:, 1])
        if (logy == 1) and (logx == 1):
            plt.loglog(err[:, 0], err[:, 1])
        plt.xlabel('n')
        plt.ylabel('e(n)')
        plt.title(title)
        plt.margins(0.01, 0.1)
        plt.grid()
        plt.show()
        return

class SolveLLS(SolveMinProbl):
    def run(self):
        A=self.matr
        y=self.vect
        w=np.dot(np.dot(np.linalg.inv(np.dot(A.T, A)), A.T), y)
        self.sol=w
        self.min=np.linalg.norm(np.dot(A, w)-y)


class SolveGrad(SolveMinProbl):
    def run(self, gamma, Nit=1000):
        self.err=np.zeros((Nit, 2), dtype=float)
        A=self.matr
        y=self.vect
        w=np.random.rand(self.Nf, 1)
        for it in range(Nit):
            grad=2*np.dot(A.T, (np.dot(A,w)-y))
            w=w-gamma*grad
            self.err[it, 0]=it
            self.err[it, 1]=np.linalg.norm(np.dot(A, w)-y)
        self.sol=w
        self.min=self.err[it, 1]


class SolveSteep(SolveMinProbl):
    def run(self, Nit=1000):
        self.err = np.zeros((Nit, 2), dtype=float)
        A = self.matr
        y = self.vect
        w = np.random.rand(self.Nf, 1)
        for it in range(Nit):
            grad = 2 * np.dot(A.T, (np.dot(A, w) - y))
            H=4*np.dot(A.T, A)
            w=w-np.linalg.norm(grad)**2/np.dot(np.dot(grad.T, H), grad)*grad
            self.err[it, 0] = it
            self.err[it, 1] = np.linalg.norm(np.dot(A, w) - y)
        self.sol = w
        self.min = self.err[it, 1]

class SolveStocha(SolveMinProbl):
    def run(self,Nit=100, gamma=1e-3):
        self.err=np.zeros((Nit, 2), dtype=float)
        A=self.matr
        y=self.vect
        w=np.random.rand(self.Nf, 1)
        for it in range(Nit):
            for i in range(self.Nf):
                grad=gamma*(np.dot(A[i], w)-y[i])*A[i]
                w=w-grad
            self.err[it, 0]=it
            self.err[it, 1]= np.linalg.norm(np.dot(A, w) - y)
        self.sol=w
        self.min=self.err[it, 1]

if __name__=="__main__":
    Np=7
    Nf=7
    A=np.random.randn(Np, Nf)
    y=np.random.randn(Np, 1)
    m=SolveLLS(y, A)
    m.run()
    m.print_result('LLS')
    m.plot_w('LLS')

    gamma=1e-2
    g=SolveGrad(y, A)
    g.run(gamma)
    g.print_result('Gradient_algo')
    logx=0
    logy=1
    g.plot_err('Gradient_algo:_square_error', logx, logy)

    s=SolveSteep(y, A)
    s.run()
    s.print_result('SDA')
    s.plot_err('Steepest_decent_algo:_square_error', logx, logy)

    st=SolveStocha(y, A)
    st.run()
    s.plot_err('Stochastic Algorithm', logx, logy)
    s.print_result('SA')

#with no.random.seed(N) i obtain also the same random values
