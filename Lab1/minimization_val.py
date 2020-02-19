import numpy as np
import matplotlib.pyplot as plt

class SolveMinProbl:
    def __init__(self, y, A, y_val, A_val, y_test, A_test): #initialization
        self.matr=A
        self.Np=y.shape[0]
        self.Nf=A.shape[1]
        self.vect=y
        self.vect_val=y_val
        self.matr_val=A_val
        self.matr_test=A_test
        self.vect_test=y_test
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

    def plot_err(self, title='Square_error', logx=0, logy=0):
        err=self.err
        plt.figure()
        if (logy==0) and (logx==0):
            plt.plot(err[:, 0], err[:, 1])
        if (logy == 1) and (logx == 0):
            plt.semilogy(err[:, 0], err[:, 1], label='Train')
        if (logy == 0) and (logx == 1):
            plt.semilogx(err[:, 0], err[:, 1])
        if (logy == 1) and (logx == 1):
            plt.loglog(err[:, 0], err[:, 1])
        if (logy==0) and (logx==0):
            plt.plot(err[:, 0], err[:, 2])
        if (logy == 1) and (logx == 0):
            plt.semilogy(err[:, 0], err[:, 2], label='Validation')
        if (logy == 0) and (logx == 1):
            plt.semilogx(err[:, 0], err[:, 2])
        if (logy == 1) and (logx == 1):
            plt.loglog(err[:, 0], err[:, 2])
        plt.xlabel('n')
        plt.ylabel('e(n)')
        plt.legend()
        plt.title(title)
        plt.margins(0.01, 0.1)
        plt.grid()
        plt.show()
        return

    def plotyhattest(self, title):
        w=self.sol
        A_test=self.matr_test
        y_test=self.vect_test
        y_hat_test=np.dot(A_test, w)
        plt.figure()
        plt.scatter(y_test, y_hat_test)
        plt.xlabel('y_test')
        plt.ylabel('y_hat_test')
        plt.title(title)
        plt.grid()
        plt.show()

    def plotyhattrain(self, title):
        w=self.sol
        A_train=self.matr
        y_train=self.vect
        y_hat_train=np.dot(A_train, w)
        plt.figure()
        plt.scatter(y_train, y_hat_train)
        plt.xlabel('y_train')
        plt.ylabel('y_hat_train')
        plt.title(title)
        plt.grid()
        plt.show()

class SolveGrad(SolveMinProbl):
    def run(self, gamma=1e-5, Nit=1000):
        np.random.seed(2)
        self.err=np.zeros((Nit, 3), dtype=float)
        A=self.matr
        y=self.vect
        A_val = self.matr_val
        y_val = self.vect_val
        w=np.random.rand(self.Nf, 1)
        for it in range(Nit):
            grad=2*np.dot(A.T, (np.dot(A,w)-y))
            w=w-gamma*grad
            self.err[it, 0]=it
            self.err[it, 1]=np.linalg.norm(np.dot(A, w)-y)
            self.err[it, 2]=np.linalg.norm(np.dot(A_val, w)-y_val)
        self.sol=w
        self.min=self.err[it, 1]

class SolveSteep(SolveMinProbl):
    def run(self, Nit=1000):
        np.random.seed(2)
        self.err = np.zeros((Nit, 3), dtype=float)
        #self.err_val[0, 1]=10000
        A_val=self.matr_val
        y_val=self.vect_val
        A = self.matr
        y = self.vect
        w = np.random.rand(self.Nf, 1)
        for it in range(Nit):
            grad = 2 * np.dot(A.T, (np.dot(A, w) - y))
            H=2*np.dot(A.T, A)
            w=w-np.linalg.norm(grad)**2/np.dot(np.dot(grad.T, H), grad)*grad
            self.err[it, 0] = it
            self.err[it, 1] = np.linalg.norm(np.dot(A, w) - y)
            self.err[it, 2]=np.linalg.norm(np.dot(A_val, w) - y_val)
        self.sol = w
        self.min = self.err[it, 1]

class SolveStocha(SolveMinProbl):
    def run(self, Nit=100, gamma=1e-2):
        np.random.seed(2)
        self.err = np.zeros((Nit, 3), dtype=float)
        A_val = self.matr_val
        y_val = self.vect_val
        A = self.matr
        y = self.vect
        w = np.random.rand(self.Nf, 1)
        Ac = np.zeros((self.Nf, 1), dtype=float)
        for it in range(Nit):
            for i in range(self.Np):
                for j in range(self.Nf):
                    Ac[j, 0] = A[i, j]
                grad = (gamma * (np.dot(A[i], w) - y[i])) * Ac
                w = w - grad
            self.err[it, 0] = it
            self.err[it, 1] = np.linalg.norm(np.dot(A, w) - y)
            self.err[it, 2] = np.linalg.norm(np.dot(A_val, w) - y_val)
        self.sol = w
        self.min = self.err[it, 1]