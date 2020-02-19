from sub.minimization import *

Np=4
Nf=4
A=np.random.randn(Np, Nf)
y=np.random.randn(Np, 1)

m=SolveLLS(y, A)
m.run()
m.print_result('LLS')
m.plot_w('LLS')

logx=0
logy=1
mb=MiniBatch(y, A)
mb.run(1000, 1e-1, 2)
mb.plot_err(logx, logy, 'MiniBatches')
mb.print_result('MiniBatches')


class MiniBatch(SolveMinProbl):
    def run(self, Nit, gamma, n_divisions): #n_divisions is the maximum number of rows for each sub-matrix
        A=self.matr
        y=self.vect
        self.err = np.zeros((Nit, 2), dtype=float)
        w = np.random.rand(self.Nf, 1)
        for it in range(Nit):
            n=0
            while n<=self.Np:
                if n+n_divisions<=self.Np:
                    Ai=A[n:n+n_divisions, :]
                    yi=y[n:n+n_divisions]
                    n=n+n_divisions
                else:
                    Ai=A[n:, :]
                    yi=y[n:]
                    n=self.Np+1
                grad=2*np.dot(Ai.T, (np.dot(Ai,w)-yi))
                w=w-gamma*grad
            self.err[it, 0]=it
            self.err[it, 1]=np.linalg.norm(np.dot(A, w) - y)
        self.sol=w
        self.min=self.err[it, 1]
        return