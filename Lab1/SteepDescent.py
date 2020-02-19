from sub.minimization import *

Np=8
Nf=4
A=np.random.randn(Np, Nf)
y=np.random.randn(Np, 1)

m=SolveLLS(y, A)
m.run()
m.print_result('LLS')
m.plot_w('LLS')

logx=0
logy=1
s=SolveSteep(y, A)
s.run(100)
s.print_result('SDA')
s.plot_err( logx, logy, 'Steepest_decent_algo:_square_error')