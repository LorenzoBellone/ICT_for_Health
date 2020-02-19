from sub.minimization import *

Np=100
Nf=5
A=np.random.randn(Np, Nf)
y=np.random.randn(Np, 1)
np.insert(A, Nf, 1, axis=1)

m=SolveLLS(y, A)
m.run()
m.print_result('LLS')
m.plot_w('LLS')

gamma=1e-3
g=SolveGrad(y, A)
g.run(gamma)
g.print_result('Gradient_algo')
logx=0
logy=1
g.plot_err('Gradient_algo:_square_error',  logx, logy)

'''s=SolveSteep(y, A)
s.run(100)
s.print_result('SDA')
s.plot_err('Steepest_decent_algo:_square_error',  logx, logy)'''

st=SolveStocha(y, A)
st.run()
st.plot_err('Stochastic_algo:_square_error', logx, logy)
st.print_result('SA')

mb=MiniBatch(y, A)
mb.run(100, 1e-3, 2)
mb.plot_err('MiniBatches_algo:_square_error',  logx, logy)
mb.print_result('MiniBatches')

con=Conjugate(y, A)
con.run()
con.plot_err("Conjugate_algo:_square_error", logx, logy)
con.print_result('CNJ')

r=SolveRidge(y, A)
r.run()
r.print_result('Ridge_Regression')
r.plot_w('RRA')