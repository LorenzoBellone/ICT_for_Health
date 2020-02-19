from sub.minimization import *

Np=5
Nf=5
A=np.random.randn(Np, Nf)
y=np.random.randn(Np, 1)

m=SolveLLS(y, A)
m.run()
m.print_result('LLS')
m.plot_w('LLS')

logx=0
logy=1
st=SolveStocha(y, A)
st.run()
st.plot_err('Stochastic Algorithm', 0, 1)
st.print_result('SA')