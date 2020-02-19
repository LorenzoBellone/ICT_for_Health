from sub.minimization import *

Np=2939
Nf=17
A=np.random.rand(Np, Nf)
y=np.random.rand(Np, 1)

m=SolveLLS(y, A)
m.run()
m.print_result('LLS')
m.plot_w('LLS')

con=Conjugate(y, A)
con.run()
con.plot_err("Conjugate Algorithm", 0, 1)
con.print_result('CNJ')
con.print_err("Conjugate")