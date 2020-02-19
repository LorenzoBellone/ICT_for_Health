import pandas as pd
from sub.minimization import *
np.random.seed(3)

data_input=pd.read_csv("parkinsons_updrs.csv")
statistics=data_input.describe()
data_input.info()
data=data_input.values  # Build a matrix from the data frame
np.random.shuffle(data)  # Shuffle the rows of the matrix data
data=data[:, 4:]  # Delete the first 4 columns from the matrix data, they are not needed

mean_array = statistics.values[1, 4:]  # An array with all the means of each column of matrix data
standard_deviation=statistics.values[2, 4:]  # An array with all the standard deviations of each column of matrix data
mean_array.shape = (1, 18)
standard_deviation.shape = (1, 18)

data_train=data[:2939, :]  # Define the training set
data_val=data[2939:4407, :]  # Define the validation set
data_test=data[4407:, :]  # Define the test set

# Evaluation of the normalized data
data_norm=np.ones((5875, 18), dtype=float)
for i in range(np.shape(data)[0]):
    for j in range(np.shape(data)[1]):
        data_norm[i, j] = (data[i, j]-mean_array[0, j])/standard_deviation[0,j]

# Verify the normalization
# for i in range(data_norm.shape[1]):
#     print(np.mean(data_norm[:, i]), np.std(data_norm[:, i]))

# Inizialization of the normalized data
data_train_norm=data_norm[:2939, :]
data_val_norm=data_norm[2939:4407, :]
data_test_norm=data_norm[4407:, :]

F0=1
y_train=data_train_norm[:, F0]  # An array with the values of the F0 feature of the train set
y_train.shape=(2939, 1)
X_train=np.delete(data_train_norm, F0, 1)  # Define the matrix X without the regressand, for the train set
X_test=np.delete(data_test_norm, F0, 1)  # Define the matrix X without the regressand, for the test set
X_val=np.delete(data_val_norm, F0, 1)  # Define the matrix X without the regressand, for the validation set
y_test=data_test_norm[:, F0]  # An array with the values of the F0 feature of the test set
y_test.shape=(1468, 1)
y_val=data_val_norm[:, F0]  # An array with the values of the F0 feature of the validation set
y_val.shape=(1468, 1)
mean_feature=mean_array[0, F0]  # The mean value of the feature F0
standard_deviation_feature=standard_deviation[0, F0]  # The standard deviation of the feature F0

logx = 0
logy = 1

#Linear least square
m=SolveLLS(y_train, X_train, y_val, X_val, y_test, X_test, mean_feature, standard_deviation_feature)
m.run()
m.print_result('LLS')
m.plot_w('LLS - weight vector')
m.plotyhattrain('Total UPDRS errors - LLS - Train')
m.plotyhattest('Total UPDRS errors - LLS - Test')
m.print_err('LLS')

#Ridge Regression
r=SolveRidge(y_train, X_train, y_val, X_val, y_test, X_test, mean_feature, standard_deviation_feature)
r.run()
r.print_result('Ridge_Regression')
r.plot_w('RR - weight vector')
r.plotyhattrain('Total UPDRS errors - RR - Train')
r.plotyhattest('Total UPDRS errors - RR - Test')
r.plot_err('Find_lambda', logx, logy)
r.print_err('RR')

#Gradient Algorithm
g=SolveGrad(y_train, X_train, y_val, X_val, y_test, X_test, mean_feature, standard_deviation_feature)
g.run()
g.print_result('Gradient_algo')
g.plot_err('Gradient_algo:_square_error',  logx, logy)
g.plot_w('GA - weight vector')
g.plotyhattrain('Total UPDRS errors - GA - Train')
g.plotyhattest('Total UPDRS errors - GA - Test')
g.print_err('GA')

#Steepest descent algorithm
s=SolveSteep(y_train, X_train, y_val, X_val, y_test, X_test, mean_feature, standard_deviation_feature)
s.run(500)
s.print_result('SDA')
s.plot_err('Steepest_decent_algo:_square_error',  logx, logy)
s.plot_w('SDA - weight vector')
s.plotyhattrain('Total UPDRS errors - SDA - Train')
s.plotyhattest('Total UPDRS errors - SDA - Test')
s.print_err('SDA')

#Stochastic Algorithm
st=SolveStocha(y_train, X_train, y_val, X_val, y_test, X_test, mean_feature, standard_deviation_feature)
st.run(100, 1e-3)
st.plot_err('Stochastic_algo:_square_error', logx, logy)
st.plot_w('SGA - weight vector')
st.print_result('SGA')
st.plotyhattrain('Total UPDRS errors - SGA - Train')
st.plotyhattest('Total UPDRS errors - SGA - Test')
st.print_err('SGA')

#Conjugate algorithm
con=Conjugate(y_train, X_train, y_val, X_val, y_test, X_test, mean_feature, standard_deviation_feature)
con.run()
con.plot_err("Conjugate_algo:_square_error", logx, logy)
con.plot_w('CGA - weight vector')
con.print_result('CGA')
con.plotyhattrain('Total UPDRS errors - CGA - Train')
con.plotyhattest('Total UPDRS errors - CGA - Test')
con.print_err('CGA')

# Represent the different w in the same plot
# plt.figure()
# n = np.arange(X_train.shape[1])
# plt.plot(n, m.run(), label='LLS')
# plt.plot(n, r.run(), label='RR')
# plt.plot(n, g.run(), label='GA')
# plt.plot(n, s.run(), label='SDA')
# plt.plot(n, st.run(), label='SGA')
# plt.plot(n, con.run(), label='CGA')
# plt.xlabel('n')
# plt.ylabel('w(n)')
# plt.legend()
# plt.title('All the weight vectors')
# plt.show()
