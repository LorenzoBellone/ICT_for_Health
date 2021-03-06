import pandas as pd
import numpy as np
from sklearn import svm

# Define the name of the feature
feat_names = ['Age', 'Blood Pressure', 'Specific Gravity', 'Albumin', 'Sugar', 'Red Blood Cells', 'Pus Cell',
              'Pus Cell Clumps', 'Bacteria', 'Blood Glucose Random', 'Blood Urea', 'Serum Creatinine', 'Sodium',
              'Potassium', 'Hemoglobin', 'Pacet Cell Volume', 'White Blood Cell Count', 'Red Blood Cell Count',
              'Hypertension', 'Diabetes Mellitus', 'Coronary Artery Disease', 'Appetite', 'Pedal Edema', 'Anemia', 'Class']
# Generate the DataFrame, considering that the dataset is not clean
x = pd.read_csv('chronic_kidney_disease.arff', skiprows=29, sep=',', header=None, na_values=['?', '\t?'],
                names=feat_names)
# Replace the categorical features into numerical data
data_fr = x.replace(['normal', 'present', 'yes', 'good', 'ckd', ' ckd', ' yes', '	yes', 'yes ', 'yes	'], 1)
data_fr = data_fr.replace(['abnormal', 'notpresent', 'no', 'poor', 'notckd', ' no', '	no', 'no ', 'no	' 'notckd '], 0)
# Define the matrix X from the DataFrame with 25 valid data
X = data_fr.dropna(thresh=25).values
# Define the matrix X from the DataFrame with 20 valid data
X_20 = data_fr.dropna(thresh=20).values
# Normalization of the data
mean_X = np.mean(X, axis=0)
std_X = np.std(X, axis=0)
X_25_norm = (X - mean_X)/std_X

X_20_test = X_20.copy()
X_20_norm = (X_20 - mean_X)/std_X

# Ridge Regression for missing data
for i in range(X_20.shape[0]):  # Search how many missing value there are row by row
    nancolumns = []
    for j in range(X_20.shape[1]):
        if np.isnan(X_20[i, j]):
            nancolumns.append(j)

    X_train = np.delete(X_25_norm, nancolumns, 1)  # Define the data train
    X_test = np.delete(X_20_norm, nancolumns, 1)  # Define the data test
    lamb = 10  # The value of the Lagrangian Multiplier
    for k in nancolumns:
        y_train = X_25_norm[:, k]  # The column for training the model
        y_train.shape=(X_25_norm.shape[0], 1)
        I = np.eye(X_train.shape[1])  # Identity matrix
        w = np.dot(np.dot(np.linalg.inv(np.dot(X_train.T, X_train) + lamb * I), X_train.T), y_train)  # Evaluation of the weight vector
        y_hat_test = np.dot(X_test, w)  # Regression of column k
        y_hat_test_unnorm = y_hat_test.T*std_X[k] + mean_X[k]  # Denormalization of the column k
        missing_value = y_hat_test_unnorm[0, i]  # The missing value is evaluated as the i entry of the regressed column
        # The missing value has to be in the same range of the other values for a specific feature
        if k in [3, 4, 5, 6, 7, 8, 18, 19, 20, 21, 22, 23, 24]:
            missing_value = int(round(missing_value))
        elif k == 2:
            missing_value = round(missing_value, 3)
        else:
            missing_value = round(missing_value, 1)
        X_20_test[i, k] = missing_value

#%% SVM PART
n, f = np.shape(X_20_test)
X_svm_train = X_20_test[0:int(n*0.8), 0:f-1]
X_svm_test = X_20_test[int(n-n*0.8):n, 0:f-1]
y_svm_train = X_20_test[0:int(n*0.8), f-1]
for i in range(y_svm_train.shape[0]):
    if y_svm_train[i] == 0:
        y_svm_train[i] = -1
y_svm_test = X_20_test[int(n-n*0.8):n, f-1]
for i in range(y_svm_test.shape[0]):
    if y_svm_test[i] == 0:
        y_svm_test[i] = -1
clf = svm.SVC(gamma='auto')
clf.fit(X_svm_train, y_svm_train)
y_svm_predict = clf.predict(X_svm_test)
accuracy = 0
for i in range(y_svm_test.shape[0]):
    if y_svm_test[i] == y_svm_predict[i]:
        accuracy += 1
print("The accuracy of the SVM is: " + str(accuracy/y_svm_test.shape[0]*100) + "%")

