import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
# Importing the dataset
dataset = pd.read_csv('train.csv')
X = dataset.iloc[:, 4:6].values
y = dataset.iloc[:, 6].values

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y = sc_y.fit_transform(y)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

#RMSE 
mse = sklearn.metrics.mean_squared_error(y_test, y_pred)

#TEST set
dataset_test = pd.read_csv('test.csv')
X_hacker = dataset_test.iloc[:, 6:].values

Hacker_pred = regressor.predict(X_hacker)

#to CSV
dfNew.to_csv (r'C:\Users\nldos\Desktop\UStesting\submission.csv', index = False, header=True)

dfNew = pd.DataFrame(Hacker_pred)

##################################################

# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor.fit(X, y)

Hacker_pred = regressor.predict(X_hacker)
Hacker_pred_new = regressor.predict(X_hacker)
dfNew = pd.DataFrame(Hacker_pred)
dfNew.to_csv (r'C:\Users\nldos\Desktop\UStesting\RFsubmission.csv', index = False, header=True)








