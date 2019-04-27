# Simple Linear Regression
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 09:52:38 2017

@author: bshekhawat
"""

# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import data
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values

#splitting dataset into training and test
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.33, random_state = 0)

# Scaling the data
#from sklearn.preprocessing import StandardScaler
#sc_X = StandardScaler()
#X_train = sc_X.fit_transform(X_train)
#X_test = sc_X.fit_transform(X_test)

# Fitting simple LR to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predict the test results
y_pred = regressor.predict(X_test)

# Visualizing
plt.scatter(X_train, Y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title("Salary vs Exp")
plt.xlabel("Year of Exp")
plt.ylabel("Salary")
plt.show()

