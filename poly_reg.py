import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('Position_Salaries.csv')
x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2:3].values

#Linear Model processing
from sklearn.linear_model import LinearRegression
linear_reg=LinearRegression()
linear_reg.fit(x,y)

#Polynomial Model processing
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
x_poly=poly_reg.fit_transform(x)

#fitting polynomial model
linear_reg_2=LinearRegression()
linear_reg_2.fit(x_poly,y)

#Visualising the polynomial model
x_grid=np.arange(min(x),max(x),0.1)
x_grid=x_grid.reshape((len(x_grid),1))
plt.scatter(x,y,color='red')
plt.plot(x_grid,linear_reg_2.predict(poly_reg.fit_transform(x_grid)),color='blue')
plt.show()

#Predicting the value
linear_reg_2.predict(poly_reg.fit_transform(6.5))