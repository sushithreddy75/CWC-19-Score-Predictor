import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
kiwi=pd.read_csv('nz.csv')
ko=kiwi.iloc[:,:-1].values
kr=kiwi.iloc[:,-1].values
from sklearn.model_selection import train_test_split
#ko_train,ko_test,kr_train,kr_test=train_test_split(ko,kr,test_size=0.4,random_state=0)
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=6)
ko_2=poly_reg.fit_transform(ko)
ko_train,ko_test,kr_train,kr_test=train_test_split(ko_2,kr,test_size=0.4,random_state=0)
from sklearn.linear_model import LinearRegression
lin_reg1=LinearRegression()
lin_reg1.fit(ko_train,kr_train)
plt.scatter(ko,kr,color='red')
plt.plot(ko,lin_reg1.predict(ko_2),color='black')
plt.title('NEW ZEALAND INNINGS')
plt.xlabel('OVERS')
plt.ylabel('RUNS')
plt.show()
ind=pd.read_csv('IND.csv')
io=ind.iloc[:,:-1].values
ir=ind.iloc[:,-1].values
io_2=poly_reg.fit_transform(io)
io_train,io_test,ir_train,ir_test=train_test_split(io_2,ir,test_size=0.4,random_state=0)
lin_reg2=LinearRegression()
lin_reg2.fit(io_train,ir_train)
plt.scatter(io,ir,color='orange')
plt.plot(io,lin_reg2.predict(io_2),color='blue')
plt.title('INDIAN INNINGS')
plt.xlabel('OVERS')
plt.ylabel('RUNS')
plt.show()
plt.plot(ko,lin_reg1.predict(ko_2),color='black')
plt.plot(io,lin_reg2.predict(io_2),color='blue')
plt.title('INDIAN VS NEW ZEALAND INNINGS(predicted)')
plt.xlabel('OVERS')
plt.ylabel('RUNS')
plt.show()
plt.plot(ko,kr,color='black')
plt.plot(io,ir,color='blue')
plt.title('INDIAN VS NEW ZEALAND INNINGS(actual)')
plt.xlabel('OVERS')
plt.ylabel('RUNS')
plt.show()
plt.plot(ko,kr,color='black')
plt.plot(io,ir,color='blue')
plt.title('INDIAN VS NEW ZEALAND INNINGS(actual vs predicted)')
plt.xlabel('OVERS')
plt.ylabel('RUNS')
plt.plot(ko,lin_reg1.predict(ko_2),ls='--',color='black')
plt.plot(io,lin_reg2.predict(io_2),ls='--',color='blue')
plt.show()