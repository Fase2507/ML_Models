import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import  LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.ensemble import RandomForestRegressor

#LOAD DATA
df=pd.read_csv('delaney.csv')
# print(df.head(10)) #checking data set

#CHOOSE X AND y VALUES
y=df['logS']
X=df.drop('logS',axis=1)

#SEPERATE TEST & TRAIN DATAS
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=100)

#CHOOSE YOUR MODULE
lr=LinearRegression()
lr.fit(X_train,y_train)

#PREDICTION
y_lr_train_pred=lr.predict(X_train)
y_lr_test_pred=lr.predict(X_test)

#EVALUATE
lr_train_mse=mean_squared_error(y_train,y_lr_train_pred)*100
lr_train_r2=r2_score(y_train,y_lr_train_pred)*100

lr_test_mse=mean_squared_error(y_test,y_lr_test_pred)*100
lr_test_r2=r2_score(y_test,y_lr_test_pred)*100
#RESULTS

lr_result=pd.DataFrame(["Linear Regression",lr_train_mse,lr_train_r2,lr_test_mse,lr_test_r2]).transpose()
lr_result.columns=['Method','Training MSE','Training R2','Test MSE','Test R2']


#RANDOM FOREST
rf=RandomForestRegressor()
rf.fit(X_train,y_train)
#RF PREDICTION
y_rf_train_pred=rf.predict(X_train)
y_rf_test_pred=rf.predict(X_test)
#TRAIN
rf_train_mse=mean_squared_error(y_train,y_rf_train_pred)*100
rf_train_r2=r2_score(y_train,y_rf_train_pred)*100
#TEST
rf_test_mse=mean_squared_error(y_test,y_rf_test_pred)*100
rf_test_r2=r2_score(y_test,y_rf_test_pred)*100
rf_result=pd.DataFrame(["Random Forest",rf_train_mse,rf_train_r2,rf_test_mse,rf_test_r2]).transpose()
rf_result.columns=['Method','Training MSE','Training R2','Test MSE','Test R2']
print('Random forest: \n',rf_result)
print('Linear regression \n',lr_result)
