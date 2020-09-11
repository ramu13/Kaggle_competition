import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

#pandasが省略せずに表示してくれる
pd.set_option("display.max_columns",100)
pd.set_option("display.max_rows",100)
#print(df_train)

y_train = df_train["SalePrice"]
test_id = df_test["Id"]
x_train = df_train.drop(["Id","SalePrice"],axis=1)
x_test = df_test.drop(["Id"],axis=1)

x_train = x_train.fillna(x_train.median())
x_test = x_test.fillna(x_test.median())

for i in range(x_train.shape[1]):
	if x_train.iloc[:,i].dtypes == object:
		mode = x_train.iloc[:,i].mode().values
		for j in range(x_train.shape[0]):
			if x_train.isnull().iloc[j,i] ==True:
				x_train.iloc[j,i] = mode
print(x_train.isnull().sum().sum())
for i in range(x_test.shape[1]):
	if x_test.iloc[:,i].dtypes == object:
		mode = x_test.iloc[:,i].mode().values
		for j in range(x_test.shape[0]):
			if x_test.isnull().iloc[j,i] ==True:
				x_test.iloc[j,i] = mode
print(x_test.isnull().sum().sum())