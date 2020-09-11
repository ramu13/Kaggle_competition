import numpy as np
import pandas as pd

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

drop_list = ['Alley','FireplaceQu','PoolQC','Fence','MiscFeature']
train = train.drop(drop_list,axis=1)
test = test.drop(drop_list,axis=1)

train_Id = train['Id']
test_Id = test['Id']

y_train = train['SalePrice']
x_train = train.drop(['Id','SalePrice'],axis=1)
x_test = test.drop('Id',axis=1)

x_train = x_train.fillna(x_train.median())
x_test = x_test.fillna(x_test.median())
#x_train.median(),x_test.median()は共にprintするだけで何かのオブジェクトを返しているわけではない

for i in range(x_train.shape[1]):
	if x_train.iloc[:,i].dtypes == object:
		mode = x_train.iloc[:,i].mode().values
		#x_train.iloc[:,i].mode()まででシリーズを返す(長さ1しかない)
		#x_train.ilo[:,i].mode().valuesでリストを返す
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


