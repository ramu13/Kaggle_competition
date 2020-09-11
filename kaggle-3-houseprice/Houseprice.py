import numpy as numpy
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

#ちょっと復習も兼ねて応用してみた。本来ならfillnaのより便利な使い方調べてやるだろうな
df = pd.read_csv('train.csv')
train_y = df["SalePrice"]
df = df.drop(["Street","Utilities","Alley","LandContour","Condition2","RoofMatl","SalePrice","Id"],axis=1)
empty_series = df.isnull().sum()
empty_columns = empty_series.index#SeriesのindexがとDataframeのcolumn
print(empty_columns)
print('\n\n')
#多分seriesをイテレーションとすると
#for v in empty_series:(values)
#for i, v in empty_series:(columnsとvaluesからなるtuple)
empty_columns = {key: empty_series[key]  for key in empty_columns if empty_series[key]!=0}
full_object_columns = {key: df[key] for key in df.columns if empty_series[key]==0}
print(empty_columns)
print('\n\n')
print(full_object_columns.keys())
print('\n\n')
empty_float_column = list()
empty_object_column = list()
for key in empty_columns:
	if df[key].dtype == float:
		empty_float_column.append(key)
	else:
		empty_object_column.append(key)

print(empty_float_column)
print('\n\n')
print(empty_object_column)
print('\n\n')

#float型のデータの補完する
for v in empty_float_column:
	df[v].fillna(df[v].median(),inplace=True)
	print(df[v])
	print(df[v].isnull().sum())
print('\n\n')
#文字列で表現されているデータを補完し、ラベル及びワンホット化する
for key in empty_object_column:
	df[key].fillna(df[key].mode())
	dum = pd.get_dummies(df[key])
	skip = dum.keys()[0]#あるカラムから生成したラベルn個のうち１つは不要になりこの列をskipとする
	print(dum)
	print("skip :", skip)
	df = pd.concat((df,dum),axis=1)
	df = df.drop(skip,axis=1)
	df = df.drop([key],axis=1)
	print('\n\n')
for key in full_object_columns.keys():
	dum = pd.get_dummies(df[key])
	skip = dum.keys()[0]#あるカラムから生成したラベルn個のうち１つは不要になりこの列をskipとする
	print(dum)
	print("skip :", skip)
	df = pd.concat((df,dum),axis=1)
	df = df.drop(skip,axis=1)
	df = df.drop([key],axis=1)
	print('\n\n')
#試しに次の行を実行するとリストが帰ってくると思っていたから驚いた。どうやら独自の型らしい。ただしオフセットによる抽出可能
#print(dum.keys())
#多重回帰の実装
print(df,'\n\n')
X_train = df
print(X_train,'\n\n')
_x = PolynomialFeatures(degree = 3)
model = LinearRegression()
X_train = _x.fit_transform(X_train)
model1.fit(X_train,y_train)#ここでMAC先輩がエンドレスエイトに陥った



#testデータも同じ形にする
test_df = pd.read_csv("test.csv")
test_y = test_df["SalePrice"]
test_df = test_df.drop(["Street","Utilities","Alley","LandContour","Condition2","RoofMatl","SalePrice","Id"],axis=1)
empty_series = test_df.isnull().sum()
empty_columns = empty_series.index
empty_columns = {key: empty_series[key]  for key in empty_columns if empty_series[key]!=0}
full_object_columns = {key: test_df[key] for key in test_df.columns if empty_series[key]==0}
empty_float_column = list()
empty_object_column = list()
for key in empty_columns:
	if test_df[key].dtype == float:
		empty_float_column.append(key)
	else:
		empty_object_column.append(key)
for v in empty_float_column:
	test_df[v].fillna(test_df[v].median(),inplace=True)
	print(test_df[v])
	print(test_df[v].isnull().sum())
#文字列で表現されているデータを補完する
for key in empty_object_column:
	test_df[key].fillna(df[key].mode())
	dum = pd.get_dummies(test_df[key])
	skip = dum.keys()[0]
	print(dum)
	print("skip :", skip)
	test_df = pd.concat((test_df,dum),axis=1)
	test_df = test_df.drop(skip,axis=1)
	test_df = test_df.drop([key],axis=1)
	print('\n\n')
for key in full_object_columns.keys():
	dum = pd.get_dummies(test_df[key])
	skip = dum.keys()[0]
	print(dum)
	print("skip :", skip)
	test_df = pd.concat((test_df,dum),axis=1)
	test_df = test_df.drop(skip,axis=1)
	test_df = test_df.drop([key],axis=1)
	print('\n\n')

test__X = test_df
print(test_X,'ここまで')
output = model1.predict(test__X)
format_df = pd.read_csv('test.csv')
sub = pd.concat([format_df['PassengerId'],pd.Series(output)],axis=1)
sub = sub.rename(columns = {0: "SalePrice"})
sub.to_csv("submission.csv",index = False)
