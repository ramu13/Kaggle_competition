import numpy as np
import pandas as pd
TRAIN_DATA='./train.csv'
TEST_DATA='./test.csv'
def load_train_data():
    df = pd.read_csv(TRAIN_DATA)
    return df
def load_test_data():
    df = pd.read_csc(TRAIN_DATA)
    return df


from sklearn.linear_model import LogisticRegression
logistic = LogisticRegression()
df = load_train_data()
X_train = df.drop(['Ticket','Survived','Name','Fare','Embarked','Cabin'], axis=1)
#なんだこれ（）いらないの
#ああ、コラム(項目であるSurvivedも入れているのか)
y_train = df['Survived'].values

#前処理のライブラリー
#LabelEncoderはラベルつまり項目を互いに異なる整数値で表現する
#そんなオブジェクトを返す
from sklearn import preprocessing as sp
#pandasが用意する1次元データを返す函数
#load_train_date()はpandasの関数を使ってpandasのデータ型を返す
#つまりX_trainはpandasのデータ型でありリストなどの汎用型では無い
#性別の値、ここではStringである'男'と'女'を数値０と１として変換する
#その情報をle内部で保持しており
#fit_transformにより渡した列('男'と'女'しか含まない)をarray([1,0,0,1.....])のような数列に変換する
le = sp.LabelEncoder()
le.fit(X_train.Sex.unique())
print(X_train.Sex)
X_train.Sex = le.fit_transform(X_train.Sex)
print(X_train.Sex)

#OneHotEncoder()を例で理解する
#多次元配列[0,2,1,1]を列ベクトルにしてOneHotEncoderに入れると
one = sp.OneHotEncoder()
print('ここがonehotencoder')
print(X_train.Sex.values)
print(X_train.Sex.values.reshape(-1,1))
print(X_train.Sex.values.reshape(-1,1).transpose())
enced = one.fit_transform(X_train.Sex.values.reshape(1,-1).transpose())
print('ここからがenced')
print(enced)
print(enced.toarray())
#index=df.Sex.indexがわからん
temp = pd.DataFrame(index=df.Sex.index, columns='Sex-' + le.classes_, data=enced.toarray())
print('ここからがtemp')
print(temp)
enced_data = pd.concat([X_train, temp], axis=1)
del enced_data['Sex']

from sklearn.impute import SimpleImputer
im = SimpleImputer(missing_values=np.nan, strategy='mean')
im.fit(enced_data)
im.transform(enced_data)
print('ここがenced_data')
print(enced_data)
enced_data = im.fit_transform(enced_data)
print(enced_data)
