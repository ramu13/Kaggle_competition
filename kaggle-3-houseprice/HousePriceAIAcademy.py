#0.18789
import numpy as np
import pandas as pd
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
#print(train.head())

#print(train.info())
#print(train.shape, test.shape)

train=train.drop('Alley',axis=1).drop('FireplaceQu',axis=1).drop('PoolQC',axis=1).drop('Fence',axis=1).drop('MiscFeature',axis=1)
test=test.drop('Alley',axis=1).drop('FireplaceQu',axis=1).drop('PoolQC',axis=1).drop('Fence',axis=1).drop('MiscFeature',axis=1)
#print(train)
train_id = train['Id']
test_id = test['Id']#テストセットのIDは提出用のファイルを作る際に必要

y_train = train['SalePrice']
x_train = train.drop(['Id','SalePrice'], axis=1)
x_test = test.drop('Id', axis=1)

x_train = x_train.fillna(x_train.median())#float型だけやってくれるのか
x_test = x_test.fillna(x_test.median())
print(x_train.info())

#print(x_train.loc[:,"MSSubClass"])
#print(x_train.iloc[:,0])
#print(x_train.shape[0])
#print(x_train.shape[1])
#print(x_train.shape[2])エラー
mode = x_train.mode()
print(mode)
#x_train.mode()が
#   MSSubClass MSZoning  LotFrontage  ...  YrSold SaleType SaleCondition
#          20       RL         69.0  ...    2009       WD        Normal
#x_train.columnsが
#None
#Index(['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
#       'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',
#       'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
#       'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle',
#       'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea',
#       'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond',
#       'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2',
#       'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC',
#       'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
#       'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
#       'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd',
#       'Functional', 'Fireplaces', 'GarageType', 'GarageYrBlt', 'GarageFinish',
#       'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive',
#       'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
#       'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',
#       'SaleCondition'],
#      dtype='object')
#mode = x_train.columns.valuesが上のIndexをListに変える
count = 0
for i in range(x_train.shape[1]):#iは何列目のデータ
	if x_train.iloc[:,i].dtypes == object:#[x_train.columns.values[i]]はカラム名
		mode = x_train.mode()[x_train.columns.values[i]].values#.valuesと言ってもリストではなく一つのシーケンスを返す
		print()
		print(mode)
		count=count+1
		for j in range(x_train.shape[0]):
			if x_train.isnull().iloc[j,i] == True:
				x_train.iloc[j,i] = mode
print(x_train.isnull().sum().sum())


#ラベル
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for i in range(x_train.shape[1]):
	if x_train.iloc[:,i].dtypes == object:
		le.fit(list(x_train[x_train.columns.values[i]].values))
		x_train[x_train.columns.values[i]] = le.transform(list(x_train[x_train.columns.values[i]].values))
print(x_train.info())
for i in range(x_test.shape[1]):
	if x_test.iloc[:,i].dtypes == object:
		le.fit(list(x_test[x_train.columns.values[i]].values))
		x_test[x_test.columns.values[i]] = le.transform(list(x_test[x_test.columns.values[i]].values))
print(x_train.info)


#データの前処理の最終段階です。
#現段階で74種類の特徴量がありますが、これを大幅に削減します。
#今回は単純にscikit-learnのfeature_selectionを用いて、
#目的変数に最も影響を与えていると考えられる5つの特徴量を抽出したいと思います。
from sklearn.feature_selection import SelectKBest, f_regression
selector = SelectKBest(score_func=f_regression, k=5)
selector.fit(x_train,y_train)
print(selector.get_support())

x_train_selected = pd.DataFrame({
	'OverallQual':x_train['OverallQual'],
	'ExterQual':x_train['ExterQual'],
	'GrLivArea':x_train['GrLivArea'],
	'GarageCars':x_train['GarageCars'],
	'GarageArea':x_train['GarageArea']
	})
x_test_selected=pd.DataFrame({
	'OverallQual':x_test['OverallQual'],
	'ExterQual':x_test['ExterQual'],
	'GrLivArea':x_test['GrLivArea'],
	'GarageCars':x_test['GarageCars'],
	'GarageArea':x_test['GarageArea']
	})
print(x_train_selected.head())

#データの分割
from sklearn.model_selection import train_test_split
xp_train, xp_test,yp_train,yp_test=train_test_split(x_train_selected,y_train,test_size=0.3,random_state=1)

#モデルの導入とグリッドサーチ
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor()
svr=SVR()
parameters_forest = {'n_estimators':[100,500,1000,3000],'max_depth':[3,6,12]}
parameters_svr = {'C':[0.1, 10, 1000], 'epsilon':[0.01, 0.1, 0.5]}
from sklearn.model_selection import GridSearchCV
grid_forest = GridSearchCV(forest, parameters_forest)
grid_forest.fit(xp_train, yp_train)
grid_svr = GridSearchCV(svr, parameters_svr)
grid_svr.fit(xp_train, yp_train)
#結果をMSEで見る
from sklearn.metrics import mean_squared_error
yp_pred_forest = grid_forest.predict(xp_test)
print(mean_squared_error(yp_test, yp_pred_forest))
yp_pred_svr=grid_svr.predict(xp_test)
print(mean_squared_error(yp_test,yp_pred_svr))

#どうやらランダムフォレストの方が誤差が小さいようなので今回二つのモデル比較ではランダムフォレストを使おうという結論に至る
print(grid_forest.best_params_)
#一番良かったときのハイパーパラメータを見て実装する
best_forest = RandomForestRegressor()
best_forest.fit(x_train_selected, y_train)
result = np.array(best_forest.predict(x_test_selected))
df_result = pd.DataFrame(result,columns=['SalePrice'])
df_result = pd.concat([test_id, df_result],axis=1)
df_result.to_csv('submission.csv',index=False)
print("完了しました")
