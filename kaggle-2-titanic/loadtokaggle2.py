import pandas as pd

df = pd.read_csv('train.csv')
df['Age'].fillna(df['Age'].median(),inplace=True)
del df['Cabin']
import matplotlib.pyplot as plt
import seaborn as sns
sns.countplot(x = df['Pclass'],hue=df["Survived"])

import numpy as np
edge = np.arange(0,100,10)
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
pd.crosstab(df['FamilySize'],df['Survived'],normalize='index').plot(kind='bar',stacked=True)

sex_dum = pd.get_dummies(df['Sex'])
df = pd.concat((df,sex_dum),axis=1)
df = df.drop(['Sex'],axis=1)
df = df.drop('female',axis=1)

emb_dum = pd.get_dummies(df['Embarked'])
df = pd.concat((df,emb_dum),axis=1)
df = df.drop(['Embarked','S'],axis=1)
df = df.drop(['Name','Ticket','PassengerId','Parch','SibSp'],axis=1)


x = df.iloc[:,1:]
y = df['Survived']
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit(x,y)

test_df = pd.read_csv("test.csv")
sex_dum = pd.get_dummies(test_df['Sex'])
test_df = pd.concat((test_df,sex_dum),axis=1)
test_df = test_df.drop('Sex',axis=1)
test_df = test_df.drop('female',axis=1)

emb_dum = pd.get_dummies(test_df['Embarked'])
test_df = pd.concat((test_df,emb_dum),axis=1)
test_df = test_df.drop(['Embarked','S'],axis=1)

test_df['FamilySize'] = test_df['SibSp'] + test_df['Parch'] + 1
test_df = test_df.drop(['Name','Ticket','Cabin','PassengerId','Parch','SibSp'],axis=1)

test_df['Age'].fillna(test_df.Age.median(),inplace=True)
test_df['Fare'].fillna(test_df.Fare.median(),inplace=True)
output = forest.predict(test_df)

format_df = pd.read_csv('test.csv')
sub = pd.concat([format_df['PassengerId'],pd.Series(output)],axis=1)
sub = sub.rename(columns = {0:'Survived'})
sub.to_csv("submission.csv",index = False)
