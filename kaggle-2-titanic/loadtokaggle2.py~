import pandas as pd

df = pd.read_csv('train.csv')
print(df.head())
#print(df.insull().sum())

df['Age'].fillna(df['Age'].median(),inplace=True)
#df.drop("Column",axis=1) df.drop(List) でそれぞれ列もしくは行の削除
del df['Cabin']
print(df)

import matplotlib.pyplot as plt
import seaborn as sns
sns.countplot(x = df['Pclass'],hue=df["Survived"])
plt.show()

import numpy as np
edge = np.arange(0,100,10)
print(edge)
plt.pause(1)
plt.hist(
    (df[df['Survived']==0]['Age'],df[df['Survived']==1]['Age']),
    histtype='barstacked',
    bins=edge,
    label=[0,1])
plt.legend(title='Survived')
plt.show()
plt.pause(1)
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
pd.crosstab(df['FamilySize'],df['Survived'],normalize='index').plot(kind='bar',stacked=True)
plt.show()

sex_dum = pd.get_dummies(df['Sex'])
df = pd.concat((df,sex_dum),axis=1)
df = df.drop(['Sex'],axis=1)
df = df.drop('female',axis=1)

emb_dum = pd.get_dummies(df['Embarked'])
df = pd.concat((df,emb_dum),axis=1)
df = df.drop(['Embarked','S'],axis=1)
df = df.drop(['Name','Ticket','PassengerId','Parch','SibSp'],axis=1)
print(df.head())



