import pandas as pd
from sklearn.naive_bayes import GaussianNB

df_1 = pd.read_csv('./train.csv')
df_2 = pd.read_csv('./test.csv')
df_train = df_1.copy()
df_test = df_2.copy()

features = ['Age', 'Sex', 'Pclass']

def clean_sex(sex):
    if sex == 'male' or sex == 1:
        return 1
    elif sex =='female' or sex == 0:
        return 0

df_train['Sex'] = df_train['Sex'].apply(lambda x: clean_sex(x))
df_train['Age'].fillna(df_train['Age'].mean(), inplace=True)

df_test['Sex'] = df_test['Sex'].apply(lambda x: clean_sex(x))
df_test['Age'].fillna(df_test['Age'].mean(), inplace=True)

X_train = df_train[['Sex', 'Age']].to_numpy()
y_train = df_train['Survived'].to_numpy()
X_test = df_test[['Sex', 'Age']].to_numpy()

model = GaussianNB()
y_pred = model.fit(X_train, y_train).predict(X_test)

output = pd.DataFrame({'PassengerId': df_test['PassengerId'], 'Survived':y_pred})
output.to_csv('my_first_submission.csv',index = False)
