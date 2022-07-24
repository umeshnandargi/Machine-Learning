from sklearn.utils import shuffle
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing, linear_model
import pandas as pd
import numpy as np

df = pd.read_csv('car.data')

# def f(df):
#     if df.safety =='low':
#         return 0
#     elif df.safety == 'med':
#         return 1
#     else:
#         return 2
#
# df['safety']= df.apply(f, axis =1)
# print(df)

print(df)

le = preprocessing.LabelEncoder()
df['buying'] = le.fit_transform(list(df.buying))
df['maint'] = le.fit_transform(list(df.maint))
df['door'] = le.fit_transform(list(df.door))
df['persons'] = le.fit_transform(list(df.persons))
df['lug_boot'] = le.fit_transform(list(df.lug_boot))
df['safety'] = le.fit_transform(list(df.safety))
# cls = le.fit_transform(list(df['class']))
df['class'] = le.fit_transform(list(df['class']))

# cls2 = le.inverse_transform(cls)
# print(cls)
# print(cls2)

print(df)

X = np.array(df.drop('class', axis=1))
y = np.array(df['class'])
# print(X,y)


X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1)

model = KNeighborsClassifier(n_neighbors=9)
model.fit(X_train, y_train)

prediction = model.predict(X_test)
acc = model.score(X_test, y_test)
names = ['acc', 'good', 'unacc', 'vgood']
# [0 1 2 3]
# ['acc' 'good' 'unacc' 'vgood']


for i in range(len(prediction)):
    print('Prediction: ', names[prediction[i]], 'Data: ', X_test[i], 'Actual:', names[y_test[i]])
    # print(model.kneighbors([X_test[i]], 9, True))

print(f'accuracy = {acc*100} %')
