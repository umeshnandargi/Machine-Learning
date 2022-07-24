import numpy as np
from sklearn import model_selection
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras.datasets import mnist
# import os
import pandas as pd
from sklearn.metrics import accuracy_score
import pickle
import matplotlib.pyplot as plt


st = pd.read_csv('student-mat.csv', sep=';')
st = st[['G1', 'G2', 'G3', 'studytime', 'failures']]

X = np.array(st.drop(['G3'], axis=1))
y = np.array(st[['G3']])
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1)

# best = 0
# for i in range(50):
#     X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1)
#
#     model = linear_model.LinearRegression()
#     model.fit(X_train, y_train)
#
#     acc = model.score(X_test, y_test)
#     print(f'{acc * 100} %')
#
#     if acc>best:
#         best = acc
#         with open('Studentmodel.pickle', 'wb') as f:
#             pickle.dump(model, f)
#
#     # Test with following code if the models is saved correctly:
#     # Input : [13.69723528] [13 14  3  0] [14]
#     # Output : print(model.predict([[13, 14, 3, 0]]))


pickle_in = open('Studentmodel.pickle', 'rb')
model = pickle.load(pickle_in)

print('slopes:\n', model.coef_)
print('intercept:\n', model.intercept_)

prediction = model.predict(X_test)

for x in range(len(prediction)):
    print(prediction[x], X_test[x], y_test[x])

plt.scatter(st.G1, y)
plt.scatter(st.G2, y)
plt.scatter(st.failures, y)
plt.scatter(st.studytime, y)
plt.legend(['G1', 'G2', 'failures', 'studytime'])

plt.show()
