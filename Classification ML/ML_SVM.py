from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn import model_selection

cancer_df = datasets.load_breast_cancer()
# print(cancer_df.feature_names)
# print(cancer_df.target_names)


# # View Merged Data
# dt= np.array(cancer_df.target)
# dt2= np.array(cancer_df.data)
# cols=np.array(list(cancer_df.feature_names)+ ['m/b'])
# data1 = pd.DataFrame(data= np.c_[dt2, dt] , columns= cols)
# print(data1)


X = cancer_df.data
y = cancer_df.target

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
classes = ['malignant', 'benign']

model = svm.SVC(kernel='linear', C=3)
model.fit(X_train,y_train)
prediction = model.predict(X_test)
# acc = model.score(X_test, y_test)
acc = metrics.accuracy_score(y_test, prediction)


for i in range(len(prediction)):
    print('Prediction: ', classes[prediction[i]], '; Data: ', X_test[i], '; Actual:', classes[y_test[i]])

print(f'Accuracy: {acc*100} %')