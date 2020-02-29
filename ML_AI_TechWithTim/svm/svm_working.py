import sklearn.model_selection
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier


cancer = datasets.load_breast_cancer()

# print(cancer.feature_names)
# print(cancer.target_names)

x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)
# print(x_train, y_train)

classes = ['malignant', 'benign']

clf = svm.SVC()     # try different parameters
model = KNeighborsClassifier()


clf.fit(x_train, y_train)
model.fit(x_train, y_train)

y_pred = clf.predict(x_test)
model_pred = model.predict(x_test)

acc = clf.score(x_test, y_test)     # acc = metrics.accuracy_score(y_test, y_pred), both same
acc1 = model.score(x_test, y_test)

print('Accuracy for SVM: \n', acc)
print('Accuracy for KNN: \n', acc1)
