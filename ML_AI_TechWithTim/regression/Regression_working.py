import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model,preprocessing
from sklearn.utils import shuffle
import pickle
import matplotlib.pyplot as plt
from matplotlib import style


data = pd.read_csv('student-mat.csv', sep=';')

le = preprocessing.LabelEncoder()
romantic = le.fit_transform(list(data['romantic']))
G1 = list(data['G1'])
G2 = list(data['G2'])
G3 = list(data['G3'])
studytime = list(data['studytime'])
failures = list(data['failures'])
absences = list(data['absences'])

data = data[["G1", "G2", "G3", "studytime", "failures", "absences", "romantic"]]        # now we need only these attributes


# print(data.head())

predict = "G3"      # we are going to predict G3
X = list(zip(G1, G2, studytime, failures, absences, romantic))
# X = np.array(data.drop([predict], 1))       # drops G3 from training
y = np.array(data[predict])                 # we have to predict G3, so output


best = 0
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)
for _ in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)        # we are training the data, i.e getting the equation of line

    acc = linear.score(x_test, y_test)          # checking the accuracy on test set

    print(acc)

    if acc > best:
        best = acc
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)


pickle_in = open('studentmodel.pickle', 'rb')
linear = pickle.load(pickle_in)

# printing the coefficient of the line we got after model training
print("co: \n", linear.coef_)
print("Intercept: \n", linear.intercept_)

predictions = linear.predict(x_test)

for i in range(len(predictions)):
    print(predictions[i], x_test[i], y_test[i])

# print(best)

p = "romantic"
style.use('ggplot')
plt.scatter(data[p], data["G3"])
plt.xlabel(p)
plt.ylabel("Final Grade")
plt.show()
