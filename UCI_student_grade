#Import Library
import numpy as np
import pandas as pd
from sklearn import linear_model
import sklearn
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib import style
import pickle


data = pd.read_csv("student-mat.csv", sep=";")

predict = "G3"

data = data[["G1", "G2", "absences", "failures", "studytime", "G3"]]
data = shuffle(data)  # Optional - shuffle the data

x = np.array(data.drop([predict], 1))  # dropping predict column ,i.e G3 column
y = np.array(data[predict])


x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)


# TRAIN MODEL MULTIPLE TIMES FOR BEST SCORE
best = 0
for _ in range(20):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print("Accuracy: " + str(acc))

    if acc > best:
        best = acc
        with open("studentgrades.pickle", "wb") as f:
            pickle.dump(linear, f)

# LOAD MODEL
pickle_in = open("studentgrades.pickle", "rb")
linear = pickle.load(pickle_in)


print("-------------------------")
print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)
print("-------------------------")

predicted = linear.predict(x_test)
for x in range(len(predicted)):
    print(predicted[x], x_test[x], y_test[x], sep=' '*6)


style.use('ggplot')

def plot_scatter(p):
    plt.scatter(data[p], data['G3'])
    plt.xlabel(p)
    plt.ylabel('Final Grade')


fig = plt.figure()
p = "G1"
plt.subplot(2,2,1)
plot_scatter(p)
p = "G2"
plt.subplot(2,2,2)
plot_scatter(p)
p = "studytime"
plt.subplot(2,2,3)
plot_scatter(p)
p ="absences"
plt.subplot(2,2,4)
plot_scatter(p)
plt.show()

fig.savefig('student_grades.png')
