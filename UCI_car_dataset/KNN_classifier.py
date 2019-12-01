import numpy as np
import pandas as pd
from sklearn import linear_model,preprocessing
import sklearn
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
import csv
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv('car.txt', sep=',')
print(data.head())

# print(type(data))  # Dataframe


# transforming non-integer data to integer data
le = preprocessing.LabelEncoder()

buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
doors = le.fit_transform(list(data["doors"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class_values"]))

predict = "cls"


x = list(zip(buying, maint, doors, persons, lug_boot, safety))
y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

print(x_train)
print(y_train)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(x_train, y_train)

acc = model.score(x_test, y_test)

print(acc)


predicted = model.predict(x_test)

names = ['unacc', 'acc', 'good', 'very good']



for x in range(len(predicted)):
    print("Predicted: ", names[predicted[x]], "Data: ", x_test[x], "Actual: ", names[y_test[x]])
    n = model.kneighbors([x_test[x]], 9, True)
    # print("N: ",n)
