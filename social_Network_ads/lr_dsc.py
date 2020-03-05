import numpy as np
import pandas as pd

dataset =pd.read_csv('/home/moritz/Downloads/Social_Network_Ads.csv')
X = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:,4].values


from sklearn.model_selection import train_test_split

X_train, X_test, y_train ,y_test = train_test_split(X, y,test_size=0.2)

# normalization is being done, i.e feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)

predicted = model.predict(X_test)
acc = model.score(X_test, y_test)

print(acc)


print(predicted)
