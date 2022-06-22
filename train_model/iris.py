import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score
import pickle

df = pd.read_csv('./data/iris.csv')
x = df.drop("species", axis=1)
y = df["species"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train, y_train)
filename = './saved_model/iris.sav'
pickle.dump(knn, open(filename, 'wb'))
# y_pred = knn.predict(x_test)
# print(f"Accuracy : {knn.score(x_test, y_test) * 100} %")
# print(accuracy_score(y_test, y_pred))

