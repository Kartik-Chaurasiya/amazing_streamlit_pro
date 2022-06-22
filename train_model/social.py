import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score
import pickle

df = pd.read_csv('./data/social.csv')
x = np.array(df[["Age", "EstimatedSalary"]])
y = np.array(df[["Purchased"]])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
dTree = DecisionTreeClassifier()
dTree.fit(x_train, y_train)
filename = './saved_model/social.sav'
pickle.dump(dTree, open(filename, 'wb'))
# y_pred = dTree.predict(x_test)
# print(f"Accuracy : {dTree.score(x_test, y_test) * 100} %")
# print(accuracy_score(y_test, y_pred))

