import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# from sklearn.metrics import accuracy_score
import pickle

df = pd.read_csv('./data/advertising.csv')
x = np.array(df.drop(["Sales"], 1))
y = np.array(df["Sales"])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
lreg = LinearRegression()
lreg.fit(x_train, y_train)
filename = './saved_model/advertising.sav'
pickle.dump(lreg, open(filename, 'wb'))
# y_pred = lreg.predict(x_test)
# print(f"Accuracy : {lreg.score(x_test, y_test) * 100} %")
# print(accuracy_score(y_test, y_pred))

