import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
import pickle

df = pd.read_csv('./data/waiter_tip.csv')
df["sex"] = df["sex"].map({"Female": 0, "Male": 1})
df["smoker"] = df["smoker"].map({"No": 0, "Yes": 1})
df["day"] = df["day"].map({"Thur": 0, "Fri": 1, "Sat": 2, "Sun": 3})
df["time"] = df["time"].map({"Lunch": 0, "Dinner": 1})
x = np.array(df[["total_bill", "sex", "smoker", "day", "time", "size"]])
y = np.array(df["tip"])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
reg = RandomForestRegressor(n_estimators=10)
reg.fit(x_train, y_train)
filename = './saved_model/waiter_tip.sav'
pickle.dump(reg, open(filename, 'wb'))
# y_pred = reg.predict(x_test)
# print(f"Accuracy : {reg.score(x_test, y_test) * 100} %")
# print(accuracy_score(y_test, y_pred))

