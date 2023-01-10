import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

data_latih = pd.read_csv("data/iris-latih.csv")
data_uji = pd.read_csv("data/iris-uji.csv")

# independent
x = data_latih.drop(["variety"], axis=1)
# dependent
y = data_latih["variety"]

x_train, x_test, y_train, y_test = train_test_split(x, y)

modelnb = GaussianNB()

nbtrain = modelnb.fit(x_train, y_train)

# print(data_uji)

y_pred = nbtrain.predict(data_uji)
print(y_pred)