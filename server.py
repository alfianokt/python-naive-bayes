import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from bottle import request, route, run

data_latih = pd.read_csv("data/iris-latih.csv")

# independent
x = data_latih.drop(["variety"], axis=1)
# dependent
y = data_latih["variety"]

x_train, x_test, y_train, y_test = train_test_split(x, y)

modelnb = GaussianNB()

nbtrain = modelnb.fit(x_train, y_train)

@route('/')
def index():
    return "Hello world"

@route('/check', method="POST")
def check():
    d = {
      "sepal.length": [request.json['sepal.length']],
      "sepal.width": [request.json['sepal.width']],
      "petal.length": [request.json['petal.length']],
      "petal.width": [request.json['petal.width']]
    }

    p = pd.DataFrame(data=d)
    y_pred = nbtrain.predict(p)

    return y_pred


run(host='localhost', port=8080, debug=True)