from sklearn.datasets import load_iris
import numpy as np
from src.functions import generate_randomly_missing
from sklearn.model_selection import train_test_split
from src.WLDA import WLDA

iris = load_iris()
X, y = iris.data, iris.target

print("Bộ dữ liệu Iris: Shape X:", X.shape, " Shape y:", y.shape)
for i in np.unique(y):
  print(sum(y==i))

def experiment(X,y,missing_rate,run_time):
    #G = len(np.unique(y)) #number of labels
    for t in range(run_time):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
        X_train = generate_randomly_missing(X_train,missing_rate)
        model = WLDA()
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        print(f'{t+1:2d}-th time with accuracy = {accuracy:.4f}')
experiment(X,y,.4,10)
