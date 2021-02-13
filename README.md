# Machine-Learning
starting to explore ML and AI
import pandas as pa
import numpy as np

from sklearn.datasets import load_iris
iris_dataset = load_iris()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'],iris_dataset['target'], random_state=0)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train, y_train)

print("provide the necessary information regarding the flower "
"in CM \n  and i will tell you to which family they belong to \n")


X_new = np.array([[0.0,0.0,0.0,0.0]])
for i in range(0,4):
  if i == 0:
    X_new[0,i] = input("sepal length (cm) : ")
  if i == 1:
    X_new[0,i]  = input('sepal width (cm)  : ')
  if i == 2:
    X_new[0,i]  = input('petal length (cm) : ')
  if i == 3:
    X_new[0,i]  = input('petal width (cm)  : ')

prediction = knn.predict(X_new)

print(" \n From the given data it's a : {}".format(iris_dataset[
                                                        'target_names']
                                         [prediction]))
