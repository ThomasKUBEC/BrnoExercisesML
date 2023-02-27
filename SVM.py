
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np




iris=load_iris()
iris.feature_names
print(iris.feature_names)
print(iris.data[0:5,:])
print('Target:')
print(iris.target[:])
#print(iris.data)

print('\n')




print('\nThe first two colomuns are : ', iris.data[:, :2])
X2D = iris.data[:, :2]
Y2T = iris.target
print('TARGET', Y2T)
X_train, X_test, y_train, y_test = train_test_split(X2D,Y2T,test_size=0.2)
print(X_train.shape)
print(X_test.shape)
plt.scatter(X_train[y_train==0,0],X_train[y_train==0,1],color='green')
plt.scatter(X_train[y_train==1,0],X_train[y_train==1,1],color='red')


SVMmodel=SVC(kernel='linear', C=200)
SVMmodel.fit(X_train,y_train)
SVMmodel.get_params()
SVMmodel.score(X_test,y_test)
print('Score : ', SVMmodel.score(X_test,y_test))


supvectors=SVMmodel.support_vectors_
W=SVMmodel.coef_
b=SVMmodel.intercept_
X0 = np.linspace(min(X_train[:,0]),max(X_train[:,0]),10)
print('W =', W[0][0])
print('b =', b[0])
X1 = -(W[0][0]/W[0][1])*X0-(b[0]/W[0][1])
#X1 = -(W[:,0]/W[:,1])*X0-(b/W[:,1])
plt.scatter(X1,X0)
plt.show()




from sklearn.svm import OneClassSVM
from sklearn.datasets import make_blobs
from numpy import quantile, where, random

random.seed(11)
x, _ = make_blobs(n_samples=300, centers=1, cluster_std=.3, center_box=(4, 4))

plt.scatter(x[:,0], x[:,1])
plt.show()



SVMmodelOne = OneClassSVM(kernel='rbf', gamma=0.001, nu=0.03)


SVMmodelOne.fit(x)
pred = SVMmodelOne.predict(x)
anom_index = where(pred==-1)
values = x[anom_index]

plt.scatter(x[:,0], x[:,1])
plt.scatter(values[:,0], values[:,1], color='red')
plt.axis('equal')
plt.show()

supvectorsOne = SVMmodelOne.support_vectors_
plt.scatter()