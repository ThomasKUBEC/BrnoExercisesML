import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn import preprocessing,decomposition
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


X=np.array([[2, 1, 0],[4, 3, 0]])

#We calculate the covariance
R=np.matmul(X,X.T)/3

#Calculate the SVD decomposition and new basis vectors:
[U,D,V]=np.linalg.svd(R)
u1=U[:,0] # new basis vectors
u2=U[:,1]

print('\n',U)

# Calculate the coordinates in new orthonormal basis:
Xi1=np.matmul(np.transpose(X),u1)
Xi2=np.matmul(np.transpose(X),u2)

Xaprox = np.matmul(u1[:,None],Xi1[None,:])+np.matmul(u2[:,None],Xi2[None,:])
print('Xaprox',Xaprox)


print('\n\n\nPCA on iris data')
iris=load_iris()
iris.feature_names
print(iris.feature_names)
print(iris.data[0:5,:])
print(iris.target[:])
X=iris.data
y=iris.target
axes1=plt.axes(projection='3d')
axes1.scatter3D(X[y==0,1],X[y==0,1],X[y==0,2],color='green')
axes1.scatter3D(X[y==1,1],X[y==1,1],X[y==1,2],color='blue')
axes1.scatter3D(X[y==2,1],X[y==2,1],X[y==2,2],color='magenta')
#plt.show()


Xscaler = StandardScaler()
Xpp=Xscaler.fit_transform(X)

# define PCA object (three components), fit and transform the data
pca = decomposition.PCA(n_components=3)
pca.fit(Xpp)
Xpca = pca.transform(Xpp)
print(pca.get_covariance())
# you can plot the transformed feature space in 3D:
axes2=plt.axes(projection='3d')
axes2.scatter3D(Xpca[y==0,0],Xpca[y==0,1],Xpca[y==0,2],color='green')
axes2.scatter3D(Xpca[y==1,0],Xpca[y==1,1],Xpca[y==1,2],color='blue')
axes2.scatter3D(Xpca[y==2,0],Xpca[y==2,1],Xpca[y==2,2],color='magenta')
plt.show()

exp_var_pca = pca.explained_variance_
print('Explained Variance ',exp_var_pca)
    
exp_var_ratio_pca = pca.explained_variance_ratio_
print('Explained Variance ratio',exp_var_ratio_pca)

plt.scatter(Xpca[y==0,0],Xpca[y==0,1],color='blue')
plt.scatter(Xpca[y==1,0],Xpca[y==1,1],color='green')
plt.scatter(Xpca[y==2,0],Xpca[y==2,1],color='red')
plt.show()

print('\n\n\nKNN CLASSIFIER')

X2D = iris.data
Y2T = iris.target
X_train, X_test, y_train, y_test = train_test_split(X2D,Y2T,test_size=0.3)

knn1=KNeighborsClassifier(n_neighbors = 3)
knn1.fit(X_train,y_train)
Ypred=knn1.predict(X_test)
print(Ypred)

cm = confusion_matrix(y_test,Ypred)
display = ConfusionMatrixDisplay(confusion_matrix=cm)
display.plot()
plt.show()



