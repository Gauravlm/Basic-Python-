import numpy as np
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
import seaborn as sn
iris = load_iris()

#print(iris)

print(iris.feature_names)
print(iris.target_names)
print(iris.data[0])
print(iris.target[0])

 x = iris.data  
 y = iris.target 


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.5)
x_train.hist(bins=10)
##############################################################################
# using decission Tree
from sklearn.tree import DecisionTreeClassifier
clf_DT = DecisionTreeClassifier()

# fit the model using decission tree
clf_DT.fit(x_train, y_train)  

# predict 
pred_DT = clf.predict(x_test)
#print(pred_DT)

# checking the accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,pred_DT))

#############################################################################
from sklearn.neighbors import KNeighborsClassifier
clf_KN= KNeighborsClassifier(n_neighbors=4)
#fit the model
clf_KN.fit(x_train,y_train)

#predict
pred_KN=clf_KN.predict(x_test)
# checking the accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,pred_KN))   
# got 0.96 accuracy if test_size = 0.5 in train_test_split
# got 0.9714 accuracy if test_size = 0.7 in train_test_split

##############################################################################
# using k-fold cross validation
from sklearn.cross_validation import cross_val_score
clf_KN = KNeighborsClassifier(n_neighbors=5)
score = cross_val_score(clf_KN,x,y,cv=10,scoring='accuracy')
print(score)
print(score.mean())


# using optimal k values for cross validation
k_val =range(1,31)
k_score= []

for k in k_val:
    KNN_k = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(KNN_k, x, y, cv=10, scoring='accuracy')
    k_score.append(score.mean())

print(k_score)


# plot k_score and k_val
import  matplotlib.pyplot as plt
plt.plot(k_val,k_score)
plt.xlabel('Values of k')
plt.ylabel('cross validity accuracy')
plt.show()


# after ploting ghaph k_val Vs K-score
# we at value of k=20 we get maximum accuracy so put n_neighbors =20
KNN = KNeighborsClassifier(n_neighbors =20)
score = cross_val_score(KNN, x, y, cv=10,scoring = 'accuracy').mean()
print(score)   # got 0.98 accuracy


