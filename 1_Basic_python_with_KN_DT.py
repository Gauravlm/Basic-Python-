import numpy as np
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
iris = load_iris()

#print(iris)

print(iris.feature_names)
print(iris.target_names)
print(iris.data[0])
print(iris.target[0])

 x = iris.data  
 y = iris.target 

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.7)
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


# with 