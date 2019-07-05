# machine-learning-1

"""
a simlple regression analysis on the bostan housing date
here we perform a simple regression analysis on the boston housing
data exploring two tyes of regresssion
"""


from sklearn.datasets import load_boston
data=load_boston()
#print a histogram of the quality to predict :price

import matplotlib.pyplot as plt
plt.figure(figsize=(4,3))
plt.hist(data.target)
plt.xlabel('price ($1000s)')
plt.ylabel('count')
plt.tight_layout()
plt.show()
#print the join histogram for each features

for index, feature_name in enumerate(data.feature_names):
    plt.figure(figsize=(4,3))
    plt.scatter(data.data[:,index],data.target)
    plt.ylabel('price',size=15)
    plt.xlabel(feature_name, size=15)
    plt.tight_layout()
    plt.show()

#simple prediction
##############################################################
from sklearn.model_selection import train_test_split
X_train, X_test,Y_train,Y_test= train_test_split(data.data,data.target)

from sklearn.linear_model import LinearRegression
clf= LinearRegression()

clf.fit(X_train,Y_train)
predicted=clf.predict(X_test)
expected= Y_test

plt.figure(figsize=(4,3))
plt.scatter(expected,predicted)
plt.plot([10,50],[0,50],'--k')
plt.axis('tight')
plt.xlabel('True price($1000s)')
plt.ylabel('predicted prices ($1000s)')
plt.tight_layout()
plt.show()

####################################################
# Prediction with gradient boosted tree

from sklearn.ensemble import GradientBoostingRegressor

clf = GradientBoostingRegressor()
clf.fit(X_train ,Y_train)

predicted=clf.predict(X_test)
expected = Y_test

plt.figure(figsize=(4,3))
plt.scatter(expected,predicted)
plt.plot([10,50],[0,50],'--k')
plt.axis('tight')
plt.xlabel('True price ($1000s)')
plt.ylabel('predicted price ($1000s)')
plt.tight_layout()
plt.show()


                                

                                

