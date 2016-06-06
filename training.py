# edited by Leiya Ma at 2016-06-05

import numpy as np
import matplotlib.pyplot as plt
import random

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree     import DecisionTreeClassifier

from sklearn     import cross_validation
from scipy.stats import randint as sp_randint

# load training data
print "Loading training data..."
x_train = np.genfromtxt('BinaryTrain_data.csv', delimiter = ',')
train_labels_raw = np.genfromtxt('BinaryTrain_sbj_list.csv', delimiter = ',', skip_header = 1)
y_train = train_labels_raw[:, 1]

# partition the data into training and testing sets 
# (this is not done before submission as we train on the whole dataset)
# test_size_percentage = 0.20            # size of test set as a percentage in [0., 1.]
# seed = random.randint(0, 2**30)        # pseudo-random seed for split 
# x_train, x_test, y_train, y_test = cross_validation.train_test_split(
# 	x, y, test_size=test_size_percentage, random_state=seed)

# load testing data
print "Loading test data..."
x_test = np.genfromtxt('BinaryTest_data.csv', delimiter = ',')

# adaboost classifier
# initialize the  (defaults to using shallow decision trees
# as the weak learner to boost) and optimize parameters using random search
print "Training adaboost DT..."
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                         n_estimators=400)
bdt.fit(x_train, y_train)

# set up bagging around each AdaBoost set
print "Training bagged DT..."
bagged = BaggingClassifier(n_estimators=201,
                           max_samples=0.1)
bagged.fit(x_train, y_train)


# initialize a random forest classifier 
print 'Training random forest...'
rfc = RandomForestClassifier(n_estimators=200,
                             max_features=40,
                             min_samples_split=2,
                             min_samples_leaf=1)
rfc.fit(x_train, y_train)

# training scores
print "Training scores..."
print bdt.score(x_train, y_train)
print bagged.score(x_train, y_train)
print rfc.score(x_train, y_train)

# score the classfier on the test set 
# print "Scoring..."
# print bdt.score(x_test, y_test)
# print bagged.score(x_test, y_test)
# print rfc.score(x_test, y_test)

# print "Writing predictions..."
predictions1 = bdt.predict(x_test)
predictions2 = bagged.predict(x_test)
predictions3 = rfc.predict(x_test)
predictions = []

for i in range(100):
	if predictions1[i] + predictions2[i] + predictions3[i] > 1:
		predictions.append(1)
	else:
		predictions.append(0)

f = open('/Users/LeiyaMa/Desktop/binary/predictions.csv', 'w')
f.write('SID,Label\n')
for i in range(100):
	f.write('Sbj' + str(i+1) + ',' + str(int(predictions[i])) + '\n')


################################################################################
# RESULTS:
################################################################################

# with original training data:
# training with 75%, testing on 25% yields score = 0.662213740458

# with transformed tf-idf data (not sure if this is working yet 
# because this seems signifigantly worse than with the original data):
# training with 75%, testing on 25% yields score = 0.644083969466

# submission using all training data yields about 0.53 on the leaderboard

################################################################################
# RANDOM DISORGANIZED CODE BELOW: 
################################################################################

####
# FOR PRINTING CROSS VALIDATION SCORES:
####
# print "Generating CV scores..."
# scores = cross_validation.cross_val_score(random_search, x_train, y_train, cv=5)
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

####
# FOR PLOTTING:
####
# perform the plot 
# plt.plot(estimators, validation_means, label='Train')
# plt.errorbar(estimators, validation_means, validation_std_devs)
# plt.legend(loc='best')
# plt.title('n_estimators vs. validation mean score')
# plt.xlabel('number of estimators')
# plt.ylabel('mean validation score')
# plt.show()

####
# FOR PLOTTING ESTIMATORS VS VALIDATION MEANS:
####
# estimator_range = [i for i in range(1, 600 + 1)]
# estimators = estimator_range[::100]
# validation_means = [] 
# validation_std_devs = []
# print estimators 
# for n in estimators:
# 	print "Training with n_estimators = " + str(n)
# 	bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),
# 	                         n_estimators=n,
#                              learning_rate=1.)
# 	print "Computing validation mean..."
# 	scores = cross_validation.cross_val_score(bdt, x_train, y_train, cv=5)
# 	validation_means.append(scores.mean())
# 	validation_std_devs.append(scores.std() * 2)


# In[ ]:




# In[ ]:



