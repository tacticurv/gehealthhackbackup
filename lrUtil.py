import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

def lrClassifier(trainData, testData, yPos, n_folds):
	#  # outcome and parameters
    y = np.array([x[yPos] for x in trainData])
    X = np.array([x[0:yPos] for x in trainData])
    # fit and calculate
    model = LogisticRegression()
    model = model.fit(X, y)
    model.score(X, y)
    predicted = model.predict([x[0:yPos] for x in testData])
    #Predict Output
    return predicted


def adaboost_gridSearch_lrClassifier(trainData, testData, yPos):
	"""
		adaboosting --best out-of-the-box classifiers used on weak learning models to boost the learning,
		along with gridsearch makes the parameters distribution more relevant
	"""
	# outcome and parameters
	y = np.array([x[yPos] for x in trainData])
	X = np.array([x[0:yPos] for x in trainData])

    # parameters included here for gridSearch
	param_grid = {
					"n_estimators": range(1,100,10)
                }

	tree = DecisionTreeClassifier(max_depth = 5)

	ABC = AdaBoostClassifier(base_estimator = tree)
	grid_search_ABC = GridSearchCV(ABC, param_grid=param_grid, scoring = 'roc_auc')
	if len(np.unique(y)) > 2:
		grid_search_ABC = AdaBoostClassifier(base_estimator=tree, n_estimators=100)

	grid_search_ABC.fit(X, y)
	return grid_search_ABC
