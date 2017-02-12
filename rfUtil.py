from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
import numpy as np

def rfClassifier(trainData, testData, yPos, n_estimators, n_folds):
    #  # outcome and parameters
    y = np.array([x[yPos] for x in trainData])
    X = np.array([x[0:yPos] for x in trainData])
    
    # fit and calculate
    cfr = RandomForestClassifier(n_estimators=n_estimators)
    cfr.fit(X, y)
    #Predict Output
    return cfr
    # cv = cross_validation.KFold(len(train), n_folds=n_folds)
    
    # results = []
    # for traincv, testcv in cv:
    #     probas = cfr.fit(train[traincv], target[traincv]).predict_proba(train[testcv])
    #     results.append(logloss.llfun(target[testcv], [x[1] for x in probas]))
    # return results


    
    
