import numpy as np
import pandas as pd

from sklearn import svm, grid_search


def remove_missing_data(fileName, fillNa, df_return):
    df = pd.read_csv(fileName, delimiter="\t", index_col = [0])

    # drop rows, columns only if all values are nan
    df.dropna(how='all')
    # replace null, n/a
    df.replace('null', np.nan)
    df.replace('n/a', np.nan)
    # df.applymap(lambda x: 1 if (x==True) else 0)
    # df.replace('', np.nan)
    df.to_csv(fileName, sep=',',index=False)
    if(fillNa):
        df.fillna(fillNa)
    if(df_return):
        return df
    else:
        df.to_csv(fileName, sep=',',index=False)
        return None

def svc_param_selection(dataset, nfolds):
    # outcome and parameters
    y = np.array([x[yPos] for x in dataset])
    X = np.array([x[0:yPos] for x in dataset])

    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    kernels = ["linear", "rbf"]
    param_grid = {'C': Cs, 'gamma' : gammas, 'kernel': kernels}
    grid_search = GridSearchCV(svm.SVC(), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_

# def test_missing_data():
#     raw_data = {'first_name': ['Jason', np.nan, 'Tina', 'Jake', 'Amy'],
#         'last_name': ['Miller', np.nan, 'Ali', 'Milner', 'Cooze'],
#         'age': [42, np.nan, 36, 24, 73],
#         'sex': ['m', np.nan, 'f', 'm', 'f'],
#         'preTestScore': [4, np.nan, np.nan, 2, 3],
#         'postTestScore': [25, np.nan, np.nan, 62, 70]}
    
#     df = pd.DataFrame(raw_data, columns = ['first_name', 'last_name', 'age', 'sex', 'preTestScore', 'postTestScore'])
#     df_no_missing = df.dropna()
    
#     df.fillna(0)

    




    
