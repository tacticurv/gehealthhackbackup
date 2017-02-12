from sklearn import svm, datasets
import numpy as np



def svmClassifier(trainData, testData ,yPos, kernel, c, gamma):
    """
        # sklearn.svm.SVC(C=1.0, kernel='rbf', degree=3, gamma=0.0, coef0=0.0, shrinking=True, probability=False,tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None)
    """
    model = svm.SVC(kernel=kernel, C=c, gamma=gamma)
    # outcome and parameters
    y = np.array([x[yPos] for x in trainData])
    X = np.array([x[0:yPos] for x in trainData])
    # fit and calculate
    model.fit(X, y)
    # model.score(X, y)
    #Predict Output
    return model


def testsvmClassifier():
    iris = datasets.load_iris()
    X = iris.data[:, :2]
    y = iris.target
    svc = svm.SVC(kernel='linear', C=1,gamma=0).fit(X, y)
    
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = (x_max / x_min)/100
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    plt.subplot(1, 1, 1)
    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.title('SVC with linear kernel')
    plt.show()
    
