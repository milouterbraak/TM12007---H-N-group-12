from load_data import load_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2, SequentialFeatureSelector, SelectFromModel
from sklearn.linear_model import Lasso
import seaborn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from scipy import stats
from statsmodels.stats import weightstats
import numpy as np
import statistics
from time import time
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# Classifiers and kernels

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA, KernelPCA
from sklearn.kernel_approximation import RBFSampler
from sklearn.metrics.pairwise import rbf_kernel, sigmoid_kernel
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection
from sklearn import metrics


def prep(features, label):
    # Splitting data in train and test group
    X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=.2)

    # functie van maken??

    def binner(y, class1):
        y_bin = []
        for val in y:
            if val == class1:
                y_bin.append(0)
            else:
                y_bin.append(1) 
        return y_bin
    
    y_train_bin = binner(y_train, 'T12')
    y_test_bin = binner(y_test, 'T12')



    # Create the dataframe
    outlier_feat = []
    for feature in X_train.columns:
        # IQR
        Q1 = np.percentile(X_train[feature], 25,
                        interpolation = 'midpoint')
        
        Q3 = np.percentile(X_train[feature], 75,
                        interpolation = 'midpoint')
        IQR = Q3 - Q1
    
        if not IQR == 0:
            # Upper bound
            X_train.loc[X_train[feature] > (Q3+1.5*IQR),feature] = Q3
            # Lower bound
            X_train.loc[X_train[feature] < (Q1-1.5*IQR),feature] = Q1


    for feature in X_test.columns:
        
        # IQR
        Q1 = np.percentile(X_test[feature], 25,
                        interpolation = 'midpoint')
        
        Q3 = np.percentile(X_test[feature], 75,
                        interpolation = 'midpoint')
        IQR = Q3 - Q1
        
        if not IQR == 0:
            # Upper bound
            X_test.loc[X_test[feature] > (Q3+1.5*IQR),feature] = Q3
            # Lower bound
            X_test.loc[X_test[feature] < (Q1-1.5*IQR),feature] = Q1


    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns = features.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns = features.columns)


    coefs = []
    accuracies = []
    times = []

    n_alphas = 100
    alphas = np.logspace(-10, -1, n_alphas)

    for a in alphas:
        # Fit classifier
        clf = Lasso(alpha=a, fit_intercept=False)
        t0 = time()
        clf.fit(X_train_scaled, y_train_bin)
        duration = time() - t0
        y_pred = clf.predict(X_test_scaled)
        
        # Append statistics
        accuracy = float((y_test_bin != y_pred).sum()) / float(len(y_test_bin))
        times.append(duration)
        accuracies.append(accuracy)
        coefs.append(clf.coef_)

    # #############################################################################
    # Display results

    # # Weights
    # plt.figure()
    # ax = plt.gca()
    # ax.plot(alphas, np.squeeze(coefs))
    # ax.set_xscale('log')
    # ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
    # plt.xlabel('alpha')
    # plt.ylabel('weights')
    # plt.title('Lasso coefficients as a function of the regularization')
    # plt.axis('tight')
    # plt.show()

    # # Performance
    # plt.figure()
    # ax = plt.gca()
    # ax.plot(alphas, accuracies)
    # ax.set_xscale('log')
    # ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
    # plt.xlabel('alpha')
    # plt.ylabel('accuracies')
    # plt.title('Performance as a function of the regularization')
    # plt.axis('tight')
    # plt.show()

    # # Times
    # plt.figure()
    # ax = plt.gca()
    # ax.plot(alphas, times)
    # ax.set_xscale('log')
    # ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
    # plt.xlabel('alpha')
    # plt.ylabel('times (s)')
    # plt.title('Fitting time as a function of the regularization')
    # plt.axis('tight')
    # plt.show()


    selector = SelectFromModel(estimator=Lasso(alpha=10**(-7)), threshold='median')
    selector.fit(X_train_scaled, y_train_bin)
    n_original = X_train_scaled.shape[1]
    X_train_fs = selector.transform(X_train_scaled)
    X_test_fs = selector.transform(X_test_scaled)
    n_selected = X_train_fs.shape[1]
    print(f"Selected {n_selected} from {n_original} features.")


    # selector.threshold_

    N_COMP = .9
    pca = PCA(n_components=N_COMP)
    pca.fit(X_train_fs)
    X_train_pca = pca.transform(X_train_fs)
    X_test_pca = pca.transform(X_test_fs)

    return X_train_pca, y_train, y_train_bin, X_test_pca, y_test, y_test_bin