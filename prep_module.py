from load_data import load_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
import seaborn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn import model_selection
from sklearn import metrics
from sklearn.metrics import auc
from ROC_curve import plot_roc_curve as ROC

def prep(features, label):
    # Splitting data in train and test group
    X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=.2, stratify=label)


    def binner(y, class1):
        y_bin = []
        for val in y:
            if val == class1:
                y_bin.append(0)
            else:
                y_bin.append(1) 
        return y_bin
    
    y_train_bin = binner(y_train, 'T12') # make binary y_score
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

    # Construct classifiers
    coefs = []
    accuracies = []
    times = []

    n_alphas = 100
    alphas = np.logspace(-7, -1, n_alphas)

    for a in alphas:
        # Fit classifier
        clf = Lasso(alpha=a, fit_intercept=False)
        clf.fit(X_train_scaled, y_train_bin) 
        y_pred = clf.predict(X_test_scaled)
        
        # Append statistics
        accuracy = clf.score(X_train_scaled,y_train_bin)
        accuracies.append(accuracy)
        coefs.append(clf.coef_)

    # Weights
    plt.figure()
    ax = plt.gca()
    ax.plot(alphas, np.squeeze(coefs))
    ax.set_xscale('log')
    ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
    plt.xlabel('alpha')
    plt.ylabel('weights')
    plt.title('Lasso coefficients as a function of the regularization')
    plt.axis('tight')
    plt.show()

    # Performance
    plt.figure()
    ax = plt.gca()
    ax.plot(alphas, accuracies)
    ax.set_xscale('log')
    ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
    plt.xlabel('alpha')
    plt.ylabel('accuracies')
    plt.title('Performance as a function of the regularization')
    plt.axis('tight')
    plt.show()


    selector = SelectFromModel(estimator=Lasso(alpha=10**(-6)), threshold='median')
    selector.fit(X_train_scaled, y_train_bin)
    n_original = X_train_scaled.shape[1]
    X_train_fs = selector.transform(X_train_scaled)
    X_test_fs = selector.transform(X_test_scaled)
    n_selected = X_train_fs.shape[1]
    print(f"Selected {n_selected} from {n_original} features.")

    N_COMP = .9
    pca = PCA(n_components=N_COMP)
    pca.fit(X_train_fs)
    X_train_pca = pca.transform(X_train_fs)
    X_test_pca = pca.transform(X_test_fs)

    print(f'Number of features extracted from the PCA: {X_test_pca.shape[1]}')



    # SVC


    # Create a 3 fold stratified CV iterator
    cv_3fold = model_selection.StratifiedKFold(n_splits=3)
    results = []
    best_cls = []
    y_train_a = y_train.to_numpy()

    # Loop over the folds
    for train_opt_index, validation_index in cv_3fold.split(X_train_pca,y_train_a):
        # Split the data properly
        X_train_opt = X_train_pca[train_opt_index]
        y_train_opt = y_train_a[train_opt_index]
        
        X_validation = X_train_pca[validation_index]
        y_validation = y_train_a[validation_index]
        
        # Create a grid search to find the optimal k using a gridsearch and 3-fold cross validation
        # Same as above
        parameters = {
            # 'coef0':list(np.linspace(0.01,1,11)),
            'C':list(np.linspace(0.01,1,100)),
            # 'kernel' : ['linear','sigmoid'],
            # 'gamma' : ['scale','auto'],
            # 'degree' : [1, 2, 3, 4, 5]
        }
        svm_clf = SVC(kernel = 'linear', probability=True)
        cv_3fold = model_selection.StratifiedKFold(n_splits=3)
        grid_search = model_selection.GridSearchCV(svm_clf, parameters, cv=cv_3fold, scoring='roc_auc')
        grid_search.fit(X_train_opt, y_train_opt)
        
        # Get resulting classifier
        clf = grid_search.best_estimator_
        print(f'Best classifier: C={clf.C}')
        best_cls.append(clf.C)
        
        # Test the classifier on the train_opt data
        probabilities_train_opt = clf.predict_proba(X_train_opt)
        scores_train_opt = probabilities_train_opt[:, 1]
        
        # Get the auc
        auc_train_opt = metrics.roc_auc_score(y_train_opt, scores_train_opt)
        results.append({
            'auc': auc_train_opt,
            'n': clf.C,
            'set': 'train_opt'
        })

        # Test the classifier on the test data
        probabilities = clf.predict_proba(X_validation)
        scores = probabilities[:, 1]
        
        # Get the auc
        auc = metrics.roc_auc_score(y_validation, scores)
        results.append({
            'auc': auc,
            'n': clf.C,
            'set': 'validation'
        })
        
    # Create results dataframe and plot it
    results = pd.DataFrame(results)
    seaborn.boxplot(y='auc', x='set', data=results)

    optimal_c = float(np.mean(best_cls))
    print(f"The optimal C={optimal_c}")
    print(results)


    # Use the optimal parameters without any tuning to validate the optimal classifier
    clf = SVC(kernel = 'linear', probability=True, C=optimal_c)
    # Fit on the entire dataset
    clf.fit(X_train_pca, y_train)

    # Test the classifier on the indepedent test data
    probabilities = clf.predict_proba(X_test_pca)
    scores = probabilities[:, 1]

    # Get the auc
    auc_svc = metrics.roc_auc_score(y_test, scores)
    print(f'The AUC on the test set is {auc_svc} using a {optimal_c} Slack')






    score_train_svc = clf.score(X_train_pca, y_train)
    score_test_svc = clf.score(X_test_pca, y_test)
    print(f"Training result svc: {score_train_svc}")
    print(f"Test result svc: {score_test_svc}")

    # ROC(probabilities,y_test_bin)

#  KNN

    # Create a 3 fold stratified CV iterator
    cv_3fold = model_selection.StratifiedKFold(n_splits=3)
    results = []
    best_cls = []
    y_train_a = y_train.to_numpy()

    # Loop over the folds
    for train_opt_index, validation_index in cv_3fold.split(X_train_pca,y_train_a):
        # Split the data properly
        X_train_opt = X_train_pca[train_opt_index]
        y_train_opt = y_train_a[train_opt_index]
        
        X_validation = X_train_pca[validation_index]
        y_validation = y_train_a[validation_index]
        
        # Create a grid search to find the optimal k using a gridsearch and 3-fold cross validation
        # Same as above
        parameters = {
            'n_neighbors': list(range(1,31)),
        }
        knn_clf = KNeighborsClassifier(weights='distance')
        cv_3fold = model_selection.StratifiedKFold(n_splits=3)
        grid_search = model_selection.GridSearchCV(knn_clf, parameters, cv=cv_3fold, scoring='roc_auc')
        grid_search.fit(X_train_opt, y_train_opt)
        
        # Get resulting classifier
        clf = grid_search.best_estimator_
        print(f'Best classifier: n={clf.n_neighbors}')
        best_cls.append(clf.n_neighbors)
        
        # Test the classifier on the train_opt data
        probabilities_train_opt = clf.predict_proba(X_train_opt)
        scores_train_opt = probabilities_train_opt[:, 1]
        
        # Get the auc
        auc_train_opt = metrics.roc_auc_score(y_train_opt, scores_train_opt)
        results.append({
            'auc': auc_train_opt,
            'k': clf.n_neighbors,
            'set': 'train_opt'
        })

        # Test the classifier on the test data
        probabilities = clf.predict_proba(X_validation)
        scores = probabilities[:, 1]
        
        # Get the auc
        auc = metrics.roc_auc_score(y_validation, scores)
        results.append({
            'auc': auc,
            'k': clf.n_neighbors,
            'set': 'validation'
        })
        
    # Create results dataframe and plot it
    results = pd.DataFrame(results)
    seaborn.boxplot(y='auc', x='set', data=results)

    optimal_n = int(np.mean(best_cls))
    print(f"The optimal k={optimal_n}")
    print(results)


    # Use the optimal parameters without any tuning to validate the optimal classifier
    clf = SVC(kernel = 'linear', probability=True, C=optimal_n)
    # Fit on the entire dataset
    clf.fit(X_train_pca, y_train)

    # Test the classifier on the indepedent test data
    probabilities = clf.predict_proba(X_test_pca)
    scores = probabilities[:, 1]

    # Get the auc
    auc_knn = metrics.roc_auc_score(y_test, scores)
    print(f'The AUC on the test set is {auc_knn} using a {optimal_n}-NN')

    score_train_knn = clf.score(X_train_pca, y_train)
    score_test_knn = clf.score(X_test_pca, y_test)
    print(f"Training result knn: {score_train_knn}")
    print(f"Test result knn: {score_test_knn}")

    # ROC(probabilities,y_test_bin)
    return score_train_svc, score_test_svc, auc_svc,score_train_knn, score_test_knn, auc_knn