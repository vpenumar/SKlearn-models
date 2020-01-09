
import random
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from sklearn import datasets
from sklearn.linear_model import SGDClassifier, LogisticRegression, Perceptron, PassiveAggressiveClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid, RadiusNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, DotProduct, Matern, StationaryKernelMixin, WhiteKernel
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import ParameterSampler

from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin
from sklearn.base import is_classifier

from core import *
from params import *


linear_models_n_params = [
    (SGDClassifier,
     {'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge'],
      'alpha': [0.0001, 0.001, 0.1],
      'penalty': penalty_12none
      }),

    (LogisticRegression,
     {'penalty': penalty_12, 'max_iter': max_iter, 'tol': tol,  'warm_start': warm_start, 'C':C, 'solver': ['liblinear']
      }),

    (Perceptron,
     {'penalty': penalty_all, 'alpha': alpha, 'n_iter': n_iter, 'eta0': eta0, 'warm_start': warm_start
      }),

    (PassiveAggressiveClassifier,
     {'C': C, 'n_iter': n_iter, 'warm_start': warm_start,
      'loss': ['hinge', 'squared_hinge'],
      })
]

linear_models_n_params_small = linear_models_n_params

svm_models_n_params = [
    (SVC,
     {'C':C, 'kernel': kernel, 'degree': degree, 'gamma': gamma, 'coef0': coef0, 'shrinking': shrinking, 'tol': tol, 'max_iter': max_iter_inf2}),

    (NuSVC,
     {'nu': nu, 'kernel': kernel, 'degree': degree, 'gamma': gamma, 'coef0': coef0, 'shrinking': shrinking, 'tol': tol
      }),

    (LinearSVC,
     { 'C': C, 'penalty_12': penalty_12, 'tol': tol, 'max_iter': max_iter,
       'loss': ['hinge', 'squared_hinge'],
       })
]

svm_models_n_params_small = [
    (SVC,
     {'C':C, 'kernel': kernel, 'degree': degree, 'gamma': gamma, 'coef0': coef0, 'shrinking': shrinking, 'tol': tol, 'max_iter': max_iter_inf2}),

    (NuSVC,
     {'nu': nu, 'kernel': kernel, 'degree': degree, 'gamma': gamma, 'coef0': coef0, 'shrinking': shrinking, 'tol': tol
      }),

    (LinearSVC,
     { 'C': C, 'penalty': penalty_12, 'tol': tol, 'max_iter': max_iter,
       'loss': ['hinge', 'squared_hinge'],
       })
]

neighbor_models_n_params = [

    (KMeans,
     {'algorithm': ['auto', 'full', 'elkan'],
      'init': ['k-means++', 'random']}),

    (KNeighborsClassifier,
     {'n_neighbors': n_neighbors, 'algo': neighbor_algo, 'leaf_size': neighbor_leaf_size, 'metric': neighbor_metric,
      'weights': ['uniform', 'distance'],
      'p': [1, 2]
      }),

    (NearestCentroid,
     {'metric': neighbor_metric,
      'shrink_threshold': [1e-3, 1e-2, 0.1, 0.5, 0.9, 2]
      }),

    (RadiusNeighborsClassifier,
     {'radius': neighbor_radius, 'algo': neighbor_algo, 'leaf_size': neighbor_leaf_size, 'metric': neighbor_metric,
      'weights': ['uniform', 'distance'],
      'p': [1, 2],
      'outlier_label': [-1]
      })
]

gaussianprocess_models_n_params = [
    (GaussianProcessClassifier,
     {'warm_start': warm_start,
      'kernel': [RBF(), ConstantKernel(), DotProduct(), WhiteKernel()],
      'max_iter_predict': [500],
      'n_restarts_optimizer': [3],
      })
]

bayes_models_n_params = [
    (GaussianNB, {})
]

nn_models_n_params = [
    (MLPClassifier,
     { 'hidden_layer_sizes': [(16,), (64,), (100,), (32, 32)],
       'activation': ['identity', 'logistic', 'tanh', 'relu'],
       'alpha': alpha, 'learning_rate': learning_rate, 'tol': tol, 'warm_start': warm_start,
       'batch_size': ['auto', 50],
       'max_iter': [1000],
       'early_stopping': [True, False],
       'epsilon': [1e-8, 1e-5]
       })
]

nn_models_n_params_small = [
    (MLPClassifier,
     { 'hidden_layer_sizes': [(64,), (32, 64)],
       'batch_size': ['auto', 50],
       'activation': ['identity', 'tanh', 'relu'],
       'max_iter': [500],
       'early_stopping': [True],
       'learning_rate': learning_rate_small
       })
]

tree_models_n_params = [

    (RandomForestClassifier,
     {'criterion': ['gini', 'entropy'],
      'max_features': max_features, 'n_estimators': n_estimators, 'max_depth': max_depth,
      'min_samples_split': min_samples_split, 'min_impurity_split': min_impurity_split, 'warm_start': warm_start, 'min_samples_leaf': min_samples_leaf,
      }),

    (DecisionTreeClassifier,
     {'criterion': ['gini', 'entropy'],
      'max_features': max_features, 'max_depth': max_depth, 'min_samples_split': min_samples_split, 'min_impurity_split':min_impurity_split, 'min_samples_leaf': min_samples_leaf
      }),

    (ExtraTreesClassifier,
     {'n_estimators': n_estimators, 'max_features': max_features, 'max_depth': max_depth,
      'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf, 'min_impurity_split': min_impurity_split, 'warm_start': warm_start,
      'criterion': ['gini', 'entropy']})
]

tree_models_n_params_small = [

    (RandomForestClassifier,
     {'max_features_small': max_features_small, 'n_estimators_small': n_estimators_small, 'min_samples_split': min_samples_split, 'max_depth_small': max_depth_small, 'min_samples_leaf': min_samples_leaf
      }),

    (DecisionTreeClassifier,
     {'max_features_small': max_features_small, 'max_depth_small': max_depth_small, 'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf
      }),

    (ExtraTreesClassifier,
     {'n_estimators_small': n_estimators_small, 'max_features_small': max_features_small, 'max_depth_small': max_depth_small,
      'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf})
]


def run_all_classifiers(x, y, small = True, normalize_x = True, n_jobs=cpu_count()-1, brain=False, test_size=0.2, n_splits=5, upsample=True, scoring=None, verbose=False, grid_search=True):
    all_params = (linear_models_n_params_small if small else linear_models_n_params) +  (nn_models_n_params_small if small else nn_models_n_params) + ([] if small else gaussianprocess_models_n_params) + neighbor_models_n_params + (svm_models_n_params_small if small else svm_models_n_params) + (tree_models_n_params_small if small else tree_models_n_params)
    return main_loop(all_params, StandardScaler().fit_transform(x) if normalize_x else x, y, isClassification=True, n_jobs=n_jobs, verbose=verbose, brain=brain, test_size=test_size, n_splits=n_splits, upsample=upsample, scoring=scoring, grid_search=grid_search)

def run_one_classifier(x, y, small = True, normalize_x = True, n_jobs=cpu_count()-1, brain=False, test_size=0.2, n_splits=5, upsample=True, scoring=None, verbose=False, grid_search=True):
    all_params = (linear_models_n_params_small if small else linear_models_n_params) +  (nn_models_n_params_small if small else nn_models_n_params) + ([] if small else gaussianprocess_models_n_params) + neighbor_models_n_params + (svm_models_n_params_small if small else svm_models_n_params) + (tree_models_n_params_small if small else tree_models_n_params)
    all_params = random.choice(all_params)
    return all_params[0](**(list(ParameterSampler(all_params[1], n_iter=1))[0]))


class HungaBungaClassifier(ClassifierMixin):
    def __init__(self, brain=False, test_size = 0.2, n_splits = 5, random_state=None, upsample=True, scoring=None, verbose=False, normalize_x = True, n_jobs =cpu_count() - 1, grid_search=True):
        self.model = None
        self.brain = brain
        self.test_size = test_size
        self.n_splits = n_splits
        self.random_state = random_state
        self.upsample = upsample
        self.scoring = None
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.normalize_x = normalize_x
        self.grid_search = grid_search
        super(HungaBungaClassifier, self).__init__()

    def fit(self, x, y):
        self.model = run_all_classifiers(x, y, normalize_x=self.normalize_x, test_size=self.test_size, n_splits=self.n_splits, upsample=self.upsample, scoring=self.scoring, verbose=self.verbose, brain=self.brain, n_jobs=self.n_jobs, grid_search=self.grid_search)[0]
        return self

    def predict(self, x):
        return self.model.predict(x)


class HungaBungaRandomClassifier(ClassifierMixin):
    def __init__(self, brain=False, test_size = 0.2, n_splits = 5, random_state=None, upsample=True, scoring=None, verbose=False, normalize_x = True, n_jobs =cpu_count() - 1, grid_search=True):
        self.model = None
        self.brain = brain
        self.test_size = test_size
        self.n_splits = n_splits
        self.random_state = random_state
        self.upsample = upsample
        self.scoring = None
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.normalize_x = normalize_x
        self.grid_search = grid_search
        super(HungaBungaRandomClassifier, self).__init__()

    def fit(self, x, y):
        self.model = run_one_classifier(x, y, normalize_x=self.normalize_x, test_size=self.test_size, n_splits=self.n_splits, upsample=self.upsample, scoring=self.scoring, verbose=self.verbose, brain=self.brain, n_jobs=self.n_jobs, grid_search=self.grid_search)
        self.model.fit(x, y)
        return self

    def predict(self, x):
        return self.model.predict(x)


if __name__ == '__main__':
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    clf = HungaBungaClassifier()
    clf.fit(X, y)
    print(clf.predict(X).shape)

