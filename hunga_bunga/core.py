
import warnings
warnings.filterwarnings('ignore')

import sklearn.model_selection
import numpy as np
nan = float('nan')
import traceback

from pprint import pprint
from collections import Counter
from multiprocessing import cpu_count
from time import time
from tabulate import tabulate
try: from tqdm import tqdm
except: tqdm = lambda x: x

from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedShuffleSplit as sss, ShuffleSplit as ss, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, ExtraTreesRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn import model_selection


TREE_N_ENSEMBLE_MODELS = [RandomForestClassifier, GradientBoostingClassifier, DecisionTreeClassifier, DecisionTreeRegressor,ExtraTreesClassifier, ExtraTreesRegressor, AdaBoostClassifier, AdaBoostRegressor]


class GridSearchCVProgressBar(sklearn.model_selection.GridSearchCV):
    def _get_param_iterator(self):
        iterator = super(GridSearchCVProgressBar, self)._get_param_iterator()
        iterator = list(iterator)
        n_candidates = len(iterator)
        cv = sklearn.model_selection._split.check_cv(self.cv, None)
        n_splits = getattr(cv, 'n_splits', 3)
        max_value = n_candidates * n_splits
        class ParallelProgressBar(sklearn.model_selection._search.Parallel):
            def __call__(self, iterable):
                iterable = tqdm(iterable, total=max_value)
                iterable.set_description("GridSearchCV")
                return super(ParallelProgressBar, self).__call__(iterable)
        sklearn.model_selection._search.Parallel = ParallelProgressBar
        return iterator


class RandomizedSearchCVProgressBar(sklearn.model_selection.RandomizedSearchCV):
    def _get_param_iterator(self):
        iterator = super(RandomizedSearchCVProgressBar, self)._get_param_iterator()
        iterator = list(iterator)
        n_candidates = len(iterator)
        cv = sklearn.model_selection._split.check_cv(self.cv, None)
        n_splits = getattr(cv, 'n_splits', 3)
        max_value = n_candidates * n_splits
        class ParallelProgressBar(sklearn.model_selection._search.Parallel):
            def __call__(self, iterable):
                iterable = tqdm(iterable, total=max_value)
                iterable.set_description("RandomizedSearchCV")
                return super(ParallelProgressBar, self).__call__(iterable)
        sklearn.model_selection._search.Parallel = ParallelProgressBar
        return iterator


def upsample_indices_clf(inds, y):
    assert len(inds) == len(y)
    countByClass = dict(Counter(y))
    maxCount = max(countByClass.values())
    extras = []
    for klass, count in countByClass.items():
        if maxCount == count: continue
        ratio = int(maxCount / count)
        cur_inds = inds[y == klass]
        extras.append(np.concatenate( (np.repeat(cur_inds, ratio - 1), np.random.choice(cur_inds, maxCount - ratio * count, replace=False))))
    return np.concatenate([inds] + extras)


def cv_clf(x, y, test_size = 0.2, n_splits = 5, random_state=None, doesUpsample = True):
    sss_obj = sss(n_splits, test_size, random_state=random_state).split(x, y)
    if not doesUpsample: yield sss_obj
    for train_inds, valid_inds in sss_obj: yield (upsample_indices_clf(train_inds, y[train_inds]), valid_inds)


def cv_reg(x, test_size = 0.2, n_splits = 5, random_state=None): return ss(n_splits, test_size, random_state=random_state).split(x)


def timeit(klass, params, x, y):
    start = time()
    clf = klass(**params)
    clf.fit(x, y)
    return time() - start


def main_loop(models_n_params, x, y, isClassification, test_size = 0.2, n_splits = 5, random_state=None, upsample=True, scoring=None, verbose=True, n_jobs =cpu_count() - 1, brain=False, grid_search=True):
    def cv_(): return cv_clf(x, y, test_size, n_splits, random_state, upsample) if isClassification else cv_reg(x, test_size, n_splits, random_state)
    res = []
    num_features = x.shape[1]
    scoring = scoring or ('accuracy' if isClassification else 'neg_mean_squared_error')
    if brain: print('Scoring criteria:', scoring)
    for i, (clf_Klass, parameters) in enumerate(tqdm(models_n_params)):
        try:
            if brain: print('-'*15, 'model %d/%d' % (i+1, len(models_n_params)), '-'*15)
            if brain: print(clf_Klass.__name__)
            if clf_Klass == KMeans: parameters['n_clusters'] = [len(np.unique(y))]
            elif clf_Klass in TREE_N_ENSEMBLE_MODELS: parameters['max_features'] = [v for v in parameters['max_features'] if v is None or type(v)==str or v<=num_features]
            if grid_search: clf_search = GridSearchCVProgressBar(clf_Klass(), parameters, scoring, cv=cv_(), n_jobs=n_jobs)
            else: clf_search = RandomizedSearchCVProgressBar(clf_Klass(), parameters, scoring, cv=cv_(), n_jobs=n_jobs)
            clf_search.fit(x, y)
            timespent = timeit(clf_Klass, clf_search.best_params_, x, y)
            if brain: print('best score:', clf_search.best_score_, 'time/clf: %0.3f seconds' % timespent)
            if brain: print('best params:')
            if brain: pprint(clf_search.best_params_)
            if verbose:
                print('validation scores:', clf_search.cv_results_['mean_test_score'])
                print('training scores:', clf_search.cv_results_['mean_train_score'])
            res.append((clf_search.best_estimator_, clf_search.best_score_, timespent))
        except Exception as e:
            if verbose: traceback.print_exc()
            res.append((clf_Klass(), -np.inf, np.inf))
    if brain: print('='*60)
    if brain: print(tabulate([[m.__class__.__name__, '%.3f'%s, '%.3f'%t] for m, s, t in res], headers=['Model', scoring, 'Time/clf (s)']))
    winner_ind = np.argmax([v[1] for v in res])
    winner = res[winner_ind][0]
    if brain: print('='*60)
    if brain: print('The winner is: %s with score %0.3f.' % (winner.__class__.__name__, res[winner_ind][1]))
    return winner, res



if __name__ == '__main__':
    y = np.array([0,1,0,0,0,3,1,1,3])
    x = np.zeros(len(y))
    for t, v in cv_reg(x): print(v,t)
    for t, v in cv_clf(x, y, test_size=5): print(v,t)