
import warnings
warnings.filterwarnings('ignore')
from multiprocessing import cpu_count
from sklearn.base import BaseEstimator
from regression import HungaBungaRegressor, HungaBungaRandomRegressor
from classification import HungaBungaClassifier, HungaBungaRandomClassifier


class HungaBungaZeroKnowledge(BaseEstimator):
    def __init__(self, brain=False, test_size = 0.2, n_splits = 5, random_state=None, upsample=True, scoring=None, verbose=True, normalize_x = True, n_jobs =cpu_count() - 1, grid_search=True):
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
        self.problem_type = 'Unknown'
        self.grid_search = grid_search
        super(HungaBungaZeroKnowledge, self).__init__()

    def fit(self, X, y):
        try:
            self.model = HungaBungaClassifier(normalize_x=self.normalize_x, test_size=self.test_size, n_splits=self.n_splits, upsample=self.upsample, scoring=self.scoring, verbose=self.verbose, brain=self.brain, n_jobs=self.n_jobs, grid_search=self.grid_search)
            self.model.fit(X, y)
            self.problem_type = 'Classification'
        except:
            self.model = HungaBungaRegressor(normalize_x=self.normalize_x, test_size=self.test_size, n_splits=self.n_splits, upsample=self.upsample, scoring=self.scoring, verbose=self.verbose, brain=self.brain, n_jobs=self.n_jobs, grid_search=self.grid_search)
            self.model.fit(X, y)
            self.problem_type = 'Regression'
        return self

    def predict(self, x):
        return self.model.predict(x)

