

Hunga-Bunga
============

Brute Force all scikit-learn models and all scikit-learn parameters with **fit** **predict**.



-----
##### Lets brute force all sklearn models with all of sklearn parameters!  Ahhh Hunga Bunga!!

```python
from hunga_bunga import HungaBungaClassifier, HungaBungaRegressor
```

##### And then simply: 

<p align="center">
  <img src="https://github.com/ypeleg/HungaBunga/blob/master/HungaBunga.png?raw=true" width="400">
</p>

-----



#### What?
Yes.

#### No! Really! What?
Many believe that

> most of the work of supervised (non-deep) Machine Learning lies in feature engineering, whereas the model-selection process is just running through all the models or just take xgboost.

So here is an automation for that.

## HOW IT WORKS
Runs through all `sklearn` models (both classification and regression), with **all possible hyperparameters**, and rank using cross-validation.

## MODELS
Runs **all the model** available on `sklearn` for supervised learning [here](http://scikit-learn.org/stable/supervised_learning.html). The categories are:

* Generalized Linear Models
* Kernel Ridge
* Support Vector Machines
* Nearest Neighbors
* Gaussian Processes
* Naive Bayes
* Trees
* Neural Networks
* Ensemble methods

Note: Some models were dropped out (nearly none of them..) and some crash or cause exceptions from time to time. It takes REALLY long to test this out so clearing exceptions took me a while.

## Installation 

```python
pip install hunga-bunga
```

Dependencies
~~~~~~~~~~~~

- Python (>= 2.7)
- NumPy (>= 1.11.0)
- SciPy (>= 0.17.0)
- joblib (>= 0.11)
- scikit-learn (>=0.20.0)
- tabulate (>=0.8.2)
- tqdm (>=4.28.1)

~~~~~~~~~~~~



## Option I (Recommended): brain = False


As any other sklearn model 

```python
clf = HungaBungaClassifier()
clf.fit(x, y)
clf.predict(x)
```
    
And import from here

```python
from hunga_bunga import HungaBungaClassifier, HungaBungaRegressor
```

## Option II: brain = True


As any other sklearn model 

```
clf = HungaBungaClassifier(brain=True)
clf.fit(x, y)
```

The output looks this:

| Model                       |  accuracy     |  Time/clf (s)|
|---------------------------- |:-------------:|:-------------:|
|SGDClassifier                |     0.967     |      0.001   |
|LogisticRegression           |     0.940      |      0.001   |
|Perceptron                   |     0.900       |      0.001   |
|PassiveAggressiveClassifier  |     0.967     |      0.001   |
|MLPClassifier                |     0.827     |      0.018   |
|KMeans                       |     0.580      |      0.010    |
|KNeighborsClassifier         |     0.960      |      0.000       |
|NearestCentroid              |     0.933     |      0.000       |
|RadiusNeighborsClassifier    |     0.927     |      0.000       |
|SVC                          |     0.960      |      0.000       |
|NuSVC                        |     0.980      |      0.001   |
|LinearSVC                    |     0.940      |      0.005   |
|RandomForestClassifier       |     0.980      |      0.015   |
|DecisionTreeClassifier       |     0.960      |      0.000       |
|ExtraTreesClassifier         |     0.993     |      0.002   |

*The winner is: ExtraTreesClassifier with score 0.993.*

