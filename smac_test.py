import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from smac import HyperparameterOptimizationFacade, Scenario

iris = datasets.load_iris()


def train(config: Configuration, seed: int = 0) -> float:
    classifier = SVC(C=config["C"], random_state=seed)
    scores = cross_val_score(classifier, iris.data, iris.target, cv=5)
    return 1 - np.mean(scores)


configspace = ConfigurationSpace({"C": (0.100, 1000.0)})

scenario = Scenario(configspace, deterministic=True, n_trials=300)

smac = HyperparameterOptimizationFacade(scenario, train)
incumbent = smac.optimize()
incumbent
