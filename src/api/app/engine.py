
from abc import ABC, abstractmethod
from sklearn.model_selection import GridSearchCV
from datetime import datetime, timedelta
import concurrent.futures
from collections import Counter
import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from xgboost import XGBClassifier
from sklearn.metrics import mean_absolute_error
from pyswarm import pso
from scipy.optimize import minimize
from abc import ABC, abstractmethod
from collections import Counter, defaultdict

from sklearn.model_selection import RandomizedSearchCV
from skopt import BayesSearchCV
# Import necessary libraries
from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.svm import SVR, SVC
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge, ARDRegression, SGDRegressor, PassiveAggressiveRegressor, HuberRegressor, TheilSenRegressor, RANSACRegressor, OrthogonalMatchingPursuit, Lars, LassoLars
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from statsmodels.tsa.arima.model import ARIMA
from sklearn.mixture import GaussianMixture
import numpy as np
from sklearn.neighbors import KNeighborsClassifier  # Import here to avoid circular dependency
from sklearn.neural_network import MLPClassifier  # Example with Multi-layer Perceptron classifier

# Define classes and functions you want to expose
__all__ = [
   
    'DataLoader',
    'OptimizerFactory',
    'GridSearchOptimizer',
    'ModelBuilder',
    'BaseModel',
    'RandomForest',  # Assuming RandomForest is a defined class
    'SVM',
    'GBM',
    'LogReg',
    'KNN',
    'NeuralNetwork',
    'XGBoost',
    'Optimizer',
    'RandomSearchOptimizer',
    'BayesianOptimizer',
    'PSOOptimizer',
    'SimulatedAnnealingOptimizer',
    'TPEOptimizer',
    'OptunaCMAESOptimizer',
    'DEAPGAOptimizer',
    'OptunaTPEOptimizer',
    'EnsembleStrategy',
    'ThresholdVotingStrategy',
    'BordaCountStrategy',
    'SoftVotingStrategy',
    'MaxVotingStrategy',
    'MinVotingStrategy',
    'ProductStrategy',
    'RankAveragingStrategy',
    'MajorityVoteStrategy',
    'AverageStrategy',
    'MappingLayer',
    # Add more class
    'Ensemble',
    'DataPreparation',
    'Pipeline2',
    'WalkForwardValidation',
    'TrainingCoordinator',
    # Add more as needed...
]

# BaseModel class
class BaseModel(ABC):
    def __init__(self, name, algorithm_instance):
        self.name = name
        self.algorithm_instance = algorithm_instance

    @abstractmethod
    def train(self, X_train, y_train):
        pass

    def predict(self, X):
        return self.algorithm_instance.predict(X)

# Model implementations
class RandomForestRegressorModel(BaseModel):
    def __init__(self, params=None):
        super().__init__("RandomForestRegressor", RandomForestRegressor(**(params or {})))

    def train(self, X_train, y_train):
        self.algorithm_instance.fit(X_train, y_train)

class GradientBoostingRegressorModel(BaseModel):
    def __init__(self, params=None):
        super().__init__("GradientBoostingRegressor", GradientBoostingRegressor(**(params or {})))

    def train(self, X_train, y_train):
        self.algorithm_instance.fit(X_train, y_train)

class AdaBoostRegressorModel(BaseModel):
    def __init__(self, params=None):
        super().__init__("AdaBoostRegressor", AdaBoostRegressor(**(params or {})))

    def train(self, X_train, y_train):
        self.algorithm_instance.fit(X_train, y_train)


class RandomForestClassifierModel(BaseModel):
    def __init__(self, params=None):
        super().__init__("RandomForestClassifier", RandomForestClassifier(**(params or {})))

    def train(self, X_train, y_train):
        self.algorithm_instance.fit(X_train, y_train)

class AdaBoostClassifierModel(BaseModel):
    def __init__(self, params=None):
        super().__init__("AdaBoostClassifier", AdaBoostClassifier(**(params or {})))

    def train(self, X_train, y_train):
        self.algorithm_instance.fit(X_train, y_train)

class GradientBoostingClassifierModel(BaseModel):
    def __init__(self, params=None):
        super().__init__("GradientBoostingClassifier", GradientBoostingClassifier(**(params or {})))

    def train(self, X_train, y_train):
        self.algorithm_instance.fit(X_train, y_train)

class KMeansModel(BaseModel):
    def __init__(self, params=None):
        super().__init__("KMeans", KMeans(**(params or {})))

    def train(self, X_train, y_train=None):
        self.algorithm_instance.fit(X_train)

class PCAModel(BaseModel):
    def __init__(self, params=None):
        super().__init__("PrincipalComponentAnalysis", PCA(**(params or {})))

    def train(self, X_train, y_train=None):
        self.algorithm_instance.fit(X_train)

# Gaussian Mixture Model (GaussianHMM)
class GaussianMixtureModel(BaseModel):
    def __init__(self, params=None):
        super().__init__("GaussianHMM", GaussianMixture(**(params or {})))

    def train(self, X_train, y_train=None):
        self.algorithm_instance.fit(X_train)

# ARIMA Model
class ARIMAModel(BaseModel):
    def __init__(self, order=(1, 1, 1)):
        super().__init__("ARIMA", ARIMA(None, order=order))

    def train(self, X_train, y_train=None):
        self.algorithm_instance = self.algorithm_instance.fit(X_train)

class SVMModel(BaseModel):
    def __init__(self, params=None):
        super().__init__("SVM", SVC(**(params or {})))

    def train(self, X_train, y_train):
        self.algorithm_instance.fit(X_train, y_train)

class LogRegModel(BaseModel):
    def __init__(self, params=None):
        super().__init__("LogReg", LogisticRegression(**(params or {})))

    def train(self, X_train, y_train):
        self.algorithm_instance.fit(X_train, y_train)

class KNNModel(BaseModel):
    def __init__(self, params=None):
        super().__init__("KNN", KNeighborsClassifier(**(params or {})))

    def train(self, X_train, y_train):
        self.algorithm_instance.fit(X_train, y_train)

class NeuralNetworkModel(BaseModel):
    def __init__(self, params=None):
        super().__init__("NeuralNetwork", MLPClassifier(**(params or {})))

    def train(self, X_train, y_train):
        self.algorithm_instance.fit(X_train, y_train)

class XGBoostModel(BaseModel):
    def __init__(self, params=None):
        from xgboost import XGBClassifier  # Make sure to install xgboost
        super().__init__("XGBoost", XGBClassifier(**(params or {})))

    def train(self, X_train, y_train):
        self.algorithm_instance.fit(X_train, y_train)

# ... (other models)


class EnsembleStrategy(ABC):
    @abstractmethod
    def combine(self, predictions):
        pass

# --- Optimizers ---
class Optimizer(ABC):
    @abstractmethod
    def optimize(self, model, X_train, y_train):
        pass

class OptimizerFactory:
    @staticmethod
    def create_optimizer(optimizer_name, **kwargs):
        """
        Factory method to create optimizer instances.

        :param optimizer_name: Name of the optimizer.
        :param kwargs: Additional keyword arguments specific to each optimizer.
        :return: An instance of the requested optimizer.
        """

        if optimizer_name == "GridSearchOptimizer":
            return GridSearchOptimizer(param_grid=kwargs.get('param_grid', {}))
        elif optimizer_name == "RandomSearchOptimizer":
            return RandomSearchOptimizer(param_distributions=kwargs.get('param_distributions', {}),
                                         n_iter=kwargs.get('n_iter', 100))
        elif optimizer_name == "BayesianOptimizer":
            return BayesianOptimizer(param_space=kwargs.get('param_space', {}),
                                     n_iter=kwargs.get('n_iter', 50))
        # Add similar elif blocks for other optimizers with their specific parameters
        elif optimizer_name == "PSOOptimizer":
            # Add required parameters for PSOOptimizer
            return PSOOptimizer(**kwargs)
        elif optimizer_name == "SimulatedAnnealingOptimizer":
            # Add required parameters for SimulatedAnnealingOptimizer
            return SimulatedAnnealingOptimizer(**kwargs)
        elif optimizer_name == "TPEOptimizer":
            return TPEOptimizer(space=kwargs.get('space', {}))
        elif optimizer_name == "OptunaCMAESOptimizer":
            return OptunaCMAESOptimizer(objective_function=kwargs.get('objective_function'))
        elif optimizer_name == "DEAPGAOptimizer":
            return DEAPGAOptimizer(objective_function=kwargs.get('objective_function'),
                                   toolbox=kwargs.get('toolbox'))
        elif optimizer_name == "OptunaTPEOptimizer":
            return OptunaTPEOptimizer(objective_function=kwargs.get('objective_function'))
        else:
            raise ValueError(f"Optimizer '{optimizer_name}' not recognized.")


# Example usage


class RandomForest(BaseModel):
    def __init__(self, params=None):
        if not params:
            super().__init__("RandomForest", RandomForestRegressor())
        else:
            super().__init__("RandomForest", RandomForestRegressor(**params))
    
    def is_temporal():
        return False

    def train(self, X_train, y_train):
        self.algorithm_instance.fit(X_train, y_train)


class GridSearchOptimizer(Optimizer):
    def __init__(self, param_grid):
        self.param_grid = param_grid

    def optimize(self, model, X_train, y_train):
        grid_search = GridSearchCV(model.algorithm_instance, self.param_grid)
        grid_search.fit(X_train, y_train)
        model.algorithm_instance = grid_search.best_estimator_

class RandomSearchOptimizer(Optimizer):
    def __init__(self, param_distributions, n_iter=100):
        self.param_distributions = param_distributions
        self.n_iter = n_iter

    def optimize(self, model, X_train, y_train):
        random_search = RandomizedSearchCV(model.algorithm_instance, self.param_distributions, n_iter=self.n_iter)
        random_search.fit(X_train, y_train)
        model.algorithm_instance = random_search.best_estimator_

class BayesianOptimizer(Optimizer):
    def __init__(self, param_space, n_iter=50):
        self.param_space = param_space
        self.n_iter = n_iter

    def optimize(self, model, X_train, y_train):
        bayes_search = BayesSearchCV(model.algorithm_instance, self.param_space, n_iter=self.n_iter)
        bayes_search.fit(X_train, y_train)
        model.algorithm_instance = bayes_search.best_estimator_


class PSOOptimizer(Optimizer):
    def optimize(self, model, X_train, y_train):
        # Define a loss function to minimize. Note: This is a simple illustrative implementation and might need adjustments for real-world tasks.
        def loss_function(params):
            model.algorithm_instance.set_params(**params)
            model.algorithm_instance.fit(X_train, y_train)
            # Here, -1 is used because we want to maximize accuracy (or minimize -accuracy)
            return -1 * model.algorithm_instance.score(X_train, y_train)

        lb = [0.001, 1] # lower bounds for parameters, e.g., learning rate and regularization
        ub = [0.1, 100] # upper bounds 

        xopt, fopt = pso(loss_function, lb, ub)
        optimal_parameters = dict(zip(['param1', 'param2'], xopt))
        model.algorithm_instance.set_params(**optimal_parameters)

class SimulatedAnnealingOptimizer(Optimizer):
    def optimize(self, model, X_train, y_train):
        def loss_function(params):
            model.algorithm_instance.set_params(**params)
            model.algorithm_instance.fit(X_train, y_train)
            return -1 * model.algorithm_instance.score(X_train, y_train)

        result = minimize(loss_function, [0.01, 10], method='SLSQP')
        optimal_parameters = dict(zip(['param1', 'param2'], result.x))
        model.algorithm_instance.set_params(**optimal_parameters)
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

class TPEOptimizer(Optimizer):
    def __init__(self, space):
        # space is the search space definition using hyperopt's hp module
        self.space = space

    def optimize(self, model, X_train, y_train):
        def objective(params):
            model.algorithm_instance.set_params(**params)
            model.algorithm_instance.fit(X_train, y_train)
            loss = -1 * model.algorithm_instance.score(X_train, y_train)
            return {'loss': loss, 'status': STATUS_OK}

        trials = Trials()
        best = fmin(fn=objective,
                    space=self.space,
                    algo=tpe.suggest,
                    max_evals=100,
                    trials=trials)

        model.algorithm_instance.set_params(**best)

class OptunaCMAESOptimizer(Optimizer):
    def __init__(self, objective_function):
        self.objective_function = objective_function

    def optimize(self, model, X_train, y_train):
        sampler = optuna.samplers.CmaEsSampler()
        study = optuna.create_study(sampler=sampler, direction='minimize')
        study.optimize(self.objective_function, n_trials=100)
        
        best_params = study.best_params
        model.algorithm_instance.set_params(**best_params)
from deap import base, creator, tools, algorithms

class DEAPGAOptimizer:
    def __init__(self, objective_function, toolbox):
        self.objective_function = objective_function
        self.toolbox = toolbox  # DEAP's toolbox

    def optimize(self, model, X_train, y_train):
        # Assuming objective_function takes in an individual from the population
        # and evaluates it based on X_train and y_train
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        
        # Sample a population and evolve it using genetic algorithm
        population = self.toolbox.population(n=100)
        algorithms.eaSimple(population, self.toolbox, cxpb=0.5, mutpb=0.2, ngen=100)
        
        # Extract the best individual's parameters and set them to the model
        best_individual = tools.selBest(population, k=1)[0]
        best_params = dict(zip(model.parameters_keys, best_individual))
        model.algorithm_instance.set_params(**best_params)


import optuna

class OptunaTPEOptimizer:
    def __init__(self, objective_function):
        self.objective_function = objective_function

    def optimize(self, model, X_train, y_train):
        # TPE is the default sampler in Optuna
        study = optuna.create_study(direction='minimize')
        study.optimize(self.objective_function, n_trials=100)
        
        best_params = study.best_params
        model.algorithm_instance.set_params(**best_params)


from sklearn.svm import SVC

class SVM(BaseModel):
    def __init__(self, params=None):
        super().__init__("SVM", SVC(**params if params else {}))

    def train(self, X_train, y_train):
        self.algorithm_instance.fit(X_train, y_train)
        
from sklearn.ensemble import GradientBoostingClassifier

class GBM(BaseModel):
    def __init__(self, params=None):
        super().__init__("GBM", GradientBoostingClassifier(**params if params else {}))

    def train(self, X_train, y_train):
        self.algorithm_instance.fit(X_train, y_train)

from sklearn.linear_model import LogisticRegression

class LogisticRegressionModel(BaseModel):
    def __init__(self, params=None):
        super().__init__("LogReg", LogisticRegression(**params if params else {}))

    def train(self, X_train, y_train):
        self.algorithm_instance.fit(X_train, y_train)

from sklearn.neighbors import KNeighborsClassifier

class KNN(BaseModel):
    def __init__(self, params=None):
        super().__init__("KNN", KNeighborsClassifier(**params if params else {}))

    def train(self, X_train, y_train):
        self.algorithm_instance.fit(X_train, y_train)

from sklearn.neural_network import MLPClassifier

class NeuralNetwork(BaseModel):
    def __init__(self, params=None):
        super().__init__("NeuralNetwork", MLPClassifier(**params if params else {}))

    def train(self, X_train, y_train):
        self.algorithm_instance.fit(X_train, y_train)

from xgboost import XGBClassifier

class XGBoost(BaseModel):
    def __init__(self, params=None):
        super().__init__("XGBoost", XGBClassifier(**params if params else {}))

    def train(self, X_train, y_train):
        self.algorithm_instance.fit(X_train, y_train)
# Ridge Regression Model
class RidgeRegressionModel(BaseModel):
    def __init__(self, params=None):
        super().__init__("RidgeRegression", Ridge(**(params or {})))

    def train(self, X_train, y_train):
        self.algorithm_instance.fit(X_train, y_train)

# Lasso Regression Model
class LassoRegressionModel(BaseModel):
    def __init__(self, params=None):
        super().__init__("LassoRegression", Lasso(**(params or {})))

    def train(self, X_train, y_train):
        self.algorithm_instance.fit(X_train, y_train)

# Elastic Net Regression Model
class ElasticNetRegressionModel(BaseModel):
    def __init__(self, params=None):
        super().__init__("ElasticNetRegression", ElasticNet(**(params or {})))

    def train(self, X_train, y_train):
        self.algorithm_instance.fit(X_train, y_train)

# Bayesian Ridge Regression Model
class BayesianRidgeRegressionModel(BaseModel):
    def __init__(self, params=None):
        super().__init__("BayesianRidgeRegression", BayesianRidge(**(params or {})))

    def train(self, X_train, y_train):
        self.algorithm_instance.fit(X_train, y_train)

# ARD Regression Model
class ARDRegressionModel(BaseModel):
    def __init__(self, params=None):
        super().__init__("ARDRegression", ARDRegression(**(params or {})))

    def train(self, X_train, y_train):
        self.algorithm_instance.fit(X_train, y_train)

# ... (implement other regression models in a similar fashion)

# SGD Regressor Model
class SGDRegressorModel(BaseModel):
    def __init__(self, params=None):
        super().__init__("SGDRegressor", SGDRegressor(**(params or {})))

    def train(self, X_train, y_train):
        self.algorithm_instance.fit(X_train, y_train)

# ... (implement any other specific models you need)


# Additional regression models
class SVRModel(BaseModel):
    def __init__(self, params=None):
        super().__init__("SVR", SVR(**(params or {})))

    def train(self, X_train, y_train):
        self.algorithm_instance.fit(X_train, y_train)

# Passive Aggressive Regressor Model
class PassiveAggressiveRegressorModel(BaseModel):
    def __init__(self, params=None):
        super().__init__("PassiveAggressiveRegressor", PassiveAggressiveRegressor(**(params or {})))

    def train(self, X_train, y_train):
        self.algorithm_instance.fit(X_train, y_train)

# Huber Regressor Model
class HuberRegressorModel(BaseModel):
    def __init__(self, params=None):
        super().__init__("HuberRegressor", HuberRegressor(**(params or {})))

    def train(self, X_train, y_train):
        self.algorithm_instance.fit(X_train, y_train)

# Theil-Sen Regressor Model
class TheilSenRegressorModel(BaseModel):
    def __init__(self, params=None):
        super().__init__("TheilSenRegressor", TheilSenRegressor(**(params or {})))

    def train(self, X_train, y_train):
        self.algorithm_instance.fit(X_train, y_train)

# RANSAC Regressor Model
class RANSACRegressorModel(BaseModel):
    def __init__(self, params=None):
        super().__init__("RANSACRegressor", RANSACRegressor(**(params or {})))

    def train(self, X_train, y_train):
        self.algorithm_instance.fit(X_train, y_train)

# Orthogonal Matching Pursuit Model
class OrthogonalMatchingPursuitModel(BaseModel):
    def __init__(self, params=None):
        super().__init__("OrthogonalMatchingPursuit", OrthogonalMatchingPursuit(**(params or {})))

    def train(self, X_train, y_train):
        self.algorithm_instance.fit(X_train, y_train)

# Lars Model
class LarsModel(BaseModel):
    def __init__(self, params=None):
        super().__init__("Lars", Lars(**(params or {})))

    def train(self, X_train, y_train):
        self.algorithm_instance.fit(X_train, y_train)

# Lasso Lars Model
class LassoLarsModel(BaseModel):
    def __init__(self, params=None):
        super().__init__("LassoLars", LassoLars(**(params or {})))

    def train(self, X_train, y_train):
        self.algorithm_instance.fit(X_train, y_train)

from sklearn.linear_model import LinearRegression

# Linear Regression Model
class LinearRegressionModel(BaseModel):
    def __init__(self, params=None):
        super().__init__("LinearRegression", LinearRegression(**(params or {})))

    def train(self, X_train, y_train):
        self.algorithm_instance.fit(X_train, y_train)


# K-Neighbors Regressor Model
class KNeighborsRegressorModel(BaseModel):
    def __init__(self, params=None):
        super().__init__("KNeighborsRegressor", KNeighborsRegressor(**(params or {})))

    def train(self, X_train, y_train):
        self.algorithm_instance.fit(X_train, y_train)

# Decision Tree Regressor Model
class DecisionTreeRegressorModel(BaseModel):
    def __init__(self, params=None):
        super().__init__("DecisionTreeRegressor", DecisionTreeRegressor(**(params or {})))

    def train(self, X_train, y_train):
        self.algorithm_instance.fit(X_train, y_train)

# Decision Tree Classifier Model
class DecisionTreeClassifierModel(BaseModel):
    def __init__(self, params=None):
        super().__init__("DecisionTreeClassifier", DecisionTreeClassifier(**(params or {})))

    def train(self, X_train, y_train):
        self.algorithm_instance.fit(X_train, y_train)

# ... (add any other specific model classes if needed)


# Additional classification models
class SVCModel(BaseModel):
    def __init__(self, params=None):
        super().__init__("SVC", SVC(**(params or {})))

    def train(self, X_train, y_train):
        self.algorithm_instance.fit(X_train, y_train)

class LogisticRegressionClassifierModel(BaseModel):
    def __init__(self, params=None):
        super().__init__("LogisticRegressionClassifier", LogisticRegression(**(params or {})))

    def train(self, X_train, y_train):
        self.algorithm_instance.fit(X_train, y_train)

class GaussianNaiveBayesModel(BaseModel):
    def __init__(self, params=None):
        super().__init__("GaussianNaiveBayes", GaussianNB(**(params or {})))

    def train(self, X_train, y_train):
        self.algorithm_instance.fit(X_train, y_train)

class ThresholdVotingStrategy(EnsembleStrategy):
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        
    def combine(self, predictions):
        """Assumes predictions are probability values."""
        mean_prob = sum(predictions) / len(predictions)
        return 1 if mean_prob > self.threshold else 0

class BordaCountStrategy(EnsembleStrategy):
    def combine(self, predictions):
        """Assumes each classifier ranks each class."""
        scores = defaultdict(int)
        for prediction in predictions:
            for rank, class_ in enumerate(prediction):
                scores[class_] += rank
        return min(scores, key=scores.get)

class SoftVotingStrategy(EnsembleStrategy):
    def combine(self, predictions):
        """Assumes predictions are probability values."""
        summed = sum(predictions)
        avg = summed / len(predictions)
        return round(avg)

class MaxVotingStrategy(EnsembleStrategy):
    def combine(self, predictions):
        flattened_predictions = [pred for sublist in predictions for pred in sublist]
        return max(set(flattened_predictions), key=flattened_predictions.count)

class MinVotingStrategy(EnsembleStrategy):
    def combine(self, predictions):
        flattened_predictions = [pred for sublist in predictions for pred in sublist]
        return min(set(flattened_predictions), key=flattened_predictions.count)

class ProductStrategy(EnsembleStrategy):
    def combine(self, predictions):
        """Assumes predictions are probability values between 0 and 1."""
        flattened_predictions = [pred for sublist in predictions for pred in sublist]
        product = 1
        for pred in flattened_predictions:
            product *= pred
        return product

class RankAveragingStrategy(EnsembleStrategy):
    def combine(self, predictions):
        """Assumes classifiers return a ranking for each class."""
        averaged_rank = [sum(rank) / len(rank) for rank in zip(*predictions)]
        return averaged_rank.index(min(averaged_rank))

# Stacking would need a separate implementation since it would involve training another model.


class MajorityVoteStrategy(EnsembleStrategy):
    def combine(self, predictions):
        """Return the mode (most common) prediction."""
        flattened_predictions = [pred for sublist in predictions for pred in sublist]
        return Counter(flattened_predictions).most_common(1)[0][0]

class AverageStrategy(EnsembleStrategy):
    def combine(self, predictions):
        """Assumes predictions are continuous values."""
        flattened_predictions = [pred for sublist in predictions for pred in sublist]
        return sum(flattened_predictions) / len(flattened_predictions)
    

MODEL_REGISTRY = {
    'RandomForestRegressor': {
        'class': RandomForestRegressorModel,
        'param_grid': {'n_estimators': [100, 200], 'max_depth': [5, 10, None]},
        'optimizer_type': 'RandomSearchOptimizer'
    },
    'DecisionTreeRegressor': {
        'class': DecisionTreeRegressorModel,
        'param_grid': {'max_depth': [5, 10, None], 'min_samples_split': [2, 5, 10]},
        'optimizer_type': 'GridSearchOptimizer'
    },
    'SVR': {
        'class': SVRModel,
        'param_grid': {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']},
        'optimizer_type': 'GridSearchOptimizer'
    },
    'GradientBoostingRegressor': {
        'class': GradientBoostingRegressorModel,
        'param_grid': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5, 7]},
        'optimizer_type': 'RandomSearchOptimizer'
    },
    'AdaBoostRegressor': {
        'class': AdaBoostRegressorModel,
        'param_grid': {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]},
        'optimizer_type': 'GridSearchOptimizer'
    },
    'LinearRegression': {
        'class': LinearRegressionModel,
        'param_grid': {},  # Linear Regression typically does not need hyperparameter tuning
        'optimizer_type': None
    },
    'LogisticRegression': {
        'class': LogisticRegressionModel,
        'param_grid': {'C': [0.1, 1, 10], 'penalty': ['l2']},
        'optimizer_type': 'GridSearchOptimizer'
    },
    'KNeighborsRegressor': {
        'class': KNeighborsRegressorModel,
        'param_grid': {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']},
        'optimizer_type': 'GridSearchOptimizer'
    },
    'RidgeRegression': {
        'class': RidgeRegressionModel,
        'param_grid': {'alpha': [0.1, 1, 10]},
        'optimizer_type': 'GridSearchOptimizer'
    },
    'LassoRegression': {
        'class': LassoRegressionModel,
        'param_grid': {'alpha': [0.1, 1, 10]},
        'optimizer_type': 'GridSearchOptimizer'
    },
    'ElasticNetRegression': {
        'class': ElasticNetRegressionModel,
        'param_grid': {'alpha': [0.1, 1, 10], 'l1_ratio': [0.2, 0.5, 0.8]},
        'optimizer_type': 'GridSearchOptimizer'
    },
    'SGDRegressor': {
        'class': SGDRegressorModel,
        'param_grid': {'penalty': ['l2', 'l1', 'elasticnet'], 'alpha': [0.0001, 0.001, 0.01]},
        'optimizer_type': 'RandomSearchOptimizer'
    },
    #'XGBoostRegressor': {
      #  'class': XGBRegressor,
      #  'param_grid': {'n_estimators': [100, 200], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.1]},
      #  'optimizer_type': 'RandomSearch'
  #  },
    'RandomForestClassifier': {
        'class': RandomForestClassifierModel,
        'param_grid': {'n_estimators': [100, 200], 'max_depth': [5, 10, None]},
        'optimizer_type': 'RandomSearchOptimizer'
    },
    'GradientBoostingClassifier': {
        'class': GradientBoostingClassifierModel,
        'param_grid': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5, 7]},
        'optimizer_type': 'RandomSearchOptimizer'
    },
    'AdaBoostClassifier': {
        'class': AdaBoostClassifierModel,
        'param_grid': {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]},
        'optimizer_type': 'GridSearchOptimizer'
    },
    'SVMClassifier': {
        'class': SVCModel,
        'param_grid': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf', 'poly']},
        'optimizer_type': 'GridSearchOptimizer'
    },
    'KNeighborsClassifier': {
        'class': KNNModel,
        'param_grid': {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']},
        'optimizer_type': 'GridSearchOptimizer'
    },
    'GaussianNaiveBayes': {
        'class': GaussianMixtureModel,
        'param_grid': {},  # GaussianNB typically does not require hyperparameter tuning
        'optimizer_type': None
    },
}

class DataLoader:
    @staticmethod
    def fetch_stock_data_from_yahoo(symbol, start_date, end_date, interval="1d"):
        try:
            logging.info(f"Attempting to fetch data for symbol: {symbol} from {start_date} to {end_date} with interval {interval}")
            df = yf.download(symbol, start=start_date, end=end_date, interval=interval)
            
            # Ensure data is sorted in ascending order by 'Date'
            df.sort_values(by='Date', inplace=True)

            df.reset_index(inplace=True)  # Reset the index
            df['Formatted_Date'] = df['Date'].dt.strftime("%Y-%m-%d")  # Format the 'Date' column as 'YYYY-MM-DD'
            
            logging.info(f"Data fetched successfully for symbol: {symbol}. Shape: {df.shape}")
            return df
        except Exception as e:
            logging.error(f"Error fetching data for symbol {symbol}: {str(e)}")
            raise


# --- Model Builder ---
class ModelBuilder:
    def __init__(self, model_class, optimizer=None):
        self.model_class = model_class
        self.optimizer = optimizer
        logging.info(f"model class: { self.model_class}. optiizer class: { self.optimizer}")

    def build(self, X_train, y_train):
        if isinstance(X_train, (list, tuple)):
            print(f"[ModelBuilder.build] Before any operations - X_train length: {len(X_train)}, y_train length: {len(y_train)}")
        else:
            # Assuming they are numpy arrays or pandas data structures
            print(f"[ModelBuilder.build] Before any operations - X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        logging.info(f"model class: { self.model_class}. optiizer class: { self.optimizer}")

        model = self.model_class()
        
        if self.optimizer:
            self.optimizer.optimize(model, X_train, y_train)
        else:
            model.train(X_train, y_train)

        return model



class Ensemble:
    def __init__(self, models, strategy=None):
        if not models:
            raise ValueError("At least one model must be provided.")
        if isinstance(models, dict):
            self.models = list(models.values())
        else:
            self.models = models  # We ensure that models are stored as a list.
        
        if strategy:
            self.strategy = strategy
        else:
            self.strategy = AverageStrategy()

    def train(self, X_train, y_train):
        for model in self.models:
            model.train(X_train, y_train)

    def predict(self, X):
        if len(self.models) == 1:
            return self.models[0].predict(X)  # Accessing the first model directly, since it's a list.

        predictions = [model.predict(X) for model in self.models]
        return self.strategy.combine(predictions)

    def run(self):
        try:
            X, y = self.prep()
            X_train, X_val, X_test, y_train, y_val, y_test = sequential_split(X, y)

            self.models = [builder.build(X_train, y_train) for builder in self.model_builders]
            self.ensemble = Ensemble(self.models)
            self.ensemble.train(X_train, y_train)

            val_predictions = self.ensemble.predict(X_val)
            mae_val = mean_absolute_error(y_val, val_predictions)
            logging.info(f"Validation Mean Absolute Error: {mae_val}")

            test_predictions = self.ensemble.predict(X_test)
            mae_test = mean_absolute_error(y_test, test_predictions)
            logging.info(f"Test Mean Absolute Error: {mae_test}")

        except Exception as e:
            logging.error(f"Error in Pipeline run: {e}")
            logging.error(traceback.format_exc())
            return None

        return test_predictions

    def begin_predict(self, new_data):             
        return self.ensemble.predict(new_data)


      
class DataPreparation:
    LOOKBACK_PERIOD = 14
    PREDICTION_PERIOD = 3

    def __init__(self, dataset, indicators):
        self.dataset = dataset
        self.indicators = indicators  # List of tuples, e.g., [('SMA', 14), ('EMA', 10)]
        self.calculate_technical_indicators()

    def calculate_sma(self, data, window):
        return data['Close'].rolling(window=window, min_periods=1).mean()

    def calculate_ema(self, data, window):
        return data['Close'].ewm(span=window, adjust=False).mean()
    
    def calculate_rsi(self, data, period=14):
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_macd(self, data, short_period=12, long_period=26, signal_period=9):
        short_ema = data['Close'].ewm(span=short_period, adjust=False).mean()
        long_ema = data['Close'].ewm(span=long_period, adjust=False).mean()
        macd = short_ema - long_ema
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        return macd, signal

    def calculate_bollinger_bands(self, data, period=20, std_dev=2):
        sma = data['Close'].rolling(window=period).mean()
        rstd = data['Close'].rolling(window=period).std()
        upper_band = sma + std_dev * rstd
        lower_band = sma - std_dev * rstd
        return upper_band, lower_band

    def calculate_technical_indicators(self):
        for indicator_name, params in self.indicators:
            if indicator_name == 'SMA':
                self.dataset[f'{indicator_name}_{params}'] = self.calculate_sma(self.dataset, params)
            elif indicator_name == 'EMA':
                self.dataset[f'{indicator_name}_{params}'] = self.calculate_ema(self.dataset, params)
            elif indicator_name == 'RSI':
                self.dataset[f'{indicator_name}_{params}'] = self.calculate_rsi(self.dataset, params)
            elif indicator_name == 'MACD':
                macd, signal = self.calculate_macd(self.dataset, *params)
                self.dataset[f'MACD_{params}'] = macd
                self.dataset[f'MACDSignal_{params}'] = signal
            elif indicator_name == 'Bollinger':
                upper_band, lower_band = self.calculate_bollinger_bands(self.dataset, *params)
                self.dataset[f'BollingerUpper_{params}'] = upper_band
                self.dataset[f'BollingerLower_{params}'] = lower_band
            # Add conditions for other indicators here...

    def process_day(self, i):
        try:
            current_date = self.dataset.iloc[i]['Date']
            historical_data = self.dataset.iloc[i - self.LOOKBACK_PERIOD:i + 1]

            if len(historical_data) < self.LOOKBACK_PERIOD + 1:
                return None

            future_data = self.dataset.iloc[i:i + self.PREDICTION_PERIOD]
            historical_data = historical_data.sort_values(by='Date', ascending=False)
            future_data = future_data.sort_values(by='Date')

            # Prepare the historical and future data entries
            historical_entries = []
            for idx, row in historical_data.iterrows():
                entry = {
                    'Date': row['Date'].strftime("%Y-%m-%d"),
                    'Open': row['Open'],
                    'High': row['High'],
                    'Low': row['Low'],
                    'Volume': row['Volume'],
                    'Close': row['Close']
                }
                # Add technical indicators to the entry
                for indicator_name, params in self.indicators:
                    if indicator_name in ['SMA', 'EMA', 'RSI']:
                        entry[f'{indicator_name}_{params}'] = row[f'{indicator_name}_{params}']
                    elif indicator_name == 'MACD':
                        entry[f'MACD_{params}'] = row[f'MACD_{params}']
                        entry[f'MACDSignal_{params}'] = row[f'MACDSignal_{params}']
                    elif indicator_name == 'Bollinger':
                        entry[f'BollingerUpper_{params}'] = row[f'BollingerUpper_{params}']
                        entry[f'BollingerLower_{params}'] = row[f'BollingerLower_{params}']
                historical_entries.append(entry)

            # Prepare the future data entries
            future_entries = [{'Date': row['Date'].strftime("%Y-%m-%d"), 'Close': row['Close']}
                            for _, row in future_data.iterrows()]

            day_data = {
                'Current Date': current_date.strftime("%Y-%m-%d"),
                'Historical Data': historical_entries,
                'Future Data': future_entries,
                'Target Prices': future_data['Close'].tolist()[:self.PREDICTION_PERIOD]
            }

            return day_data if len(day_data['Target Prices']) == self.PREDICTION_PERIOD else None

        except Exception as e:
            print(f"Error processing data for {current_date}: {str(e)}")
            return None

    def extract_features_and_labels(self, processed_data, is_temporal_model):
            features = []
            labels = []

            for day_data in processed_data:
                historical_data = day_data['Historical Data']
                
                feature_vector = []
                for row in historical_data:
                    # Basic features: Open, High, Low, Close, Volume
                    basic_features = [
                        row['Open'], 
                        row['High'], 
                        row['Low'], 
                        row['Volume'],
                        row['Close']
                    ]

                    # Extracting technical indicators dynamically
                    indicator_features = []
                    for indicator_name, params in self.indicators:
                        if indicator_name in ['SMA', 'EMA', 'RSI']:
                            # Handle single parameter indicators
                            indicator_key = f'{indicator_name}_{params}'
                            indicator_features.append(row.get(indicator_key, 0))
                        elif indicator_name == 'MACD':
                            # Handle multi-parameter indicators like MACD
                            macd_key = f'MACD_{params}'
                            signal_key = f'MACDSignal_{params}'
                            indicator_features.extend([
                                row.get(macd_key, 0),
                                row.get(signal_key, 0)
                            ])
                        elif indicator_name == 'Bollinger':
                            upper_key = f'BollingerUpper_{params}'
                            lower_key = f'BollingerLower_{params}'
                            indicator_features.extend([
                                row.get(upper_key, 0),
                                row.get(lower_key, 0)
                            ])
                    
                    # Combine basic and indicator features
                    combined_features = basic_features + indicator_features
                    feature_vector.append(combined_features)

                # Flatten feature vector if the model is not temporal
                if not is_temporal_model:
                    feature_vector = [item for sublist in feature_vector for item in sublist]

                features.append(feature_vector)
                labels.append(day_data['Target Prices'][0])

            print(f"[extract_features_and_labels] features: {len(features)}, labels: {len(labels)}")
            return features, labels

    def process_days_concurrently(self):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(self.process_day, range(len(self.dataset))))

        # Filter out None values (skipped days)
        processed_data = [result for result in results if result is not None]

        return processed_data


    def prepare_data_structure(self):
        data_structure = []

        for i in range(self.LOOKBACK_PERIOD, len(self.dataset) - self.PREDICTION_PERIOD):
            day_data = self.process_day(i)
            data_structure.append(day_data)

        data_structure.sort(key=lambda x: x['Date'])

        print("Data structure preparation completed.")
        print(f"Data structure:\n{data_structure[:5]}")  # Print the first 5 entries for illustration
        return data_structure

class Pipeline2:
    def __init__(self, stocks_info):
        self.stocks_info = [(symbol, self.adjust_start_date(user_start_date, lookback_period), end_date, lookback_period, prediction_period, model_builders)
                            for symbol, user_start_date, end_date, lookback_period, prediction_period, model_builders in stocks_info]
        self.models = {}
        self.ensemble = None

    @staticmethod
    def adjust_start_date(user_start_date_str, lookback_period):
        user_start_date = pd.to_datetime(user_start_date_str)
        dates = pd.date_range(end=user_start_date, periods=lookback_period*2, freq='B')
        system_start_date = dates[0].date()
        print(f"[adjust_start_date] system_start_date: {system_start_date}")
        return system_start_date

    def load_data_serially(self):
        data_tuples = []
        for stock_info in self.stocks_info:
            print(f"[load_data_serially] Processing stock_info: {stock_info}")
            symbol, user_start_date, end_date, lookback_period, prediction_period, model_builder = stock_info
            system_start_date = self.adjust_start_date(user_start_date, lookback_period)
            dataset = DataLoader.fetch_stock_data_from_yahoo(symbol, system_start_date, end_date)
            if dataset is not None:
                data_tuples.append((stock_info, dataset))
        return data_tuples

    def load_data_concurrently(self):
        loaded_data = []
        with ThreadPoolExecutor() as executor:
            futures = {}
            for stock_info in self.stocks_info:
                symbol, user_start_date, end_date, lookback_period, prediction_period, model_builder = stock_info
                system_start_date = self.adjust_start_date(user_start_date, lookback_period)
                future = executor.submit(DataLoader.fetch_stock_data_from_yahoo, symbol, system_start_date, end_date)
                futures[future] = stock_info

            for future in futures:
                stock_info = futures[future]
                try:
                    dataset = future.result()
                    if dataset is not None:
                        loaded_data.append((stock_info, dataset))
                except Exception as e:
                    print(f"Error fetching data for {stock_info[0]}: {e}")

        return loaded_data

    def prep_data(self, dataset):
        indicators = [
            ('SMA', 14), 
            ('EMA', 10),
            ('RSI', 14),
            ('MACD', (12, 26, 9)),  # Note that MACD has three parameters
            ('Bollinger', (20, 2))  # Bollinger Bands have two parameters
        ]
        data_prep = DataPreparation(dataset, indicators)
        data_structure = data_prep.process_days_concurrently()
        print(f"[prep_data] Prepared data structure length: {len(data_structure)}")
        return data_structure

    def prepare_data_concurrently(self, data_tuples):
        prepped_data = []
        with ThreadPoolExecutor() as executor:
            for stock_info, dataset in data_tuples:
                future = executor.submit(self.prep_data, dataset)
                prepped_data.append((stock_info, future))

        prepped_data = [(stock_info, future.result()) for stock_info, future in prepped_data]
        return prepped_data
    

    def train_and_evaluate(self, stock_info, X_train, y_train, X_val, y_val, X_test, y_test):
        """Trains models and evaluates their performance on validation and test data."""
        symbol, _, _, _, _, model_builders = stock_info

        try:
            # Initialize an empty list for storing models for the current symbol
            self.models[symbol] = []

            # Iterate over each model_builder and build the models
            for model_builder in model_builders:
                model = model_builder.build(X_train, y_train)
                self.models[symbol].append(model)

            # Create an ensemble with all the built models
            all_models = [model for models in self.models.values() for model in models]
            self.ensemble = Ensemble(all_models)

            # Train the ensemble model
            self.ensemble.train(X_train, y_train)

            # Evaluate on validation and test data
            val_predictions = self.ensemble.predict(X_val)
            mae_val = mean_absolute_error(y_val, val_predictions)
            mse_val = mean_squared_error(y_val, val_predictions)
            r2_val = r2_score(y_val, val_predictions)
            mape_val = mean_absolute_percentage_error(y_val, val_predictions)

            test_predictions = self.ensemble.predict(X_test)
            mae_test = mean_absolute_error(y_test, test_predictions)
            mse_test = mean_squared_error(y_test, test_predictions)
            r2_test = r2_score(y_test, test_predictions)
            mape_test = mean_absolute_percentage_error(y_test, test_predictions)

            # Log the evaluation metrics
            logging.info(f"Validation metrics for {symbol} - MAE: {mae_val}, MSE: {mse_val}, R2: {r2_val}, MAPE: {mape_val}")
            logging.info(f"Test metrics for {symbol} - MAE: {mae_test}, MSE: {mse_test}, R2: {r2_test}, MAPE: {mape_test}")

            # Log the test predictions
            logging.info(f"Test predictions for {symbol}: {test_predictions}")

            return test_predictions, mae_val, mae_test, mse_val, mse_test, r2_val, r2_test, mape_val, mape_test

        except Exception as e:
            logging.error(f"Error in Pipeline run for {symbol}: {e}")
            logging.error(traceback.format_exc())

        return None, None, None, None, None, None, None, None, None  # Return None values if there is an error




    def extract_features_and_labels(self, processed_data, is_temporal_model):
        features = []
        labels = []

        for day_data in processed_data:
            historical_data = day_data['Historical Data']
            feature_vector = [[historical_datum['SMA_14'], historical_datum['Open'], 
                            historical_datum['High'], historical_datum['Low'], 
                            historical_datum['Volume']] 
                            for historical_datum in historical_data]

            if not is_temporal_model:
                feature_vector = [item for sublist in feature_vector for item in sublist]

            features.append(feature_vector)
            labels.append(day_data['Target Prices'][0])

        print(f"[extract_features_and_labels] features: {len(features)}, labels: {len(labels)}")
        return features, labels

    def time_series_split(self, data, train_size, val_size):
        n = len(data)
        train_end = int(train_size * n)
        val_end = int(val_size * n) + train_end

        train = data[:train_end]
        val = data[train_end:val_end]
        test = data[val_end:]

        print(f"[time_series_split] train: {len(train)}, val: {len(val)}, test: {len(test)}")
        return train, val, test

    def split_data(self, stock_info, data, is_temporal_model, train_size=0.7, val_size=0.15):
        train_data, val_data, test_data = self.time_series_split(data, train_size, val_size)

        X_train, y_train = self.extract_features_and_labels(train_data, is_temporal_model)
        print(f"[split_data] X_train: {len(X_train)}, y_train: {len(y_train)}")

        X_val, y_val = self.extract_features_and_labels(val_data, is_temporal_model)
        print(f"[split_data] X_val: {len(X_val)}, y_val: {len(y_val)}")

        X_test, y_test = self.extract_features_and_labels(test_data, is_temporal_model)
        print(f"[split_data] X_test: {len(X_test)}, y_test: {len(y_test)}")

        return stock_info, (X_train, y_train, X_val, y_val, X_test, y_test)

    def prepare_data_for_ml(self, X_train, X_val, X_test):
        X_train_flattened = [self.flatten_data(sample) for sample in X_train]
        X_val_flattened = [self.flatten_data(sample) for sample in X_val]
        X_test_flattened = [self.flatten_data(sample) for sample in X_test]

        print(f"[prepare_data_for_ml] X_train_flattened: {len(X_train_flattened)}, X_val_flattened: {len(X_val_flattened)}, X_test_flattened: {len(X_test_flattened)}")
        return X_train_flattened, X_val_flattened, X_test_flattened


    @staticmethod
    def flatten_data(data):
        return [item for sublist in data for item in sublist]

    def process_data_pipeline(self, load_concurrently=False, train_size=0.7, val_size=0.15):
        if load_concurrently:
            data_tuples = self.load_data_concurrently()
        else:
            data_tuples = self.load_data_serially()

        prepped_data = self.prepare_data_concurrently(data_tuples)
        test_predictions_list = []

        for stock_info, data in prepped_data:
            split_result = self.split_data(stock_info, data, train_size, val_size)
            si, (X_train, y_train, X_val, y_val, X_test, y_test) = split_result

            print(f"[process_data_pipeline] Data dimensions BEFORE preprocessing - X_train: {len(X_train)}, X_val: {len(X_val)}, X_test: {len(X_test)}")

            X_train, X_val, X_test = self.prepare_data_for_ml(X_train, X_val, X_test)

            print(f"[process_data_pipeline] Data dimensions AFTER preprocessing - X_train: {len(X_train)}, X_val: {len(X_val)}, X_test: {len(X_test)}")
            test_predictions, mae_val, mae_test, mse_val, mse_test, r2_val, r2_test, mape_val, mape_test = self.train_and_evaluate(si, X_train, y_train, X_val, y_val, X_test, y_test)
        # Now you can use these metrics as needed
            test_predictions_list.append((si, test_predictions, mae_val, mae_test, mse_val, mse_test, r2_val, r2_test, mape_val, mape_test))

        return test_predictions_list

    def begin_predict(self, new_data):             
        return self.ensemble.predict(new_data)


class WalkForwardValidation:
    def __init__(self, initial_train_size, validation_size, pipeline):
        self.initial_train_size = initial_train_size
        self.validation_size = validation_size
        self.pipeline = pipeline  # Instance of the Pipeline2 class
    
    def validate(self, data):
        performances = []
        predictions_list = []

        train_start = 0
        train_end = int(self.initial_train_size * len(data))
        val_end = train_end + int(self.validation_size * len(data))

        while val_end <= len(data):
            train_data = data[train_start:train_end]
            val_data = data[train_end:val_end]

            # Extract features and labels from train_data and val_data
            X_train, y_train = self.extract_features_labels(train_data)
            X_val, y_val = self.extract_features_labels(val_data)

            # Train and evaluate the model
            predictions = self.pipeline.train_and_evaluate(X_train, y_train, X_val, y_val, [], [])
            mae = mean_absolute_error(y_val, predictions)
            performances.append(mae)
            predictions_list.append(predictions)

            # Slide or expand the window
            train_end = val_end
            val_end = train_end + int(self.validation_size * len(data))

        return performances, predictions_list
    
    def extract_features_labels(self, data):
        # Define this method to appropriately extract features and labels from your data
        # This is just a placeholder and needs to be adapted based on your dataset
        X = [item['Historical Data'] for item in data]
        y = [item['Target Prices'] for item in data]
        return X, y







