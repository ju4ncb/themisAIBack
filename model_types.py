from sklearn.linear_model import LinearRegression # Lineal
from sklearn.preprocessing import PolynomialFeatures # Polinomial
from sklearn.svm import SVR # Regresión por soporte de vectores
from sklearn.tree import DecisionTreeRegressor # Árbol de decisión
from sklearn.ensemble import RandomForestRegressor # Random forest
from sklearn.linear_model import BayesianRidge # Regresión bayesiana
from sklearn.linear_model import SGDRegressor # Gradiente estocástico
from sklearn.gaussian_process import GaussianProcessRegressor # Proceso gaussiano

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

model_types = {
    "linear": LinearRegression(),
    "poly": make_pipeline(PolynomialFeatures(degree=2), LinearRegression()),
    "svr": make_pipeline(StandardScaler(), SVR(kernel='rbf')),
    "decision_tree": DecisionTreeRegressor(),
    "random_forest": RandomForestRegressor(),
    "bayesian": BayesianRidge(),
    "sgd": make_pipeline(StandardScaler(), SGDRegressor()),
    "gaussian": GaussianProcessRegressor()
}